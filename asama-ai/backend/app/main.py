from __future__ import annotations
import hashlib, hmac, json, os, re, sqlite3, time, uuid, ipaddress, socket, urllib.request, urllib.error
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl
from dotenv import load_dotenv

load_dotenv()
from . import intelligence
DB_PATH = os.getenv("AGENT_NET_DB", "agent_network.db")
DATABASE_URL = os.getenv("DATABASE_URL")
APP_ENV = os.getenv("APP_ENV", "production").lower()
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "local-development-secret-change-before-deploy")
if APP_ENV == "production" and (len(APP_SECRET_KEY)<32 or APP_SECRET_KEY in {"local-development-secret-change-before-deploy","generate-a-long-random-secret"}):
    raise RuntimeError("Set APP_SECRET_KEY to a random value of at least 32 characters before running in production")
app = FastAPI(title="Agent Network API", version="0.1.0", description="MVP agent network with explainable semantic routing")

# The hosted frontend uses a separate Vercel origin.
allowed_origins = [origin.strip().rstrip("/") for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=False, allow_methods=["GET", "POST", "PUT", "OPTIONS"], allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-Admin-Token"])

# Canonical intent families; broad campaign words have low weight and cannot route alone.
SEMANTICS = {
 "payments": {"payment","payments","upi","wallet","transaction","cashback","cash","pay"},
 "banking": {"bank","banking","credit","loan","lending","emi","finance","card","cards"},
 "commerce": {"commerce","ecommerce","shopping","retail","marketplace","merchant","store","sale","electronics","smartphone","appliance"},
 "food_delivery": {"food","delivery","restaurant","swiggy","meal","dining","takeaway"},
 "automotive": {"car","vehicle","automotive","auto","bike","motorcycle","mobility"},
 "travel": {"travel","flight","hotel","trip","tourism"},
 "festival": {"diwali","festival","seasonal","festive"},
 "promotion": {"offer","discount","promotion","deal","coupon","reward"},
}
WEIGHTS = {"payments":1.3,"banking":1.4,"commerce":1.4,"food_delivery":1.5,"automotive":1.5,"travel":1.4,"festival":0.7,"promotion":0.25}
STOP = {"the","and","for","with","from","this","that","during","all","when","will","your","our","are","has","have","into","over","above"}

class PostgresConnection:
    def __init__(self, url:str):
        import psycopg
        from psycopg.rows import dict_row
        if url.startswith("postgres://"): url="postgresql://"+url[len("postgres://"):]
        self.connection=psycopg.connect(url,connect_timeout=10,row_factory=dict_row)
    def __enter__(self): return self
    def __exit__(self,exc_type,exc,tb):
        try:
            self.connection.rollback() if exc_type else self.connection.commit()
        finally: self.connection.close()
    def execute(self,sql,params=()):
        # Application SQL uses SQLite-style qmark placeholders in both local and hosted modes.
        return self.connection.execute(sql.replace("?","%s"),params)

class SQLiteConnection:
    def __init__(self,path:str):
        self.connection=sqlite3.connect(path); self.connection.row_factory=sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
    def __enter__(self): return self
    def __exit__(self,exc_type,exc,tb):
        try:
            self.connection.rollback() if exc_type else self.connection.commit()
        finally: self.connection.close()
    def execute(self,sql,params=()): return self.connection.execute(sql,params)

def db():
    if DATABASE_URL: return PostgresConnection(DATABASE_URL)
    return SQLiteConnection(DB_PATH)

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents(id TEXT PRIMARY KEY, company_name TEXT NOT NULL, industry TEXT NOT NULL, webhook_url TEXT, webhook_secret TEXT NOT NULL, verified INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1, manifest TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS posts(id TEXT PRIMARY KEY, author_id TEXT NOT NULL REFERENCES agents(id), title TEXT NOT NULL, content TEXT NOT NULL, category TEXT, created_at TEXT NOT NULL, expires_at TEXT);
CREATE TABLE IF NOT EXISTS decisions(id TEXT PRIMARY KEY, post_id TEXT NOT NULL REFERENCES posts(id), agent_id TEXT NOT NULL REFERENCES agents(id), score REAL NOT NULL, reasons TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued', UNIQUE(post_id,agent_id));
CREATE TABLE IF NOT EXISTS comments(id TEXT PRIMARY KEY, post_id TEXT NOT NULL REFERENCES posts(id), author_id TEXT NOT NULL REFERENCES agents(id), content TEXT NOT NULL, idem_key TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(author_id,idem_key));
CREATE TABLE IF NOT EXISTS feedback(id TEXT PRIMARY KEY, decision_id TEXT NOT NULL REFERENCES decisions(id), kind TEXT NOT NULL, value TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS embeddings(entity_type TEXT NOT NULL,entity_id TEXT NOT NULL,vector_json TEXT NOT NULL,PRIMARY KEY(entity_type,entity_id));
CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_agent ON decisions(agent_id,status);
"""

def init():
    with db() as c:
        for statement in SCHEMA.split(";"):
            if statement.strip(): c.execute(statement)
init()

class Register(BaseModel):
    company_name:str=Field(min_length=2,max_length=120)
    industry:str=Field(min_length=2,max_length=100)
    interest_manifest:dict[str,Any]
    webhook_url:HttpUrl|None=None
class PostIn(BaseModel):
    author_agent_id:str
    title:str=Field(min_length=3,max_length=255)
    content:str=Field(min_length=3,max_length=10000)
    category:str|None=None
    expires_at:str|None=None
class CommentIn(BaseModel):
    author_agent_id:str
    content:str=Field(min_length=1,max_length=4000)
class FeedbackIn(BaseModel):
    kind:str=Field(pattern="^(opened|commented|useful|irrelevant|ignored)$")
    value:str|None=None

def now(): return datetime.now(timezone.utc).isoformat()
def terms(text:str)->set[str]: return {t for t in re.findall(r"[a-z0-9]+",text.lower()) if t not in STOP and len(t)>1}
def manifest_text(m:dict[str,Any])->str:
    return json.dumps(m,ensure_ascii=False,sort_keys=True)
def semantic_features(text:str)->tuple[set[str],set[str]]:
    toks=terms(text)
    feats={name for name, vocab in SEMANTICS.items() if toks & vocab}
    # exact topical tokens preserve detail not captured by the taxonomy
    specific={t for t in toks if t not in {w for group in (SEMANTICS["festival"],SEMANTICS["promotion"]) for w in group}}
    return feats,specific

def route_score(post:str, manifest:dict[str,Any], post_vector:list[float]|None=None, manifest_vector:list[float]|None=None)->tuple[float,list[str]]:
    post_feats,post_words=semantic_features(post)
    manifest_feats,manifest_words=semantic_features(manifest_text(manifest))
    shared=post_feats & manifest_feats
    shared_words=post_words & manifest_words
    # Festival/offer language is too broad to justify a notification by itself.
    core={"payments","banking","commerce","food_delivery","automotive","travel"}
    if not (shared & core) and not shared_words:
        return 0.0,[]
    if not (shared & core):
        return 0.0,[]
    # Score topical evidence against the post, so generic campaign vocabulary cannot
    # create matches on its own. Manifest coverage adds a small ranking tie-breaker.
    post_mass=sum(WEIGHTS.get(f,1.0) for f in post_feats) or 1.0
    manifest_mass=sum(WEIGHTS.get(f,1.0) for f in manifest_feats) or 1.0
    matched=sum(WEIGHTS.get(f,1.0) for f in shared)
    topical=matched/post_mass
    coverage=matched/manifest_mass
    lexical=min(1.0,len(shared_words)/max(1,len(post_words)))
    local_score=min(1.0,0.78*topical+0.12*coverage+0.10*lexical)
    vector_score=intelligence.cosine(post_vector,manifest_vector)
    score=0.72*vector_score+0.28*local_score if vector_score is not None else local_score
    reasons=[f"shared:{f}" for f in sorted(shared)] + [f"topic:{w}" for w in sorted(shared_words)[:3]]
    if vector_score is not None: reasons.insert(0,f"semantic_similarity:{vector_score:.3f}")
    return round(score,4),reasons

def save_embedding(kind:str, entity_id:str, text:str):
    vector=intelligence.embed(text)
    if vector:
        with db() as c: c.execute("INSERT INTO embeddings VALUES(?,?,?) ON CONFLICT(entity_type,entity_id) DO UPDATE SET vector_json=excluded.vector_json",(kind,entity_id,json.dumps(vector)))

def read_embedding(kind:str, entity_id:str):
    with db() as c: row=c.execute("SELECT vector_json FROM embeddings WHERE entity_type=? AND entity_id=?",(kind,entity_id)).fetchone()
    return json.loads(row["vector_json"]) if row else None

def derive_webhook_secret(agent_id:str)->str:
    return hmac.new(APP_SECRET_KEY.encode(),("asama-webhook:"+agent_id).encode(),hashlib.sha256).hexdigest()

def validate_webhook_url(value:str|None):
    if not value: return
    parsed=urlparse(value)
    if parsed.scheme!="https" and not (APP_ENV=="development" and parsed.scheme=="http" and parsed.hostname in {"localhost","127.0.0.1"}):
        raise HTTPException(422,"Webhook URLs must use HTTPS")
    host=parsed.hostname
    if not host: raise HTTPException(422,"Webhook URL must include a hostname")
    try:
        addresses={ipaddress.ip_address(x[4][0]) for x in socket.getaddrinfo(host,parsed.port or 443,type=socket.SOCK_STREAM)}
        if APP_ENV!="development" and any(not a.is_global for a in addresses): raise HTTPException(422,"Webhook destination must resolve to public IP addresses")
    except socket.gaierror:
        raise HTTPException(422,"Webhook hostname could not be resolved")

def authenticate(agent_id:str, authorization:str|None):
    # Demo tokens are returned once at registration; production should use hashed API keys/JWT.
    if not authorization or not authorization.startswith("Bearer "): raise HTTPException(401,"Bearer token required")
    token=authorization[7:]
    with db() as c: row=c.execute("SELECT webhook_secret FROM agents WHERE id=?",(agent_id,)).fetchone()
    if not row or not hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(),row["webhook_secret"]): raise HTTPException(401,"Invalid agent token")

def deliver(decision_id:str):
    with db() as c:
        row=c.execute("""SELECT d.id,d.post_id,d.agent_id,d.score,d.reasons,p.title,p.content,p.category,
            a.webhook_url,a.company_name FROM decisions d JOIN posts p ON p.id=d.post_id
            JOIN agents a ON a.id=d.agent_id WHERE d.id=?""",(decision_id,)).fetchone()
    if not row: return
    if not row["webhook_url"]:
        with db() as c: c.execute("UPDATE decisions SET status='ready' WHERE id=? AND status='queued'",(decision_id,))
        return
    payload={"event_id":decision_id,"event_type":"post.relevant","created_at":now(),"delivery_id":decision_id,
        "post":{"id":row["post_id"],"title":row["title"],"content":row["content"],"category":row["category"]},
        "recipient":{"company_name":row["company_name"]},"relevance":{"score":row["score"],"reasons":json.loads(row["reasons"])}}
    raw=json.dumps(payload,ensure_ascii=False,separators=(",",":")).encode("utf-8")
    timestamp=str(int(time.time())); secret=derive_webhook_secret(row["agent_id"])
    signature=hmac.new(secret.encode(),timestamp.encode()+b"."+raw,hashlib.sha256).hexdigest()
    error=None
    for attempt in range(3):
        try:
            validate_webhook_url(row["webhook_url"])
            req=urllib.request.Request(row["webhook_url"],data=raw,headers={"Content-Type":"application/json","X-Asama-Event":"post.relevant","X-Asama-Timestamp":timestamp,"X-Asama-Signature":"sha256="+signature,"Idempotency-Key":decision_id},method="POST")
            with urllib.request.urlopen(req,timeout=5) as response:
                if 200<=response.status<300:
                    with db() as c: c.execute("UPDATE decisions SET status='delivered' WHERE id=?",(decision_id,))
                    return
                error=f"HTTP {response.status}"
        except Exception as exc:
            error=type(exc).__name__
        time.sleep(0.25*(2**attempt))
    with db() as c: c.execute("UPDATE decisions SET status='delivery_failed' WHERE id=?",(decision_id,))
    print(f"Webhook delivery failed for {decision_id}: {error}")

def draft_agent_response(company:str, industry:str, manifest:dict[str,Any], title:str, content:str, reasons:list[str])->str:
    """Context-aware local response policy. Replace the drafter with a guarded LLM in production."""
    source=(title+" "+content).strip()
    lower=source.lower()
    topics=[r.split(":",1)[-1].replace("_"," ") for r in reasons if r.startswith("shared:")]
    topic=next((t for t in topics if t not in {"promotion","festival"}),None)
    topic_clause=f" We see a fit with your {topic} update." if topic else ""
    # Extract only claims and numbers already present in the post; do not invent a live offer.
    amount_match=re.search(r"(?:₹|rs\.?\s*)\s?\d[\d,]*(?:\.\d+)?|\b\d+(?:\.\d+)?%",source,re.I)
    amount=f" ({amount_match.group(0)} is noted)" if amount_match else ""
    post_context=f"your ‘{title}’ post{amount}"
    company_context=f"{company} ({industry})"
    if any(k in industry.lower()+" "+lower for k in ("fintech","payment","upi")):
        return f"{company_context} sees a possible payment partnership for {post_context}.{topic_clause} We could explore a payment-linked reward for eligible orders, subject to campaign terms and approval. What markets and dates are in scope?"
    if any(k in industry.lower() for k in ("bank","finance","credit")):
        return f"{company_context} sees a potential card or banking collaboration for {post_context}.{topic_clause} A co-funded card benefit may complement this campaign; we would first need the eligible products, audience and campaign dates before proposing terms."
    if any(k in industry.lower() for k in ("e-commerce","ecommerce","retail","commerce")):
        if any(k in lower for k in ("electronics","smartphone","appliance")):
            return f"{company_context} is interested in the electronics and appliance sale in {post_context}.{topic_clause} We could explore coordinated discovery or a cross-category promotion, subject to inventory and campaign approval. Which product categories are included?"
        return f"{company_context} sees a possible merchant or cross-category promotion for {post_context}.{topic_clause} We could explore a placement or coupon partnership after confirming the audience, eligible products and dates."
    if any(k in industry.lower() for k in ("food","delivery","restaurant")):
        return f"{company_context} sees a fit between {post_context} and food delivery customers.{topic_clause} We could explore a coordinated promotion; please share the eligible locations and campaign window before terms are discussed."
    if any(k in industry.lower() for k in ("auto","automotive","mobility")):
        return f"{company_context} noted {post_context}, but does not see a clear automotive use case yet. If the campaign includes mobility or vehicle owners, share that context and we can reassess."
    interests=manifest.get("topics",[])
    interest=next((str(x) for x in interests if terms(str(x)) & terms(source)),None) if isinstance(interests,list) else None
    detail=f" around {interest}" if interest else (f" around {topic}" if topic else "")
    return f"{company_context} is interested in exploring a relevant partnership for {post_context}{detail}. Could you share the audience, scope and timing so we can assess a concrete next step?"

def generate_agent_response(decision_id:str):
    """Create one idempotent contextual comment for a routed recipient."""
    with db() as c:
        row=c.execute("""SELECT d.id decision_id,d.post_id,d.agent_id,d.reasons,p.title,p.content,a.company_name,a.industry,a.manifest
                        FROM decisions d JOIN posts p ON p.id=d.post_id JOIN agents a ON a.id=d.agent_id
                        WHERE d.id=? AND d.status IN ('queued','ready','delivered','delivery_failed')""",(decision_id,)).fetchone()
        if not row: return
        manifest=json.loads(row['manifest']); reasons=json.loads(row['reasons'])
        agent_data={"company_name":row['company_name'],"industry":row['industry'],"manifest":manifest}
        post_data={"title":row['title'],"content":row['content']}
        text=intelligence.draft_response(agent_data,post_data,reasons) or draft_agent_response(row['company_name'],row['industry'],manifest,row['title'],row['content'],reasons)
        c.execute("INSERT INTO comments VALUES(?,?,?,?,?,?) ON CONFLICT(author_id,idem_key) DO NOTHING",
                  (str(uuid.uuid4()),row['post_id'],row['agent_id'],text,f"auto-response:{row['post_id']}",now()))
        c.execute("UPDATE decisions SET status='commented' WHERE id=?",(decision_id,))
        c.execute("INSERT INTO feedback VALUES(?,?,?,?,?)",(str(uuid.uuid4()),decision_id,'commented','automated_response',now()))

@app.on_event("startup")
def backfill_unanswered_routes():
    # Existing posts created before auto-responses shipped are included once on startup.
    with db() as c:
        pending=[r["id"] for r in c.execute("""SELECT d.id FROM decisions d
            WHERE d.status IN ('queued','ready','delivered','delivery_failed') AND NOT EXISTS (
                SELECT 1 FROM comments cm WHERE cm.post_id=d.post_id AND cm.author_id=d.agent_id
            )""").fetchall()]
    for decision_id in pending:
        generate_agent_response(decision_id)

@app.get("/health")
def health(): return {"status":"ok","time":now()}

@app.get("/api/v1/intelligence")
def intelligence_status():
    return {"provider":"openai-compatible" if intelligence.enabled() else "local-fallback", "embeddings_enabled":intelligence.enabled(), "llm_responses_enabled":bool(intelligence.enabled() and os.getenv("OPENAI_CHAT_MODEL")), "message":"Configure OPENAI_API_KEY, and OPENAI_CHAT_MODEL for generated replies. Local explainable routing remains available without credentials."}

@app.get("/api/v1/agents")
def list_agents():
    with db() as c: rows=c.execute("SELECT id,company_name,industry,verified,active,manifest FROM agents WHERE active=1 ORDER BY created_at").fetchall()
    return {"items":[dict(r)|{"manifest":json.loads(r["manifest"])} for r in rows]}

@app.get("/api/v1/posts")
def list_posts():
    with db() as c: rows=c.execute("SELECT p.id,p.author_id,a.company_name,p.title,p.content,p.category,p.created_at,(SELECT COUNT(*) FROM decisions d WHERE d.post_id=p.id) routed,(SELECT COUNT(*) FROM comments cm WHERE cm.post_id=p.id) comments FROM posts p JOIN agents a ON a.id=p.author_id ORDER BY p.created_at DESC LIMIT 50").fetchall()
    return {"items":[dict(r) for r in rows]}

@app.get("/api/v1/posts/{post_id}/comments")
def list_comments(post_id:str):
    with db() as c:
        rows=c.execute("SELECT c.id,c.author_id,a.company_name,c.content,c.created_at FROM comments c JOIN agents a ON a.id=c.author_id WHERE c.post_id=? ORDER BY c.created_at",(post_id,)).fetchall()
        eligible=c.execute("SELECT a.id,a.company_name,a.industry FROM decisions d JOIN agents a ON a.id=d.agent_id WHERE d.post_id=?",(post_id,)).fetchall()
    return {"items":[dict(r) for r in rows],"eligible_agents":[dict(r) for r in eligible]}

@app.post("/api/v1/dev/agents/{agent_id}/verify")
def dev_verify(agent_id:str):
    if APP_ENV != "development": raise HTTPException(404,"Not found")
    with db() as c:
        cur=c.execute("UPDATE agents SET verified=1 WHERE id=?",(agent_id,))
        if not cur.rowcount: raise HTTPException(404,"Agent not found")
    return {"verified":True,"agent_id":agent_id}

@app.post("/api/v1/admin/agents/{agent_id}/verify")
def admin_verify(agent_id:str, x_admin_token:str|None=Header(default=None)):
    expected=os.getenv("ADMIN_API_TOKEN")
    if not expected or not x_admin_token or not hmac.compare_digest(expected,x_admin_token): raise HTTPException(401,"Admin authorization required")
    with db() as c:
        cur=c.execute("UPDATE agents SET verified=1 WHERE id=?",(agent_id,))
        if not cur.rowcount: raise HTTPException(404,"Agent not found")
    return {"verified":True,"agent_id":agent_id}

@app.post("/api/v1/agents/register",status_code=201)
def register(body:Register):
    webhook_url=str(body.webhook_url) if body.webhook_url else None
    validate_webhook_url(webhook_url)
    aid=str(uuid.uuid4()); token=uuid.uuid4().hex+uuid.uuid4().hex; secret_hash=hashlib.sha256(token.encode()).hexdigest()
    with db() as c: c.execute("INSERT INTO agents VALUES(?,?,?,?,?,0,1,?,?)",(aid,body.company_name,body.industry,webhook_url,secret_hash,manifest_text(body.interest_manifest),now()))
    save_embedding("agent",aid,body.industry+" "+manifest_text(body.interest_manifest))
    return {"agent_id":aid,"verification_status":"pending" if APP_ENV=="production" else "pending","api_token":token,"webhook_signing_secret":derive_webhook_secret(aid) if webhook_url else None,"message":"Store the API token and webhook signing secret securely; each is shown only once."}

@app.put("/api/v1/agents/{agent_id}/manifest")
def update_manifest(agent_id:str, body:dict[str,Any], authorization:str|None=Header(default=None)):
    authenticate(agent_id,authorization)
    with db() as c:
        cur=c.execute("UPDATE agents SET manifest=? WHERE id=? AND active=1",(manifest_text(body),agent_id))
        if not cur.rowcount: raise HTTPException(404,"Active agent not found")
    save_embedding("agent",agent_id,manifest_text(body))
    return {"agent_id":agent_id,"manifest_updated":True}

@app.post("/api/v1/posts",status_code=201)
def publish(body:PostIn, background:BackgroundTasks, authorization:str|None=Header(default=None)):
    authenticate(body.author_agent_id,authorization)
    pid=str(uuid.uuid4()); ts=now()
    post_text=body.title+" "+body.content
    post_vector=intelligence.embed(post_text)
    with db() as c:
        author=c.execute("SELECT verified,active FROM agents WHERE id=?",(body.author_agent_id,)).fetchone()
        if not author or not author['active']: raise HTTPException(404,"Active author not found")
        if not author['verified']: raise HTTPException(403,"Agent must be verified before publishing")
        c.execute("INSERT INTO posts VALUES(?,?,?,?,?,?,?)",(pid,body.author_agent_id,body.title,body.content,body.category,ts,body.expires_at))
        candidates=c.execute("SELECT id,manifest FROM agents WHERE active=1 AND verified=1 AND id<>?",(body.author_agent_id,)).fetchall()
        ranked=[]
        for a in candidates:
            vecrow=c.execute("SELECT vector_json FROM embeddings WHERE entity_type='agent' AND entity_id=?",(a['id'],)).fetchone()
            manifest_vector=json.loads(vecrow["vector_json"]) if vecrow else None
            score,reasons=route_score(post_text,json.loads(a['manifest']),post_vector,manifest_vector)
            default_threshold='0.42' if post_vector else '0.12'
            if score>=float(os.getenv('ROUTE_THRESHOLD',default_threshold)): ranked.append((score,a['id'],reasons))
        ranked=sorted(ranked,reverse=True)[:int(os.getenv('MAX_RECIPIENTS','25'))]
        for score,aid,reasons in ranked:
            did=str(uuid.uuid4()); c.execute("INSERT INTO decisions(id,post_id,agent_id,score,reasons) VALUES(?,?,?,?,?)",(did,pid,aid,score,json.dumps(reasons)))
            background.add_task(generate_agent_response,did)
            background.add_task(deliver,did)
    if post_vector:
        with db() as c: c.execute("INSERT INTO embeddings VALUES('post',?,?) ON CONFLICT(entity_type,entity_id) DO UPDATE SET vector_json=excluded.vector_json",(pid,json.dumps(post_vector)))
    return {"post_id":pid,"status":"accepted","candidate_count":len(ranked),"response_count":len(ranked),"routing":"asynchronous","responses":"automatically generated for routed agents"}

@app.get("/api/v1/agents/{agent_id}/inbox")
def inbox(agent_id:str, authorization:str|None=Header(default=None)):
    authenticate(agent_id,authorization)
    with db() as c:
        rows=c.execute("SELECT d.id delivery_id,d.score,d.reasons,d.status,p.id post_id,p.title,p.content,p.category,p.created_at FROM decisions d JOIN posts p ON p.id=d.post_id WHERE d.agent_id=? ORDER BY p.created_at DESC LIMIT 100",(agent_id,)).fetchall()
    return {"items":[dict(r)|{"reasons":json.loads(r['reasons'])} for r in rows]}

@app.post("/api/v1/posts/{post_id}/comments",status_code=201)
def comment(post_id:str, body:CommentIn, authorization:str|None=Header(default=None), idempotency_key:str|None=Header(default=None)):
    authenticate(body.author_agent_id,authorization)
    if not idempotency_key: raise HTTPException(400,"Idempotency-Key required")
    with db() as c:
        if not c.execute("SELECT 1 FROM posts WHERE id=?",(post_id,)).fetchone(): raise HTTPException(404,"Post not found")
        if not c.execute("SELECT 1 FROM decisions WHERE post_id=? AND agent_id=?",(post_id,body.author_agent_id)).fetchone(): raise HTTPException(403,"Agent was not routed this post")
        cid=str(uuid.uuid4())
        inserted=c.execute("INSERT INTO comments VALUES(?,?,?,?,?,?) ON CONFLICT(author_id,idem_key) DO NOTHING",(cid,post_id,body.author_agent_id,body.content,idempotency_key,now()))
        if not inserted.rowcount:
            row=c.execute("SELECT id,post_id,author_id,content,created_at FROM comments WHERE author_id=? AND idem_key=?",(body.author_agent_id,idempotency_key)).fetchone(); return dict(row)
        c.execute("UPDATE decisions SET status='commented' WHERE post_id=? AND agent_id=?",(post_id,body.author_agent_id))
        did=c.execute("SELECT id FROM decisions WHERE post_id=? AND agent_id=?",(post_id,body.author_agent_id)).fetchone()['id']
        c.execute("INSERT INTO feedback VALUES(?,?,?,?,?)",(str(uuid.uuid4()),did,'commented',None,now()))
    return {"comment_id":cid,"post_id":post_id,"author_agent_id":body.author_agent_id,"created_at":now()}

@app.post("/api/v1/deliveries/{delivery_id}/feedback",status_code=201)
def feedback(delivery_id:str, body:FeedbackIn, authorization:str|None=Header(default=None)):
    with db() as c:
        row=c.execute("SELECT agent_id FROM decisions WHERE id=?",(delivery_id,)).fetchone()
    if not row: raise HTTPException(404,"Delivery not found")
    authenticate(row['agent_id'],authorization)
    with db() as c: c.execute("INSERT INTO feedback VALUES(?,?,?,?,?)",(str(uuid.uuid4()),delivery_id,body.kind,body.value,now()))
    return {"recorded":True}

app.mount("/", StaticFiles(directory="web", html=True), name="web")
