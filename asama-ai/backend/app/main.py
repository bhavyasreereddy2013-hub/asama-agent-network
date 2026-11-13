from fastapi import FastAPI
from app.database import engine, Base
from app.models.agent import Agent
from app.routes import agents

app = FastAPI(
    title="Asama.ai",
    description="Autonomous Agent Network",
    version="1.0.0"
)

Base.metadata.create_all(bind=engine)
app.include_router(agents.router)


@app.get("/")
def home():
    return {
        "message": "Welcome to Asama.ai",
        "status": "Running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}