from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.agent import Agent
from app.schemas.agent import AgentCreate, AgentResponse
from app.services.agent_service import create_agent


router = APIRouter(
    prefix="/agents",
    tags=["Agents"]
)


# 1. Create a new agent
@router.post("/", response_model=AgentResponse)
def create_new_agent(
    agent_data: AgentCreate,
    db: Session = Depends(get_db)
):
    return create_agent(db, agent_data)


# 2. Get all agents
@router.get("/", response_model=list[AgentResponse])
def get_all_agents(db: Session = Depends(get_db)):
    return db.query(Agent).all()


# 3. Get agent by ID
@router.get("/{agent_id}", response_model=AgentResponse)
def get_agent(
    agent_id: int,
    db: Session = Depends(get_db)
):
    agent = db.query(Agent).filter(
        Agent.id == agent_id
    ).first()

    if not agent:
        raise HTTPException(
            status_code=404,
            detail="Agent not found"
        )

    return agent