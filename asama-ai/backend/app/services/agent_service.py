
from sqlalchemy.orm import Session

from app.models.agent import Agent
from app.schemas.agent import AgentCreate


def create_agent(db: Session, agent_data: AgentCreate):

    new_agent = Agent(
        company_name=agent_data.company_name,
        industry_category=agent_data.industry_category,
        webhook_url=str(agent_data.webhook_url),
        interest_manifesto=agent_data.interest_manifest
    )

    db.add(new_agent)
    db.commit()
    db.refresh(new_agent)

    return new_agent