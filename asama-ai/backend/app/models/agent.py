import uuid

from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.sql import func

from app.database import Base


class Agent(Base):
    __tablename__ = "agents"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    company_name = Column(String(255), nullable=False)

    industry_category = Column(String(100), nullable=False)

    webhook_url = Column(String(500), nullable=False)

    interest_manifest = Column("interest_manifesto", Text, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )