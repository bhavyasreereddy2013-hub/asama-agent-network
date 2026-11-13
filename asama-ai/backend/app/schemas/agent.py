from pydantic import BaseModel, Field, HttpUrl, ConfigDict, Field


class AgentCreate(BaseModel):
    company_name: str
    industry_category: str
    webhook_url: HttpUrl
    interest_manifest: str


class AgentResponse(BaseModel):
    id: str
    company_name: str
    industry_category: str
    webhook_url: str
    interest_manifest: str = Field(
        validation_alias="interest_manifesto"
    )


    model_config = ConfigDict(from_attributes=True)