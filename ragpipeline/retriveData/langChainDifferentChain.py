from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class ResumeItem(BaseModel):
    resume_id: str = Field(..., description="ID of the resume, e.g. 'Emily_Green_Resume_46'")
    filename: Optional[str] = Field(None, description="Resume file name, if present in context")
    link: Optional[str] = Field(None, description="Path/URL to the resume file, if present in context")
    jd_relevance: float = Field(..., description="Relevance to the query/JD, between 0 and 1")
    profile_summary: str = Field(..., description="2–4 line summary of this candidate vs the query")
    key_skills: List[str] = Field(default_factory=list, description="Key skills relevant to the query")
    risks_or_flags: List[str] = Field(
        default_factory=list,
        description="Any risks, gaps, or misfits vs the query. Empty if none."
    )

class ResumeResponse(BaseModel):
    query: str = Field(..., description="Original user query or JD")
    resumes: List[ResumeItem] = Field(
        ...,
        description=(
            "List of ALL resumes present in the context. "
            "Include exactly one object per resume snippet."
        )
    )

resume_parser = PydanticOutputParser(pydantic_object=ResumeResponse)