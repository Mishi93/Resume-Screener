from pydantic import BaseModel
from typing import Optional, Any

# Job Description Schemas
class JDBase(BaseModel):
    title: str
    description_text: str

class JDCreate(JDBase):
    pass

class JDResponse(JDBase):
    id: int

    class Config:
        from_attributes = True

# Resume Schemas
class ResumeBase(BaseModel):
    candidate_name: str
    resume_text: str
    jd_id: int

class ResumeCreate(ResumeBase):
    pass

class ResumeResponse(ResumeBase):
    id: int
    status: str
    match_score: Optional[int] = None

    class Config:
        from_attributes = True

# Custom Status Response Schema
class ResumeStatusResponse(BaseModel):
    id: int
    candidate_name: str
    status: str
    match_score: Optional[int] = None
    analysis_report: Optional[Any] = None