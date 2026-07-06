from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description_text = Column(Text, nullable=False)
    
    resumes = relationship("Resume", back_populates="jd")

class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    candidate_name = Column(String, index=True)
    resume_text = Column(Text, nullable=False)
    status = Column(String, default="Pending")  # Pending, Processing, Completed, Failed
    match_score = Column(Integer, nullable=True)   # Score out of 100
    analysis_report = Column(Text, nullable=True)
    
    jd_id = Column(Integer, ForeignKey("job_descriptions.id"))
    jd = relationship("JobDescription", back_populates="resumes")