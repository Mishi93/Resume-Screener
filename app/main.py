from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
import json
import io
from pypdf import PdfReader
from docx import Document

from app.database import engine, Base, get_db
from app import models, schemas
from app.services.rag_service import analyze_resume_against_jd

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Resume Screening API", version="2.0")

# Helper function to extract text from files
def extract_text_from_file(file: UploadFile) -> str:
    filename = file.filename.lower()
    contents = file.file.read()
    
    if filename.endswith('.pdf'):
        pdf_stream = io.BytesIO(contents)
        reader = PdfReader(pdf_stream)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        return text
        
    elif filename.endswith('.docx'):
        docx_stream = io.BytesIO(contents)
        doc = Document(docx_stream)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
        
    else:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Please upload a PDF or DOCX file."
        )

@app.post("/jd/", response_model=schemas.JDResponse, status_code=status.HTTP_201_CREATED)
def create_job_description(jd: schemas.JDCreate, db: Session = Depends(get_db)):
    db_jd = models.JobDescription(title=jd.title, description_text=jd.description_text)
    db.add(db_jd)
    db.commit()
    db.refresh(db_jd)
    return db_jd

@app.post("/resume/", response_model=schemas.ResumeResponse, status_code=status.HTTP_202_ACCEPTED)
def upload_and_process_resume(
    jd_id: int = Form(...),
    candidate_name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. Verify Job Description exists
    jd = db.query(models.JobDescription).filter(models.JobDescription.id == jd_id).first()
    if not jd:
        raise HTTPException(status_code=404, detail="Target Job Description not found")
    
    # 2. Extract text content from the uploaded file
    try:
        resume_text = extract_text_from_file(file)
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="The uploaded file seems to be empty or unreadable.")
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=f"File parsing failed: {str(e)}")

    # 3. Initialize DB record
    db_resume = models.Resume(
        candidate_name=candidate_name,
        resume_text=resume_text,
        jd_id=jd_id,
        status="Processing"
    )
    db.add(db_resume)
    db.commit()
    db.refresh(db_resume)
    
    # 4. Execute RAG Pipeline
    try:
        analysis = analyze_resume_against_jd(db_resume.resume_text, jd.description_text)
        
        db_resume.match_score = analysis.get("match_score", 0)
        db_resume.analysis_report = json.dumps(analysis)
        db_resume.status = "Completed"
    except Exception as e:
        db_resume.status = "Failed"
        db_resume.analysis_report = json.dumps({"error": str(e)})
    
    db.commit()
    db.refresh(db_resume)
    return db_resume

@app.get("/resume/{resume_id}", response_model=schemas.ResumeStatusResponse)
def get_resume_status(resume_id: int, db: Session = Depends(get_db)):
    resume = db.query(models.Resume).filter(models.Resume.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume record not found")
    
    parsed_report = None
    if resume.analysis_report:
        try:
            parsed_report = json.loads(resume.analysis_report)
        except json.JSONDecodeError:
            parsed_report = {"raw": resume.analysis_report}

    return {
        "id": resume.id,
        "candidate_name": resume.candidate_name,
        "status": resume.status,
        "match_score": resume.match_score,
        "analysis_report": parsed_report
    }