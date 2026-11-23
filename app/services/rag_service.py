import os
from typing import List

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Embeddings now run LOCALLY via sentence-transformers -- no API calls, no
# credits. The model (~80 MB) downloads once on first run and is cached.
# Loaded at import time so Streamlit reruns don't reload it.
_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# LLM now runs on Groq's free tier. Requires GROQ_API_KEY in .env (locally)
# or in Streamlit secrets (deployed). Model ID is overridable via
# GROQ_MODEL so a future deprecation is a config change, not a code change.
# Check active models at https://console.groq.com/docs/models
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


class ScreeningAnalysis(BaseModel):
    match_score: int = Field(description="Overall compatibility score from 0 to 100")
    matched_skills: List[str] = Field(description="Skills found matching the JD")
    missing_skills: List[str] = Field(description="Critical skills missing from the resume")
    experience_fit: str = Field(description="Evaluation of the experience level matching the JD")
    verdict: str = Field(description="Final summary statement or hiring recommendation")


def analyze_resume_against_jd(resume_text: str, jd_text: str) -> dict:
    # 1. Build the vector store for the resume and retrieve relevant chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([resume_text])

    vectorstore = FAISS.from_documents(docs, _embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    relevant_docs = retriever.invoke(jd_text)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    # 2. LLM
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
    )

    # 3. Prompt + parser
    parser = JsonOutputParser(pydantic_object=ScreeningAnalysis)

    template = """
    You are an expert HR screening system. Analyze the candidate's resume context against the Job Description (JD).

    Job Description:
    {jd_text}

    Candidate Resume Context:
    {context}

    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["jd_text", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    # No try/except here on purpose: if the LLM call fails, the exception
    # propagates to streamlit_app.py, which marks the resume "Failed".
    # Previously, errors were returned as a normal result with score 0, so
    # broken analyses were saved as "Completed" and showed up on the
    # leaderboard.
    return chain.invoke({"jd_text": jd_text, "context": context})
