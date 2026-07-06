import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Loaded once at import time and reused across requests. Calls
# router.huggingface.co under the hood -- the current, supported HF
# inference gateway -- unlike the old api-inference.huggingface.co endpoint
# (deprecated) that caused the original DNS/connection failure. Requires
# HF_TOKEN to be set in the environment.
_embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)

# Define the expected JSON output structure
class ScreeningAnalysis(BaseModel):
    match_score: int = Field(description="Overall compatibility score from 0 to 100")
    matched_skills: List[str] = Field(description="Skills found matching the JD")
    missing_skills: List[str] = Field(description="Critical skills missing from the resume")
    experience_fit: str = Field(description="Evaluation of the experience level matching the JD")
    verdict: str = Field(description="Final summary statement or hiring recommendation")

def analyze_resume_against_jd(resume_text: str, jd_text: str) -> dict:
    hf_token = os.getenv("HF_TOKEN")

    # 1. Build the Vector Store for the Resume context using the embeddings
    #    model initialized at module load time above.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([resume_text])

    vectorstore = FAISS.from_documents(docs, _embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Retrieve relevant sections of the resume based on the JD requirements
    relevant_docs = retriever.invoke(jd_text)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 2. Setup the cloud LLM. HuggingFaceEndpoint from langchain_huggingface
    #    (NOT langchain_community) calls router.huggingface.co under the
    #    hood, which is HF's current, supported inference gateway.
    #
    #    Model history on this endpoint, for future reference:
    #    - Mixtral-8x7B-Instruct-v0.1: not deployed by ANY provider anymore.
    #    - Mistral-7B-Instruct-v0.3: provider(s) rejected it as "not a chat
    #      model" despite conversational task being requested -- provider
    #      support for this exact model was inconsistent/unreliable.
    #    Qwen/Qwen2.5-7B-Instruct is confirmed as a live, standard chat
    #    completion model on the router and is NOT gated (no license
    #    acceptance required on your HF account, unlike meta-llama models).
    #    If you switch models again, check availability first with:
    #      huggingface_hub.model_info(repo_id, expand="inferenceProviderMapping")
    base_llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="conversational",
        huggingfacehub_api_token=hf_token,
        temperature=0.1
    )
    llm = ChatHuggingFace(llm=base_llm)

    # 3. Prompt Template orchestration
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
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"jd_text": jd_text, "context": context})
        return result
    except Exception as e:
        # repr(e) always includes the exception type even when str(e) is
        # empty (some HF/HTTP client errors have blank messages), so the
        # cause stays visible instead of showing "Failed: " with nothing.
        error_detail = repr(e) if not str(e) else str(e)
        return {
            "match_score": 0,
            "matched_skills": [],
            "missing_skills": [],
            "experience_fit": "Error during analysis parsing.",
            "verdict": f"Failed: {error_detail}"
        }