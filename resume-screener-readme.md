# Resume-Screener — README.md

Paste this into the repo `Mishi93/Resume-Screener` as `README.md` (the repo currently has no README). Add screenshots later — see the Screenshots section.

---

# AI Resume Screener

**Live demo:** https://resume-screener-g5fopzhz696w4itcxryyot.streamlit.app/

Match candidate resumes against a job description, score the fit, see matched and missing skills, and rank candidates — with a Streamlit dashboard and a FastAPI API over the same code.

Upload a resume (PDF or DOCX), pick a job description, and get:

- **Match score, 0–100** — with a donut chart (green ≥ 70, orange ≥ 40, red below)
- **Matched skills** (green) and **missing skills** (red), with a skills-gap chart
- **Experience fit** and a **verdict** in plain words
- **Leaderboard** — all candidates for a job, ranked best first

## How it works

1. **Extract** — resume text is read from PDF (`pypdf`) or DOCX (`python-docx`).
2. **Retrieve** — the resume is split into chunks (500 characters, 50 overlap) and stored in an in-memory FAISS vector store, embedded with `sentence-transformers/all-MiniLM-L6-v2`. The top 3 chunks most relevant to the job description are retrieved as context.
3. **Score** — `Qwen/Qwen2.5-7B-Instruct` (via Hugging Face, temperature 0.1) acts as an expert HR screening system and returns structured JSON: `match_score`, `matched_skills`, `missing_skills`, `experience_fit`, `verdict`.
4. **Store** — results are saved in SQLite (SQLAlchemy) with a status of Pending / Processing / Completed / Failed, so you can check a candidate later by Resume ID.

The score comes entirely from the LLM's returned `match_score` — there is no separate keyword-counting step.

## The app — 4 pages (Streamlit)

| Page | What you do |
|---|---|
| Create Job Description | Enter a job title and description text. You get a JD ID — use it on the next page. |
| Upload & Analyze Resume | Pick a JD, enter the candidate name, upload a PDF or DOCX, click Analyze. |
| Leaderboard | Pick a JD and see all completed candidates ranked by match score. |
| Check Status | Enter a Resume ID to see that candidate's status, score, skills, and verdict. |

## The API (FastAPI)

Same database and RAG code, as an API (`app/main.py`, "AI Resume Screening API" v2.0):

| Endpoint | What it does |
|---|---|
| `POST /jd/` | Create a job description — `{title, description_text}` → 201 |
| `POST /resume/` | Upload a resume for a JD — form fields `jd_id`, `candidate_name`, `file` → 202, runs the analysis and returns the row |
| `GET /resume/{resume_id}` | Check a candidate's status, score, and analysis → 404 if not found |

## Tech stack

Python · Streamlit · FastAPI · LangChain · Hugging Face (embeddings + LLM) · FAISS · SQLAlchemy / SQLite · Plotly · pypdf · python-docx

## Run it locally

```bash
git clone https://github.com/Mishi93/Resume-Screener.git
cd Resume-Screener
pip install -r requirements.txt
```

Create a `.env` file (never commit it) with your Hugging Face token:

```
HF_TOKEN=your_hugging_face_token_here
```

`HF_TOKEN` is required — it powers both the embeddings and the LLM calls. `DATABASE_URL` is optional; if unset, a local SQLite file `resume_screener.db` is created.

**Streamlit dashboard:**

```bash
streamlit run streamlit_app.py
```

**FastAPI (alternative):**

```bash
uvicorn app.main:app --reload
```

Or open the repo in VS Code with the included dev container (Python 3.11, port 8501 forwarded).

## Project structure

```
Resume-Screener/
├── streamlit_app.py        # Streamlit dashboard (4 pages)
├── app/
│   ├── main.py             # FastAPI app (POST /jd/, POST /resume/, GET /resume/{id})
│   ├── database.py         # SQLAlchemy engine + session (SQLite by default)
│   ├── models.py           # JobDescription, Resume
│   ├── schemas.py          # Pydantic schemas
│   └── services/
│       └── rag_service.py  # RAG analysis: chunk → FAISS → Qwen2.5-7B-Instruct
├── requirements.txt
└── .devcontainer/
    └── devcontainer.json
```

## Screenshots

Add 2–3 screenshots here once taken from the live demo:

1. Upload & Analyze page with a completed result (donut chart + skills)
2. Leaderboard with ranked candidates
3. Check Status page

Take them from the live app, save them in the repo (e.g. `screenshots/`), and embed with `![description](screenshots/...)`.
