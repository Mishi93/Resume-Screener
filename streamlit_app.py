"""
AI Resume Screener - Streamlit GUI
------------------------------------
An interactive web-based GUI (runs in your browser via Streamlit) with a
donut chart visualizing the match score. Self-contained: no separate
FastAPI backend needed, calls the database and RAG pipeline directly --
same approach as standalone_app.py, just with a Streamlit front-end
instead of Tkinter.

REQUIRED PROJECT LAYOUT (place this file at your project root, next to app/):
    your_project/
    ├── app/
    │   ├── __init__.py
    │   ├── database.py
    │   ├── models.py
    │   ├── schemas.py
    │   └── services/
    │       ├── __init__.py
    │       └── rag_service.py
    ├── .env                  <- must contain HF_TOKEN=...
    └── streamlit_app.py      <- this file

Install requirements:
    pip install streamlit plotly

Run with:
    streamlit run streamlit_app.py

NOTE: Still requires internet access at runtime for the Hugging Face
embeddings + LLM calls, and a valid HF_TOKEN in your .env file.
"""

import io
import json

import streamlit as st
import plotly.graph_objects as go
from pypdf import PdfReader
from docx import Document

from app.database import engine, Base, SessionLocal
from app import models
from app.services.rag_service import analyze_resume_against_jd

# Create tables on first run (safe to call every time -- no-op if they exist)
Base.metadata.create_all(bind=engine)

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def extract_text_from_uploaded(uploaded_file) -> str:
    """Reads a Streamlit UploadedFile (PDF or DOCX) and returns plain text."""
    name = uploaded_file.name.lower()
    contents = uploaded_file.read()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(contents))
        return "".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(contents))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")


def score_color(score: int) -> str:
    if score >= 70:
        return "#2ecc71"   # green
    elif score >= 40:
        return "#f39c12"   # orange
    return "#e74c3c"       # red


def make_donut_chart(score: int) -> go.Figure:
    color = score_color(score)
    fig = go.Figure(
        data=[
            go.Pie(
                values=[score, 100 - score],
                hole=0.72,
                marker=dict(colors=[color, "#2a2a2a"], line=dict(color="rgba(0,0,0,0)", width=0)),
                textinfo="none",
                sort=False,
                direction="clockwise",
                hoverinfo="skip",
            )
        ]
    )
    fig.add_annotation(
        text=f"<b>{score}</b>",
        x=0.5, y=0.54,
        font=dict(size=44, color=color),
        showarrow=False,
    )
    fig.add_annotation(
        text="MATCH SCORE",
        x=0.5, y=0.40,
        font=dict(size=13, color="#888"),
        showarrow=False,
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_skill_pills(skills: list, kind: str):
    """Renders a list of skills as colored, rounded 'pill' badges via HTML."""
    if not skills:
        st.caption("None listed")
        return
    bg, fg = ("#1e4d2b", "#7CFFA0") if kind == "matched" else ("#4d1e1e", "#FF9E9E")
    pills_html = "".join(
        f'<span style="display:inline-block;background:{bg};color:{fg};'
        f'padding:4px 12px;margin:3px;border-radius:999px;font-size:13px;'
        f'font-weight:500;">{skill}</span>'
        for skill in skills
    )
    st.markdown(pills_html, unsafe_allow_html=True)


def make_skills_gap_chart(matched: list, missing: list) -> go.Figure:
    """Horizontal bar chart: every required skill, green if matched, red if missing."""
    skills = list(matched) + list(missing)
    colors = ["#2ecc71"] * len(matched) + ["#e74c3c"] * len(missing)
    labels = ["Matched"] * len(matched) + ["Missing"] * len(missing)

    fig = go.Figure(
        go.Bar(
            x=[1] * len(skills),
            y=skills,
            orientation="h",
            marker=dict(color=colors),
            text=labels,
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="%{y}: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=max(120, 38 * len(skills)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_leaderboard_chart(candidates: list) -> go.Figure:
    """Horizontal bar chart ranking candidates by match score, best on top."""
    candidates = sorted(candidates, key=lambda c: c["match_score"] or 0, reverse=True)
    names = [c["candidate_name"] for c in candidates]
    scores = [c["match_score"] or 0 for c in candidates]
    colors = [score_color(s) for s in scores]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=names,
            orientation="h",
            marker=dict(color=colors),
            text=[str(s) for s in scores],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis=dict(title="Match Score", range=[0, 105]),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=40),
        height=max(160, 46 * len(names)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------

st.sidebar.title("📄 Resume Screener")
page = st.sidebar.radio(
    "Go to",
    ["Create Job Description", "Upload & Analyze Resume", "Leaderboard", "Check Status"],
)

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# --------------------------------------------------------------------------
# Page: Create Job Description
# --------------------------------------------------------------------------

if page == "Create Job Description":
    st.header("1. Create a Job Description")

    with st.form("jd_form"):
        title = st.text_input("Job Title")
        description = st.text_area("Job Description Text", height=300)
        submitted = st.form_submit_button("Create Job Description")

    if submitted:
        if not title.strip() or not description.strip():
            st.warning("Please fill in both the title and description.")
        else:
            db = SessionLocal()
            try:
                jd = models.JobDescription(title=title.strip(), description_text=description.strip())
                db.add(jd)
                db.commit()
                db.refresh(jd)
                st.success(f"Created! JD ID = **{jd.id}** — use this ID on the Upload & Analyze page.")
            except Exception as e:
                st.error(f"Database error: {e}")
            finally:
                db.close()

    # Show existing JDs for convenience
    db = SessionLocal()
    try:
        jds = db.query(models.JobDescription).all()
        if jds:
            st.subheader("Existing Job Descriptions")
            st.table([{"ID": jd.id, "Title": jd.title} for jd in jds])
    finally:
        db.close()


# --------------------------------------------------------------------------
# Page: Upload & Analyze Resume
# --------------------------------------------------------------------------

elif page == "Upload & Analyze Resume":
    st.header("2. Upload & Analyze a Resume")

    db = SessionLocal()
    try:
        jds = db.query(models.JobDescription).all()
    finally:
        db.close()

    if not jds:
        st.info("No Job Descriptions exist yet. Create one first on the 'Create Job Description' page.")
    else:
        jd_options = {f"#{jd.id} — {jd.title}": jd.id for jd in jds}
        jd_label = st.selectbox("Job Description", list(jd_options.keys()))
        jd_id = jd_options[jd_label]

        candidate_name = st.text_input("Candidate Name")
        uploaded_file = st.file_uploader("Resume file", type=["pdf", "docx"])

        if st.button("Analyze", type="primary", disabled=not (candidate_name and uploaded_file)):
            db = SessionLocal()
            try:
                jd = db.query(models.JobDescription).filter(models.JobDescription.id == jd_id).first()

                with st.spinner("Extracting text from resume..."):
                    resume_text = extract_text_from_uploaded(uploaded_file)
                    if not resume_text.strip():
                        raise ValueError("The uploaded file seems to be empty or unreadable.")

                db_resume = models.Resume(
                    candidate_name=candidate_name.strip(),
                    resume_text=resume_text,
                    jd_id=jd_id,
                    status="Processing",
                )
                db.add(db_resume)
                db.commit()
                db.refresh(db_resume)

                with st.spinner("Running AI analysis (embeddings + LLM)... this can take a while."):
                    try:
                        analysis = analyze_resume_against_jd(resume_text, jd.description_text)
                        db_resume.match_score = analysis.get("match_score", 0)
                        db_resume.analysis_report = json.dumps(analysis)
                        db_resume.status = "Completed"
                    except Exception as e:
                        db_resume.status = "Failed"
                        db_resume.analysis_report = json.dumps({"error": repr(e) if not str(e) else str(e)})

                db.commit()
                db.refresh(db_resume)

                st.session_state.last_result = {
                    "id": db_resume.id,
                    "candidate_name": db_resume.candidate_name,
                    "status": db_resume.status,
                    "match_score": db_resume.match_score,
                    "analysis_report": json.loads(db_resume.analysis_report) if db_resume.analysis_report else None,
                }
            except Exception as e:
                st.error(f"Analysis failed: {e}")
            finally:
                db.close()

    # Display the most recent result, including the donut chart
    result = st.session_state.last_result
    if result:
        st.divider()
        st.subheader(f"Result for {result['candidate_name']} (Resume ID #{result['id']})")

        if result["status"] == "Failed":
            st.error("Analysis failed for this resume.")
            st.json(result["analysis_report"])
        else:
            report = result["analysis_report"] or {}
            score = result["match_score"] or 0
            matched = report.get("matched_skills", [])
            missing = report.get("missing_skills", [])

            if score >= 70 and not st.session_state.get(f"celebrated_{result['id']}"):
                st.balloons()
                st.session_state[f"celebrated_{result['id']}"] = True

            col1, col2 = st.columns([1, 2])
            with col1:
                st.plotly_chart(make_donut_chart(score), use_container_width=True)
                total = len(matched) + len(missing)
                coverage = round(len(matched) / total * 100) if total else 0
                st.caption(f"**{len(matched)}/{total}** required skills covered ({coverage}%)")
            with col2:
                st.markdown(f"**Status:** {result['status']}")
                st.markdown(f"**Experience Fit:** {report.get('experience_fit', '—')}")
                st.markdown(f"**Verdict:** {report.get('verdict', '—')}")

                st.markdown("✅ **Matched Skills**")
                render_skill_pills(matched, "matched")
                st.markdown("❌ **Missing Skills**")
                render_skill_pills(missing, "missing")

            if matched or missing:
                st.markdown("#### 📊 Skills Gap")
                st.plotly_chart(make_skills_gap_chart(matched, missing), use_container_width=True)

            with st.expander("View extracted resume text"):
                db = SessionLocal()
                try:
                    resume_row = db.query(models.Resume).filter(models.Resume.id == result["id"]).first()
                    st.text(resume_row.resume_text if resume_row else "Not available.")
                finally:
                    db.close()


# --------------------------------------------------------------------------
# Page: Leaderboard
# --------------------------------------------------------------------------

elif page == "Leaderboard":
    st.header("🏆 Candidate Leaderboard")

    db = SessionLocal()
    try:
        jds = db.query(models.JobDescription).all()
    finally:
        db.close()

    if not jds:
        st.info("No Job Descriptions exist yet. Create one first on the 'Create Job Description' page.")
    else:
        jd_options = {f"#{jd.id} — {jd.title}": jd.id for jd in jds}
        jd_label = st.selectbox("Compare candidates for", list(jd_options.keys()))
        jd_id = jd_options[jd_label]

        db = SessionLocal()
        try:
            resumes = (
                db.query(models.Resume)
                .filter(models.Resume.jd_id == jd_id, models.Resume.status == "Completed")
                .all()
            )
            candidates = [{"candidate_name": r.candidate_name, "match_score": r.match_score} for r in resumes]
        finally:
            db.close()

        if not candidates:
            st.info("No completed analyses yet for this Job Description.")
        else:
            st.plotly_chart(make_leaderboard_chart(candidates), use_container_width=True)

            ranked = sorted(candidates, key=lambda c: c["match_score"] or 0, reverse=True)
            st.table(
                [
                    {"Rank": i + 1, "Candidate": c["candidate_name"], "Match Score": c["match_score"]}
                    for i, c in enumerate(ranked)
                ]
            )


# --------------------------------------------------------------------------
# Page: Check Status
# --------------------------------------------------------------------------

elif page == "Check Status":
    st.header("3. Check a Resume's Status by ID")

    resume_id = st.number_input("Resume ID", min_value=1, step=1, format="%d")
    if st.button("Fetch Status"):
        db = SessionLocal()
        try:
            resume = db.query(models.Resume).filter(models.Resume.id == int(resume_id)).first()
            if not resume:
                st.error(f"No resume found with ID {int(resume_id)}.")
            else:
                parsed_report = None
                if resume.analysis_report:
                    try:
                        parsed_report = json.loads(resume.analysis_report)
                    except json.JSONDecodeError:
                        parsed_report = {"raw": resume.analysis_report}

                st.markdown(f"**Candidate:** {resume.candidate_name}")
                st.markdown(f"**Status:** {resume.status}")

                if resume.status == "Completed" and resume.match_score is not None:
                    report = parsed_report or {}
                    matched = report.get("matched_skills", [])
                    missing = report.get("missing_skills", [])

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.plotly_chart(make_donut_chart(resume.match_score), use_container_width=True)
                    with col2:
                        st.markdown(f"**Experience Fit:** {report.get('experience_fit', '—')}")
                        st.markdown(f"**Verdict:** {report.get('verdict', '—')}")
                        st.markdown("✅ **Matched Skills**")
                        render_skill_pills(matched, "matched")
                        st.markdown("❌ **Missing Skills**")
                        render_skill_pills(missing, "missing")

                    if matched or missing:
                        st.markdown("#### 📊 Skills Gap")
                        st.plotly_chart(make_skills_gap_chart(matched, missing), use_container_width=True)
                else:
                    st.json(parsed_report)
        except Exception as e:
            st.error(f"Database error: {e}")
        finally:
            db.close()