"""JobForge AI: upload a resume, get ranked job matches with skill gaps and explanations.

    streamlit run app/streamlit_app.py
"""

import streamlit as st

from jobforge.config import get_settings
from jobforge.db import connect
from jobforge.explain import DbCache, explain_matches, make_llm
from jobforge.match import DEFAULT_W_SEM, load_descriptions, match_resume
from jobforge.resume import pdf_to_text
from jobforge.skills import SkillExtractor, load_vocab

st.set_page_config(page_title="JobForge AI", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading skill vocabulary...")
def get_extractor() -> SkillExtractor:
    return SkillExtractor(load_vocab())


@st.cache_resource(show_spinner="Loading embedding model...")
def warm_embedding_model() -> None:
    from jobforge.embed import get_model
    get_model()


@st.cache_resource
def get_llm():
    return make_llm(get_settings())


def read_resume(upload, pasted: str) -> str:
    if upload is not None:
        if upload.name.lower().endswith(".pdf"):
            return pdf_to_text(upload)
        return upload.getvalue().decode("utf-8", errors="replace")
    return pasted.strip()


def skill_chips(skills: list[str], color: str) -> str:
    return " ".join(f":{color}-badge[{s}]" for s in skills) if skills else "_none_"


st.title("JobForge AI")
st.caption("Upload a resume to find the best-matching job postings, see which skills match and which are missing.")

with st.sidebar:
    st.header("Filters")
    location = st.text_input("Location contains", placeholder="e.g. Remote, New York, London")
    min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.05)
    top_n = st.slider("Results", 5, 25, 10)
    with st.expander("Ranking weights"):
        w_sem = st.slider("Semantic similarity weight", 0.0, 1.0, DEFAULT_W_SEM, 0.1,
                          help="The rest of the score is skill coverage: the share of the job's skills your resume has.")

upload = st.file_uploader("Resume (PDF or text)", type=["pdf", "txt", "md"])
pasted = st.text_area("...or paste resume text", height=150, key="pasted")
run = st.button("Find matches", type="primary")

if run:
    resume_text = read_resume(upload, pasted)
    if len(resume_text.split()) < 20:
        st.warning("That resume looks empty or too short. Upload a PDF with selectable text, or paste the text.")
        st.stop()

    extractor = get_extractor()
    warm_embedding_model()
    llm = get_llm()
    settings = get_settings()
    try:
        with st.spinner("Matching..."), connect() as conn:
            resume_skills, results = match_resume(conn, resume_text, extractor, top_n=top_n, w_sem=w_sem,
                                                  location=location or None)
            results = [r for r in results if r["score"] >= min_score]
            descriptions = load_descriptions(conn, [r["job_id"] for r in results])
            results = explain_matches(results, descriptions, resume_text, extractor, llm=llm,
                                      model_name=settings.llm_model or "", cache=DbCache(conn))
    except Exception as e:  # database down, etc.: explain instead of a stack trace
        st.error(f"Matching failed: {e}. Is the database running (`docker compose up -d db`)?")
        st.stop()

    st.subheader(f"Skills found in your resume ({len(resume_skills)})")
    st.markdown(skill_chips(sorted(resume_skills), "blue"))
    if not llm:
        st.info("Explanations are generated from the skill lists. Set LLM_PROVIDER, LLM_MODEL and an API key "
                "in .env for LLM-written explanations.")

    if not results:
        st.warning("No matches with these filters. Try a broader location or a lower minimum score.")
    for i, r in enumerate(results, 1):
        with st.container(border=True):
            left, right = st.columns([4, 1])
            more = f" (+{len(r['other_locations'])} more)" if r["other_locations"] else ""
            left.markdown(f"**{i}. [{r['title']}]({r['url']})**  \n{r['company']} · {r['location']}{more}")
            right.metric("Score", f"{r['score']:.2f}",
                         help=f"semantic {r['semantic']:.2f} · coverage {r['coverage']:.0%}")
            st.markdown(f"**Matched:** {skill_chips(r['matched'], 'green')}")
            st.markdown(f"**Missing:** {skill_chips(r['missing'], 'orange')}")
            st.write(r["explanation"])
            st.caption("LLM explanation" if r["explanation_source"] == "llm" else "Explanation from skill lists")
