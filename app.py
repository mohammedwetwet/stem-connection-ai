import os
import re
import html
import textwrap
from typing import List, Optional, Literal

import fitz
import streamlit as st

from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="STEM Connect AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# HTML RENDERING
# =========================================================

def render_html(content):
    """
    Render complete HTML blocks directly.
    Do not use this function for partial/open HTML tags.
    """
    cleaned = textwrap.dedent(content).strip()
    st.html(cleaned)


def safe(value):
    return html.escape(str(value or ""))


# =========================================================
# DESIGN SYSTEM
# =========================================================

LEVEL_COLORS = {
    "BLUE": "#3B82F6",
    "GREEN": "#22C55E",
    "YELLOW": "#EAB308",
    "RED": "#EF4444",
    "Insufficient evidence": "#94A3B8",
}


# =========================================================
# CUSTOM CSS
# =========================================================

render_html("""
<style>
@import url(
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

:root {
    color-scheme: dark;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #0B1120 !important;
    color: #F1F5F9 !important;
}

[data-testid="stHeader"] {
    background: rgba(11,17,32,0.96) !important;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.7rem;
    padding-bottom: 3rem;
}

/* GENERAL TEXT */

.stApp p,
.stApp label,
.stApp li {
    color: #F1F5F9;
}

.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p {
    color: #A8B7CC !important;
}

h1, h2, h3, h4 {
    color: #F8FAFC !important;
    letter-spacing: -0.025em;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #101A2C !important;
    border-right: 1px solid #263650;
}

section[data-testid="stSidebar"] * {
    color: #F1F5F9 !important;
}

section[data-testid="stSidebar"] input {
    color: #111827 !important;
    background: #F8FAFC !important;
    border: 1px solid #CBD5E1 !important;
}

section[data-testid="stSidebar"] input::placeholder {
    color: #64748B !important;
}

/* INPUTS */

div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stNumberInput"] input {
    background: #17243A !important;
    color: #F8FAFC !important;
    -webkit-text-fill-color: #F8FAFC !important;
    border: 1px solid #354761 !important;
    border-radius: 12px !important;
    caret-color: #2DD4BF !important;
}

div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stTextArea"] textarea::placeholder {
    color: #A8B7CC !important;
    -webkit-text-fill-color: #A8B7CC !important;
    opacity: 1 !important;
}

div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: #2DD4BF !important;
    box-shadow: 0 0 0 2px rgba(45,212,191,0.15) !important;
}

/* RADIO */

div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span {
    color: #F1F5F9 !important;
    opacity: 1 !important;
}

/* SELECT */

div[data-baseweb="select"] > div {
    background: #17243A !important;
    border-color: #354761 !important;
    color: #F8FAFC !important;
}

/* FILE UPLOADER */

div[data-testid="stFileUploader"] {
    background: #111C30 !important;
    border: 1px dashed #3B5973 !important;
    border-radius: 16px !important;
    padding: 12px !important;
}

div[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
}

div[data-testid="stFileUploader"] button {
    background: #1D3049 !important;
    color: #F8FAFC !important;
    border: 1px solid #42617E !important;
    border-radius: 10px !important;
}

div[data-testid="stFileUploader"] small {
    color: #A8B7CC !important;
}

/* BUTTONS */

.stButton > button,
div[data-testid="stDownloadButton"] button {
    min-height: 46px;
    border-radius: 12px !important;
    font-weight: 700 !important;
    transition: 0.2s ease;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(
        100deg,
        #0D9488,
        #0F766E
    ) !important;
    color: #FFFFFF !important;
    border: 1px solid #2DD4BF !important;
    box-shadow: 0 5px 20px rgba(13,148,136,0.18);
}

.stButton > button[kind="primary"]:hover {
    background: #14B8A6 !important;
    color: #06201E !important;
    transform: translateY(-1px);
}

.stButton > button[kind="secondary"],
div[data-testid="stDownloadButton"] button {
    background: #17243A !important;
    color: #F8FAFC !important;
    border: 1px solid #354761 !important;
}

/* HERO */

.hero {
    padding: 38px;
    border-radius: 24px;
    background: linear-gradient(
        125deg,
        #13233B 0%,
        #123B4A 55%,
        #115E59 100%
    );
    border: 1px solid #285566;
    box-shadow: 0 18px 50px rgba(0,0,0,0.22);
    margin-bottom: 24px;
}

.hero-badge {
    display: inline-block;
    padding: 7px 12px;
    border: 1px solid #3A817D;
    border-radius: 30px;
    background: rgba(45,212,191,0.10);
    color: #99F6E4 !important;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
}

.hero h1 {
    font-size: clamp(32px, 4vw, 49px);
    line-height: 1.2;
    margin: 18px 0 12px;
    color: #FFFFFF !important;
    font-weight: 800;
}

.hero p {
    max-width: 740px;
    font-size: 15px;
    line-height: 1.8;
    color: #D5E7F1 !important;
}

.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 22px;
}

.hero-pill {
    display: inline-block;
    border: 1px solid #385D70;
    border-radius: 30px;
    padding: 8px 12px;
    background: rgba(255,255,255,0.05);
    color: #E0F2FE !important;
    font-size: 12px;
    font-weight: 600;
}

/* SECTIONS */

.section-title {
    font-size: 23px;
    font-weight: 800;
    color: #F8FAFC !important;
    margin: 30px 0 5px;
}

.section-subtitle {
    font-size: 13px;
    color: #A8B7CC !important;
    line-height: 1.8;
    margin-bottom: 18px;
}

/* CARDS */

.workflow-card,
.document-card {
    padding: 20px;
    background: #111C30;
    border: 1px solid #263650;
    border-radius: 17px;
    height: 100%;
}

.workflow-card {
    min-height: 145px;
}

.workflow-number {
    width: 35px;
    height: 35px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    background: #123B43;
    color: #5EEAD4 !important;
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 13px;
}

.workflow-title,
.document-title {
    font-size: 16px;
    font-weight: 800;
    color: #F8FAFC !important;
    margin-bottom: 9px;
}

.workflow-description,
.document-description {
    color: #A8B7CC !important;
    font-size: 13px;
    line-height: 1.8;
}

.document-tag {
    font-size: 11px;
    color: #5EEAD4 !important;
    letter-spacing: 1.3px;
    font-weight: 800;
    margin-bottom: 12px;
}

/* RESULT CARDS */

.result-card {
    background: #111C30;
    border: 1px solid #263650;
    border-radius: 18px;
    padding: 23px;
    margin: 12px 0;
    color: #F1F5F9 !important;
    line-height: 1.95;
}

.result-title {
    font-size: 23px;
    font-weight: 800;
    color: #99F6E4 !important;
    margin: 10px 0 16px;
}

.eyebrow {
    color: #5EEAD4 !important;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.tip-card {
    background: #112B38;
    border: 1px solid #24515B;
    border-radius: 13px;
    padding: 14px 16px;
    margin-top: 28px;
    color: #D9F7F2 !important;
    font-size: 13px;
    line-height: 1.8;
}

/* METRICS */

div[data-testid="stMetric"] {
    background: #111C30 !important;
    border: 1px solid #263650 !important;
    border-radius: 15px !important;
    padding: 18px !important;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #A8B7CC !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div {
    color: #5EEAD4 !important;
}

/* EXPANDERS */

div[data-testid="stExpander"] {
    background: #111C30 !important;
    border: 1px solid #263650 !important;
    border-radius: 13px !important;
    overflow: hidden;
}

div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary span,
div[data-testid="stExpander"] summary p {
    color: #F1F5F9 !important;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #8495AD !important;
    font-size: 12px;
    line-height: 1.9;
    padding: 30px 0 8px;
    border-top: 1px solid #1E2C42;
    margin-top: 40px;
}

@media (max-width: 700px) {
    .hero {
        padding: 23px;
        border-radius: 18px;
    }

    .hero h1 {
        font-size: 31px;
    }

    .block-container {
        padding-left: 16px;
        padding-right: 16px;
    }

    .workflow-card {
        margin-bottom: 12px;
    }
}
</style>
""")


# =========================================================
# RUBRIC
# =========================================================

RUBRIC_TEXT = """
CAPSTONE JOURNAL RUBRIC

1. Learning Transfer

BLUE: Response shows strong evidence of synthesis of content presented
and insights gained throughout the entire Capstone.

GREEN: Response shows evidence of synthesis of content presented and
insights gained throughout the entire Capstone.

YELLOW: Response shows little evidence of synthesis of content presented
and insights gained throughout the entire Capstone.

RED: Response shows no evidence of synthesis of content presented and
insights gained throughout the entire Capstone.


2. Team Collaboration

BLUE: Response demonstrates understanding of the role of the individual
in a team setting, including insights about effective collaborative behavior.

GREEN: Response demonstrates understanding of the role of the individual
in a team setting.

YELLOW: Response demonstrates a vague understanding of the role of the
individual in a team setting.

RED: Response demonstrates no understanding of the role of the individual
in a team setting.


3. Using the Engineering Design Process

BLUE: Response provides a clear, insightful and well-developed reflection
on, and value judgment of, the step being addressed in the project.
The reflection would be clearly useful to others attempting the same
or similar project.

GREEN: Response provides a generally clear and adequately developed
reflection on, and value judgment of, the step being addressed in the project.
The reflection would be generally useful to others attempting the same
or similar project.

YELLOW: Response provides a general reflection of the step being addressed
in the project. The reflection would not be useful to others attempting
the same or similar project.

RED: Response does not address the question asked.


4. Personal Reflection

BLUE: Response reflects insightful viewpoints and interpretations.
Clear, detailed examples describe how the student understands their
personal growth.

GREEN: Response reflects viewpoints and interpretations.
Some examples describe how the student understands their personal growth.

YELLOW: Response reflects viewpoints and interpretations that are
unsupported or supported with flawed arguments.

RED: Response demonstrates a lack of understanding of personal growth.
"""


# =========================================================
# STRUCTURED OUTPUT MODELS
# =========================================================

class ConceptAnalysis(BaseModel):
    concept: str
    status: Literal["supported", "limited", "unsupported"]
    explanation: str
    evidence_quote: Optional[str] = None
    evidence_page: Optional[int] = None
    limitation: str


class ConnectionAnalysis(BaseModel):
    lo_title: str
    summary: str
    concept_analyses: List[ConceptAnalysis]
    strongest_connection: str
    missing_information: List[str]


class EssayDraft(BaseModel):
    title: str
    essay: str
    concepts_used: List[str]
    evidence_pages_used: List[int]
    personalization_notes: List[str]


class CriterionFeedback(BaseModel):
    criterion: str
    estimated_level: Literal[
        "BLUE",
        "GREEN",
        "YELLOW",
        "RED",
        "Insufficient evidence",
    ]
    evidence_from_essay: str
    reason: str
    improvement_suggestion: str
    reflective_question: str


class EssayFeedback(BaseModel):
    overall_estimated_level: Literal[
        "BLUE",
        "GREEN",
        "YELLOW",
        "RED",
        "Insufficient evidence",
    ]
    overall_summary: str
    criteria: List[CriterionFeedback]
    next_steps: List[str]
    caution: str


# =========================================================
# PDF PROCESSING
# =========================================================

def extract_pdf_pages(uploaded_file):

    doc = fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf",
    )

    pages = []

    for index, page in enumerate(doc):
        pages.append({
            "source": uploaded_file.name,
            "page": index + 1,
            "text": page.get_text("text").strip(),
        })

    doc.close()

    return pages


def all_text(pages):

    return "\n\n".join(
        f"[PDF page {page['page']}]\n{page['text']}"
        for page in pages
        if page["text"]
    )


def find_learning_outcome(curriculum_pages, lo_id):

    text = all_text(curriculum_pages)

    pattern = (
        r"(?m)^\s*"
        r"([A-Z]{2,5}\.\d{1,2}\.\d{2})"
        r"\s*[-–—:]?"
    )

    matches = list(re.finditer(pattern, text))
    normalized = lo_id.strip().upper()

    for index, match in enumerate(matches):

        if match.group(1).upper() == normalized:

            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            )

            return text[match.start():end].strip()[:10000]

    index = text.upper().find(normalized)

    if index >= 0:
        return text[
            max(0, index - 500):
            min(len(text), index + 5000)
        ]

    raise ValueError(
        f"Learning Outcome '{normalized}' was not found "
        "in the uploaded curriculum PDF. Check the code "
        "and ensure the PDF contains selectable text."
    )


def extract_concepts(lo_section):

    match = re.search(
        r"(?is)Key\s+Concepts\s*:\s*(.*?)"
        r"(?=\n\s*(?:Skills|Essential Questions|"
        r"Textbook and Resource Materials|Evidence of Learning|"
        r"Capstone Connection|Relations|SEC Topic|$))",
        lo_section,
    )

    if not match:
        return []

    concepts = []
    seen = set()

    for line in match.group(1).splitlines():

        line = re.sub(
            r"^\s*[-•*]?\s*\d+[.)]?\s*",
            "",
            line,
        ).strip(" \t:;")

        if not line or len(line) > 140:
            continue

        key = re.sub(r"\W+", "", line.casefold())

        if key and key not in seen:
            seen.add(key)
            concepts.append(line)

    return concepts[:40]


# =========================================================
# EVIDENCE RETRIEVAL
# =========================================================

def retrieve_evidence(capstone_pages, query, top_k=5):

    chunks = [
        {
            "source": page["source"],
            "page": page["page"],
            "text": page["text"],
        }
        for page in capstone_pages
        if page["text"].strip()
    ]

    if not chunks:
        raise ValueError(
            "No selectable text was extracted from the "
            "Capstone PDF. Scanned documents may require OCR."
        )

    corpus = [chunk["text"] for chunk in chunks]

    if len(corpus) == 1:
        return chunks

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000,
    )

    matrix = vectorizer.fit_transform(corpus)

    scores = cosine_similarity(
        vectorizer.transform([query]),
        matrix,
    ).ravel()

    order = scores.argsort()[::-1]

    selected = [
        dict(chunks[index], score=float(scores[index]))
        for index in order[:top_k]
        if scores[index] > 0
    ]

    if selected:
        return selected

    return [
        dict(chunk, score=0.0)
        for chunk in sorted(
            chunks,
            key=lambda item: len(item["text"]),
            reverse=True,
        )[:min(3, len(chunks))]
    ]


def format_evidence(evidence):

    return "\n\n---\n\n".join(
        (
            f"[SOURCE: {item['source']} "
            f"| PDF PAGE: {item['page']}]\n"
            f"{item['text'][:7000]}"
        )
        for item in evidence
    )


def word_count(text):

    return len(
        re.findall(
            r"\b[\w]+(?:['’\-][\w]+)*\b",
            text,
        )
    )


# =========================================================
# GEMINI
# =========================================================

def make_llm(api_key, model_name):

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
        max_retries=2,
    )


# =========================================================
# GENERATE CONNECTION ESSAY
# =========================================================

def run_generation(
    llm,
    lo_id,
    lo_section,
    concepts,
    evidence,
):

    analysis_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a careful STEM curriculum-to-capstone analyst.

Use only the supplied curriculum excerpt, concepts, and Capstone evidence.

Do not invent project design, materials, measurements, results,
or team actions.

A shared word is not proof of a scientific connection.

Distinguish direct evidence from cautious inference.

Return one analysis for each supplied concept.

Use:
- supported when documents directly support the relationship
- limited when a plausible connection needs missing details
- unsupported when no defensible connection is established

Evidence quotes must be copied verbatim from the Capstone evidence.

If no quote supports a claim, return null quote and page.

Never overclaim that an indicator proves food safety.
            """,
        ),
        (
            "human",
            """
LO ID:
{lo_id}

CURRICULUM EXCERPT:
{lo_section}

EXACT CONCEPT LIST:
{concepts}

CAPSTONE EVIDENCE:
{evidence}

Return concise structured analysis.
            """,
        ),
    ])

    analysis = (
        analysis_prompt
        | llm.with_structured_output(ConnectionAnalysis)
    ).invoke({
        "lo_id": lo_id,
        "lo_section": lo_section,
        "concepts": (
            "\n".join(f"- {concept}" for concept in concepts)
            if concepts
            else "No reliable concept list could be extracted."
        ),
        "evidence": format_evidence(evidence),
    })

    essay_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
Write a model essay for a STEM high-school Capstone Journal.

The essay body MUST contain 200–250 words, excluding the title.

Use only the curriculum excerpt, analysis, and Capstone evidence
for claims about this project.

Do not invent materials, measurements, results, personal actions,
or team experiences.

Write clear, formal, accessible English.

Explain the scientific idea, connect it to an actual Capstone
requirement or design choice, propose measurements or control
comparisons that could test the connection, acknowledge limitations,
and conclude with the value of evidence-based engineering.

If the connection is weak, frame it as a possible application
or investigation, not a proven result.

Do not assign a rubric color.

Do not write unsupported first-person claims.
            """,
        ),
        (
            "human",
            """
LO ID:
{lo_id}

CURRICULUM:
{lo_section}

CONCEPT ANALYSIS:
{analysis}

CAPSTONE EVIDENCE:
{evidence}

Create one integrated 200–250 word essay.

Return source page numbers and details the student should personalize.
            """,
        ),
    ])

    draft = (
        essay_prompt
        | llm.with_structured_output(EssayDraft)
    ).invoke({
        "lo_id": lo_id,
        "lo_section": lo_section,
        "analysis": analysis.model_dump_json(indent=2),
        "evidence": format_evidence(evidence),
    })

    count = word_count(draft.essay)

    if not 200 <= count <= 250:

        repair_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
Revise the essay body to 200–250 words.
Do not add unsupported project facts.
Return the same structured schema.
                """,
            ),
            (
                "human",
                """
Current word count: {count}

ESSAY:
{essay}

EVIDENCE CONTEXT:
{context}
                """,
            ),
        ])

        draft = (
            repair_prompt
            | llm.with_structured_output(EssayDraft)
        ).invoke({
            "count": count,
            "essay": draft.essay,
            "context": (
                analysis.model_dump_json(indent=2)
                + "\n"
                + format_evidence(evidence)
            ),
        })

    final_count = word_count(draft.essay)

    if not 200 <= final_count <= 250:
        raise ValueError(
            f"The model returned {final_count} words. "
            "Expected 200–250 words. Please try again."
        )

    return analysis, draft


# =========================================================
# RUBRIC EVALUATION
# =========================================================

def run_feedback(llm, essay):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a formative writing coach.

Use only the supplied rubric and student's essay.

Evaluate all four criteria:
1. Learning Transfer
2. Team Collaboration
3. Using the Engineering Design Process
4. Personal Reflection

Use the exact levels:
BLUE, GREEN, YELLOW, RED.

Use Insufficient evidence when the essay does not provide enough
information for a defensible level.

For every criterion:
- identify evidence from the essay
- explain the estimated level using the rubric
- give a concrete improvement suggestion
- ask a reflective question

Do not invent the student's actions, role, feelings, growth,
or project results.

Do not rewrite the essay.

The overall level is formative feedback, not an official grade.
            """,
        ),
        (
            "human",
            """
RUBRIC:
{rubric}

STUDENT ESSAY:
{essay}

Provide structured formative feedback for all four criteria.
            """,
        ),
    ])

    return (
        prompt
        | llm.with_structured_output(EssayFeedback)
    ).invoke({
        "rubric": RUBRIC_TEXT,
        "essay": essay,
    })


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    render_html("""
    <div style="padding:8px 0 22px">

        <div style="
            font-size:22px;
            font-weight:800;
            color:#5EEAD4;
            margin-top:5px">
            STEM Connect AI
        </div>

        <div style="
            color:#A8B7CC;
            font-size:12px;
            line-height:1.7;
            margin-top:6px">
            Your curriculum-to-capstone writing partner.
        </div>

    </div>
    """)

    st.markdown("### AI Configuration")

    api_key = st.text_input(
        "Google Gemini API key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Keep your API key private.",
    )

    model_name = st.text_input(
        "Gemini model",
        value=os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        ),
    )

    st.divider()

    st.markdown("### How it works")

    st.markdown("""
    **01 — Upload**

    Add your subject and Capstone PDFs.

    **02 — Connect**

    Enter the exact Learning Outcome code.

    **03 — Create**

    Generate an essay or review your own.
    """)

    st.divider()

    render_html("""
    <div style="
        background:#16243B;
        border:1px solid #30445F;
        border-radius:12px;
        padding:13px;
        color:#CBD5E1;
        font-size:12px;
        line-height:1.7">
        Keep your API key private. Never share it with anyone.
    </div>
    """)


# =========================================================
# HERO
# =========================================================

render_html("""
<div class="hero">

    <span class="hero-badge">
        BUILT FOR STEM STUDENTS
    </span>

    <h1>
        Turn what you learn<br>
        into what you build.
    </h1>

    <p>
        Connect your scientific Learning Outcomes to your
        Capstone project. Generate a focused 200–250 word
        connection essay or get thoughtful feedback using
        your Capstone Journal Rubric.
    </p>

    <div class="hero-pills">
        <span class="hero-pill">PDF-based workflow</span>
        <span class="hero-pill">LO + Capstone connection</span>
        <span class="hero-pill">Rubric-guided feedback</span>
    </div>

</div>
""")


# =========================================================
# WORKFLOW
# =========================================================

render_html("""
<div class="section-title">Your workflow</div>

<div class="section-subtitle">
    Three simple steps to connect scientific knowledge
    with your engineering project.
</div>
""")

columns = st.columns(3, gap="medium")

workflow_items = [
    (
        "01",
        "Upload your documents",
        "Add the scientific subject PDF and your Capstone project PDF.",
    ),
    (
        "02",
        "Select your Learning Outcome",
        "Enter the exact LO code found in your curriculum document.",
    ),
    (
        "03",
        "Generate or evaluate",
        "Create a connection essay or receive rubric-based feedback.",
    ),
]

for col, item in zip(columns, workflow_items):

    number, title, description = item

    with col:

        render_html(f"""
        <div class="workflow-card">

            <div class="workflow-number">
                {number}
            </div>

            <div class="workflow-title">
                {safe(title)}
            </div>

            <div class="workflow-description">
                {safe(description)}
            </div>

        </div>
        """)


# =========================================================
# DOCUMENT UPLOADS
# =========================================================

render_html("""
<div class="section-title">Your workspace</div>

<div class="section-subtitle">
    Upload the two documents you want STEM Connect AI to analyze.
    Your files determine the content of the analysis.
</div>
""")

up1, up2 = st.columns(2, gap="large")

with up1:

    render_html("""
    <div class="document-card">

        <div class="document-tag">DOCUMENT 01</div>

        <div class="document-title">
            Scientific subject
        </div>

        <div class="document-description">
            Curriculum, Learning Outcomes, scientific concepts,
            and skills.
        </div>

    </div>
    """)

    curriculum_file = st.file_uploader(
        "Upload scientific subject PDF",
        type=["pdf"],
        key="curriculum",
    )

    if curriculum_file:
        st.success(f"Ready: {curriculum_file.name}")


with up2:

    render_html("""
    <div class="document-card">

        <div class="document-tag">DOCUMENT 02</div>

        <div class="document-title">
            Capstone project
        </div>

        <div class="document-description">
            Project requirements, design choices, evidence,
            and project progress.
        </div>

    </div>
    """)

    capstone_file = st.file_uploader(
        "Upload Capstone project PDF",
        type=["pdf"],
        key="capstone",
    )

    if capstone_file:
        st.success(f"Ready: {capstone_file.name}")


# =========================================================
# WORKFLOW SELECTION
# =========================================================

render_html("""
<div class="section-title">
    What would you like to do?
</div>

<div class="section-subtitle">
    Choose a workflow and enter the Learning Outcome code.
</div>
""")

mode = st.radio(
    "Select a workflow",
    [
        "Generate a connection essay",
        "Evaluate my own essay",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

lo_col, tip_col = st.columns(
    [1, 1.2],
    gap="large",
)

with lo_col:

    lo_id = st.text_input(
        "Learning Outcome code",
        placeholder="e.g. CH.2.04",
    ).strip().upper()

with tip_col:

    render_html("""
    <div class="tip-card">
        <b>Tip:</b> Enter the code exactly as it appears
        in your subject PDF. The app uses the document you
        upload rather than a fixed Chemistry file.
    </div>
    """)


# =========================================================
# ESSAY INPUT
# =========================================================

student_essay = ""

if mode == "Evaluate my own essay":

    render_html("""
    <div class="section-title">Your essay</div>

    <div class="section-subtitle">
        Paste your own work. The app will evaluate it using
        the rubric without rewriting your essay.
    </div>
    """)

    student_essay = st.text_area(
        "Essay text",
        height=240,
        placeholder="Paste your essay here...",
        label_visibility="collapsed",
    )


# =========================================================
# ACTION BUTTON
# =========================================================

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

button_label = (
    "Generate my connection essay"
    if mode == "Generate a connection essay"
    else "Evaluate my essay with the rubric"
)


# =========================================================
# PROCESS REQUEST
# =========================================================

if st.button(
    button_label,
    type="primary",
    use_container_width=True,
):

    if not api_key:
        st.error(
            "Please add your Google Gemini API key in the sidebar."
        )
        st.stop()

    if not curriculum_file or not capstone_file:
        st.error(
            "Please upload both the scientific subject PDF "
            "and the Capstone PDF."
        )
        st.stop()

    if not lo_id:
        st.error(
            "Please enter the Learning Outcome code."
        )
        st.stop()

    if (
        mode == "Evaluate my own essay"
        and not student_essay.strip()
    ):
        st.error(
            "Please paste your essay before requesting feedback."
        )
        st.stop()

    try:

        with st.spinner(
            "Reading your PDFs and preparing the analysis..."
        ):

            curriculum_pages = extract_pdf_pages(
                curriculum_file
            )

            capstone_pages = extract_pdf_pages(
                capstone_file
            )

            if not any(
                page["text"].strip()
                for page in curriculum_pages
            ):
                raise ValueError(
                    "No selectable text was extracted from the "
                    "subject PDF. It may require OCR."
                )

            if not any(
                page["text"].strip()
                for page in capstone_pages
            ):
                raise ValueError(
                    "No selectable text was extracted from the "
                    "Capstone PDF. It may require OCR."
                )

            lo_section = find_learning_outcome(
                curriculum_pages,
                lo_id,
            )

            concepts = extract_concepts(
                lo_section
            )

            llm = make_llm(
                api_key,
                model_name,
            )

        with st.expander(
            "Review extracted Learning Outcome",
            expanded=False,
        ):

            st.code(
                lo_section[:8000],
                language="text",
            )

            if concepts:

                st.markdown("**Detected key concepts**")

                for concept in concepts:
                    st.markdown(f"- {safe(concept)}")

            else:

                st.warning(
                    "The app could not reliably detect a Key "
                    "Concepts list. It will still use the LO excerpt."
                )

        # =================================================
        # GENERATE ESSAY
        # =================================================

        if mode == "Generate a connection essay":

            with st.spinner(
                "Finding relevant Capstone evidence and "
                "generating your essay..."
            ):

                query = (
                    f"{lo_id} {lo_section} "
                    f"{' '.join(concepts)} "
                    "Capstone project design materials "
                    "measurements experiment control variables results"
                )

                evidence = retrieve_evidence(
                    capstone_pages,
                    query,
                    top_k=5,
                )

                analysis, draft = run_generation(
                    llm,
                    lo_id,
                    lo_section,
                    concepts,
                    evidence,
                )

            st.success(
                "Your essay is ready. Review it and personalize "
                "details that only you know."
            )

            render_html("""
            <div class="section-title">
                Your connection essay
            </div>

            <div class="section-subtitle">
                Review the generated draft before using it.
            </div>
            """)

            render_html(f"""
            <div class="result-card">

                <div class="eyebrow">GENERATED DRAFT</div>

                <div class="result-title">
                    {safe(draft.title)}
                </div>

                <div style="
                    color:#E2E8F0;
                    font-size:15px;
                    line-height:1.95">
                    {safe(draft.essay).replace(chr(10), '<br>')}
                </div>

            </div>
            """)

            count = word_count(draft.essay)

            evidence_pages = len({
                item["page"]
                for item in evidence
            })

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Word count",
                count,
                "Target: 200–250",
            )

            m2.metric(
                "Evidence pages",
                evidence_pages,
                "Retrieved from Capstone",
            )

            m3.metric(
                "Concepts checked",
                len(analysis.concept_analyses),
                "LO connection review",
            )

            st.download_button(
                "Download essay as TXT",
                data=(
                    f"{draft.title}\n\n"
                    f"{draft.essay}\n\n"
                    f"Word count: {count}\n"
                ),
                file_name="stem_connection_essay.txt",
                mime="text/plain",
                use_container_width=True,
            )

            if draft.personalization_notes:

                render_html("""
                <div class="section-title">
                    Before you submit
                </div>

                <div class="section-subtitle">
                    Check these details and personalize them
                    using your real project experience.
                </div>
                """)

                for note in draft.personalization_notes:
                    st.markdown(f"- {safe(note)}")

            # CONNECTION ANALYSIS

            render_html("""
            <div class="section-title">
                Connection analysis
            </div>
            """)

            render_html(f"""
            <div class="result-card">

                <div class="eyebrow">SUMMARY</div>

                <p>{safe(analysis.summary)}</p>

                <div class="eyebrow">
                    STRONGEST DEFENSIBLE CONNECTION
                </div>

                <p>{safe(analysis.strongest_connection)}</p>

            </div>
            """)

            if analysis.missing_information:

                st.markdown("### Missing information and limitations")

                for item in analysis.missing_information:
                    st.markdown(f"- {safe(item)}")

            # CONCEPT ANALYSIS

            render_html("""
            <div class="section-title">
                Concept-by-concept check
            </div>

            <div class="section-subtitle">
                Review which scientific connections are supported
                by your project evidence.
            </div>
            """)

            for item in analysis.concept_analyses:

                with st.container(border=True):

                    badge = {
                        "supported": "SUPPORTED",
                        "limited": "LIMITED",
                        "unsupported": "UNSUPPORTED",
                    }.get(item.status, item.status.upper())

                    st.markdown(f"### {safe(item.concept)}")
                    st.markdown(f"**Status:** {badge}")
                    st.write(item.explanation)

                    if item.evidence_quote:

                        st.markdown(
                            f"> {item.evidence_quote}"
                        )

                        if item.evidence_page:
                            st.caption(
                                f"Capstone PDF page {item.evidence_page}"
                            )

                    st.markdown(
                        f"**Limitation:** {item.limitation}"
                    )

            # SOURCE EVIDENCE

            render_html("""
            <div class="section-title">
                Retrieved Capstone evidence
            </div>

            <div class="section-subtitle">
                Source pages selected as potentially relevant
                to your Learning Outcome.
            </div>
            """)

            for item in evidence:

                with st.expander(
                    f"{item['source']} | "
                    f"Page {item['page']} | "
                    f"Match score {item.get('score', 0):.3f}"
                ):

                    st.write(item["text"][:5000])

        # =================================================
        # EVALUATE ESSAY
        # =================================================

        else:

            with st.spinner(
                "Comparing your essay with all four rubric criteria..."
            ):

                feedback = run_feedback(
                    llm,
                    student_essay,
                )

            st.success(
                "Your rubric-based feedback is ready."
            )

            render_html("""
            <div class="section-title">
                Your rubric feedback
            </div>

            <div class="section-subtitle">
                A formative estimate to help you improve.
                This is not an official grade.
            </div>
            """)

            level = feedback.overall_estimated_level

            color = LEVEL_COLORS.get(
                level,
                "#94A3B8",
            )

            render_html(f"""
            <div class="result-card"
                 style="border-color:{color}">

                <div class="eyebrow">
                    OVERALL ESTIMATED LEVEL
                </div>

                <div style="
                    color:{color};
                    font-size:38px;
                    font-weight:800;
                    margin:12px 0">
                    {safe(level)}
                </div>

                <div style="
                    color:#A8B7CC;
                    font-size:13px">
                    Formative feedback — Not an official grade
                </div>

                <p>{safe(feedback.overall_summary)}</p>

            </div>
            """)

            render_html("""
            <div class="section-title">
                Criteria breakdown
            </div>

            <div class="section-subtitle">
                Each criterion is evaluated separately
                against the supplied Capstone Journal Rubric.
            </div>
            """)

            for item in feedback.criteria:

                color = LEVEL_COLORS.get(
                    item.estimated_level,
                    "#94A3B8",
                )

                with st.container(border=True):

                    left, right = st.columns([3, 1])

                    with left:
                        st.markdown(f"### {safe(item.criterion)}")

                    with right:

                        render_html(f"""
                        <div style="
                            text-align:right;
                            padding-top:10px">

                            <span style="
                                display:inline-block;
                                background:{color};
                                color:#FFFFFF;
                                border-radius:30px;
                                padding:7px 11px;
                                font-size:11px;
                                font-weight:800">
                                {safe(item.estimated_level)}
                            </span>

                        </div>
                        """)

                    st.markdown("**Evidence from your essay**")
                    st.write(item.evidence_from_essay)

                    st.markdown("**Why this level?**")
                    st.write(item.reason)

                    st.markdown("**How to improve**")
                    st.write(item.improvement_suggestion)

                    st.markdown("**Question to think about**")
                    st.info(item.reflective_question)

            render_html("""
            <div class="section-title">
                Your next steps
            </div>
            """)

            for step in feedback.next_steps:
                st.markdown(f"- {safe(step)}")

            st.warning(feedback.caution)

            report_parts = [
                f"Overall estimated level: {level}",
                feedback.overall_summary,
            ]

            for criterion in feedback.criteria:

                report_parts.append(
                    f"\n{criterion.criterion}: "
                    f"{criterion.estimated_level}\n"
                    f"Evidence: {criterion.evidence_from_essay}\n"
                    f"Reason: {criterion.reason}\n"
                    f"Suggestion: {criterion.improvement_suggestion}\n"
                    f"Question: {criterion.reflective_question}"
                )

            report_parts.append(
                "\nNext steps:\n"
                + "\n".join(
                    f"- {step}"
                    for step in feedback.next_steps
                )
            )

            report_parts.append(feedback.caution)

            st.download_button(
                "Download rubric feedback as TXT",
                data="\n".join(report_parts),
                file_name="stem_essay_feedback.txt",
                mime="text/plain",
                use_container_width=True,
            )

    except Exception as exc:

        st.error(f"Something went wrong: {exc}")

        st.caption(
            "Check your API key, model name, PDF text extraction, "
            "and exact LO code. Scanned PDFs may require OCR."
        )


# =========================================================
# FOOTER
# =========================================================

render_html("""
<div class="footer">
    STEM CONNECT AI
    <br>
    Student writing support | Evidence-based learning
    <br><br>
    Always verify scientific claims and personalize
    reflections using your real project experience.
</div>
""")