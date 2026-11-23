"""RECCALL: PDF-to-quiz generation plus a separately trained Bloom classifier."""

import json
from pathlib import Path

import fitz  # PyMuPDF
from google import genai
import joblib
import streamlit as st

from quiz_utils import parse_quiz

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "bloom_classifier.joblib"
METRICS_FILE = BASE_DIR / "models" / "metrics.json"
BLOOM_EXPLANATIONS = {
    "Remember": "Recall facts, definitions, or basic information.",
    "Understand": "Explain or interpret the meaning of a concept.",
    "Apply": "Use learned information in a practical situation.",
    "Analyze": "Examine patterns, relationships, or components.",
    "Evaluate": "Make and justify a judgement using criteria.",
    "Create": "Develop or design something new.",
}


@st.cache_resource
def load_classifier(model_mtime: float):
    """Cache the trained local model across Streamlit reruns."""
    return joblib.load(MODEL_FILE)  # Load only a model trained by your team.


def extract_text(uploaded_pdf) -> str:
    with fitz.open(stream=uploaded_pdf.getvalue(), filetype="pdf") as document:
        return "\n".join(page.get_text() for page in document).strip()


def make_prompt(course_text: str) -> str:
    return f"""Based ONLY on the course text below, generate exactly 3 distinct
multiple-choice study questions (4 options each). Vary cognitive demands where possible.
Do not invent answers that cannot be supported by the text.

Return only a JSON array of exactly three objects. Each object must have:
- question: the question text
- options: an object with keys A, B, C, D, each with answer text
- answer: one of A, B, C, D
- hint: short hint that does not give away the answer
- explanation: short explanation of the correct answer

Course material:
{course_text[:12000]}"""


st.set_page_config(page_title="RECCALL AI", page_icon="📚")
st.title("RECCALL")
st.caption("PDF Study Assistant | Gemini quiz generation + our trained Bloom's Taxonomy classifier")

api_key = st.sidebar.text_input("Gemini API key", type="password")
model_name = st.sidebar.text_input("Gemini model", value="gemini-3.6-flash")

classifier = None
if MODEL_FILE.is_file():
    try:
        classifier = load_classifier(MODEL_FILE.stat().st_mtime)
        st.sidebar.success("Trained Bloom classifier loaded")
    except Exception as exc:
        st.sidebar.error(f"Could not load trained model: {exc}")
else:
    st.sidebar.warning("Classifier not trained yet. Run: python train_model.py")

if METRICS_FILE.is_file():
    try:
        scores = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        gold = scores["gold_evaluation"]
        st.sidebar.caption(
            f"Held-out human-label accuracy: {gold['accuracy']:.1%} "
            f"({gold['samples']} questions) | Macro F1: {gold['macro_f1']:.3f}"
        )
    except (ValueError, KeyError, OSError):
        st.sidebar.caption("Evaluation metrics unavailable")

# A standalone demo makes the ML contribution visible even without Gemini/API access.
if classifier is not None:
    with st.sidebar.expander("Try our trained model (no API key needed)"):
        demo_question = st.text_area(
            "Enter an educational question",
            value="Explain how a decision tree makes predictions.",
        )
        if demo_question.strip():
            demo_label = str(classifier.predict([demo_question.strip()])[0])
            st.write(f"Predicted Bloom level: **{demo_label}**")
            st.caption(BLOOM_EXPLANATIONS.get(demo_label, ""))

uploaded_file = st.file_uploader("Upload course material (PDF)", type=["pdf"])
if uploaded_file is not None:
    try:
        text = extract_text(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read this PDF: {exc}")
        text = ""

    if text:
        st.success(f"Extracted {len(text):,} characters from {uploaded_file.name}")
        if len(text) > 12000:
            st.caption("Quiz generation currently uses the first 12,000 characters of this PDF.")
    else:
        st.warning("No selectable text was found. Scanned/image-only PDFs are not supported yet.")

    if not api_key:
        st.info("Enter a Gemini API key in the sidebar to generate a quiz.")
    if classifier is None:
        st.info("Run 'python train_model.py' once before generating labelled quizzes.")

    if st.button(
        "Generate Practice Quiz",
        type="primary",
        disabled=not (bool(text) and bool(api_key) and classifier is not None),
    ):
        with st.spinner("Generating and classifying questions..."):
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=model_name.strip(),
                    contents=make_prompt(text),
                    config={"response_mime_type": "application/json"},
                )
                quiz_data = parse_quiz(response.text)
                # This is OUR scikit-learn model — Gemini does not assign these labels.
                predictions = classifier.predict([q["question"] for q in quiz_data])
                for question, label in zip(quiz_data, predictions):
                    question["bloom_level"] = str(label)
                st.session_state["quiz_data"] = quiz_data
                st.session_state["quiz_pdf_name"] = uploaded_file.name
                st.session_state["quiz_nonce"] = st.session_state.get("quiz_nonce", 0) + 1
                st.success("Quiz generated and classified!")
            except Exception as exc:
                st.error(f"Could not generate a quiz: {exc}")
                st.caption("Check the Gemini API key, model name and your network connection.")

# Hide an old quiz if the user has uploaded a different PDF.
if "quiz_data" in st.session_state and (
    uploaded_file is None or st.session_state.get("quiz_pdf_name") == uploaded_file.name
):
    st.subheader("Practice Quiz")
    st.caption("Bloom levels are predicted by our independently trained TF-IDF + Logistic Regression model.")
    nonce = st.session_state.get("quiz_nonce", 0)
    for index, question in enumerate(st.session_state["quiz_data"]):
        st.markdown(f"#### Question {index + 1}")
        st.write(question["question"])
        level = question.get("bloom_level", "Unclassified")
        st.caption(f"**Predicted Bloom level: {level}** — {BLOOM_EXPLANATIONS.get(level, '')}")

        answer = st.radio(
            "Choose your answer:",
            options=list("ABCD"),
            format_func=lambda option, opts=question["options"]: f"{option}) {opts[option]}",
            index=None,
            key=f"question_{nonce}_{index}",
        )
        if st.button("Submit Answer", key=f"submit_{nonce}_{index}"):
            if answer is None:
                st.warning("Please choose an answer first.")
            elif answer == question["answer"]:
                st.success("Correct!")
                st.info(question["explanation"])
            else:
                st.warning("Not quite. Try again!")
                st.info(f"Hint: {question['hint']}")
        st.divider()
