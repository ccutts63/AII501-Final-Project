import streamlit as st
import fitz  # PyMuPDF
from google import genai
import json # Used to process Gemini's structured quiz response

st.set_page_config(page_title="RECCALL AI")

st.title("RECCALL")
st.caption("AI-Powered Spaced Repetition & Study Assistant")

# Sidebar for API Key
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

# File Upload
uploaded_file = st.file_uploader("Upload Course Material (PDF)", type=["pdf"])

if uploaded_file:
    # Extract Text using PyMuPDF
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    
    st.success(f"Extracted {len(text)} characters from {uploaded_file.name}")
    
    if api_key:
        # Generate Practice Questions via official Gemini SDK
        if st.button("Generate Practice Quiz"):
            with st.spinner("AI is generating questions..."):
                try:
                    # Initialize the official Google GenAI client
                    client = genai.Client(api_key=api_key)
                    prompt = f"""
                    Based on the following course material, generate exactly 3 multiple-choice study questions.

                    Return ONLY valid JSON in this exact format:

                    [
                    {{
                        "question": "Question text",
                        "options": {{
                        "A": "Option A",
                        "B": "Option B",
                        "C": "Option C",
                        "D": "Option D"
                        }},
                        "answer": "A",
                        "hint": "A short hint that helps without revealing the answer",
                        "explanation": "A short explanation of why the answer is correct"
                    }}
                    ]

                    Do not include markdown or any text outside the JSON.

                    Course material:
                    {text[:3000]}
                    """
                    response = client.models.generate_content(
                        model="gemini-3.6-flash",  # Switched models after encountering high-demand errors
                        contents=prompt
                    )

                    # Convert Gemini's JSON response into Python data
                    quiz_data = json.loads(response.text)

                    # Store the quiz so it remains available when Streamlit reruns the page
                    st.session_state["quiz_data"] = quiz_data

                    st.success("Quiz generated successfully!")
                except Exception as e:
                    st.error(f"Error generating quiz: {e}")
    else:
        st.info("Please enter your Gemini API key in the sidebar to generate quiz questions.")

# Display the generated quiz after it has been saved
if "quiz_data" in st.session_state:
    st.markdown("### Practice Quiz")

    # Loop through each generated question
    for i, question in enumerate(st.session_state["quiz_data"]):
        st.markdown(f"#### Question {i + 1}")
        st.write(question["question"])

        # Let the student choose one of the four possible answers
        selected_answer = st.radio(
            "Choose your answer:",
            options=["A", "B", "C", "D"],
            format_func=lambda option: f"{option}) {question['options'][option]}",
            index=None,
            key=f"question_{i}"
        )

        # Check the student's answer when Submit Answer is clicked
        if st.button("Submit Answer", key=f"submit_{i}"):

            # Make sure the student selected an answer first
            if selected_answer is None:
                st.warning("Please choose an answer first.")

            # Correct answer
            elif selected_answer == question["answer"]:
                st.success("Correct!")
                st.info(question["explanation"])

            # Wrong answer: provide a hint without revealing the answer
            else:
                st.warning("Not quite. Try again!")
                st.info(f"Hint: {question['hint']}")

        # Separate each quiz question visually
        st.divider()