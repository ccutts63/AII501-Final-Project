import streamlit as st
import fitz  # PyMuPDF
from google import genai

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
                    prompt = f"Based on the following course material, generate 3 multiple-choice study questions with answers:\n\n{text[:3000]}"
                    
                    response = client.models.generate_content(
                        model="gemini-3.5-flash-lite",  # Switched to flash-lite to avoid high-demand queues
                        contents=prompt
                    )
                    
                    st.markdown("### Generated Practice Quiz")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error generating quiz: {e}")
    else:
        st.info("Please enter your Gemini API key in the sidebar to generate quiz questions.")