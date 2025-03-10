import streamlit as st
import os
import re
import pdfplumber
import fitz  # PyMuPDF
import json
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("Error: GROQ_API_KEY is missing. Please set it in your .env file.")
    st.stop()

client = Groq(api_key=api_key)

def extract_text_with_pdfplumber(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
            if text.strip():
                return text
    except Exception as e:
        st.warning(f"pdfplumber failed: {e}")
    return None

def extract_text_with_fitz(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)  # type: ignore
        if text.strip():
            return text 
    except Exception as e:
        st.warning(f"fitz (PyMuPDF) failed: {e}")
    return None

def extract_entities_with_groq(text):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI that extracts structured data from resumes. Output should be in JSON format only. Exclude descriptions and work done in job. Do not give anything else as output."},
                {"role": "user", "content": f"Extract key information (like name, contact, skills, education, projects, certifications, and experience) from the following resume:\n{text}"}
            ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling Groq API: {e}")
    return None


def convert_llm_output_to_dict(llm_output):
    try:
        cleaned_output = re.sub(r"```(json)?", "", llm_output).strip()
        return json.loads(cleaned_output)
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None

def save_to_json(data, output_path="extracted_resumes.json"):
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as file:
            try:
                resumes = json.load(file)
            except json.JSONDecodeError:
                resumes = []
    else:
        resumes = []

    resumes.append(data)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(resumes, file, indent=4)

    st.success(f"Resume data saved to {output_path}")

# Streamlit UI
st.title("📄 Resume Scanner with Groq & Streamlit")
st.write("Upload a PDF resume, and I’ll extract the structured data for you!")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.info("File uploaded successfully!")
    with st.spinner("Extracting text..."):
        # Save the uploaded file to a temporary location
        temp_path = "temp_resume.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Extract text from PDF
        resume_text = extract_text_with_pdfplumber(temp_path)
        
        if not resume_text:
            st.warning("pdfplumber extraction failed or returned empty text. Trying fitz...")
            resume_text = extract_text_with_fitz(temp_path)
        
        if not resume_text:
            st.error("Error: No text extracted from the PDF. Check the file content.")
        else:
            # Extract structured entities with Groq
            extracted_info = extract_entities_with_groq(resume_text)
            structured_data = convert_llm_output_to_dict(extracted_info)

            if structured_data:
                st.success("✅ Resume information extracted successfully!")
                st.json(structured_data)

                # Save the structured data to a JSON file
                save_to_json(structured_data)

                # Provide a download link for the extracted JSON
                st.download_button(
                    label="Download Extracted JSON",
                    data=json.dumps(structured_data, indent=4),
                    file_name="extracted_resume.json",
                    mime="application/json"
                )

    # Cleanup
    os.remove(temp_path)
