import streamlit as st
import os
import re
import pdfplumber
import fitz  # PyMuPDF
import json
from groq import Groq
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions


chroma_client = chromadb.PersistentClient(path="chroma_resume.db")
embedding_func = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="resumes2",
    embedding_function=embedding_func # type: ignore
)

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

def save_to_chromadb(data):
    try:
        if not data:
            st.error("NO DATA")
            return
        resume_dict = convert_llm_output_to_dict(data)
        if not resume_dict:
            st.error("Failed to parse LLM output")
            return
        
        candidate_name = resume_dict.get("name", "Unknown").replace(' ','_').lower()
        existing_ids = collection.peek()["ids"]

        if candidate_name in existing_ids:
            st.warning(f"Resume for '{candidate_name}' already exists. Skipping insert.")
            return

        summary = json.dumps(resume_dict)
        collection.add(
            documents=[summary],
            metadatas=[{"name": resume_dict.get("name", "Unknown")}],
            ids=[candidate_name]
        )
        st.write(f"Resume data for '{candidate_name}; saved to ChromaDB")
    except Exception as e:
        st.error(f"Error saving to ChromaDB: {e}")


def chroma_query():
    results = collection.query(
        query_texts=[""],
        n_results=5,
        include=["embeddings","distances","documents"] # type: ignore
    )
    return results


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

            if extracted_info:
                st.success("✅ Resume information extracted successfully!")
                #st.json(extracted_info)

                # Save the structured data to a JSON file
                save_to_chromadb(extracted_info)
                st.write(chroma_query())
                # Provide a download link for the extracted JSON
                st.download_button(
                    label="Download Extracted JSON",
                    data=json.dumps(extracted_info, indent=4),
                    file_name="extracted_resume.json",
                    mime="application/json"
                )

    # Cleanup
    os.remove(temp_path)
