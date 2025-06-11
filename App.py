# ========= Environment Setup =========
import os
import sys
import asyncio
from pathlib import Path

os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
os.environ['LANGUAGETOOL_HOME'] = str(Path.home() / '.cache' / 'language_tool')
os.environ['LANGUAGETOOL_SERVER'] = 'http://localhost:8081'

if sys.platform.startswith('win') and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ========= Imports =========
from dotenv import load_dotenv
load_dotenv()
import re
import io
from io import BytesIO
import json
import base64
import datetime
from typing import List

# PDF & Resume Handling
import pdfplumber
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from xhtml2pdf import pisa

# Data & NLP
import pandas as pd
import numpy as np
import nltk
import yake
import spacy
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# Visualization
import plotly.express as px

# LLM API
import google.generativeai as genai

# Web App
import streamlit as st

# ========= Streamlit Config =========
st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📄",
    layout="wide"
)
# Database
import pymysql

# ========= NLTK Setup =========
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ========= NumPy Check =========
try:
    print("NumPy version:", np.__version__)
except ImportError as e:
    st.error("NumPy is not installed. Please install it using: pip install numpy")
    sys.exit(1)

# ========= Model Loaders =========
nlp = spacy.load("en_core_web_sm")

@st.cache_resource
def load_models():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading SentenceTransformer: {e}")
        return None

sentence_model = load_models()

@st.cache_resource
def load_keyword_models():
    try:
        kw_model = KeyBERT()
        yake_extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, top=30)
        return kw_model, yake_extractor
    except Exception as e:
        st.warning(f"Keyword model loading failed: {e}")
        return None, None

kw_model, yake_extractor = load_keyword_models()

def preprocess(text: str) -> str:
    doc = nlp(text.lower())
    return " ".join(token.lemma_ for token in doc if token.is_alpha and not token.is_stop)

def extract_keywords(text: str) -> List[str]:
    if kw_model is None or yake_extractor is None:
        return []
    kb_keywords = [kw[0].lower().strip() for kw in kw_model.extract_keywords(text, top_n=50)]
    yk_keywords = [kw.lower().strip() for kw, _ in yake_extractor.extract_keywords(text)]
    return list(set(kb_keywords + yk_keywords))

def fuzzy_contains(text, keyword, threshold=0.85):
    return SequenceMatcher(None, text, keyword).ratio() > threshold
def get_resume_score(resume_text, job_description):
    try:
        # === Preprocessing ===
        resume_clean = preprocess(resume_text)
        jd_clean = preprocess(job_description)

        # === Keyword Match Score (Weight: 45%) ===
        vectorizer = CountVectorizer().fit([resume_clean, jd_clean])
        vectors = vectorizer.transform([resume_clean, jd_clean])
        keyword_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 45

        # === Semantic Similarity (Weight: 25%) ===
        if sentence_model:
            embeddings = sentence_model.encode([resume_clean, jd_clean], convert_to_tensor=False)
            semantic_score = float(util.cos_sim(embeddings[0], embeddings[1])[0]) * 25
        else:
            semantic_score = 0

        # === Structure Score (Weight: 15%) ===
        structure_score = 0
        sections = extract_resume_sections(resume_text)
        required = {'Summary', 'Experience', 'Education', 'Skills'}
        found = {sec for sec in sections if sections[sec].strip()}
        structure_score = (len(found & required) / len(required)) * 15

        # === Experience Relevance (Weight: 5%) ===
        experience_score = 0
        if 'Experience' in sections and sections['Experience']:
            exp_text = sections['Experience'].lower()
            jd_keywords = set(extract_keywords(jd_clean))
            exp_terms = {kw for kw in jd_keywords if any(x in kw for x in ['experience', 'managed', 'developed', 'years', 'worked'])}
            matches = sum(1 for kw in exp_terms if fuzzy_contains(exp_text, kw))
            experience_score = min(5, matches)

        # === Bonus: Achievements (Max: 5 pts) ===
        achievement_patterns = [
            r'\d+%', r'\$\d+', r'\d+\s*(million|billion)',
            r'increased by \d+', r'reduced by \d+',
            r'\d+\s*(users|clients|projects|applications)'
        ]
        achievement_count = sum(1 for pat in achievement_patterns if re.search(pat, resume_text, re.I))
        achievement_score = min(5, achievement_count)

        # === Bonus: Action Verbs (Max: 5 pts) ===
        action_verbs = {
            'developed', 'implemented', 'created', 'designed', 'managed', 'led', 'improved',
            'optimized', 'increased', 'reduced', 'achieved', 'delivered', 'established',
            'maintained', 'coordinated', 'collaborated', 'analyzed', 'resolved'
        }
        verb_matches = sum(1 for verb in action_verbs if verb in resume_clean)
        action_score = min(5, verb_matches / 2)

        # === Bonus: Core Skills Coverage (Max: 5 pts) ===
        core_skills = {'python', 'sql', 'machine learning', 'nlp', 'streamlit', 'data analysis'}  # customize as needed
        skill_matches = sum(1 for skill in core_skills if skill in resume_clean)
        skill_score = min(5, skill_matches)

        # === Final Score ===
        final_score = (
            keyword_score +
            semantic_score +
            structure_score +
            experience_score +
            achievement_score +
            action_score +
            skill_score
        )
        return round(min(100, max(0, final_score)), 2)

    except Exception as e:
        st.error(f"Error calculating resume score: {str(e)}")
        return 0
def pdf_reader(uploaded_file):
    try:
        resource_manager = PDFResourceManager()
        output_string = io.StringIO()
        laparams = LAParams()

        # Reset file pointer before reading
        uploaded_file.seek(0)
        with TextConverter(resource_manager, output_string, laparams=laparams) as converter:
            interpreter = PDFPageInterpreter(resource_manager, converter)
            for page in PDFPage.get_pages(uploaded_file, caching=True, check_extractable=True):
                interpreter.process_page(page)

        return output_string.getvalue()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

#extracting resume sections
def extract_resume_sections(text):
    sections={
        'Summary':'',
        'Education':'',
        'Experience':'',
        'Projects':'',
        'Skills':'',
        'Certifications':'',
        'Achievements':''
    }

    #Normalize the text to avoid inconsistent formatting issues like skills, SKILLS ,etc

    text = re.sub(r'\n+','\n',text) #remove extra new lines
    text = re.sub(r'[ ]{2,}',' ',text) #collapse/remove multiple white spaces


    section_headers = list(sections.keys())
    pattern = r"(?i)" + '|'.join([rf'\b{header}\b\s*:?[\n]?' for header in section_headers])

    #split resume into parts based on headers
    parts = re.split(pattern,text,flags=re.IGNORECASE) #list containing content split by dedicated headers
    headers = re.findall(pattern,text)#getting all the headers present in resume

    for i in range(1,len(parts)):
        section = re.sub(r'[^a-zA-Z]','',headers[i-1]).strip().title()
        #each parts[i] holds text content for a section, and headers[i-1] is matching header that came just before it
        if section in sections:
            sections[section] = parts[i].strip()
    return sections

#Database Connection
#Connectiong to sql servel
connection = pymysql.connect(host='localhost',user='root',password='Krish@2003',db='resume_analyzer')
cursor=connection.cursor()

def insert_data(cursor, name, email, resume_score, timestamp, no_of_pages, missing_skills, experience, skills, jd_skills, overall_summary):
    try:
        DB_table_name = 'user_data'
        insert_sql = f"""
            INSERT INTO {DB_table_name} 
            (Name, Email_id, resume_score, Timestamp, Page_nos, missing_skills, experience, resume_skills, jd_skills, overall_summary)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        recommended_values = (name, email, str(resume_score), timestamp, str(no_of_pages), 
                            missing_skills, experience, skills, jd_skills, overall_summary)
        cursor.execute(insert_sql, recommended_values)
        cursor.connection.commit()
        return True
    except Exception as e:
        st.error(f"Error inserting data into database: {str(e)}")
        return False

def get_pdf_page_count(file_path):
    with pdfplumber.open(file_path) as pdf:
        return len(pdf.pages)

def show_pdf(uploaded_file):
    try:
        base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

def extract_basic_resume_info(text):
    try:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        name = ""
        
        # Step 1: Try to get name from first few lines
        for line in lines[:10]:  # Check first 10 lines
            if (
                len(line.split()) <= 4 and                      # Max 4 words (John A. Doe)
                not any(char.isdigit() for char in line) and    # No numbers
                not re.search(r'@|\bresume\b|\bsummary\b|\bengineer\b|\bdeveloper\b', line.lower())
            ):
                name = line
                break

        # Step 2: Fallback to first non-empty line if no match
        if not name and lines:
            name = lines[0]

        # Normalize all caps to title case
        if name.isupper():
            name = name.title()

        # Extract email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
        email = email_match.group(0) if email_match else ""

        # Extract phone
        phone_match = re.search(r'(\+?\d[\d\s\-().]{7,})', text)
        phone = phone_match.group(0) if phone_match else ""

        # Extract experience (optional)
        exp_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience', text.lower())
        experience = exp_match.group(1) if exp_match else "0"

        return {
            'name': name.strip(),
            'email': email.strip(),
            'phone': phone.strip(),
            'experience': experience.strip()
        }

    except Exception as e:
        return {
            'name': "",
            'email': "",
            'phone': "",
            'experience': "0"
        }

    except Exception as e:
        st.error(f"Error extracting basic info: {str(e)}")
        return {
            'name': "",
            'email': "",
            'phone': "",
            'experience': "0"
        }    

def list_to_ul(lst):
    return "<ul>" + "".join(f"<li>{item}</li>" for item in lst) + "</ul>"


def generate_resume_report_pdf(data: dict, template_path: str) -> bytes:
    # Load HTML template file
    with open(template_path, "r", encoding="utf-8") as file:
        html_template = file.read()

    # Replace placeholders with actual data
    for key, value in data.items():
        html_template = html_template.replace(f"{{{{{key}}}}}", str(value))

    # Convert to PDF
    result = BytesIO()
    pisa_status = pisa.CreatePDF(html_template, dest=result)
    if pisa_status.err:
        return None
    return result.getvalue()

# Scoped CSS styles
st.markdown("""
    <style>
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        padding: 0.75rem 1.2rem;
        border-left: 5px solid #2ecc71;
        background-color: #ecfdf5;
        border-radius: 8px;
        color: #1e272e;
        font-family: 'Segoe UI', sans-serif;
        margin: 1.5rem 0 1rem 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.04);
    }

    .custom-table-wrapper {
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', sans-serif;
    }

    .custom-table {
        width: 100%;
        border-collapse: collapse;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        background: #fdfefe;
    }

    .custom-table thead {
        background-color: #1abc9c;
        color: black;
    }

    .custom-table th, .custom-table td {
        padding: 14px 20px;
        text-align: left;
        font-size: 15px;
        border-bottom: 1px solid #e1e1e1;
        color : black;
    }

    .custom-table tbody tr:nth-child(even) {
        background-color: #f7fdfc;
    }

    .custom-table tbody tr:hover {
        background-color: #e0f7f1;
        transition: background-color 0.2s ease-in-out;
    }

    .chart-box {
        background: linear-gradient(to right, #f8fefc, #f2fcf7);
        padding: 2rem 1.5rem;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.08);
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    st.title("Resume Analyzer")
    st.sidebar.markdown("Choose User")
    activities = ["User", "Admin"]
    ADMIN_PASSWORD = "admin"
    choice = st.sidebar.selectbox("Choose among the given options: ", activities)

    connection = None
    try:
        # Database setup
        connection = pymysql.connect(
            host=os.getenv('host'),
            user=os.getenv('user'),
            password=os.getenv('password'),
            db='resume_analyzer'
        )
        cursor = connection.cursor()

        # Create database and table
        cursor.execute("CREATE DATABASE IF NOT EXISTS resume_analyzer")
        cursor.execute("USE resume_analyzer")
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS user_data (
            ID INT NOT NULL AUTO_INCREMENT,
            Name VARCHAR(500) NOT NULL,
            Email_id VARCHAR(500) NOT NULL,
            resume_score VARCHAR(10) NOT NULL,
            Timestamp VARCHAR(50) NOT NULL,
            Page_nos VARCHAR(5) NOT NULL,
            missing_skills TEXT NOT NULL,
            experience TEXT NOT NULL,
            resume_skills TEXT NOT NULL,
            jd_skills TEXT NOT NULL,
            overall_summary TEXT NOT NULL,
            PRIMARY KEY(ID)
        )
        """
        cursor.execute(create_table_sql)

        if choice == "User":
            st.markdown("""<h2 style='text-align:left; color:#73a1e2;'>Upload your resume</h2>""", 
                       unsafe_allow_html=True)
            
            pdf_file = st.file_uploader("Upload your resume", type=["pdf"])
            job_description_input = st.text_area('Enter the Job Description(JD)')
            
            if pdf_file and job_description_input:
                with st.spinner('Processing your resume...'):
                    # Show PDF
                    show_pdf(pdf_file)
                    
                    # Extract and analyze resume
                    resume_text = pdf_reader(pdf_file)
                    resume_data = extract_basic_resume_info(resume_text)

                    API_KEY = os.getenv("API_KEY")
                    genai.configure(api_key=API_KEY)
                    ai_model = genai.GenerativeModel(model_name="gemini-2.0-flash")
                    prompt = f"""
                    You are an expert AI career assistant simulating a modern Applicant Tracking System (ATS).

                    Given the following Job Description and Resume Text, perform these tasks:

                    1. Summarize the resume into exactly 10 crisp, professional, and **paraphrased** bullet points.
                    - Do not copy lines directly from the resume.
                    - Use formal language and combine related ideas.
                    - Focus on key skills, achievements, technologies, and traits.
                    - Make the summary useful for recruiters to quickly assess the candidate.

                    2. Extract all technical, soft, domain-specific, and managerial skills and keywords **explicitly or implicitly** mentioned in the resume.

                    3. Extract all required skills and keywords from the job description (JD).

                    4. Suggest 5 to 10 **strong, crisp, resume-style bullet points** that the candidate should add to improve their resume.
                    - Use **action verbs** and keep them **professional and relevant**.
                    - These points should be derived by analyzing both the resume and the JD.
                    - Focus on **missing but important** skills, tools, or responsibilities aligned with the job role.
                    - Do not repeat existing content from the resume.

                    5. Write a **formal, concise, and compelling cover letter** tailored for the job.
                    - Highlight the candidate's key technical strengths, academic background, and project experience relevant to the job role.
                    - Mention their interest in internships and readiness to contribute.
                    - Begin the cover letter with "Dear Hiring Manager," and end with "Sincerely, [Candidate Name]".
                    - Keep the tone professional and confident.

                    6. Calculate a **resume_score (out of 100%)** based on HR-style ATS practices, using the following weighted criteria:
                    - **Keyword Match (40%)**: Match between resume and JD using both exact terms and similar concepts.
                    - **Experience Alignment (25%)**: Relevance of prior roles, responsibilities, and projects to the job.
                    - **Education & Certifications (10%)**: Match with required degrees, domains, or certifications.
                    - **Soft Skills & Communication (10%)**: Evidence of interpersonal, leadership, and teamwork skills.
                    - **Resume Quality (15%)**: Structure, grammar, bulleting, clarity, action verbs, and formatting.

                    Use intelligent scoring — do not reward keyword stuffing or irrelevant matches.
                    Return a float value rounded to 1 decimal place in the field `resume_score`.

                    7. Return all output in valid JSON format as follows:

                    {{
                    "resume_summary": [
                        "First summarized bullet point...",
                        "..."
                    ],
                    "skills_from_resume": [
                        "Skill1", "Skill2", "Skill3"
                    ],
                    "skills_from_jd": [
                        "Skill1", "Skill2", "Skill3"
                    ],
                    "must_add_points": [
                        "First strong bullet point to add...",
                        "..."
                    ],
                    "cover_letter": "Complete cover letter text here.",
                    "resume_score": 0.0
                    }}

                    Here is the Job Description:
                    {job_description_input}

                    Here is the Resume:
                    {resume_text}
                    """
                    response = ai_model.generate_content(prompt)
                    cleaned_response = re.sub(r"^```(?:json)?|```$", "", response.text.strip(), flags=re.MULTILINE)
                    result = json.loads(cleaned_response)
                    resume_keywords = result["skills_from_resume"]
                    jd_keywords = result["skills_from_jd"]
                    if resume_text:
                        # Display basic info
                        name = resume_data.get('name', 'User')
                        
                        if not name:  # If name is empty after cleaning, use a default
                            name = "User"
                            
                        st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #021659 0%, #0d47a1 100%); padding: 25px; border-radius: 15px; margin: 20px 0; color: white;'>
                                <h2 style='margin: 0; font-size: 28px;'> Hello, {name}!</h2>
                                <p style='margin: 10px 0 0 0; opacity: 0.9;'>Welcome to your resume analysis</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Display analysis results
                        with st.spinner('Analyzing resume...'):
                            # Calculate resume score
                            ml_score = get_resume_score(resume_text, job_description_input)
                            ai_score = result["resume_score"]
                            score = round(0.3 * ml_score + 0.7 * ai_score, 1)
                            # Format score to exactly 2 decimal places
                            formatted_score = "{:.2f}".format(float(score))
                            # Remove trailing zeros after decimal point
                            formatted_score = formatted_score.rstrip('0').rstrip('.')
                            
                            # Determine color and message based on score
                            if score >= 80:
                                color = "#28a745"  # Green
                                message = "Excellent match!"
                                icon = "🎯"
                                bg_gradient = "linear-gradient(135deg, #28a745 0%, #20c997 100%)"
                            elif score >= 60:
                                color = "#ffc107"  # Yellow
                                message = "Good match"
                                icon = "✨"
                                bg_gradient = "linear-gradient(135deg, #ffc107 0%, #ffb300 100%)"
                            else:
                                color = "#dc3545"  # Red
                                message = "Needs improvement"
                                icon = "📝"
                                bg_gradient = "linear-gradient(135deg, #dc3545 0%, #c82333 100%)"

                            # Professional ATS score display
                            st.markdown("""
                                <div style='background: white; border-radius: 12px; margin: 15px 0; 
                                         box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden;'>
                                    <div style='background: {bg_gradient}; padding: 20px; color: white;'>
                                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                                            <div>
                                                <h3 style='margin: 0; font-size: 18px; font-weight: 500; opacity: 0.9;'>ATS Score</h3>
                                                <div style='display: flex; align-items: baseline; gap: 10px; margin-top: 5px;'>
                                                    <span style='font-size: 36px; font-weight: 600;'>{formatted_score}%</span>
                                                    <span style='font-size: 16px; opacity: 0.9;'>{message}</span>
                                                </div>
                                            </div>
                                            <div style='font-size: 36px;'>{icon}</div>
                                        </div>
                                    </div>
                                    <div style='padding: 15px 20px;'>
                                        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                                            <span style='color: #666; font-size: 15px; font-weight: 500;'>Match Progress</span>
                                            <span style='color: {color}; font-size: 15px; font-weight: 600;'>{formatted_score}%</span>
                                        </div>
                                        <div style='background-color: #f0f2f5; height: 6px; border-radius: 3px; overflow: hidden;'>
                                            <div style='width: {formatted_score}%; height: 100%; background: {bg_gradient}; 
                                                      transition: width 0.3s ease-in-out;'></div>
                                        </div>
                                    </div>
                                </div>
                            """.format(
                                color=color,
                                icon=icon,
                                formatted_score=formatted_score,
                                message=message,
                                bg_gradient=bg_gradient
                            ), unsafe_allow_html=True)
                            
                            # Professional metrics display
                            st.markdown("""
                                <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 15px 0;'>
                                    <div style='background: white; padding: 15px; border-radius: 10px; 
                                             box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid #f0f2f5;'>
                                        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 8px;'>
                                            <div style='background: {color}15; padding: 8px; border-radius: 8px;'>
                                                <span style='color: {color}; font-size: 20px;'>📊</span>
                                            </div>
                                            <span style='color: #2c3e50; font-size: 16px; font-weight: 500;'>Keywords</span>
                                        </div>
                                        <div style='color: #666; font-size: 14px; line-height: 1.4;'>
                                            Match job description keywords
                                        </div>
                                    </div>
                                    <div style='background: white; padding: 15px; border-radius: 10px; 
                                             box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid #f0f2f5;'>
                                        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 8px;'>
                                            <div style='background: {color}15; padding: 8px; border-radius: 8px;'>
                                                <span style='color: {color}; font-size: 20px;'>🎯</span>
                                            </div>
                                            <span style='color: #2c3e50; font-size: 16px; font-weight: 500;'>Skills</span>
                                        </div>
                                        <div style='color: #666; font-size: 14px; line-height: 1.4;'>
                                            Align with required skills
                                        </div>
                                    </div>
                                    <div style='background: white; padding: 15px; border-radius: 10px; 
                                             box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid #f0f2f5;'>
                                        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 8px;'>
                                            <div style='background: {color}15; padding: 8px; border-radius: 8px;'>
                                                <span style='color: {color}; font-size: 20px;'>📝</span>
                                            </div>
                                            <span style='color: #2c3e50; font-size: 16px; font-weight: 500;'>Content</span>
                                        </div>
                                        <div style='color: #666; font-size: 14px; line-height: 1.4;'>
                                            Quality and relevance
                                        </div>
                                    </div>
                                </div>
                            """.format(color=color), unsafe_allow_html=True)
                            
                            # Professional tips section
                            st.markdown("""
                                <div style='background: white; padding: 15px 20px; border-radius: 10px; 
                                         margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); 
                                         border: 1px solid #f0f2f5;'>
                                    <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 12px;'>
                                        <div style='background: {color}15; padding: 8px; border-radius: 8px;'>
                                            <span style='color: {color}; font-size: 20px;'>💡</span>
                                        </div>
                                        <span style='color: #2c3e50; font-size: 18px; font-weight: 500;'>Improvement Tips</span>
                                    </div>
                                    <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;'>
                                        <div style='color: #666; font-size: 14px; line-height: 1.5;'>
                                            • Add missing keywords
                                        </div>
                                        <div style='color: #666; font-size: 14px; line-height: 1.5;'>
                                            • Highlight relevant skills
                                        </div>
                                        <div style='color: #666; font-size: 14px; line-height: 1.5;'>
                                            • Use industry terms
                                        </div>
                                        <div style='color: #666; font-size: 14px; line-height: 1.5;'>
                                            • Quantify achievements
                                        </div>
                                    </div>
                                </div>
                            """.format(color=color), unsafe_allow_html=True)
                            
                            #Printing resume skills
                            if resume_keywords:
                                st.markdown("<div class='section-header'>Resume Skills</div>", unsafe_allow_html=True)
                                keywords = [word.capitalize() for word in resume_keywords]
                                keyword_groups = [keywords[i:i+4] for i in range(0, len(keywords), 4)]
                                
                                for group in keyword_groups:
                                    cols = st.columns(4)
                                    for i, keyword in enumerate(group):
                                        with cols[i]:
                                            st.markdown(f"""
                                                <div style='background-color:rgb(225, 228, 231); padding: 12px; 
                                                         border-radius: 8px; margin-bottom: 8px; border: 0px solid #3f8af0;
                                                         text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                                                    <span style='color:rgb(7, 7, 7); font-size: 15px;'>{keyword}</span>
                                                </div>
                                            """, unsafe_allow_html=True)
                                               
                            #resume summary
                            bullet_summary = result["resume_summary"]
                            # Display bullet summary
                            st.markdown("<div class='section-header'>Resume Summary</div>", unsafe_allow_html=True)
                            for point in bullet_summary:
                                st.markdown(f"""
                                    <div style='margin-bottom: 10px; padding: 8px; background-color: #f8f9fa; border-radius: 5px;'>
                                        <p style='margin: 0; color: #2c3e50;'>{point}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            #Printing JD requirements
                            if jd_keywords:
                                st.markdown("<div class='section-header'>JD Skills Requirements</div>", unsafe_allow_html=True)
                                keywords = [word.capitalize() for word in jd_keywords]
                                keyword_groups = [keywords[i:i+4] for i in range(0, len(keywords), 4)]
                                
                                for group in keyword_groups:
                                    cols = st.columns(4)
                                    for i, keyword in enumerate(group):
                                        with cols[i]:
                                            st.markdown(f"""
                                                <div style='background-color:rgb(225, 228, 231); padding: 12px; 
                                                         border-radius: 8px; margin-bottom: 8px; border: 0px solid #3f8af0;
                                                         text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                                                    <span style='color:rgb(7, 7, 7); font-size: 15px;'>{keyword}</span>
                                                </div>
                                            """, unsafe_allow_html=True)

                            #JD vs Resume Keywords
                            resume_skills = set(skill.strip().lower() for skill in resume_keywords)
                            jd_skills = set(skill.strip().lower() for skill in jd_keywords)

                            missing_skills = list(jd_skills-resume_skills)
                            missing_skills = [skill.title() for skill in missing_skills]
                            
                            # Missing Keywords Section
                            st.markdown("<div class='section-header'>Missing Keywords</div>", unsafe_allow_html=True)
                            st.markdown("These important keywords from the job description are not found in your resume. Adding these will significantly improve your match score.")
                            
                            if missing_skills:
                                # Group keywords into sets of 3
                                keywords = [word.capitalize() for word in missing_skills]
                                keyword_groups = [keywords[i:i+4] for i in range(0, len(keywords), 4)]
                                
                                for group in keyword_groups:
                                    cols = st.columns(4)
                                    for i, keyword in enumerate(group):
                                        with cols[i]:
                                            st.markdown(f"""
                                                <div style='background-color:rgb(225, 228, 231); padding: 12px; 
                                                         border-radius: 8px; margin-bottom: 8px; border: 0px solid #3f8af0;
                                                         text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                                                    <span style='color:rgb(7, 7, 7); font-size: 15px;'>{keyword}</span>
                                                </div>
                                            """, unsafe_allow_html=True)
                            else:
                                st.success("Great job! Your resume contains all the important keywords.")
                            
                            # Tips Section
                            st.markdown("""
                                <h3 style='margin-bottom: 20px;'>💡 How to Improve Your Resume</h3>

                                <div style='background-color: #ffffff; padding: 15px 20px; border-radius: 8px; border: 2px solid #ffc107; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                                    <h4 style='color: #d39e00; margin-bottom: 10px;'>1️. Add Missing Keywords</h4>
                                    <ul style='color: #5a5a5a; font-size: 14px; padding-left: 20px;'>
                                        <li>Incorporate missing keywords naturally in your experience descriptions</li>
                                        <li>Add them to your skills section if relevant</li>
                                        <li>Include them in your resume summary</li>
                                     </ul>
                                </div>

                                <div style='background-color: #ffffff; padding: 15px 20px; border-radius: 8px; border: 2px solid #ffc107; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                                    <h4 style='color: #d39e00; margin-bottom: 10px;'>2. Highlight Priority Keywords</h4>
                                    <ul style='color: #5a5a5a; font-size: 14px; padding-left: 20px;'>
                                        <li>Place high-priority keywords in prominent positions</li>
                                        <li>Use them in bullet points describing your achievements</li>
                                        <li>Include specific examples that demonstrate these skills</li>
                                    </ul>
                                 </div>

                                <div style='background-color: #ffffff; padding: 15px 20px; border-radius: 8px; border: 2px solid #ffc107; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                                    <h4 style='color: #d39e00; margin-bottom: 10px;'>3️. Best Practices</h4>
                                    <ul style='color: #5a5a5a; font-size: 14px; padding-left: 20px;'>
                                        <li>Keep descriptions concise and impactful</li>
                                        <li>Use action verbs to start bullet points</li>
                                        <li>Quantify achievements where possible</li>
                                        <li>Ensure all keywords are contextually relevant</li>
                                    </ul>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            #JD vs Resume Difference
                            missing_points = result["must_add_points"]
                            if missing_points:
                                st.markdown("<div class='section-header'>Suggested Points/Lines to add in Resume</div>", unsafe_allow_html=True)
                                st.markdown('These are the following suggestions based on Resume and Job Description, consider adding these to improve your match')
                                for point in missing_points:
                                    st.markdown(f"""
                                        <div style='margin-bottom: 10px; padding: 10px; background-color: white; border-radius: 5px;'>
                                            <p style='margin: 0; color: #333;'>{point}</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Add improvement tips
                                st.markdown("""
                                    <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #17a2b8;'>
                                        <h4 style='color: #0c5460; margin-bottom: 10px;'>💡 How to Add Missing Requirements</h4>
                                        <ul style='color: #666; margin: 0; padding-left: 20px;'>
                                            <li>Incorporate these points naturally in your experience descriptions</li>
                                            <li>Add specific examples that demonstrate these skills</li>
                                            <li>Use action verbs and quantify achievements where possible</li>
                                            <li>Ensure the additions are relevant to your actual experience</li>
                                        </ul>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                    <div style='background-color: #d4edda; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #28a745;'>
                                        <h3 style='color: #155724; margin-bottom: 10px;'>Great Match!</h3>
                                        <p style='color: #666; margin: 0;'>Your resume appears to cover all the requirements mentioned in the job description.</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            #Cover Letter
                            cover_letter = result["cover_letter"]
                            st.markdown("<div class='section-header'>Cover Letter</div>", unsafe_allow_html=True)
                            st.markdown("""
                                    <div style='background-color: #ffffff; padding: 15px 20px; border-radius: 8px; border: 2px solid #ffc107; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                                        <span style='color: #5a5a5a; font-size: 14px; padding-left: 20px;'>{}</span>
                                    </div>
                                """.format(cover_letter), unsafe_allow_html=True)

                            # Save to database
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            experience = resume_data.get('experience', 'Not Specified')
                            
                            resume_pages = get_pdf_page_count(pdf_file)

                            if insert_data(cursor, name, resume_data.get('email', ''), score, timestamp,
                                         resume_pages, ' '.join(missing_skills),
                                         experience, ' '.join(resume_keywords),
                                         ' '.join(jd_keywords), '\n'.join(bullet_summary)):
                                st.success("Analysis results saved successfully!")

                            st.title("📄 Resume Analysis PDF Report")

                            # Replace this data with your actual resume analysis output:
                            data = {
                                "name": name,
                                "email": resume_data.get('email', ''),
                                "page_nos": resume_pages,
                                "resume_score": formatted_score,
                                "missing_points":list_to_ul(missing_points),
                                "resume_keywords":list_to_ul(resume_keywords),
                                "jd_keywords":list_to_ul(jd_keywords),
                                "job_description_input":job_description_input,
                                "missing_keywords": list_to_ul(missing_skills),
                                "bullet_summary": list_to_ul(bullet_summary),
                                "cover_letter":cover_letter,
                                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }

                            template_path = "./templates/resume_report_template.html"

                            if st.button("Generate PDF Report"):
                                pdf = generate_resume_report_pdf(data, template_path)
                                if pdf:
                                    st.download_button(
                                        label="⬇️ Download Resume Report",
                                        data=pdf,
                                        file_name="Resume_Analysis_Report.pdf",
                                        mime="application/pdf"
                                    )
                                else:
                                    st.error("Failed to generate PDF report.")

                                                        
                    else:
                        st.error("Could not extract text from the PDF")

        if choice == "Admin":
             st.title("🔐 Admin Login")
             password = st.text_input("Enter Admin Password", type="password")

             if password:
                if password == ADMIN_PASSWORD:
                    st.success("✅ Access granted. Welcome, Admin!")
                    def fetch_resume_data():
                        cursor.execute("SELECT * FROM user_data")
                        return pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

                    # --- Admin Panel ---
                    st.title("📊 Resume Analyzer - Admin Dashboard")
                    st.markdown("---")

                    # --- Load Data ---
                    data = fetch_resume_data()

                    if data.empty:
                        st.warning("⚠️ No resume data found in the database.")
                        st.stop()

                    # --- Key Stats ---
                    st.subheader("📌 Key Metrics Overview")
                    with st.container():
                        col1, col2, col3 = st.columns(3)
                        col1.metric("📄 Total Resumes", len(data))
                        col2.metric("📊 Avg ATS Score", f"{round(data['resume_score'].astype(float).mean(), 2)}%")
                        col3.metric("📃 Avg Pages/Resume", round(data["Page_nos"].astype(float).mean(), 2))

                    st.markdown("---")

                    # --- Score Histogram ---
                    st.subheader("📈 ATS Score Distribution")
                    fig = px.histogram(
                        data,
                        x="resume_score",
                        nbins=10,
                        title="Distribution of ATS Scores",
                        labels={"resume_score": "ATS Score"},
                        color_discrete_sequence=["#00BFFF"]
                    )
                    fig.update_layout(margin=dict(t=50, b=30, l=30, r=30))
                    st.plotly_chart(fig, use_container_width=True)


                    st.markdown("---")

                    # --- Filter Resumes ---
                    st.subheader("📂 Filter Resumes Based on ATS Score")
                    score_filter = st.slider("🎯 Minimum ATS Score Filter", min_value=0, max_value=100, value=60, help="Only show resumes with ATS score above this threshold.")
                    filtered_data = data[data["resume_score"].astype(float) >= score_filter]

                    st.dataframe(filtered_data, use_container_width=True, height=400)

                    # --- Download Option ---
                    st.download_button(
                        label="⬇️ Download Filtered Data as CSV",
                        data=filtered_data.to_csv(index=False),
                        file_name="filtered_resumes.csv",
                        mime="text/csv"
                    )

                    st.markdown("---")

                    # --- Resume Viewer ---
                    st.subheader("📄 Resume Summaries")

                    # --- Candidate Selection UI ---
                    st.markdown("#### 🎯 Select a Candidate")
                    selected_email = st.selectbox("📧 Choose by Email", data["Email_id"].unique())

                    # --- Display Resume Summary ---
                    resume_row = data[data["Email_id"] == selected_email].iloc[0]

                    st.markdown("### 📝 Summary Preview")
                    with st.expander("💬 View Full Resume Summary", expanded=True):
                        st.markdown(
                            f"""
                            <div style="background-color:#f9f9f9;padding:1rem;border-radius:10px;margin-bottom:5px;border:1px solid #ddd;color:black;">
                                <pre style="font-size:15px;font-family:Courier New, monospace;white-space:pre-wrap;">{resume_row['overall_summary']}</pre>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

                    # Optional: Highlight candidate stats in columns
                    st.markdown("### 📌 Candidate Stats")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("📄 Pages", resume_row["Page_nos"])
                    col2.metric("📊 ATS Score", float(resume_row["resume_score"]))
                    st.markdown("---")

                else:
                    st.error("Incorrect password. Please try again.")
                        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        if connection:
            connection.close()

if __name__ == "__main__":
    main()