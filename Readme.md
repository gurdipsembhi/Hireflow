![HireFlow-AI Demo](demo.gif)

**AI-powered recruitment platform** that semantically matches candidates to job descriptions using Gemini embeddings and FAISS vector search. Delivers recruiter-friendly insights through LLM evaluations and interactive Streamlit dashboard.[web:11][web:22]

## 🎯 Project Overview

Traditional resume screening is time-consuming and prone to bias. HireFlow-AI solves this by:

- Converting resumes and job descriptions into dense embeddings via Google Gemini
- Using Pinecone for efficient semantic similarity search (not just keyword matching)
- Generating LLM-powered evaluations with strengths, weaknesses, and fit scores
- Providing an intuitive Streamlit interface for ranking, filtering, and comparisons

The system handles diverse resume formats (PDF, DOCX, text) and scales for enterprise use.[web:11][web:22]

## ✨ Key Features

- **Semantic Matching**: Gemini embeddings + Pinecone for precise candidate-job alignment
- **LLM Evaluations**: Prompt-engineered insights on skills, gaps, and cultural fit
- **Interactive Dashboard**: Real-time ranking, filtering, and side-by-side comparisons
- **Robust Parsing**: Pandas + Pydantic for reliable multi-format resume processing
- **Scalable Architecture**: Ready for Pinecone, multilingual support, and ATS integration[web:11]

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Core ML** | Google Gemini API, Pinecone |
| **Data** | Pandas, Pydantic,Numpy |
| **UI** | Gradio |
| **Language** | Python 3.10+ |
| **Future** | Pinecone, LangChain, multilingual NLP[web:11][web:22] |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- GitHub account

### Installation
git clone https://github.com/gurdipsembhi/Hireflow.git
cd Hireflow
pip install -r requirements.txt

## For upstering the docs
python app.py

### Environment Setup
.env
Place your Key in .env file 
PINECONE_API_KEY=**************
GOOGLE_API_KEY=**********************

### Launch
python hire.py
