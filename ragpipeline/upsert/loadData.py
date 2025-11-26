import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from pipeline.llm import llm
from langchain_core.documents import Document

### Use Pydantic to format
### Template
### Functions for Step 4
def load_resume_text(path:Path):
    suffix = path.suffix.lower()
    if(suffix == '.pdf'):
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        return "\n".join(d.page_content for d in docs)

def llm_parse_resume(raw_text):
    prompt = f"""
    You are a strict JSON resume parser.

    Return ONLY valid minified JSON. No markdown, no commentary.
    In the summary generate a summary of his experience and projects
    Schema (use exactly these keys):
    {{
    "summary": "string",
    "skills": ["string", ...],
    "experiences": [
        {{
        "title": "string",
        "company": "string",
        "location": "string",
        "start_date": "string",
        "end_date": "string",
        "description": "string",
        "skills": ["string", ...]
        }}
    ],
    "education": [
        {{
        "degree": "string",
        "institution": "string",
        "year": "string"
        }}
    ],
    "projects": [
        {{
        "name": "string",
        "description": "string",
        "skills": ["string", ...]
        }}
    ]
    }}

    If something is missing, use "" or [].

    Resume:
    \"\"\"{raw_text[:12000]}\"\"\"
    """.strip()
    resp = llm.invoke(prompt)
    content = getattr(resp, "content", resp)

    # Debug print to check raw response
    print("LLM raw response:", content)

    if not content or not isinstance(content, str) or content.strip() == "":
        raise ValueError("Received empty or invalid response from LLM")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON from LLM response: {e}")

    return parsed

def build_resume_doc(parsed, resume_id, filename):
    summary = (parsed.get("summary") or "").strip()
    skills = parsed.get("skills") or []
    experiences = parsed.get("experiences") or []
    education = parsed.get("education") or []
    projects = parsed.get("projects") or []
    parts = []
    # Emebedding part
    if summary: 
        parts.append(f"Summary: {summary}")
    if skills: 
        parts.append("Skills: " + ", ".join(skills))
    full_text = "\n".join(parts).strip()
    all_skills = set()
    # Metadata part
    for s in skills:
        all_skills.add(s.strip())
    roles = set()
    companies = set()
    for e in experiences:
        roles.add((e.get("title") or "").strip())
        companies.add((e.get("company") or "").strip())
    metadata = {"resume_id": resume_id, "filename": filename, "skills": sorted(all_skills), "roles": sorted(roles), "companies": sorted(companies)}
    return Document(page_content = full_text, metadata = metadata)
    