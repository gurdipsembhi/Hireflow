### COntext building
from config import RESUME_DIR
from pathlib import Path
RESUME_DIR=Path(RESUME_DIR)

def build_resume_link(result, base_path = str(RESUME_DIR)):
    md = result.get("metadata")
    file_name = md['filename']
    if file_name:
        return f"{base_path}/{file_name}"
    else:
        return ""


def build_context_snippets(results):
    parts = []
    for i, r in enumerate(results):
        md = r.get('metadata')
        resume_id = r.get('id')
        skills = ", ".join(md.get('skills'))
        roles = ", ".join(md.get('roles'))
        snippet = (r.get("preview") or "")[:500]
        filename = r.get("filename") or ""
        if not snippet:
            continue
        link = build_resume_link(r)
        block_lines = [f"[{i}] resume_id: {resume_id}", 
                      f"File: {filename}", 
                      f"Link: {link}"]
        if roles:
            block_lines.append(roles)
        if skills:
            block_lines.append(skills)
        block = "\n".join(block_lines)
        parts.append(block)
    return "\n".join(parts)
        
        