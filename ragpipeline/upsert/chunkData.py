from ragpipeline.upsert.loadData import load_resume_text, llm_parse_resume, build_resume_doc
### Step 4: Chunk and split
def step4_load_and_split(cfg):
    resume_dir = cfg['resume_dir']
    docs = []
    for fp in resume_dir.iterdir():
        raw  = load_resume_text(fp).strip()
        resume_id = fp.stem
        parsed = llm_parse_resume(raw)
        doc = build_resume_doc(parsed, resume_id, fp.name)
        docs.append(doc)
    return {"docs": docs}