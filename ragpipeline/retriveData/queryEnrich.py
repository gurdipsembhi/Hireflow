from pipeline.llm import llm

def edit_query(query):
    prompt = f"""
Imagine you're helping retriever the most suitable candidates resumes using Multi query expansion. You may receieve a simple query or a proper JD.

Write 3 different variations of the same query focussing on 3 different important areas for retrieval. Do not explain what you're doing.
Return only the points.
Query/JD : 
{query}
"""
    resp = llm.invoke(prompt)
    content = getattr(resp, "content", resp)
    return str(content).strip()
    