from pathlib import Path
import yaml
from ragpipeline.retriveData.langChainDifferentChain import resume_parser

# DEFAULT_TEMPLATE = """
# You are an AI assistant helping resume search and candidate shortlisting

# You will be given:
# - A user question of a Job description
# - Context made of numbered resume snippets like [0], [1], [2], ....
# USE ONLY THIS CONTEXT FOR ANSWERING
# When you mention a candidate, reference the snippets with ids like [0], [1]....
# If question is JD-Based, shortlist request then given a ranked list of candidates of the most relevant candidates with short explanations for each.

# Context:
# {sources}
# Question:
# {query}

# Answer:
# """.strip()

# PROMPT_PATH = 'sdfadf'
# def load_prompt(path: str = PROMPT_PATH) -> dict:
#     cfg = {}
#     p = Path(path)
#     if p.exists():
#         cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
#     cfg.setdefault("vars", {})
#     cfg.setdefault("template", DEFAULT_TEMPLATE)
#     return cfg

# def render_prompt(cfg: dict, query: str, sources: str) -> str:
#     vars_all = dict(cfg.get("vars", {}), query=query, sources=sources)
#     return cfg["template"].format(**vars_all)

#     # New prompt loader / renderer (no YAML needed for now)
    
DEFAULT_TEMPLATE = """
    You are an assistant that converts retrieved resume snippets into a structured JSON response.
    
    You are given:
    - The user's query (question or job description).
    - A set of resume snippets in the CONTEXT. Each snippet contains:
      - a resume_id (e.g. Emily_Green_Resume_46),
      - a File line with the filename,
      - a Link line with a path/URL,
      - and a Snippet with some text.
    
    Your job:
    0. Shortlist the top n resumes basis the requirement below in the QUERY. For example if user says top 5 resumes then only pick top 5 in the order from CONTEXT provided.
    1. For EVERY resume snippet that appears in the context, create EXACTLY ONE object in the JSON.
    2. Do NOT invent new resumes. Drop Resumes basis the QUERY. Example Top 5 resumes and if you're getting 7 resumes in context then drop 2 resumes
    3. Copy `resume_id`, `filename`, and `link` EXACTLY from the context whenever they are present.
    4. Use the snippet text and the query to:
       - estimate `jd_relevance` between 0 and 1 basis no of matching keywords(0 = not relevant, 1 = perfect fit),
       - write `profile_summary` (2–4 lines),
       - fill `key_skills` and `risks_or_flags`.
    
    You MUST follow this JSON schema and formatting instructions:
    
    {format_instructions}
    
    Return ONLY valid JSON. No markdown, no comments, no extra text.
    
    CONTEXT:
    {sources}
    
    QUERY:
    {query}
    """.strip()
def load_prompt(path: str) -> dict:
    """
    For now we ignore PROMPT_PATH and just return a config dict
    with our default template.
    """
    return {
        "vars": {},
        "template": DEFAULT_TEMPLATE,
    }

def render_prompt(cfg: dict, query: str, sources: str) -> str:
    """
    Render the final prompt, injecting the Pydantic format instructions.
    """
    format_instructions = resume_parser.get_format_instructions()
    vars_all = dict(
        cfg.get("vars", {}),
        query=query,
        sources=sources,
        format_instructions=format_instructions,
    )
    return cfg["template"].format(**vars_all)
