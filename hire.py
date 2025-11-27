import gradio as gr
from typing import Dict, Any, Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Your backend imports - adjust paths as needed
from langchain_core.runnables import RunnableLambda
from ragpipeline.retriveData.langChainDifferentChain import resume_parser
from ragpipeline.retriveData.hybridQuery import hybrid_query
from ragpipeline.retriveData.rerank import rerank_docs_crossencoder
from ragpipeline.retriveData.contextBuilding import build_context_snippets
from ragpipeline.retriveData.queryEnrich import edit_query
from ragpipeline.retriveData.loadAndRenderPrompt import load_prompt, render_prompt
from config import RETRIEVAL_ALPHA, TOP_K
from pipeline.llm import llm
from langchain_core.runnables import RunnableConfig

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from ragpipeline.faq.template import resume_qa_chain
from ragpipeline.retriveData.langChainDifferentChain import ResumeResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import yaml

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)
# ---- Core functions for the resume assistant ----
def _start(query: str):
    return {"query": query}

def _extract_answer(answer_obj):
    content = getattr(answer_obj, "content", answer_obj)
    return str(content).strip()

_history_store = {}
def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _history_store:
        _history_store[session_id] = ChatMessageHistory()
    return _history_store[session_id]

CURRENT_RESUME_JSON: Optional[Dict[str, Any]] = None
QA_SESSION_ID = "resume_qa"

# Intent classification prompt
intent_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an intent classifier. Given a user message, respond with exactly one word:\n"
     "- 'parse' if the user is asking to shortlist/match resumes to a JD or requirements.\n"
     "- 'qa' if the user is asking follow-up questions about already parsed resumes."),
    ("human", "{input}")
])

intent_chain = intent_prompt | llm | StrOutputParser()

def classify_intent(user_input: str) -> str:
    intent = intent_chain.invoke({"input": user_input}).strip().lower()
    if "parse" in intent:
        return "parse"
    if "qa" in intent:
        return "qa"
    return "qa"

# Path to your prompt file (update if needed)
PROMPT_PATH = "flkhdlkhdlkh"

# QA agent with history
qa_agent = RunnableWithMessageHistory(
    resume_qa_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history",
)

# Main resume parsing chain
base_chain = (
    RunnableLambda(_start)
    .assign(query_text=RunnableLambda(lambda d: edit_query(d["query"])))
    .assign(
        results=RunnableLambda(
            lambda d: hybrid_query(
                d["query_text"] or d["query"],
                alpha=RETRIEVAL_ALPHA,
                top_k=TOP_K,
            )
        )
    )
    .assign(
        results_reranked=RunnableLambda(
            lambda d: rerank_docs_crossencoder(
                d["results"],
                d["query_text"] or d["query"],
            )
        )
    )
    .assign(
        sources=RunnableLambda(
            lambda d: build_context_snippets(d["results_reranked"])
        )
    )
    .assign(cfg=RunnableLambda(lambda _: load_prompt(PROMPT_PATH)))
    .assign(
        prompt=RunnableLambda(
            lambda d: render_prompt(d["cfg"], d["query_text"], d["sources"])
        )
    )
    .assign(answer_obj=RunnableLambda(lambda d: llm.invoke(d["prompt"])))
    .assign(answer_raw=RunnableLambda(lambda d: _extract_answer(d["answer_obj"])))
    .assign(
        answer_parsed=RunnableLambda(
            lambda d: resume_parser.parse(d["answer_raw"])
        )
    )
)

def handle_user_message(user_input: str) -> str:
    """
    Very small 'agent':
    - classify intent -> 'parse' or 'qa' 
    - call base_chain when we need to parse / shortlist resumes
    - call qa_agent when it's a follow-up question about those resumes
    """
    global CURRENT_RESUME_JSON, _history_store

    intent = classify_intent(user_input)

    # -------- PARSE branch: new JD / new shortlist --------
    if intent == "parse":
        # 1) run your full resume parsing chain
        out = base_chain.invoke(user_input)
        parsed: ResumeResponse = out["answer_parsed"]
        CURRENT_RESUME_JSON = parsed.model_dump()   # dict: { query, resumes: [...] }

        # 2) reset QA history for this new resume set
        _history_store[QA_SESSION_ID] = ChatMessageHistory()

        # 3) simple summary for the user (with links)
        data = CURRENT_RESUME_JSON
        resumes = data.get("resumes", [])

        lines = [f"Parsed {len(resumes)} resumes for your query.\n"]
        lines.append("Shortlisted resumes:\n")

        for r in resumes:
            filename = r.get("filename") or r["resume_id"]
            link = r.get("link") or ""
            rel = float(r.get("jd_relevance", 0.0))

            if link:
                # Markdown-style link -> clickable in Gradio Markdown
                lines.append(f"- [{filename}]({link})  (relevance = {rel:.2f})")
            else:
                lines.append(f"- {filename}  (relevance = {rel:.2f})")

        return "\n".join(lines)

    # -------- QA branch: follow-up questions on current resumes --------
    if CURRENT_RESUME_JSON is None:
        return (
            "I don't have any parsed resumes yet. "
            "First give me a JD or say something like 'shortlist candidates for ...'."
        )

    cfg:RunnableConfig = {"configurable": {"session_id": QA_SESSION_ID}}

    answer = qa_agent.invoke(
        {
            "question": user_input,
            "resume_json": CURRENT_RESUME_JSON,
        },
        config=cfg,
    )
    return answer

# Build markdown listing shortlisted resumes for right panel
def build_shortlist_markdown() -> str:
    if CURRENT_RESUME_JSON is None:
        return "No resumes parsed yet.\n\nPaste a JD and ask me to shortlist candidates."

    data = CURRENT_RESUME_JSON
    resumes = data.get("resumes", [])
    if not resumes:
        return "No resumes parsed for the current query."

    lines = [f"### Current shortlist ({len(resumes)} resumes)\n"]
    for r in resumes:
        filename = r.get("filename") or r.get("resume_id", "unknown")
        link = r.get("link") or ""
        rel = float(r.get("jd_relevance", 0.0))
        if link:
            lines.append(f"- [{filename}]({link}) — relevance: **{rel:.2f}**")
        else:
            lines.append(f"- {filename} — relevance: **{rel:.2f}**")
    return "\n".join(lines)

def gradio_handler(user_message, chat_history):
    reply = handle_user_message(user_message)
    chat_history = chat_history or []

    # Append user message as role:user
    chat_history.append({"role": "user", "content": user_message})

    # Append assistant reply as role:assistant
    chat_history.append({"role": "assistant", "content": reply})

    shortlist_md = build_shortlist_markdown()
    return chat_history, shortlist_md


# Gradio UI design
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Resume Assistant")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation")
            msg = gr.Textbox(
                label="Your message",
                placeholder="Paste a JD to shortlist resumes, or ask follow-up questions..."
            )
            send_btn = gr.Button("Send")

        with gr.Column(scale=2):
            gr.Markdown("### Shortlisted resumes")
            shortlist_panel = gr.Markdown(build_shortlist_markdown())

    send_btn.click(
        fn=gradio_handler,
        inputs=[msg, chatbot],
        outputs=[chatbot, shortlist_panel],
    )
    msg.submit(
        fn=gradio_handler,
        inputs=[msg, chatbot],
        outputs=[chatbot, shortlist_panel],
    )

if __name__ == "__main__":
    demo.launch()
