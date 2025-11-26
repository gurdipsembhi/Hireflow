# from langchain_core.runnables import RunnableLambda
# from dotenv import load_dotenv
# import os
# load_dotenv()
# from ragpipeline.retriveData.langChainDifferentChain import resume_parser

# from ragpipeline.retriveData.hybridQuery import hybrid_query
# from ragpipeline.retriveData.rerank import rerank_docs_crossencoder
# from ragpipeline.retriveData.contextBuilding import build_context_snippets
# from ragpipeline.retriveData.queryEnrich import edit_query
# from ragpipeline.retriveData.loadAndRenderPrompt import load_prompt, render_prompt
# from config import RETRIEVAL_ALPHA, TOP_K
# from pipeline.llm import llm
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from ragpipeline.faq.template import resume_qa_chain

# def _start(query: str):
#     return {"query": query}

# def _extract_answer(answer_obj):
#     content = getattr(answer_obj, "content", answer_obj)
#     return str(content).strip()

# PROMPT_PATH = "flkhdlkhdlkh"


# _history_store = {}

# def get_history(session_id: str) -> ChatMessageHistory:
#     if session_id not in _history_store:
#         _history_store[session_id] = ChatMessageHistory()
#     return _history_store[session_id]

# qa_agent = RunnableWithMessageHistory(
#     resume_qa_chain,
#     get_history,
#     input_messages_key="question",   # this is the user message
#     history_messages_key="history",  # goes into MessagesPlaceholder("history")
# )



# base_chain = (
#     RunnableLambda(_start)
#     # 1) edit query (HyDE or identity, whatever your edit_query does)
#     .assign(query_text=RunnableLambda(lambda d: edit_query(d["query"])))
#     # 2) hybrid retrieval
#     .assign(
#         results=RunnableLambda(
#             lambda d: hybrid_query(
#                 d["query_text"] or d["query"],
#                 alpha=RETRIEVAL_ALPHA,
#                 top_k=TOP_K,
#             )
#         )
#     )
#     # 3) rerank
#     .assign(
#         results_reranked=RunnableLambda(
#             lambda d: rerank_docs_crossencoder(
#                 d["results"],
#                 d["query_text"] or d["query"],
#             )
#         )
#     )
#     # 4) build context snippets (remember these include resume_id, File, Link)
#     .assign(
#         sources=RunnableLambda(
#             lambda d: build_context_snippets(d["results_reranked"])
#         )
#     )
#     # 5) load & render prompt
#     .assign(cfg=RunnableLambda(lambda _: load_prompt(PROMPT_PATH)))
#     .assign(
#         prompt=RunnableLambda(
#             lambda d: render_prompt(d["cfg"], d["query_text"], d["sources"])
#         )
#     )
#     # 6) call LLM
#     .assign(answer_obj=RunnableLambda(lambda d: llm.invoke(d["prompt"])))
#     # 7) extract raw text + parse with Pydantic
#     .assign(answer_raw=RunnableLambda(lambda d: _extract_answer(d["answer_obj"])))
#     .assign(
#         answer_parsed=RunnableLambda(
#             lambda d: resume_parser.parse(d["answer_raw"])
#         )
#     )
# )

# demo_jd = """We are hiring for a Senior Controller / Head of Accounting role.

# Requirements:
# - 7–10 years of progressive experience in accounting and financial management
# - Strong experience with financial reporting, month-end closing, and budget management
# - Proven team leadership and cross-functional collaboration
# - Experience driving process improvements and managing stakeholders
# From the available resumes in the system, shortlist the top 4 candidates"""

# base_out = base_chain.invoke(demo_jd)
# resume_json = base_out["answer_parsed"].model_dump()
# config = {"configurable": {"session_id": "default"}}
# qa_question_1 = "Whose skillsets are matching the most, can you do a keyword match against the initial query"
# answer_1 = qa_agent.invoke(
#     {
#         "question": qa_question_1,
#         "resume_json": resume_json,   # initial query is inside this JSON
#     },
#     config=config,
# )
# print("_________________________",answer_1)
# config = {"configurable": {"session_id": "default"}}
# qa_question_1 = "Give one candidate with highest matching keywords"
# answer_1 = qa_agent.invoke(
#     {
#         "question": qa_question_1,
#         "resume_json": resume_json,   # initial query is inside this JSON
#     },
#     config=config,
# )
# print("+++++++++++++++++++++++++",answer_1)

# from typing import Dict, Any, Optional
# from langchain_community.chat_message_histories import ChatMessageHistory

# CURRENT_RESUME_JSON: Optional[Dict[str, Any]] = None

# # We'll reuse the same store used by get_history
# # (assuming you already did `_history_store = {}` above for qa_agent)
# QA_SESSION_ID = "resume_qa"
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# intent_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system",
#          "You are an intent classifier. "
#          "Given a user message, respond with exactly one word:\n"
#          "- 'parse' if the user is asking to shortlist/match resumes to a JD or requirements.\n"
#          "- 'qa' if the user is asking follow-up questions about already parsed resumes."),
#         ("human", "{input}")
#     ]
# )

# intent_chain = intent_prompt | llm | StrOutputParser()
# def classify_intent(user_input: str) -> str:
#     intent = intent_chain.invoke({"input": user_input}).strip().lower()
#     if "parse" in intent:
#         return "parse"
#     if "qa" in intent:
#         return "qa"
#     # fallback: treat as qa
#     return "qa"
# def handle_user_message(user_input: str) -> str:
#     """
#     Very small 'agent':
#     - classify intent -> 'parse' or 'qa' 
#     - call base_chain when we need to parse / shortlist resumes
#     - call qa_agent when it's a follow-up question about those resumes
#     """
#     global CURRENT_RESUME_JSON, _history_store

#     intent = classify_intent(user_input)

#     # -------- PARSE branch: new JD / new shortlist --------
#     if intent == "parse":
#         # 1) run your full resume parsing chain
#         out = base_chain.invoke(user_input)
#         parsed: ResumeResponse = out["answer_parsed"]
#         CURRENT_RESUME_JSON = parsed.model_dump()   # dict: { query, resumes: [...] }

#         # 2) reset QA history for this new resume set
#         _history_store[QA_SESSION_ID] = ChatMessageHistory()

#         # 3) simple summary for the user (with links)
#         data = CURRENT_RESUME_JSON
#         resumes = data.get("resumes", [])

#         lines = [f"Parsed {len(resumes)} resumes for your query.\n"]
#         lines.append("Shortlisted resumes:\n")

#         for r in resumes:
#             filename = r.get("filename") or r["resume_id"]
#             link = r.get("link") or ""
#             rel = float(r.get("jd_relevance", 0.0))

#             if link:
#                 # Markdown-style link -> clickable in Gradio Markdown
#                 lines.append(f"- [{filename}]({link})  (relevance = {rel:.2f})")
#             else:
#                 lines.append(f"- {filename}  (relevance = {rel:.2f})")

#         return "\n".join(lines)

#     # -------- QA branch: follow-up questions on current resumes --------
#     if CURRENT_RESUME_JSON is None:
#         return (
#             "I don't have any parsed resumes yet. "
#             "First give me a JD or say something like 'shortlist candidates for ...'."
#         )

#     cfg = {"configurable": {"session_id": QA_SESSION_ID}}

#     answer = qa_agent.invoke(
#         {
#             "question": user_input,
#             "resume_json": CURRENT_RESUME_JSON,
#         },
#         config=cfg,
#     )
#     return answer
# demo_jd = """We are hiring for a Senior Controller / Head of Accounting role.

# Requirements:
# - 7–10 years of progressive experience in accounting and financial management
# - Strong experience with financial reporting, month-end closing, and budget management
# - Proven team leadership and cross-functional collaboration
# - Experience driving process improvements and managing stakeholders
# From the available resumes in the system, shortlist the top 1–2 candidates"""

# print(handle_user_message(demo_jd))

# question = "what missing kills do you see from these list of resumes?"
# print(handle_user_message(question))
# question = "Any other skill that you found?"
# print(handle_user_message(question))
