import json
from typing import Dict, Any, List
from pipeline.llm import llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant answering questions about candidate resumes.\n"
            "You are given:\n"
            "- The initial job description or query.\n"
            "- A structured Resume JSON.\n"
            "- Conversation history.\n\n"
            "Use ONLY the Resume JSON as ground truth. "
            "If something is not present there, say you don't know.\n\n"
            "Initial query / JD:\n{initial_query}\n\n"
            "Resume JSON:\n{resume_json}"
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

def _prep_qa_inputs(d: Dict[str, Any]) -> Dict[str, Any]:
    resume_json = d["resume_json"]
    initial_query = resume_json.get("query", "")

    return {
        "question": d["question"],
        "history": d.get("history", []),  # LangChain will fill this
        "resume_json": json.dumps(resume_json, ensure_ascii=False),
        "initial_query": initial_query,
    }
resume_qa_chain = (
    RunnableLambda(_prep_qa_inputs)
    | qa_prompt
    | llm
    | StrOutputParser()
)
