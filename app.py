from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from langchain_core.runnables import RunnableLambda
load_dotenv()  # Load .env file
from ragpipeline.upsert.upsertAndPackageData import step6_package_and_upsert
from ragpipeline.upsert.loadData import load_resume_text, llm_parse_resume, build_resume_doc
from pipeline.llm import llm
from ragpipeline.upsert.chunkData import step4_load_and_split
from config import RESUME_DIR
from ragpipeline.upsert.encodeData import step5_encode
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

RESUME_DIR=Path(RESUME_DIR)
def create_upsert_chain():
    chain = (
        RunnableLambda(lambda _: {"resume_dir": RESUME_DIR})
        .assign(raw_texts=RunnableLambda(lambda d: [load_resume_text(path) for path in Path(d["resume_dir"]).glob("*.pdf")]))
        .assign(parsed_documents=RunnableLambda(lambda d: [llm_parse_resume(text) for text in d["raw_texts"]]))
        .assign(built_docs=RunnableLambda(lambda d: [build_resume_doc(parsed, i, f"resume_{i}.pdf") for i, parsed in enumerate(d["parsed_documents"], 1)]))
        .assign(splitted=RunnableLambda(lambda d: step4_load_and_split({"resume_dir": RESUME_DIR})))
        .assign(encoded=RunnableLambda(lambda d: step5_encode(d["splitted"])))
        .assign(package_and_upsert=RunnableLambda(lambda d: step6_package_and_upsert(d["encoded"])))
    )
    return chain
# Usage example:
upsert_chain = create_upsert_chain()
result = upsert_chain.invoke({})