# import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
)