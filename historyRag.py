# import os
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_google_genai import ChatGoogleGenerativeAI

# # Load environment variables from .env (if you use one)
# load_dotenv()

# # Read Google API key from environment variable
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY not found in environment variables")

# # Initialize the Google Gemini LLM client
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     api_key=GOOGLE_API_KEY,
# )

# # Define chat prompt template
# chat_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant."),
#         MessagesPlaceholder("history"),
#         ("human", "{input}"),
#     ]
# )

# # Create the full chat chain: prompt -> llm -> output parser (string)
# chat_chain = chat_prompt | llm | StrOutputParser()

# # Manual message history to keep track of conversation context
# history = []

# # First user message
# user_1 = "I have a dog named Bruno. He is 3 years old."
# resp1 = chat_chain.invoke({"input": user_1, "history": history})
# print("A1:", resp1)

# # Update history for next turn
# history.append(HumanMessage(content=user_1))
# history.append(AIMessage(content=resp1))

# # Second user message (refers to previous context)
# user_2 = "How old is he?"
# resp2 = chat_chain.invoke({"input": user_2, "history": history})
# print("A2:", resp2)

# # Update history again
# history.append(HumanMessage(content=user_2))
# history.append(AIMessage(content=resp2))
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
# # Load environment variables from .env (if you use one)
load_dotenv()

# # Read Google API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# # Initialize the Google Gemini LLM client
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
)

# 1) Simple chat prompt with history
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

chat_chain = chat_prompt | llm | StrOutputParser()
# 2) History store + getter
# Take session id from front end.
_history_store = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _history_store:
        _history_store[session_id] = ChatMessageHistory()
    return _history_store[session_id]


# 3) Wrap with RunnableWithMessageHistory
# RunnableWithMessageHistory will keep updating _history_store on it's own, you don't have to worry about it.
chat_agent = RunnableWithMessageHistory(
    chat_chain,
    get_history,
    input_messages_key="input",     # field treated as user message
    history_messages_key="history", # passed into MessagesPlaceholder
)

# 4) Demo: ask something, then follow up with pronouns
cfg = {"configurable": {"session_id": "demo"}}

resp1 = chat_agent.invoke({"input": "I have a dog named Bruno. He is 3 years old."}, config=cfg)
print("A1:", resp1)
# resp1 = chat_agent.invoke({"input": "What was the dog again?."}, config=cfg)
# print("A1:", resp1)
resp1 = chat_agent.invoke({"input": "What was the dog again?."}, config=cfg)
print("A1:", resp1)
