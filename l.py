from typing import Annotated
from typing_extensions import TypedDict
import os
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

import os
import requests

# Ensure your GROQ_API_KEY environment variable is set
api_key = os.environ.get("GROQ_API_KEY")

response = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)
llm=ChatGroq(model="allam-2-7b", api_key=api_key)
