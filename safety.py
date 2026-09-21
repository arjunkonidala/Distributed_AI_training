# agents/safety.py

from langchain_openai import ChatOpenAI
from rag.retriever import retrieve_guidelines
from state import AgentState

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


def safety_and_rag_agent(state: AgentState):

    query = state["user_query"]

    # Retrieve only grounded clinical information
    docs = retrieve_guidelines(query)

    context = "\n\n".join(docs)

    prompt = f"""
    You are a clinical information assistant.

    Answer ONLY using the supplied clinical guidelines.

    Clinical guidelines:
    {context}

    User query:
    {query}

    Rules:
    - Do not invent medical facts.
    - Do not provide an unverified diagnosis.
    - Clearly state when information is insufficient.
    - For emergency symptoms, recommend seeking immediate
      professional medical attention.
    """

    response = llm.invoke(prompt)

    return {
        "retrieved_docs": docs,
        "response": response.content,
        "safety_flag": False
    }
