# agents/triage.py

from langchain_openai import ChatOpenAI
from state import AgentState

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


def triage_agent(state: AgentState):

    prompt = f"""
    Classify the urgency of this request.

    User:
    {state["user_query"]}

    Patient context:
    {state["patient_context"]}

    Possible outputs:
    - emergency
    - urgent
    - routine
    """

    result = llm.invoke(prompt)

    urgency = result.content.strip().lower()

    if "emergency" in urgency:
        urgency = "emergency"
    elif "urgent" in urgency:
        urgency = "urgent"
    else:
        urgency = "routine"

    return {
        "urgency": urgency
    }
