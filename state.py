# state.py

from typing import TypedDict, List


class AgentState(TypedDict):
    user_query: str
    patient_context: str

    intent: str
    urgency: str

    retrieved_docs: List[str]
    response: str

    safety_flag: bool
    tool_result: str
