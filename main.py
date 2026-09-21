# main.py

from graph import build_graph


app = build_graph()

query = """
I have been experiencing chest discomfort
and shortness of breath since this morning.
What should I do?
"""

result = app.invoke({
    "user_query": query,
    "patient_context": "",
    "intent": "",
    "urgency": "",
    "retrieved_docs": [],
    "response": "",
    "safety_flag": False,
    "tool_result": ""
})

print("Urgency:", result["urgency"])
print("\nResponse:")
print(result["response"])
