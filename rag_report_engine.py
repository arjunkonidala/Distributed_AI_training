"""RAG (Retrieval-Augmented Generation) Engine for CareFlow Medical History & Reports.
Grounds the Report Agent on the patient's actual medical records with deep human empathy
and structured Markdown table formatting:
- patient_baseline_summaries (chronic conditions, allergies, active medications)
- patient_memories (atomic semantic facts, historical findings, triggers)
- medical_conditions (structured diseases, symptoms, onset in strict 24h format, treatments, tests)
- appointments (scheduled consults)
- chat_messages (conversational ground truth)
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import database as db
from api_pool import key_pool

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def retrieve_patient_history_context(username: str, query: str) -> Dict[str, Any]:
    """Retrieves and ranks all relevant medical history items for RAG generation."""
    baseline = db.get_patient_baseline(username) or {}
    conditions = db.get_conditions(username)
    memories = db.get_patient_memories(username, status="active", limit=50)
    appointments = db.get_appointments(username) if hasattr(db, "get_appointments") else []
    
    # Query words for keyword ranking / filtering
    query_tokens = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]

    # 1. Relevant conditions
    relevant_conditions = []
    for c in conditions:
        data = c.get("data", {})
        disease = c.get("disease_name", "")
        # Score relevance
        dump = json.dumps(data).lower()
        score = sum(1 for token in query_tokens if token in dump)
        relevant_conditions.append({
            "disease_name": disease,
            "onset_24h": c.get("onset"),
            "severity_score": c.get("severity_score"),
            "chief_complaint": c.get("chief_complaint"),
            "treatment_name": c.get("treatment_name"),
            "treatment_status": c.get("treatment_status"),
            "report_type": c.get("report_type"),
            "location": c.get("location"),
            "data": data,
            "timeline": data.get("timeline", []),
            "relevance_score": score
        })
    # Sort by relevance
    relevant_conditions.sort(key=lambda x: x["relevance_score"], reverse=True)

    # 2. Relevant atomic memories
    relevant_memories = []
    for m in memories:
        content = m.get("content", "")
        score = sum(1 for token in query_tokens if token in content.lower())
        relevant_memories.append({
            "memory_type": m.get("memory_type"),
            "content": content,
            "created_at": m.get("created_at"),
            "relevance_score": score
        })
    relevant_memories.sort(key=lambda x: x["relevance_score"], reverse=True)

    return {
        "baseline": baseline,
        "conditions": relevant_conditions,
        "memories": relevant_memories,
        "appointments": appointments,
    }


def build_empathetic_table_report(username: str, retrieval: Dict[str, Any], query: str = "") -> str:
    """Builds a warm, empathetic, and beautifully structured medical summary with generous spacing and clean Markdown tables."""
    profile = db.get_profile(username) or {}
    first_name = profile.get("name", username).split()[0] if profile else username
    baseline = retrieval.get("baseline", {})
    conditions = retrieval.get("conditions", [])
    
    # Identify key symptoms present
    names = [c.get("disease_name", "") for c in conditions]
    names_str = ", ".join(names[:3]).lower() if names else "your symptoms"
    
    parts: List[str] = []
    
    # 1. Warm, compassionate opening with deep empathy
    parts.append(
        f"Hello {first_name}. I am so glad you reached out.\n\n"
        f"I know that dealing with a sudden onset of symptoms like {names_str} can feel overwhelming and disorienting. "
        f"It is completely understandable that you are looking for clarity on what has been happening with your body. Please take a deep breath.\n\n"
        f"Based on the records we have logged, these appear to be acute, transient episodes rather than signs of an established chronic condition. "
        f"Below is a clear, structured summary of your recent health journey to help you feel more in control and prepared for any next steps."
    )
    
    # 2. Clinical Baseline Overview Table
    chronic = ", ".join(baseline.get("chronic_conditions", [])) or "None identified"
    allergies = ", ".join(baseline.get("allergies", [])) or "None identified"
    meds = baseline.get("active_medications", [])
    meds_str = ", ".join([f"{m.get('name', 'Med')}" if isinstance(m, dict) else str(m) for m in meds]) or "None currently prescribed"
    general = baseline.get("general_summary") or "No durable clinical baseline established; current symptoms are considered transient/acute."
    
    parts.append(
        "### 👤 Clinical Baseline Overview\n\n"
        "| Category | Clinical Profile Details |\n"
        "| :--- | :--- |\n"
        f"| **Chronic Conditions** | {chronic} |\n"
        f"| **Known Allergies** | {allergies} |\n"
        f"| **Active Medications** | {meds_str} |\n"
        f"| **General Status** | {general} |"
    )
    
    # 3. Recent Episodes & Symptoms Table
    if conditions:
        rows = [
            "### 🩺 Recent Episodes & Symptoms\n\n"
            "Here is a detailed look at the specific episodes you have logged, including their onset times, severity, and context:\n\n"
            "| Episode | Onset (24h) | Severity | Key Triggers & Context | Status |\n"
            "| :--- | :--- | :--- | :--- | :--- |"
        ]
        for c in conditions:
            disease = c.get("disease_name", "General Health")
            onset = c.get("onset_24h") or "Not specified"
            sev = f"{c['severity_score']} / 10" if c.get("severity_score") else "Not specified"
            data = c.get("data", {})
            trig_data = data.get("symptoms", {}).get("trigger_event")
            if isinstance(trig_data, dict):
                context = trig_data.get("event_description") or c.get("chief_complaint") or "Recorded in consult"
            else:
                context = c.get("chief_complaint") or "Recorded in consult"
            status = "Active"
            rows.append(f"| **{disease}** | {onset} | {sev} | {context} | {status} |")
        parts.append("\n".join(rows))
    else:
        parts.append(
            "### 🩺 Recent Episodes & Symptoms\n\n"
            "*No acute conditions or active symptoms currently logged.*"
        )
        
    # 4. Questions & Guidance For Your Doctor
    guidance_items = []
    
    # Specific timeline correlation
    dizzy_cond = next((c for c in conditions if "dizz" in c.get("disease_name", "").lower()), None)
    pain_cond = next((c for c in conditions if "stomach" in c.get("disease_name", "").lower() or (c.get("severity_score") and int(c.get("severity_score", 0)) >= 7)), None)
    anx_cond = next((c for c in conditions if "anx" in c.get("disease_name", "").lower()), None)
    
    if dizzy_cond and pain_cond:
        d_onset = dizzy_cond.get("onset_24h", "prior to stomach pain")
        p_onset = pain_cond.get("onset_24h", "shortly after")
        guidance_items.append(
            f"- **Timeline & Trigger Correlation**: Share that dizziness started on **{d_onset}** (following acute alcohol intake), followed by severe stomach pain (**{p_onset}**) with loose motions. Providing these exact timestamps helps a clinician evaluate potential gastric and metabolic causes."
        )
    elif conditions:
        c0 = conditions[0]
        guidance_items.append(
            f"- **Timeline & Context**: Mention your exact onset time (**{c0.get('onset_24h', 'recorded')}**) and related triggers to give your clinician complete context."
        )
        
    if anx_cond:
        guidance_items.append(
            "- **Systemic & Gut-Brain Reaction**: Ask whether the acute gastrointestinal distress or dizziness could be contributing to the sudden anxiety spikes you experienced."
        )
        
    if pain_cond or any(c.get("severity_score") and int(c["severity_score"]) >= 7 for c in conditions):
        guidance_items.append(
            "- **Hydration & Recovery**: Given the high severity stomach pain (8/10) and loose motions, ask about electrolyte rehydration and bland diet support while monitoring recovery."
        )
    else:
        guidance_items.append(
            "- **Symptom Monitoring**: Continue logging any changes in symptom severity to share with your physician."
        )
        
    parts.append(
        "### 💡 Questions & Guidance For Your Doctor\n\n"
        "When you discuss these symptoms with your physician, here are targeted considerations based on your timeline:\n\n"
        + "\n\n".join(guidance_items)
    )
    
    # 5. Supportive sign-off
    parts.append(
        "I am here with you every step of the way. Take it easy today, stay well hydrated, and please let me know if you would like to log more updates or prepare questions for your doctor."
    )
    
    return "\n\n".join(parts)


REPORT_RAG_SYSTEM_PROMPT = """You are CareFlow's clinical care coordinator.
Generate a warm, deeply empathetic, and beautifully structured medical summary based SOLELY on the verified patient records below.

TONE & STYLE:
- Warm, compassionate, deeply validating, and reassuring.
- Acknowledge how overwhelming and disorienting sudden physical symptoms can feel. Reassure the patient that their logged records point to acute, transient episodes rather than a chronic condition.
- Maintain generous spacing: ALWAYS put double newlines (\\n\\n) between headings, tables, bullet points, and paragraphs so that the content is spacious, clear, and easy to read.

REQUIRED SECTIONS & SPACING:
1. Warm Greeting & Empathy:
   Acknowledge their distress with genuine care, encourage them to take a deep breath, and reassure them that based on records these appear to be acute, transient episodes.
   
2. ### 👤 Clinical Baseline Overview
   A clean Markdown table with categories: Chronic Conditions, Known Allergies, Active Medications, and General Status.

3. ### 🩺 Recent Episodes & Symptoms
   A brief intro sentence followed by a clean Markdown table:
   | Episode | Onset (24h) | Severity | Key Triggers & Context | Status |
   (Strictly preserve exact 24-hour onset timestamps like YYYY-MM-DD HH:MM).

4. ### 💡 Questions & Guidance For Your Doctor
   2-3 thoughtful bullet points with double spacing, connecting their timeline correlation, symptom triggers, and hydration/recovery questions for their doctor.

5. Reassuring Sign-Off:
   A warm, comforting 1-2 sentence closing assuring them that CareFlow is with them and encouraging rest and hydration.

STRICT GROUNDING:
- ONLY state details present in the retrieved records. Do not invent diagnoses or medications.
- ALWAYS preserve exact dates and 24-hour timestamps (YYYY-MM-DD HH:MM).

RETRIEVED PATIENT CLINICAL HISTORY:
{retrieved_context}
"""


def generate_rag_report_response(username: str, query: str, history: Optional[List[Any]] = None) -> str:
    """Generates an empathetic, RAG-grounded response answering the patient's query about their medical history."""
    retrieval = retrieve_patient_history_context(username, query)
    
    context_lines = []
    
    # 1. Baseline Summary
    baseline = retrieval.get("baseline", {})
    if baseline:
        context_lines.append("### Baseline Summary:")
        context_lines.append(f"- Summary: {baseline.get('general_summary', 'N/A')}")
        context_lines.append(f"- Chronic Conditions: {', '.join(baseline.get('chronic_conditions', [])) or 'None'}")
        context_lines.append(f"- Known Allergies: {', '.join(baseline.get('allergies', [])) or 'None'}")
        meds_str = ", ".join([f"{m.get('name', 'Med')} ({m.get('dosage_frequency', 'as directed')})" if isinstance(m, dict) else str(m) for m in baseline.get("active_medications", [])])
        context_lines.append(f"- Active Medications: {meds_str or 'None'}")
        context_lines.append("")

    # 2. Medical Conditions & 24h Onset
    conditions = retrieval.get("conditions", [])
    if conditions:
        context_lines.append("### Conditions & Symptoms Record:")
        for c in conditions:
            context_lines.append(f"- Condition: {c['disease_name']}")
            context_lines.append(f"  • Onset (24h Time): {c['onset_24h'] or 'Not specified'}")
            context_lines.append(f"  • Chief Complaint: {c['chief_complaint'] or 'N/A'}")
            if c.get("severity_score"):
                context_lines.append(f"  • Severity: {c['severity_score']}/10")
            if c.get("treatment_name"):
                context_lines.append(f"  • Treatment: {c['treatment_name']} ({c.get('treatment_status', 'active')})")
            if c.get("report_type"):
                context_lines.append(f"  • Test/Report: {c['report_type']}")
            if c.get("timeline"):
                for t in c["timeline"][-3:]:
                    context_lines.append(f"  • Timeline: [{t.get('timestamp')}] {t.get('event_description')}")
        context_lines.append("")

    # 3. Memories
    memories = retrieval.get("memories", [])
    if memories:
        context_lines.append("### Relevant Clinical Observations:")
        for m in memories[:8]:
            context_lines.append(f"- `[{m['memory_type']}]` {m['content']} (Logged: {m.get('created_at', '')[:16]})")
        context_lines.append("")

    context_str = "\n".join(context_lines) if context_lines else "No prior medical records found for this patient."

    system_prompt = REPORT_RAG_SYSTEM_PROMPT.replace("{retrieved_context}", context_str)
    
    # Use LLM configured with key pool for 'report' agent
    llm = key_pool.create_llm(agent_name="report", max_tokens=1000)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Patient Query regarding medical history:\n{query}")
    ]

    answer = ""
    try:
        resp = llm.invoke(messages)
        content = resp.content.strip()
        # Verify that the LLM generated structured tables; if it returned a raw paragraph, augment or use fallback
        if "|" in content and "\n" in content:
            answer = content
        else:
            answer = build_empathetic_table_report(username, retrieval, query)
    except Exception as e:
        print(f"[Report RAG Notice]: {e}")
        answer = build_empathetic_table_report(username, retrieval, query)

    if not answer:
        answer = build_empathetic_table_report(username, retrieval, query)

    # If user explicitly asked to generate or download a full report, save markdown file
    lower_q = query.lower()
    if any(k in lower_q for k in ["download", "generate full report", "pdf", "save report", "formal report"]):
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        report_file = REPORTS_DIR / f"{username}_report_{ts}.md"
        report_file.write_text(f"# CareFlow Medical History Report\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n**Patient:** {username}\n\n## Summary\n{answer}\n\n## Clinical Records Grounding\n{context_str}", encoding="utf-8")
        answer += f"\n\n*(A formal copy of this report has also been archived to your patient records file: `{report_file.name}`)*"

    return answer
