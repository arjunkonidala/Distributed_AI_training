"""Autonomous Long-Term Memory (LTM) Consolidation Agent for CareFlow.
Agent-written, self-consolidating memory layer that synthesizes chronic conditions,
allergies, active medications, lifestyle habits, and symptom triggers across conversations.
Maintains both database persistence (for fast RAG/retrieval) and an updated Markdown profile.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

import database as db

from api_pool import key_pool

ROOT = Path(__file__).parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")


def get_llm():
    return key_pool.create_llm(
        agent_name="memory_consolidation",
        model_name=MODEL_NAME,
        temperature=0.2,
        max_tokens=800,
    )


CONSOLIDATION_SYSTEM_PROMPT = """You are the CareFlow Clinical Memory Consolidation Agent.
Your responsibility is to analyze the conversation between a patient and the healthcare assistant,
extract durable long-term medical knowledge, resolve any conflicts or outdated facts, and synthesize
an updated clinical baseline.

Focus ONLY on long-term clinical relevance:
1. 'chronic_condition': Diagnosed or ongoing conditions (e.g. Hypertension, Gastritis, Asthma, Migraine).
2. 'allergy': Drug, environmental, or food allergies and adverse reactions.
3. 'medication_history': Daily or active medications, dosages, or discontinued therapies.
4. 'symptom_trigger': Specific triggers identified by the patient (e.g. spicy food, lack of sleep, stress).
5. 'lifestyle_habit': Relevant habits (smoking, alcohol, night shifts, physical activity).
6. 'historical_finding': Lab results, past surgeries, or diagnostic scans (e.g. endoscopy 2024).

Do NOT store temporary, transient chit-chat (e.g., "I am feeling a little tired today").
Only store verified, enduring clinical facts.

Existing Patient Baseline:
{current_baseline}

Existing Active Memories:
{active_memories}

Your Output MUST be a valid JSON object matching this schema:
{
    "new_memories": [
        {
            "memory_type": "chronic_condition | allergy | medication_history | symptom_trigger | lifestyle_habit | historical_finding",
            "content": "Clear, self-contained atomic clinical statement",
            "confidence_score": 0.8 to 1.0
        }
    ],
    "contradicted_memory_ids": ["ids of any existing memories that the patient stated are no longer true or discontinued"],
    "chronic_conditions": ["list of all currently known active chronic conditions"],
    "active_medications": [{"name": "Med name", "dosage_frequency": "e.g. 20mg daily"}],
    "allergies": ["list of known allergies"],
    "general_summary": "Concise 2-4 sentence clinical summary of the patient's baseline health status, history, and active precautions."
}
"""


def consolidate_patient_memory(username: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Autonomous agent consolidation: extracts durable facts from dialogue, resolves contradictions,
    and updates both the database LTM tables and the patient's Markdown memory file.
    """
    if not username:
        return {"status": "error", "message": "Missing username"}

    # 1. Fetch current baseline and active memories
    baseline = db.get_patient_baseline(username) or {}
    active_memories = db.get_patient_memories(username, status="active", limit=30)
    
    # 2. Fetch recent conversation messages
    if session_id:
        messages = db.get_chat_messages(session_id)
    else:
        convs = db.get_conversations(username)
        messages = db.get_chat_messages(convs[0]["id"]) if convs else []

    if not messages:
        return {"status": "noop", "message": "No conversation messages to consolidate"}

    # Format transcript for the consolidation LLM
    transcript_lines = []
    for m in messages[-20:]:  # review up to the last 20 messages of this dialogue
        role = "Patient" if m.get("role") == "user" else "Assistant"
        transcript_lines.append(f"{role}: {m.get('text', '')}")
    transcript_text = "\n".join(transcript_lines)

    # If no LLM key or call fails, gracefully fallback
    llm = get_llm()
    if not llm:
        print("[MemoryAgent] No GROQ_API_KEY set; skipping LLM consolidation.")
        return {"status": "skipped", "message": "No LLM available"}

    active_memories_summary = json.dumps([
        {"id": m["id"], "type": m["memory_type"], "content": m["content"]}
        for m in active_memories
    ], indent=2)

    prompt = CONSOLIDATION_SYSTEM_PROMPT.replace(
        "{current_baseline}", json.dumps(baseline, indent=2)
    ).replace(
        "{active_memories}", active_memories_summary
    )

    try:
        call_messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Here is the latest patient conversation transcript:\n\n{transcript_text}")
        ]
        response = llm.invoke(call_messages)
        content = response.content.strip()
        
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return {"status": "error", "message": "Could not parse JSON from consolidation agent"}
        
        data = json.loads(match.group())
    except Exception as e:
        print(f"[MemoryAgent] LLM consolidation failed: {e}")
        return {"status": "error", "message": str(e)}

    # 3. Apply updates to the database
    # Invalidate contradicted memories
    for mem_id in data.get("contradicted_memory_ids", []):
        try:
            db.update_patient_memory_status(mem_id, "contradicted")
        except Exception as e:
            print(f"[MemoryAgent] Could not invalidate memory {mem_id}: {e}")

    # Add new memories
    saved_memories = []
    for item in data.get("new_memories", []):
        content = (item.get("content") or "").strip()
        mtype = (item.get("memory_type") or "historical_finding").strip()
        conf = float(item.get("confidence_score") or 1.0)
        
        # Avoid duplicate memory insertion
        is_dup = any(
            m["content"].lower().strip() == content.lower().strip()
            for m in active_memories
        )
        if content and not is_dup:
            saved = db.save_patient_memory(
                username=username,
                memory_type=mtype,
                content=content,
                confidence_score=conf,
                status="active",
                source_session_id=session_id
            )
            saved_memories.append(saved)

    # 4. Upsert Consolidated Baseline Profile
    chronic = data.get("chronic_conditions") or baseline.get("chronic_conditions") or []
    meds = data.get("active_medications") or baseline.get("active_medications") or []
    allergies = data.get("allergies") or baseline.get("allergies") or []
    summary = data.get("general_summary") or baseline.get("general_summary") or "Patient care baseline recorded."

    updated_baseline = db.upsert_patient_baseline(
        username=username,
        chronic_conditions=chronic,
        active_medications=meds,
        allergies=allergies,
        general_summary=summary,
    )

    # 5. Export updated Markdown summary artifact for human review & export
    md_path = export_patient_summary_md(username)

    return {
        "status": "success",
        "new_memories_count": len(saved_memories),
        "contradicted_count": len(data.get("contradicted_memory_ids", [])),
        "baseline": updated_baseline,
        "markdown_path": str(md_path)
    }


def export_patient_summary_md(username: str) -> Path:
    """Generates an agent-maintained Markdown summary file for the patient's long-term health record."""
    profile = db.get_profile(username) or {}
    baseline = db.get_patient_baseline(username) or {}
    memories = db.get_patient_memories(username, status="active", limit=50)
    
    chronic = baseline.get("chronic_conditions", [])
    meds = baseline.get("active_medications", [])
    allergies = baseline.get("allergies", [])
    summary = baseline.get("general_summary", "No consolidated summary yet.")
    last_consolidated = baseline.get("last_consolidated_at", "N/A")

    lines = [
        f"# CareFlow Long-Term Clinical Memory Profile",
        f"**Patient**: {profile.get('name', username)} (`@{username}`)",
        f"**Last Consolidated**: {last_consolidated} (Version {baseline.get('version', 1)})\n",
        f"## 1. Clinical Baseline Summary",
        f"{summary}\n",
        f"## 2. Chronic Conditions",
    ]
    if chronic:
        for c in chronic:
            lines.append(f"- **{c}**")
    else:
        lines.append("- *None formally recorded*")

    lines.append("\n## 3. Active Medications & Therapies")
    if meds:
        for m in meds:
            if isinstance(m, dict):
                lines.append(f"- **{m.get('name', 'Unknown')}**: {m.get('dosage_frequency', 'as directed')}")
            else:
                lines.append(f"- {m}")
    else:
        lines.append("- *No ongoing medications reported*")

    lines.append("\n## 4. Known Allergies & Sensitivities")
    if allergies:
        for a in allergies:
            lines.append(f"- ⚠️ {a}")
    else:
        lines.append("- *No allergies reported*")

    lines.append("\n## 5. Consolidated Atomic Observations")
    if memories:
        for m in memories:
            lines.append(f"- `[{m['memory_type']}]` {m['content']} *(Recorded: {m['created_at'][:10]})*")
    else:
        lines.append("- *No individual observations logged yet*")

    lines.append("\n---\n*This document is autonomously maintained and consolidated by CareFlow's Long-Term Memory Agent.*")

    output_path = REPORTS_DIR / f"{username}_memory_profile.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
