"""Multi-agent orchestration for CareFlow using LangGraph and Groq.
Implements a Supervisor Router pattern with an Interactive Intake Agent, Triage Agent, and Scheduling Agent.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

import database as db
import guardrails as guard
from models import CareFlowState, MedicalCondition, Reports, Symptoms, Treatment

from dotenv import load_dotenv
from api_pool import key_pool
from rag_report_engine import generate_rag_report_response
from onset_normalizer import normalize_onset_24h

load_dotenv()

MODEL_NAME = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")


def safe_llm_invoke(system_prompt: str, user_prompt: str, history: Optional[List[BaseMessage]] = None, max_tokens: int = 600, agent_name: str = "general") -> str:
    """Safely invokes Groq LLM using key pool with full conversation history and error recovery."""
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
    if history:
        for m in history:
            if isinstance(m, (HumanMessage, AIMessage)) and m.content:
                messages.append(m)
        # History already ends with the current patient turn; replace it with the (delimited) prompt.
        if len(messages) > 1 and isinstance(messages[-1], HumanMessage):
            messages.pop()
        messages.append(HumanMessage(content=user_prompt))
    else:
        messages.append(HumanMessage(content=user_prompt))

    try:
        call_llm = key_pool.create_llm(agent_name=agent_name, model_name=MODEL_NAME, max_tokens=max_tokens)
        resp = call_llm.invoke(messages)
        return resp.content.strip()
    except Exception as e:
        err_msg = str(e)
        print(f"[Groq LLM Notice ({agent_name})]: {err_msg}")
        
        # Check if Groq included failed_generation with the generated text
        if "failed_generation" in err_msg:
            match = re.search(r'"arguments":\s*(?:")?(.*?)(?:")?\}', err_msg, re.DOTALL)
            if match:
                clean = match.group(1).replace('\\"', '"').replace('\\n', ' ').strip()
                if clean:
                    return clean
                    
        # Try fallback model with next available key in pool
        try:
            used_key = key_pool.get_key_for_agent(agent_name)
            fallback_model = "openai/gpt-oss-120b" if MODEL_NAME != "openai/gpt-oss-120b" else "openai/gpt-oss-20b"
            fallback_llm = key_pool.create_llm(agent_name=agent_name, model_name=fallback_model, max_tokens=max_tokens, failover_from=used_key)
            resp = fallback_llm.invoke(messages)
            return resp.content.strip()
        except Exception as e2:
            print(f"[Groq Fallback Notice ({agent_name})]: {e2}")
            return "Thank you for sharing that with me. I have recorded these details in your care record. How are you feeling right now, and would you like me to help arrange an appointment or share comfort tips?"


# ------------------ SUPERVISOR ROUTER ------------------

def supervisor_node(state: CareFlowState) -> Dict[str, Any]:
    """Supervisor Node: Context-aware evaluation of user input to route to Intake, Triage, or Scheduling agent."""
    user_input = state.get("user_input", "")
    lower_input = user_input.lower().strip()
    history = state.get("messages", [])
    
    # 1. Immediate emergency red-flag override
    is_emergency = (
        state.get("guard_action") == "emergency" or 
        guard.detect_emergency(user_input) or 
        guard.detect_self_harm(user_input)
    )

    # 2. Explicit scheduling keywords — require literal booking intent
    is_scheduling = any(phrase in lower_input for phrase in [
        "book an appointment", "schedule an appointment", "book appointment", "schedule appointment",
        "book a doctor", "doctor visit slot", "reschedule my appointment", "make an appointment",
        "i want to book", "i need an appointment", "can i book", "can i schedule"
    ])

    # 3. Report & Medical History RAG keywords
    is_report_query = any(phrase in lower_input for phrase in [
        "generate report", "health report", "medical report", "download report", "full report",
        "summary report", "give me report", "my report", "my medical history", "past medical",
        "previous scan", "last scan", "what medications was i", "timeline of my", "what were my results",
        "past symptoms", "medical record", "summarize my history", "history of my"
    ])

    # 4. Check for symptom mentions to determine if parallel execution is needed
    has_symptoms = any(phrase in lower_input for phrase in [
        "pain", "ache", "fever", "cough", "dizzy", "headache", "nausea", "burning", "vomit",
        "tired", "cramp", "swelling", "stomach", "chest", "throat", "pressure", "rash",
        "bleeding", "shortness of breath", "symptom"
    ])

    # 1. Immediate emergency red-flag override
    if is_emergency:
        return {"next_agent": "triage"}

    # 2. Report & Medical History RAG keywords
    if is_report_query:
        return {"next_agent": "report"}

    # 3. Explicit scheduling keywords — require literal booking intent
    if is_scheduling:
        return {"next_agent": "scheduling"}

    system_prompt = """You are the CareFlow Supervisor Router.
Evaluate the patient's latest message and route to the correct agent.

Agents:
- 'intake': DEFAULT. Use for symptoms, treatments, answering questions, ANY health-related message.
- 'triage': ONLY for life-threatening emergencies (chest pain, severe bleeding, unconscious).
- 'scheduling': ONLY when user explicitly says 'book' or 'schedule an appointment'. Never use for general health queries.
- 'report': ONLY when user asks for a health report or queries their past medical history.
- 'direct': ONLY for a pure greeting like 'hi'/'hello' with zero health content.

When in doubt always use 'intake'.
Reply ONLY with JSON: {"next_agent": "intake"}
""" + guard.DATA_ONLY_RULE

    try:
        content = safe_llm_invoke(system_prompt, guard.wrap_patient_message(user_input), history=history, max_tokens=60, agent_name="supervisor")
        match = re.search(r"\{.*?\}", content, re.DOTALL)
        if match:
            decision = json.loads(match.group())
            next_agent = decision.get("next_agent", "intake")
            if next_agent in ["intake", "report", "triage", "scheduling", "direct"]:
                return {"next_agent": next_agent}
    except Exception as e:
        print(f"[Supervisor] Routing fallback: {e}")

    return {"next_agent": "intake"}


# ------------------ INTAKE AGENT (INTERACTIVE & EMPATHETIC) ------------------

def intake_agent_node(state: CareFlowState) -> Dict[str, Any]:
    """Intake Agent: Empathetically engages the patient, extracts rich clinical entities, offers comfort guidance, and avoids repetitive loops."""
    username = state["username"]
    user_input = state["user_input"]
    lower_input = user_input.lower().strip()
    follow_up_tracker = dict(state.get("follow_up_tracker", {}))
    active_conditions = dict(state.get("active_conditions", {}))
    history = state.get("messages", [])
    
    # Determine disease name and get current DB state BEFORE extraction
    from datetime import datetime
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    existing_conditions = db.get_conditions(username)
    latest_known_disease = existing_conditions[0]["disease_name"] if existing_conditions else "General Health"
    current_db_state = json.dumps(existing_conditions[0].get("data", {})) if existing_conditions else "{}"
    
    # 1. Extraction step: Extract rich clinical entities across symptoms, triggers, treatments, reports
    extraction_system_prompt = """You are an expert clinical entity extractor for CareFlow.
Analyze the user's message in the context of recent chat history and extract all medical condition details.
Interpret triggers, events, preceding activities (e.g. eating spicy food, stressful events, poor sleep, workout), symptom qualities, treatments, and tests.
If user says 'no' or 'not taking any medication', mark status as 'none_reported'. If user says 'no tests done', mark report_type as 'none_reported'.
disease_name must be a condition the patient says they have been diagnosed with, or a short descriptive label of their main symptom (e.g. 'Stomach Pain'). Never put a speculative diagnosis, or a condition the patient is only asking about, in disease_name.
If the message contains no health information, return null for every field.
DATA_RULE_PLACEHOLDER

CRITICAL CONTEXT:
- Current date and 24-hour time: {current_time}
- Patient's currently saved medical record for {disease} (if any):
{db_state}

Ensure you properly extract fields to update this Pydantic-based JSON structure with any new information. Convert any relative times (e.g. '2 hours ago', 'this morning') into precise 24-hour timestamps 'YYYY-MM-DD HH:MM' based on the current time.

Return a valid JSON object matching this schema:
{
    "disease_name": "Name of condition (e.g. Gastritis, Migraine, Back Pain)",
    "symptoms": {
        "chief_complaint": "primary symptom or complaint",
        "onset": "Strict 24-hour timestamp 'YYYY-MM-DD HH:MM' (e.g. '2026-10-18 14:30') based on current time: {current_time}",
        "severity_score": 1-10 integer or null,
        "location": "body location (e.g. 'stomach', 'right temple')",
        "character_quality": "sensation quality (e.g. 'burning', 'throbbing', 'cramping', 'sharp', 'dull ache')",
        "associated_symptoms": ["list of other symptoms mentioned"],
        "trigger_event": {
            "event_description": "preceding trigger or event (e.g. 'ate noodles at fast food center', 'lack of sleep')",
            "event_category": "dietary | physical_exertion | stress_emotional | environmental | sleep | trauma | other",
            "timing_relative_to_onset": "e.g. 'right after eating', 'during the night'"
        },
        "aggravating_factors": "factors making it worse or null",
        "alleviating_factors": "factors giving relief or null",
        "daily_impact": "impact on sleep/work or null"
    },
    "treatments": {
        "intervention_name": "medication/therapy/remedy name or null",
        "treatment_type": "medication | home_remedy | therapy | surgery | null",
        "dosage_frequency": "dose and schedule if stated or null",
        "status": "active | completed | discontinued | none_reported | null",
        "effectiveness_notes": "relief or side effects or null"
    },
    "reports": {
        "report_type": "lab test/scan/endoscopy type or 'none_reported' or null",
        "provider_facility": "clinic or facility name or null",
        "key_findings": "findings stated or null"
    },
    "timeline": [
        {
            "timestamp": "Precise absolute 24h time (e.g. '2026-10-03 14:00')",
            "event_description": "Brief description of what happened at this time"
        }
    ],
    "is_requesting_full_report": false,
    "is_asking_for_solution_or_advice": true/false
}""".replace("{disease}", latest_known_disease).replace("{db_state}", current_db_state).replace("{current_time}", current_time_str).replace("DATA_RULE_PLACEHOLDER", guard.DATA_ONLY_RULE)

    extracted_data = {}
    is_asking_for_solution = False
    try:
        ext_content = safe_llm_invoke(extraction_system_prompt, guard.wrap_patient_message(user_input), history=history, max_tokens=800, agent_name="intake_extractor")
        match = re.search(r"\{.*\}", ext_content, re.DOTALL)
        if match:
            extracted_data = json.loads(match.group())
            is_asking_for_solution = extracted_data.get("is_asking_for_solution_or_advice", False)
            if "symptoms" in extracted_data and isinstance(extracted_data["symptoms"], dict):
                raw_on = extracted_data["symptoms"].get("onset")
                extracted_data["symptoms"]["onset"] = normalize_onset_24h(raw_on)
    except Exception as e:
        print(f"[Intake] Extraction error: {e}")

    # Detect if user is asking for solution/advice via keywords
    if any(phrase in lower_input for phrase in ["suggest", "solution", "what should i do", "how to cure", "remedy", "what to take", "can u help me", "help me relieve"]):
        is_asking_for_solution = True

    # Handle negative answers explicitly in tracker
    last_assistant_msg = ""
    for m in reversed(history):
        if isinstance(m, AIMessage):
            last_assistant_msg = m.content.lower()
            break
            
    if lower_input in ["no", "nope", "no not yet", "not yet", "nothing", "none"]:
        if "medication" in last_assistant_msg or "treatment" in last_assistant_msg:
            follow_up_tracker[f"{extracted_data.get('disease_name', 'General')}_treatment_asked"] = True
        elif "test" in last_assistant_msg or "scan" in last_assistant_msg or "report" in last_assistant_msg:
            follow_up_tracker[f"{extracted_data.get('disease_name', 'General')}_reports_asked"] = True

    disease_name = extracted_data.get("disease_name") or latest_known_disease
    if str(disease_name).lower() in ["none", "null", "unknown", "", "general health"] and existing_conditions:
        disease_name = existing_conditions[0]["disease_name"]

    # Determine updated categories for UI tags
    categories = []
    if any((extracted_data.get("symptoms") or {}).values()):
        categories.append("symptoms")
    if any((extracted_data.get("treatments") or {}).values()):
        categories.append("treatment")
    if any((extracted_data.get("reports") or {}).values()):
        categories.append("reports")

    # 1. DB CHECK & UPDATE: Save to database only when the turn carried clinical content,
    # so greetings, questions, and off-topic text never become a "condition".
    if categories or extracted_data.get("timeline"):
        updated_condition = db.upsert_condition(username, disease_name, extracted_data)
        cond_record = updated_condition.get("data", extracted_data)
    else:
        existing = next((c for c in existing_conditions if c["disease_name"].lower() == str(disease_name).lower()), None)
        cond_record = existing.get("data", {}) if existing else {}
    active_conditions[disease_name] = cond_record

    # 2. Comprehend & Evaluate missing clinical fields
    symptoms = cond_record.get("symptoms") or {}
    treatments = cond_record.get("treatments") or {}
    reports = cond_record.get("reports") or {}
    
    missing_items = []
    if not symptoms.get("onset") and not follow_up_tracker.get(f"{disease_name}_onset_asked"):
        missing_items.append(("onset", "when the symptom started or how long they've had it"))
    if not symptoms.get("severity_score") and not follow_up_tracker.get(f"{disease_name}_severity_asked"):
        missing_items.append(("severity", "how severe it feels on a scale from 1 to 10"))

    # Systemic conditions where asking "where in your body" is clinically unnatural
    is_systemic = any(
        s in disease_name.lower() or s in str(symptoms.get("chief_complaint", "")).lower()
        for s in ["dizzy", "dizziness", "vertigo", "lightheaded", "fever", "nausea", "fatigue", "tired", "anxiety", "insomnia", "chills", "malaise"]
    )
    if not symptoms.get("location") and not is_systemic and not follow_up_tracker.get(f"{disease_name}_location_asked"):
        missing_items.append(("location", "where exactly they are experiencing this (which part of the body)"))
    if not symptoms.get("character_quality") and not follow_up_tracker.get(f"{disease_name}_quality_asked"):
        missing_items.append(("quality", "what the symptom feels like (e.g., sharp, dull, throbbing, burning)"))
    if not symptoms.get("trigger_event") and not follow_up_tracker.get(f"{disease_name}_trigger_asked"):
        missing_items.append(("trigger", "if they noticed anything that might have triggered or caused this"))
    if not symptoms.get("aggravating_factors") and not follow_up_tracker.get(f"{disease_name}_aggravating_asked"):
        missing_items.append(("aggravating", "if anything makes it feel worse"))
    if not symptoms.get("alleviating_factors") and not follow_up_tracker.get(f"{disease_name}_alleviating_asked"):
        missing_items.append(("alleviating", "if anything makes it feel better"))
    if not symptoms.get("associated_symptoms") and not follow_up_tracker.get(f"{disease_name}_associated_asked"):
        missing_items.append(("associated", "if they are experiencing any other symptoms along with this"))
    if not symptoms.get("daily_impact") and not follow_up_tracker.get(f"{disease_name}_impact_asked"):
        missing_items.append(("impact", "how this is impacting their daily life, sleep, or work"))
    
    # Treatment follow-ups (do not ask if status is none_reported)
    if not treatments.get("intervention_name") and str(treatments.get("status", "")).lower() != "none_reported" and not follow_up_tracker.get(f"{disease_name}_treatment_asked"):
        missing_items.append(("treatment", "whether they are taking any medications or treatments for it"))
    elif treatments.get("intervention_name") and str(treatments.get("status", "")).lower() != "none_reported":
        if not treatments.get("dosage_frequency") and not follow_up_tracker.get(f"{disease_name}_dosage_asked"):
            missing_items.append(("dosage", "the dosage or frequency of the treatment they are taking"))
        if not treatments.get("effectiveness_notes") and not follow_up_tracker.get(f"{disease_name}_effectiveness_asked"):
            missing_items.append(("effectiveness", "whether the treatment has provided any relief or caused side effects"))

    # Report follow-ups (do not ask if report_type is none_reported)
    is_report_none = str(reports.get("report_type", "")).lower() in ["none_reported", "none", "null"]
    if not reports.get("report_type") and not is_report_none and not follow_up_tracker.get(f"{disease_name}_reports_asked"):
        missing_items.append(("reports", "if they have had any relevant tests, scans, or lab reports"))
    elif reports.get("report_type") and not is_report_none:
        if not reports.get("key_findings") and not follow_up_tracker.get(f"{disease_name}_findings_asked"):
            missing_items.append(("findings", "what the key findings or results of their tests were"))

    # Determine next topic to ask that hasn't been asked yet
    next_question_topic = None
    next_question_desc = None
    for topic_key, topic_desc in missing_items:
        tracker_tag = f"{disease_name}_{topic_key}_asked"
        if not follow_up_tracker.get(tracker_tag):
            next_question_topic = topic_key
            next_question_desc = topic_desc
            break

    # Formulate human-readable already answered topics
    already_answered = []
    for key, val in follow_up_tracker.items():
        if val and disease_name in key:
            topic = key.replace(f"{disease_name}_", "").replace("_asked", "")
            already_answered.append(topic)

    # Severity-based advice injection (only show once)
    severity = symptoms.get("severity_score")
    severity_instruction = ""
    if severity is not None and not follow_up_tracker.get(f"{disease_name}_severity_advice_given"):
        if int(severity) <= 4:
            severity_instruction = (
                f"Severity is low ({severity}/10). In one sentence, mention a general comfort measure "
                f"(e.g. rest, hydration, a warm compress, as appropriate) and that a pharmacist or doctor can advise "
                f"if it doesn't settle. Never name a medication or a dose."
            )
        else:
            severity_instruction = (
                f"Severity is significant ({severity}/10). In one sentence, advise them to visit their nearest "
                f"health centre or consult a doctor soon."
            )
        follow_up_tracker[f"{disease_name}_severity_advice_given"] = True

    # Determine question instruction
    if next_question_topic and not is_asking_for_solution:
        question_instruction = f"Acknowledge what they just shared in 1 natural clause, then ask ONE natural question about {next_question_desc}."
        sentence_rule = "- MAXIMUM 1-2 brief conversational sentences: acknowledge the patient's new update, then ask the ONE question. Keep it concise."
    else:
        question_instruction = "Briefly acknowledge their update and let them know their record is saved. Invite them to share any other questions."
        sentence_rule = "- MAXIMUM 1-2 brief sentences. Keep it brief. Do not ask any probing medical questions."

    # Long-term memory baseline injection
    ltm_baseline = db.get_patient_baseline(username)
    ltm_context = ""
    if ltm_baseline:
        ltm_summary = ltm_baseline.get("general_summary", "")
        ltm_allergies = ", ".join(ltm_baseline.get("allergies", [])) or "None"
        ltm_chronic = ", ".join(ltm_baseline.get("chronic_conditions", [])) or "None"
        ltm_context = f"\nLong-Term Baseline: {ltm_summary} | Allergies: {ltm_allergies} | Chronic: {ltm_chronic}"

    # Anti-repetition instructions from recent conversation history
    past_assistant_replies = [m.content for m in history if isinstance(m, AIMessage) and m.content][-3:]
    anti_repetition_text = ""
    if past_assistant_replies:
        anti_repetition_text = (
            "\nCRITICAL ANTI-REPETITION MANDATE:\n"
            "- NEVER reuse exact phrases or canned lines from earlier messages (such as 'I am sorry you are feeling unwell', 'Please rest in a safe place and drink water to help with hydration').\n"
            "- If you have already expressed empathy or offered general comfort advice previously in this chat, DO NOT repeat it. Jump straight to acknowledging their specific new detail and asking your question.\n"
            "- NEVER ask a question that was already asked or answered earlier in the chat.\n"
            "- Vary your openings naturally (e.g., 'Got it...', 'Understood...', 'Thanks for clarifying...', or reply directly).\n"
        )

    intake_system_prompt = f"""You are CareFlow, a warm care coordination assistant. Respond in plain conversational text.

{guard.SAFETY_RULES}

STRICT RULES:
- NO markdown, NO bullet points, NO bold/italic, NO asterisks, NO headers.
{sentence_rule}
- Do not repeat any topic from: {', '.join(already_answered)}
- Sound like an attentive human healthcare coordinator, not a repetitive automated script.
{anti_repetition_text}
{severity_instruction}
{question_instruction}

Context: {disease_name} | Stored: {json.dumps({k: v for k, v in cond_record.get('symptoms', {}).items() if v})}{ltm_context}"""

    # Generate conversational response — keep it short
    response_text = safe_llm_invoke(intake_system_prompt, guard.wrap_patient_message(user_input), history=history, max_tokens=220, agent_name="intake")
    
    # Mark topic as asked in tracker
    if next_question_topic:
        follow_up_tracker[f"{disease_name}_{next_question_topic}_asked"] = True
    follow_up_tracker[f"{disease_name}_asked"] = True
        
    return {
        "response_text": response_text,
        "follow_up_tracker": follow_up_tracker,
        "active_conditions": active_conditions,
        "categories": categories,
        "messages": [AIMessage(content=response_text)],
        "agent_responses": {"intake": response_text},
        "parallel_agents": ["intake"],
    }


# ------------------ TRIAGE AGENT ------------------

def triage_agent_node(state: CareFlowState) -> Dict[str, Any]:
    """Triage Agent: Assesses clinical urgency, checks red flags, and offers safety guidance."""
    user_input = state["user_input"]
    history = state.get("messages", [])
    
    triage_system_prompt = """You are the CareFlow Triage Assistant.
Your mission is to perform clinical safety triage, detect red-flag symptoms, and provide clear safety directions.

Guidelines:
1. If urgent/red-flag symptoms are present (chest pain, severe shortness of breath, sudden numbness, heavy bleeding, loss of consciousness), immediately advise the patient to call emergency services (e.g. 911 / 112) or go to the nearest emergency department.
2. For non-emergency symptoms, validate their condition empathetically, categorize urgency (Routine / Priority / Immediate), and advise on consulting a physician.
3. Keep the response caring, reassuring, and safety-focused. Never diagnose or prescribe medication.

""" + guard.SAFETY_RULES

    triage_text = safe_llm_invoke(triage_system_prompt, guard.wrap_patient_message(user_input), history=history, agent_name="triage")
    
    return {
        "response_text": triage_text,
        "categories": ["symptoms"],
        "messages": [AIMessage(content=triage_text)],
        "agent_responses": {"triage": triage_text},
        "parallel_agents": ["triage"],
    }


# ------------------ SCHEDULING AGENT ------------------

def scheduling_agent_node(state: CareFlowState) -> Dict[str, Any]:
    """Scheduling Agent: Coordinates mock appointment bookings with providers."""
    username = state["username"]
    user_input = state["user_input"]
    history = state.get("messages", [])
    
    sched_system_prompt = """You are the CareFlow Scheduling Agent.
Coordinate healthcare appointments empathetically and efficiently.
Extract the appointment intent and return JSON:
{
    "wants_to_book": true/false,
    "doctor_type": "Primary Care / Cardiologist / Specialist / General Practitioner",
    "preferred_date": "YYYY-MM-DD or Next Available",
    "preferred_time": "Time or Morning/Afternoon",
    "reason": "Brief reason from patient"
}
""" + guard.DATA_ONLY_RULE

    try:
        intent_text = safe_llm_invoke(sched_system_prompt, guard.wrap_patient_message(user_input), history=history, agent_name="scheduling")
        match = re.search(r"\{.*\}", intent_text, re.DOTALL)
        if match:
            booking_data = json.loads(match.group())
            if booking_data.get("wants_to_book"):
                doc = booking_data.get("doctor_type") or "General Practitioner"
                p_date = booking_data.get("preferred_date") or "Upcoming Monday"
                p_time = booking_data.get("preferred_time") or "10:00 AM"
                reason = booking_data.get("reason") or user_input
                
                # Book in DB
                db.book_appointment(username, doc, p_date, p_time, reason)
                confirmation_msg = f"I've scheduled a mock appointment for you with a {doc} for {p_date} at {p_time} regarding '{reason}'. You can reschedule or adjust this at any time."
                return {
                    "response_text": confirmation_msg,
                    "categories": ["treatment"],
                    "messages": [AIMessage(content=confirmation_msg)],
                    "agent_responses": {"scheduling": confirmation_msg},
                    "parallel_agents": ["scheduling"],
                }
    except Exception as e:
        print(f"[Scheduling] Booking error: {e}")

    scheduling_conv_prompt = """You are the CareFlow Scheduling Coordinator.
Help the patient arrange appointments with specialists, clinic checkups, or follow-ups.
Ask what kind of care provider or time suits them best.

""" + guard.SAFETY_RULES

    reply = safe_llm_invoke(scheduling_conv_prompt, guard.wrap_patient_message(user_input), history=history, agent_name="scheduling")
    
    return {
        "response_text": reply,
        "categories": ["treatment"],
        "messages": [AIMessage(content=reply)],
        "agent_responses": {"scheduling": reply},
        "parallel_agents": ["scheduling"],
    }


# ------------------ DIRECT / GREETING HANDLER ------------------

def direct_node(state: CareFlowState) -> Dict[str, Any]:
    username = state["username"]
    profile = db.get_profile(username)
    first_name = profile["name"].split()[0] if profile and profile.get("name") else username
    
    greeting = f"Hello {first_name}! I'm CareFlow, your care coordination companion. How can I support your health journey today? You can share any symptoms, past reports, or treatments you'd like me to track."
    return {
        "response_text": greeting,
        "categories": [],
        "messages": [AIMessage(content=greeting)],
        "agent_responses": {"direct": greeting},
        "parallel_agents": ["direct"],
    }


# ------------------ REPORT AGENT (RAG-GROUNDED) ------------------

def report_agent_node(state: CareFlowState) -> Dict[str, Any]:
    """Report Agent: RAG-grounded retrieval across the patient's medical history."""
    username = state["username"]
    user_input = state["user_input"]
    history = state.get("messages", [])
    
    report_response = generate_rag_report_response(username, user_input, history=history)
    return {
        "response_text": report_response,
        "categories": ["reports"],
        "messages": [AIMessage(content=report_response)],
        "agent_responses": {"report": report_response},
        "parallel_agents": ["report"],
    }


# ------------------ AGGREGATE PARALLEL RESPONSES ------------------

def aggregate_responses_node(state: CareFlowState) -> Dict[str, Any]:
    """Combines responses when parallel agents run concurrently (e.g. intake extraction + triage/report)."""
    agent_responses = state.get("agent_responses", {})
    if not agent_responses:
        return {}

    # Priority 1: Emergency Triage ALWAYS takes conversational precedence
    if "triage" in agent_responses:
        triage_text = agent_responses["triage"]
        if "intake" in agent_responses:
            combined = f"{triage_text}\n\n(Note: I have also recorded your reported symptom details in your medical record.)"
        else:
            combined = triage_text
        return {"response_text": combined, "categories": ["triage", "symptoms"]}

    # Priority 2: Report RAG + Intake combined response
    if "report" in agent_responses and "intake" in agent_responses:
        report_text = agent_responses["report"]
        intake_text = agent_responses["intake"]
        combined = f"{report_text}\n\n---\n**Regarding your reported symptoms:**\n{intake_text}"
        return {"response_text": combined, "categories": ["reports", "symptoms"]}

    # Default to whatever single agent responded
    first_agent = list(agent_responses.keys())[0]
    return {"response_text": agent_responses[first_agent]}


# ------------------ GUARDRAIL NODES ------------------

def llm_domain_classifier(text: str) -> Optional[bool]:
    """Fallback domain check for messages the regex rules cannot place. True = in-domain."""
    system_prompt = """You classify messages sent to a healthcare care-coordination assistant.
In-domain: the patient's own health, symptoms, wellbeing, medications they take, tests, reports, appointments, or small talk/greetings.
Off-domain: anything else (coding, sports, politics, homework, entertainment, general trivia, finance).
Reply ONLY with JSON: {"in_domain": true} or {"in_domain": false}
""" + guard.DATA_ONLY_RULE
    content = safe_llm_invoke(system_prompt, guard.wrap_patient_message(text), max_tokens=30, agent_name="domain_classifier")
    match = re.search(r"\{.*?\}", content, re.DOTALL)
    if not match:
        return None
    verdict = json.loads(match.group()).get("in_domain")
    return verdict if isinstance(verdict, bool) else None


def _log_flags(state: CareFlowState, flags: List[Dict[str, str]], excerpt: str) -> None:
    for f in flags:
        try:
            db.log_guardrail_event(state.get("session_id", ""), state.get("username", ""), f["stage"], f["rule"], f["action"], excerpt)
        except Exception as e:
            print(f"[Guardrails] Could not log event: {e}")


def input_guard_node(state: CareFlowState) -> Dict[str, Any]:
    """Input Guard: hygiene, crisis/emergency detection, prompt injection, diagnosis/prescription requests, domain."""
    has_history = len(state.get("messages", [])) > 1
    decision = guard.check_input(state.get("user_input", ""), has_history=has_history, classifier=llm_domain_classifier)
    _log_flags(state, decision.flags, decision.clean_text)
    update: Dict[str, Any] = {
        "user_input": decision.clean_text,
        "guard_action": decision.action,
        "guard_rule": decision.rule,
        "guardrail_flags": decision.flags,
    }
    if decision.action in ("block", "crisis"):
        update["response_text"] = decision.response
    return update


def guard_response_node(state: CareFlowState) -> Dict[str, Any]:
    """Replies with the fixed guardrail template chosen by input_guard (no LLM call)."""
    text = state.get("response_text") or guard.OFF_DOMAIN_RESPONSE
    return {"response_text": text, "categories": [], "messages": [AIMessage(content=text)]}


def route_after_input_guard(state: CareFlowState) -> str:
    action = state.get("guard_action", "allow")
    if action in ("block", "crisis"):
        return "guard_response"
    if action == "emergency":
        return "triage"
    return "supervisor"


def output_guard_node(state: CareFlowState) -> Dict[str, Any]:
    """Output Guard: removes diagnostic assertions and unrequested medication/dose advice, enforces emergency line."""
    username = state.get("username", "")
    allowed_context = state.get("user_input", "")
    try:
        allowed_context += " " + json.dumps([c.get("data", {}) for c in db.get_conditions(username)])
    except Exception:
        pass

    # Reports, tables, and structured summaries preserve markdown formatting
    resp_text = state.get("response_text", "")
    is_report = (
        "report" in state.get("parallel_agents", [])
        or state.get("next_agent") == "report"
        or "reports" in state.get("categories", [])
        or ("|" in resp_text and "\n" in resp_text)
    )

    safe_text, flags = guard.check_output(
        resp_text,
        guard_action=state.get("guard_action", "allow"),
        guard_rule=state.get("guard_rule", ""),
        allowed_context=allowed_context,
        preserve_markdown=is_report,
    )
    _log_flags(state, flags, resp_text)
    return {"response_text": safe_text, "guardrail_flags": flags}


def route_supervisor(state: CareFlowState) -> List[str]:
    """Routes to one or multiple nodes in parallel based on supervisor decision."""
    decision = state.get("next_agent", "intake")
    user_input = state.get("user_input", "").lower()
    has_symptoms = any(phrase in user_input for phrase in [
        "pain", "ache", "fever", "cough", "dizzy", "headache", "nausea", "burning", "vomit",
        "tired", "cramp", "swelling", "stomach", "chest", "throat", "pressure", "rash",
        "bleeding", "shortness of breath", "symptom"
    ])
    # Parallel fan-out: if emergency or report query includes symptoms, run intake extraction in parallel
    if decision == "triage" and has_symptoms:
        return ["intake", "triage"]
    if decision == "report" and has_symptoms:
        return ["intake", "report"]
    return [decision]


# ------------------ LANGGRAPH WORKFLOW BUILDER ------------------

def build_careflow_graph():
    workflow = StateGraph(CareFlowState)

    # Add nodes matching the architecture
    workflow.add_node("input_guard", input_guard_node)
    workflow.add_node("guard_response", guard_response_node)
    workflow.add_node("output_guard", output_guard_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("report", report_agent_node)
    workflow.add_node("intake", intake_agent_node)
    workflow.add_node("triage", triage_agent_node)
    workflow.add_node("scheduling", scheduling_agent_node)
    workflow.add_node("direct", direct_node)
    workflow.add_node("aggregate_responses", aggregate_responses_node)
    
    # Every turn enters through the input guard
    workflow.add_edge(START, "input_guard")
    workflow.add_conditional_edges(
        "input_guard",
        route_after_input_guard,
        {"guard_response": "guard_response", "triage": "triage", "supervisor": "supervisor"},
    )

    # Conditional routing from supervisor (supports parallel fan-out)
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "report": "report",
            "intake": "intake",
            "triage": "triage",
            "scheduling": "scheduling",
            "direct": "direct",
        }
    )
    
    # Fan-in: parallel agents aggregate before leaving through the output guard
    for node in ["report", "intake", "triage", "scheduling", "direct"]:
        workflow.add_edge(node, "aggregate_responses")
    workflow.add_edge("aggregate_responses", "output_guard")
    workflow.add_edge("guard_response", "output_guard")
    workflow.add_edge("output_guard", END)
    
    return workflow.compile()


# Global compiled graph
careflow_graph = build_careflow_graph()


def run_orchestrator(username: str, session_id: str, user_input: str, follow_up_tracker: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Runs the LangGraph multi-agent orchestrator for a single user turn."""
    # Retrieve past messages from DB for context (complete history)
    db_messages = db.get_chat_messages(session_id)
    langchain_messages: List[BaseMessage] = []
    
    for m in db_messages:
        if m["role"] == "user":
            langchain_messages.append(HumanMessage(content=m["text"]))
        else:
            langchain_messages.append(AIMessage(content=m["text"]))
            
    initial_state: CareFlowState = {
        "messages": langchain_messages,
        "active_conditions": {},
        "follow_up_tracker": follow_up_tracker or {},
        "next_agent": "",
        "username": username,
        "session_id": session_id,
        "user_input": user_input,
        "response_text": "",
        "categories": [],
        "guard_action": "allow",
        "guard_rule": "",
        "guardrail_flags": [],
        "agent_responses": {},
        "parallel_agents": [],
        "rag_report_context": "",
    }

    final_state = careflow_graph.invoke(initial_state)
    
    # Persist updated tracker to conversation in DB
    updated_tracker = final_state.get("follow_up_tracker", {})
    db.update_conversation_tracker(session_id, updated_tracker)
    
    return {
        "response_text": final_state.get("response_text", "Thank you for sharing that with CareFlow."),
        "categories": final_state.get("categories", []),
        "follow_up_tracker": updated_tracker,
        "next_agent": final_state.get("next_agent", "intake"),
        "guard_action": final_state.get("guard_action", "allow"),
        "guardrail_flags": final_state.get("guardrail_flags", []),
        "parallel_agents": final_state.get("parallel_agents", []),
    }
