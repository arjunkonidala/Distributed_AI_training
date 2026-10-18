"""Pydantic clinical schemas and LangGraph state definitions for CareFlow.
Enriched with trigger events, aggravating/alleviating factors, symptom quality, and clinical context.
"""
from __future__ import annotations

import operator
from typing import Annotated, Dict, List, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator
from onset_normalizer import normalize_onset_24h


class TriggerEvent(BaseModel):
    event_description: Optional[str] = Field(None, description="Preceding event or trigger (e.g., 'eating spicy noodles', 'heavy lifting', 'high work stress', 'lack of sleep').")
    event_category: Optional[str] = Field(None, description="Category: 'dietary', 'physical_exertion', 'stress_emotional', 'environmental', 'sleep_deprivation', 'trauma', 'other'.")
    timing_relative_to_onset: Optional[str] = Field(None, description="When the event occurred relative to symptom onset (e.g., '10 minutes before', 'during lunch', 'the night before').")


class TimelineEvent(BaseModel):
    timestamp: str = Field(..., description="The time the event occurred in strict 24-hour format 'YYYY-MM-DD HH:MM' or absolute ISO.")
    event_description: str = Field(..., description="A brief description of what happened at this time.")

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, v: Any) -> str:
        norm = normalize_onset_24h(v)
        return norm if norm else str(v)


class Symptoms(BaseModel):
    chief_complaint: Optional[str] = Field(None, description="The main concern or symptom in patient's words (e.g. 'stomach ache', 'throbbing headache').")
    onset: Optional[str] = Field(None, description="Strict 24-hour onset timestamp in 'YYYY-MM-DD HH:MM' format (e.g. '2026-10-18 14:30').")
    severity_score: Optional[int] = Field(None, ge=1, le=10, description="Patient-reported severity score from 1 to 10.")
    location: Optional[str] = Field(None, description="Anatomical location (e.g., 'upper abdomen', 'right temple', 'lower back').")
    character_quality: Optional[str] = Field(None, description="Quality of sensation: 'throbbing', 'burning', 'sharp', 'dull ache', 'cramping', 'pressure', 'stabbing'.")
    duration_pattern: Optional[str] = Field(None, description="Temporal pattern: 'constant', 'intermittent', 'worsening', 'fluctuating'.")
    associated_symptoms: List[str] = Field(default_factory=list, description="Other symptoms reported alongside the main complaint (e.g. ['nausea', 'acid reflux', 'dizziness']).")
    trigger_event: Optional[TriggerEvent] = Field(default_factory=TriggerEvent, description="Events, activities, or dietary factors that triggered or preceded the symptoms.")
    aggravating_factors: Optional[str] = Field(None, description="Factors that make the symptom worse (e.g. 'eating oily foods', 'bright light', 'moving around').")
    alleviating_factors: Optional[str] = Field(None, description="Factors that relieve the symptom (e.g. 'resting in dark room', 'drinking warm water', 'lying down').")
    daily_impact: Optional[str] = Field(None, description="Impact on daily functioning or sleep (e.g. 'unable to work', 'disrupts sleep').")

    @field_validator("onset", mode="before")
    @classmethod
    def validate_onset_24h(cls, v: Any) -> Optional[str]:
        if not v or str(v).lower() in ["none", "null", "unknown", "n/a", ""]:
            return None
        return normalize_onset_24h(v)


class Treatment(BaseModel):
    intervention_name: Optional[str] = Field(None, description="Name of medication, therapy, procedure, or home remedy (e.g. 'Ibuprofen', 'Omeprazole', 'Warm Compress').")
    treatment_type: Optional[str] = Field(None, description="Type: 'medication', 'home_remedy', 'therapy', 'surgery', 'lifestyle'.")
    dosage_frequency: Optional[str] = Field(None, description="Dose and schedule if stated (e.g. '400mg twice daily', 'after meals').")
    status: Optional[str] = Field(None, description="Status: 'active', 'completed', 'discontinued', 'planned', 'not_started'.")
    effectiveness_notes: Optional[str] = Field(None, description="Patient-reported effect or side effects (e.g. 'partial relief', 'no effect', 'caused nausea').")


class Reports(BaseModel):
    report_type: Optional[str] = Field(None, description="Type of medical test/report: 'blood test', 'endoscopy', 'MRI', 'CT scan', 'ultrasound', 'X-ray', 'lab work'.")
    provider_facility: Optional[str] = Field(None, description="Clinic, hospital, or diagnostic center name.")
    date_conducted: Optional[str] = Field(None, description="Date the test was performed.")
    key_findings: Optional[str] = Field(None, description="Findings or results stated by patient (e.g. 'normal scan', 'elevated enzymes', 'mild gastritis').")


class MedicalCondition(BaseModel):
    disease_name: str = Field(..., description="The clinical condition or problem name (e.g. 'Gastritis', 'Migraine', 'Hypertension').")
    symptoms: Symptoms = Field(default_factory=Symptoms)
    treatments: Treatment = Field(default_factory=Treatment)
    reports: Reports = Field(default_factory=Reports)
    timeline: List[TimelineEvent] = Field(default_factory=list, description="Chronological timeline of events related to the condition.")
    insights_summary: Optional[str] = Field(None, description="Concise clinical coordination summary.")


class CareFlowState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    active_conditions: Annotated[dict, operator.ior]  # Stores extracted condition schemas
    follow_up_tracker: Annotated[dict, operator.ior]  # Tracks if a follow-up was asked per condition
    next_agent: str
    username: str
    session_id: str
    user_input: str
    response_text: str
    categories: list[str]
    guard_action: str  # allow | deflect | emergency | crisis | block (set by input_guard)
    guard_rule: str  # rule that produced guard_action, e.g. 'diagnosis_request'
    guardrail_flags: Annotated[list, operator.add]  # [{stage, rule, action}] from input/output guards
    agent_responses: Annotated[dict, operator.ior]  # Stores outputs from parallel agents
    parallel_agents: Annotated[list, operator.add]  # Tracks all agents executed in parallel
    rag_report_context: Optional[str]  # Grounding context retrieved for medical report RAG
