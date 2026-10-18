# CareFlow Progress

## Current Milestone

Multi-Agent LangGraph Orchestration and Database-Backed Chat & Structured Intake.

## Complete Architecture

1. **Database Layer (`database.py`)**:
   - Supports PostgreSQL via `DATABASE_URL` / `POSTGRES_URL` with local SQLite (`careflow.db`) fallback.
   - Complete conversational chat history stored in `chat_messages` table under session IDs.
   - User messages and assistant responses are saved immediately to the database.
   - Disease-centric condition storage in `medical_conditions` table (`disease_name`, `symptoms`, `treatments`, `reports`, `raw_data`).
   - Patient profiles (`patient_profiles`), conversations (`conversations`), and mock appointments (`appointments`).
   - Removed legacy `persistent.txt`.

2. **LangGraph Multi-Agent Orchestration (`orchestrator.py`, `models.py`)**:
   - **Supervisor Router Node**: Evaluates user input and routes to specialized agents: `Intake`, `Triage`, `Scheduling`, or direct greetings.
   - **Intake Agent**:
     - Disease-centric schema extraction (`MedicalCondition`, `Symptoms`, `Treatment`, `Reports`).
     - Tool-driven DB check and entity update via `check_and_update_db`.
     - Strict **"One Probing Question"** rule per condition enforced via `follow_up_tracker`.
     - Graceful exit and deferral of comprehensive health reports.
   - **Triage Agent**: Red-flag symptom detection and urgency categorization with emergency escalation.
   - **Scheduling Agent**: Mock appointment booking with healthcare providers via `book_mock_appointment` tool.

3. **Frontend & Local Endpoints**:
   - `GET /api/profile?username=<username>`: Load patient profile, conversation threads, chat messages, and structured intake forms from DB.
   - `POST /api/profile`: Create a patient profile in DB.
   - `POST /api/conversations`: Create a new conversation thread.
   - `POST /api/conversations/select`: Select active conversation thread.
   - `POST /api/chat`: Save user message to DB $\rightarrow$ invoke LangGraph orchestrator $\rightarrow$ save assistant response to DB $\rightarrow$ return updated patient record.

4. **Safety Guardrails (`guardrails.py`)**: input guard → agents → output guard on every turn. See `docs/GUARDRAILS.md`.

## Changelog

### 2026-10-16: Safety guardrails, prompt hardening, architecture docs

| # | Change | Files | Why | Verified by |
|---|---|---|---|---|
| 1 | Removed the hardcoded Groq API key; `GROQ_API_KEY` is env-only and startup fails with a clear error if it's missing | `orchestrator.py`, `l.py` | Secret was in plaintext source. **The old key must be rotated in the Groq console** | `grep gsk_` finds nothing; importing without the key raises `RuntimeError` |
| 2 | New guardrail module: input rules (sanitize, self-harm crisis, emergency, prompt injection, diagnosis/prescription deflection, domain restriction) and output rules (diagnosis assertions, dose/medication advice, markdown, emergency line, disclaimer, length) | `guardrails.py` (new) | Project requirement: safety guardrails + domain restriction | `tests/test_guardrails.py` |
| 3 | Graph rewired: `START → input_guard → {guard_response \| triage \| supervisor}`, every agent → `output_guard → END`. Supervisor's `URGENT_TERMS` replaced by `guard.detect_emergency`/`detect_self_harm` (kept for defence in depth, and for the existing supervisor test) | `orchestrator.py` | No reply can skip the safety checks | Graph tests with mocked LLM |
| 4 | Prompt hardening: patient text moved out of system prompts into `<patient_message>`-delimited user turns; `SAFETY_RULES` / `DATA_ONLY_RULE` added to all prompts; `safe_llm_invoke` replaces the trailing history turn instead of duplicating it | `orchestrator.py` | Prompt-injection surface | Injection unit + eval cases |
| 5 | Intake no longer tells the LLM to name OTC drugs; it now suggests comfort measures plus a pharmacist/doctor | `orchestrator.py` | Unverified medication advice | Output-guard tests |
| 6 | Intake skips `upsert_condition` when nothing clinical was extracted; extraction forbids speculative `disease_name` | `orchestrator.py` | Off-topic text / questions were being saved as conditions | `test_off_domain_creates_no_condition` |
| 7 | `guardrail_events` audit table (SQLite + Postgres), `log_guardrail_event`, `get_guardrail_events` | `database.py` | Audit trail / observability | `test_guardrail_events_logged` |
| 8 | New state fields `guard_action`, `guard_rule`, `guardrail_flags`; `run_orchestrator` returns them | `models.py`, `orchestrator.py` | Carry guard decisions through the graph | Graph tests |
| 9 | `/api/chat` returns `guardrail_flags`; new `GET /api/guardrails?username=` | `server_groq.py` | Expose safety decisions to UI/reports | Server import check |
| 10 | 60-case red-team eval set + eval script (offline and full-pipeline modes) → `reports/guardrail_eval.md` | `tests/guardrail_eval.jsonl`, `scripts/eval_guardrails.py` (new) | Safety-assurance deliverable | Offline: 60/60 after fixing 2 gaps the first run found |
| 11 | Docs: architecture deep-dive and guardrail reference; README updated; `LLM_INTEGRATION.md` marked legacy | `docs/ARCHITECTURE.md`, `docs/GUARDRAILS.md` (new), `README.md`, `LLM_INTEGRATION.md` | Record what was built and how it works | This entry |

**Test status:** `GROQ_API_KEY=dummy python3 -m unittest discover tests` gives 36/36 OK. The 30 guardrail tests are fully offline. The
4 older LLM-dependent tests passed only through the LLM fallback path, so they still need a run with a real key.

**Not done yet:** full-pipeline eval with the live LLM; UI badge for `guardrail_flags` (optional in the plan); RAG; persistent cross-session memory.
