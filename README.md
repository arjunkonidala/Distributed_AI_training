# CareFlow: Multi-Agent AI Care Coordination Assistant

CareFlow is a local multi-agent digital healthcare coordination assistant powered by **LangGraph**, **Groq LLM**, and **PostgreSQL / SQLite Database Persistence**.

## Features

1. **Database Persistence (`careflow.db` / PostgreSQL)**:
   - **Complete Chat History**: All messages saved immediately to the `chat_messages` table under each `session_id`.
   - **Disease-Centric Intake**: Conditions, symptoms, treatments, and reports persisted in `medical_conditions`.
   - **Profiles & Conversations**: Multi-thread conversation management per patient record.

2. **LangGraph Multi-Agent Orchestration**:
   - **Supervisor Router**: Evaluates intent and delegates turns to specialized agents (`Intake`, `Triage`, `Scheduling`).
   - **Intake Agent**: Empathetic conversational data collection with the **"One Probing Question"** rule per condition (enforced via `follow_up_tracker`).
   - **Triage Agent**: Red-flag symptom checks, urgency categorization, and emergency guidance.
   - **Scheduling Agent**: Mock provider appointment booking recorded in the database.

3. **Safety Guardrails** (`guardrails.py`, details in [docs/GUARDRAILS.md](docs/GUARDRAILS.md)):
   - **Input guard** before any agent: self-harm crisis response, emergency fast-path to triage, prompt-injection blocking, diagnosis/prescription deflection, off-domain refusal.
   - **Output guard** after every agent: removes diagnostic claims and unrequested medication/dose advice, guarantees the emergency line, strips markdown.
   - Every decision is logged to the `guardrail_events` table (`GET /api/guardrails?username=`).

Architecture deep-dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Change history: [progress.md](progress.md).

## Quickstart

1. Set your `GROQ_API_KEY` (required, read from the environment only; the server will not start without it):
   ```bash
   export GROQ_API_KEY="your_groq_api_key"
   ```

2. (Optional) Configure PostgreSQL:
   ```bash
   export DATABASE_URL="postgresql://user:password@localhost:5432/careflow"
   ```
   *(If not set, CareFlow automatically uses local SQLite `careflow.db`)*

3. Start the server:
   ```bash
   python3 server_groq.py
   ```

4. Open `http://localhost:8000` in your browser.

## Running Tests

```bash
python3 -m unittest discover tests            # guardrail tests run offline; agent tests need GROQ_API_KEY
python3 scripts/eval_guardrails.py --offline  # red-team eval of the input guard -> reports/guardrail_eval.md
python3 scripts/eval_guardrails.py            # full-pipeline eval with the live LLM
```
