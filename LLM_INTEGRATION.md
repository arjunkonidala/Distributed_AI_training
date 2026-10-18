> **Legacy document.** This describes the original single-call design that stored data in `persistent.txt`.
> The current system uses LangGraph agents, a SQLite/Postgres database, and safety guardrails. See
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/GUARDRAILS.md](docs/GUARDRAILS.md).

# LLM Intake Flow

`server_groq.py` is the only backend. It serves the web UI and handles profile lookup, profile creation, and chat requests. It reads `GROQ_API_KEY` from the server environment and sends each chat turn to Groq's Chat Completions API using JSON mode.

The model receives the patient profile, shared forms, recent messages from the selected chat, and pending fields for that chat. It returns patches for any combination of symptom and report forms, plus zero or more independent treatment entries, an empathetic acknowledgment, and optional non-diagnostic insights. Pydantic form schemas live in `server_groq.py`, so no separate schema module is needed. The server validates each patch; if validation fails, it sends the error and Pydantic JSON schema back to Groq and retries up to two times. Only accepted values are persisted to `persistent.txt`.

## Guided Intake

The server tracks incomplete forms per chat. It asks for the first missing required field among forms that the model found in the user's message:

- Symptoms: `chief_complaint`, `onset`, `severity_score`
- Reports: `report_type`, `provider_facility`
- Treatment: a list of independent interventions, each with `intervention_name`, `treatment_type`, and `status`

The pending fields, selected pending treatment, and current chat history are sent back to the model so short answers are interpreted in context. The model can update multiple forms or treatments in one turn; it never asks the user to pick a category. Treatment details are never copied between distinct interventions. Once required fields for the discovered topics are present, the server invites the user to add another item. Optional fields are captured when provided.

Each profile owns multiple conversations with separate transcripts, titles, and pending intake state. Its structured forms are shared across those conversations. Existing records with a single message list are moved into a `Previous chat` thread the first time they are loaded.

Urgent terms are checked by the server on every message. The UI uses `app.js` for both provided HTML pages and talks only to this backend.

## Local Endpoints

- `GET /api/profile?username=<username>`: find a saved profile.
- `POST /api/profile`: create a profile and empty forms.
- `POST /api/conversations`: create a separate chat thread.
- `POST /api/conversations/select`: switch the active chat thread.
- `GET /api/conversations?username=<username>&id=<id>`: load a chat thread.
- `POST /api/chat`: extract, validate, persist, and reply to one turn.

The local JSON store is suitable only for a prototype. It does not provide authentication, encryption, concurrency control, clinical review, or emergency escalation.
