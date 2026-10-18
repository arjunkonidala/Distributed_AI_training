# CareFlow — Presentation Content

**Subtitle:** A safe, AI-powered care coordination assistant  
**Purpose:** Present the implemented prototype and its target architecture against the CareFlow Project brief.  
**Scope note:** This presentation excludes EHR integrations and triage workflows. Appointment scheduling refers to the current mock booking workflow and a future calendar integration.

---

## Slide 1 — CareFlow: AI Care Coordination Assistant

**On-slide text**

- A conversational assistant for structured patient intake and care coordination
- Persistent patient context across conversations
- Grounded responses from approved care guidance
- Safe, observable, and extensible multi-agent architecture

**Presenter note:** CareFlow aims to reduce repetitive administrative work while helping patients share information in a clear, continuous way. It supports coordination and information gathering; it does not diagnose or prescribe.

## Slide 2 — The Problem

**On-slide text**

- Patients repeat the same information across visits and conversations
- Free-form intake can leave important details incomplete
- Staff spend time collecting, organizing, and following up on information
- General-purpose AI can produce unsupported or unsafe health guidance
- Scheduling and communication tasks often require disconnected tools

**Presenter note:** The project brief frames the opportunity as making care coordination more continuous and less administratively burdensome.

## Slide 3 — Product Scope and Goals

**On-slide text**

**CareFlow provides:**

- Conversational, structured intake
- Persistent profiles, conversation history, and captured care details
- Answers grounded in an approved knowledge collection (target capability)
- Appointment request and scheduling support
- Safety guardrails, audit records, and evaluation

**CareFlow does not provide:**

- Diagnosis or prescribing
- Clinical decision-making in place of a qualified professional

**Presenter note:** The 30–40% administrative-burden reduction is an intended project outcome, not a measured result. It should be validated in a pilot against a defined baseline.

## Slide 4 — Functional Requirements

**On-slide text**

1. Collect patient details naturally and consistently
2. Store profiles, conversations, and structured intake data
3. Recall relevant context in later conversations
4. Retrieve approved guidance and cite its source
5. Coordinate specialized AI steps with a controlled workflow
6. Support appointment requests through tools
7. Block unsafe, irrelevant, or unsupported responses
8. Measure reliability, response time, and task completion

**Presenter note:** The current prototype demonstrates intake, database-backed history, mock appointment booking, and guardrail evaluation. Source-grounded RAG, durable cross-session recall in the model workflow, MCP integrations, and operational performance dashboards are target work.

## Slide 5 — System Context and Architecture

**On-slide text / diagram**

```text
Patient
  │
  ▼
CareFlow Web UI
  │  JSON over HTTP
  ▼
CareFlow API Server
  │
  ▼
LangGraph Workflow ─────── Safety Guardrails
  │                             │
  ├── Intake Agent              ├── Input checks
  ├── Scheduling Agent          ├── Output checks
  └── Knowledge/RAG Agent*      └── Auditable decisions
  │
  ├── Groq LLM
  ├── PostgreSQL / SQLite
  ├── Approved Knowledge Store* 
  └── MCP Tool Gateway* ── Calendar / communication tools
```

`* Target architecture component; not yet implemented in the current prototype.`

**Presenter note:** The existing browser UI calls a local Python HTTP server. The server loads session history and invokes the LangGraph workflow. The workflow uses Groq and database functions. RAG and MCP are shown as planned extension points, not existing integrations.

## Slide 6 — Conversational Intake and Workflow Orchestration

**On-slide text**

- A supervisor routes each turn to the appropriate task
- Intake extracts details into structured fields
- Follow-up tracking helps avoid asking the same question repeatedly
- Scheduling handles explicit appointment requests
- Safety checks run before and after the agent workflow
- The workflow ends with a clear response and updated conversation state

**Presenter note:** The current LangGraph application includes intake, scheduling, direct-response, and report-deferral paths. This presentation omits triage. The project’s safety controls remain in scope, including deflection of diagnosis and prescription requests.

## Slide 7 — Data and Memory Design

**On-slide text**

- **Patient profile:** basic profile fields
- **Conversation:** separate threads and follow-up state
- **Chat messages:** user and assistant transcript by conversation
- **Structured care details:** symptoms, treatments, reports, and timeline
- **Appointments:** mock booking records
- **Guardrail events:** safety decisions for review

**Data flow:**

```text
User message → save message → run workflow → validate response
             → save assistant reply → return updated profile to UI
```

**Presenter note:** SQLite is the local default; PostgreSQL can be configured. The database stores history and structured information. However, the current model context is limited to recent messages in the active conversation; a secure, retrieval-based cross-session memory experience remains a target enhancement.

## Slide 8 — Knowledge Retrieval (RAG) Target

**On-slide text**

1. Curate approved care guidance, policies, and patient-facing documents
2. Parse and split documents into searchable passages
3. Attach source, version, and access metadata
4. Retrieve relevant passages for the current question
5. Generate a response constrained to the retrieved evidence
6. Return source references and abstain when evidence is insufficient

**Presenter note:** Production RAG is required by the project brief but is not present in the current repository. The target design should include document governance, access control, freshness/version tracking, retrieval-quality evaluation, and citation checks.

## Slide 9 — Tools and MCP Integration Target

**On-slide text**

- MCP provides a standard boundary between the assistant and external tools
- Proposed tools: check calendar availability, create or update a booking, send a confirmation
- Validate user intent and required fields before tool execution
- Enforce least-privilege access and validate tool inputs/outputs
- Record the request, tool result, and outcome for audit
- Current prototype: mock appointment booking stored in the application database

**Presenter note:** No live calendar or communication MCP connector is currently implemented. The target architecture excludes EHR connectivity.

## Slide 10 — Safety, Privacy, and Security

**On-slide text**

- Input guard checks for prompt injection, off-topic requests, and disallowed medical requests
- Output guard filters unsupported diagnosis and medication guidance
- Patient text is treated as data, not system instructions
- Secrets are read from environment configuration
- Guardrail decisions are recorded for audit and evaluation
- Production readiness requires authentication, authorization, encryption, retention controls, and operational security review

**Presenter note:** The prototype has deterministic input and output checks and a red-team evaluation set. The offline evaluation currently reports 60/60 known cases passing with no benign false positives in that set; this measures coverage of authored cases, not general safety. Live-LLM evaluation has not yet been completed. Production privacy and security controls are also still required.

## Slide 11 — Observability and Evaluation

**On-slide text**

**Track:**

- Availability and request errors
- End-to-end and model response time
- Intake completion and appointment-task completion
- Retrieval relevance and source-grounding rate (after RAG delivery)
- Guardrail triggers, false positives, and unsafe-output findings

**Current evidence:**

- Guardrail events are stored and can be retrieved through an API
- Offline guardrail evaluation produces a report
- A consolidated operational dashboard and full-pipeline live evaluation remain target work

**Presenter note:** Metrics should be reviewed by workflow and failure category. Avoid reporting a single success rate without defining the test population and acceptance criteria.

## Slide 12 — Deployment View and Reliability

**On-slide text**

- Browser client served by the CareFlow application
- Python HTTP API for profile, conversation, chat, and audit operations
- LangGraph workflow runs synchronously in the API process today
- Groq is the configured language-model provider
- SQLite supports local prototyping; PostgreSQL is an available deployment option
- Production target: managed application runtime, managed database, secret manager, monitoring, backups, and recovery procedures

**Presenter note:** The production deployment items are recommendations for the target state, not claims about the current local prototype. External model and database availability affect response reliability.

## Slide 13 — Verification and Acceptance Criteria

**On-slide text**

- Demonstrate a complete conversational intake and structured record update
- Demonstrate a returning user’s relevant prior context
- Demonstrate an answer grounded in an approved document with source attribution
- Demonstrate a mock appointment booking and recorded result
- Demonstrate safe deflection of diagnosis and prescription requests
- Demonstrate refusal of irrelevant-domain requests
- Report response time, task completion, reliability, and evaluation results

**Presenter note:** Intake, stored history, mock booking, and offline guardrail evaluation are currently demonstrable. Grounded RAG answers, richer cross-session recall, MCP tool execution, and operational performance metrics need implementation and verification.

## Slide 14 — Delivery Roadmap and Success Measures

**On-slide text**

**Phase 1 — Stabilize prototype**

- Complete live-LLM safety evaluation
- Add authentication and access controls
- Define retention, encryption, and backup practices

**Phase 2 — Add grounded knowledge**

- Build the approved-document ingestion and retrieval pipeline
- Add citations, abstention behavior, and retrieval evaluation

**Phase 3 — Extend memory and tools**

- Add controlled cross-session context retrieval
- Connect calendar and communication tools through MCP

**Phase 4 — Pilot and measure**

- Run usability and reliability testing
- Measure administrative time against a documented baseline
- Validate the target 30–40% reduction before making it a result claim

## Slide 15 — Intended Impact and Closing

**On-slide text**

- A more continuous patient experience
- Better organized information for care coordination
- Less repetitive administrative intake
- Safer, traceable AI assistance
- A measurable path from local prototype to production-ready service

**Closing line:** CareFlow combines structured conversation, persistent data, workflow orchestration, and safety controls to make care coordination clearer and more efficient.

---

## Appendix — Architecture Status for Presenter

| Capability | Current project status |
|---|---|
| Web chat and profile/conversation management | Implemented in prototype |
| LangGraph orchestration | Implemented in prototype |
| Structured conversational intake | Implemented in prototype |
| SQLite persistence / PostgreSQL option | Implemented in prototype |
| Mock appointment booking | Implemented in prototype |
| Input/output safety guardrails | Implemented in prototype |
| Offline guardrail evaluation | Implemented; authored set reports 60/60 pass |
| Secure, production-grade persistent memory | Partial; data is stored, but model recall is limited |
| Production RAG over approved documents | Not implemented |
| MCP calendar/communication tools | Not implemented |
| Operational observability dashboard and performance report | Partial; audit events/evaluation report exist, full dashboard/metrics remain |
| Production authentication and privacy/security controls | Not implemented; required before real patient deployment |

**Source alignment:** Content follows `CareFlow Project.pdf`, `README.md`, `docs/ARCHITECTURE.md`, `docs/GUARDRAILS.md`, and `progress.md` in this repository. The PDF’s EHR and triage-related content has been excluded from the presentation scope.
