"""CareFlow Web Server.
Uses PostgreSQL / SQLite database persistence and LangGraph Multi-Agent Orchestration.
"""
from __future__ import annotations

import json
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

import database as db
import guardian_notifier
from orchestrator import run_orchestrator

ROOT = Path(__file__).parent
PORT = int(os.environ.get("CAREFLOW_PORT", "8000"))


class CareFlowHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/profile":
            self.handle_get_profile(parsed.query)
            return
        if parsed.path == "/api/conversations":
            self.handle_get_conversation(parsed.query)
            return
        if parsed.path == "/api/guardrails":
            self.handle_get_guardrail_events(parsed.query)
            return
        if parsed.path == "/api/memory":
            self.handle_get_memory(parsed.query)
            return
        if parsed.path == "/api/guardian/alerts":
            self.handle_get_guardian_alerts(parsed.query)
            return
        # Default static file handler
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/profile":
            self.handle_create_profile()
            return
        if parsed.path == "/api/profile/guardian":
            self.handle_update_guardian()
            return
        if parsed.path == "/api/conversations":
            self.handle_create_conversation()
            return
        if parsed.path == "/api/conversations/select":
            self.handle_select_conversation()
            return
        if parsed.path == "/api/chat":
            self.handle_chat()
            return
        if parsed.path == "/api/memory/consolidate":
            self.handle_consolidate_memory()
            return
        
        self.send_json({"error": "Endpoint not found"}, status=HTTPStatus.NOT_FOUND)

    def read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------ PROFILE ENDPOINTS ------------------

    def handle_get_profile(self, query_string: str):
        params = parse_qs(query_string)
        username = (params.get("username") or [""])[0].strip()
        if not username:
            self.send_json({"error": "Missing username parameter."}, status=HTTPStatus.BAD_REQUEST)
            return

        record = db.get_full_patient_record(username)
        if not record:
            self.send_json({"found": False, "username": username})
            return

        self.send_json({"found": True, "record": record})

    def handle_create_profile(self):
        body = self.read_json_body()
        username = (body.get("username") or "").strip()
        name = (body.get("name") or "").strip()
        if not username or not name:
            self.send_json({"error": "Username and name are required."}, status=HTTPStatus.BAD_REQUEST)
            return
        guardian_email = (body.get("guardian_email") or "").strip()
        if guardian_email and not guardian_notifier.is_valid_email(guardian_email):
            self.send_json({"error": "Guardian email is not a valid email address."}, status=HTTPStatus.BAD_REQUEST)
            return

        existing = db.get_profile(username)
        if existing:
            # Return existing record
            record = db.get_full_patient_record(username)
            self.send_json({"record": record})
            return

        db.create_profile(
            username=username,
            name=name,
            date_of_birth=body.get("date_of_birth"),
            height=body.get("height"),
            weight=body.get("weight"),
            guardian_name=(body.get("guardian_name") or "").strip(),
            guardian_email=guardian_email,
        )
        record = db.get_full_patient_record(username)
        self.send_json({"record": record})

    def handle_update_guardian(self):
        body = self.read_json_body()
        username = (body.get("username") or "").strip()
        guardian_name = (body.get("guardian_name") or "").strip()
        guardian_email = (body.get("guardian_email") or "").strip()
        if not username:
            self.send_json({"error": "Username is required."}, status=HTTPStatus.BAD_REQUEST)
            return
        if guardian_email and not guardian_notifier.is_valid_email(guardian_email):
            self.send_json({"error": "Guardian email is not a valid email address."}, status=HTTPStatus.BAD_REQUEST)
            return
        if not db.get_profile(username):
            self.send_json({"error": "Profile not found."}, status=HTTPStatus.NOT_FOUND)
            return
        db.update_guardian_contact(username, guardian_name, guardian_email)
        self.send_json({"record": db.get_full_patient_record(username, active_conv_id=(body.get("conversation_id") or None))})

    # ------------------ CONVERSATION ENDPOINTS ------------------

    def handle_create_conversation(self):
        body = self.read_json_body()
        username = (body.get("username") or "").strip()
        if not username:
            self.send_json({"error": "Username is required."}, status=HTTPStatus.BAD_REQUEST)
            return

        conv = db.create_conversation(username, "New chat")
        record = db.get_full_patient_record(username, active_conv_id=conv["id"])
        self.send_json({"record": record})

    def handle_select_conversation(self):
        body = self.read_json_body()
        username = (body.get("username") or "").strip()
        conv_id = (body.get("conversation_id") or "").strip()
        if not username or not conv_id:
            self.send_json({"error": "Username and conversation_id are required."}, status=HTTPStatus.BAD_REQUEST)
            return

        record = db.get_full_patient_record(username, active_conv_id=conv_id)
        if not record:
            self.send_json({"error": "Conversation not found."}, status=HTTPStatus.NOT_FOUND)
            return

        self.send_json({"record": record})

    def handle_get_conversation(self, query_string: str):
        params = parse_qs(query_string)
        username = (params.get("username") or [""])[0].strip()
        conv_id = (params.get("id") or [""])[0].strip()
        if not username or not conv_id:
            self.send_json({"error": "Missing parameters."}, status=HTTPStatus.BAD_REQUEST)
            return

        conv = db.get_conversation(username, conv_id)
        if not conv:
            self.send_json({"error": "Conversation not found."}, status=HTTPStatus.NOT_FOUND)
            return

        self.send_json({"conversation": conv})

    # ------------------ CHAT & ORCHESTRATION ------------------

    def handle_chat(self):
        body = self.read_json_body()
        username = (body.get("username") or "").strip()
        conv_id = (body.get("conversation_id") or "").strip()
        user_message = (body.get("message") or "").strip()

        if not username or not user_message:
            self.send_json({"error": "Username and message are required."}, status=HTTPStatus.BAD_REQUEST)
            return

        # Ensure conversation exists
        conv = db.get_conversation(username, conv_id) if conv_id else None
        if not conv:
            convs = db.get_conversations(username)
            conv = convs[0] if convs else db.create_conversation(username, "New chat")
            conv_id = conv["id"]

        # 1. Complete Chat History in DB: Save user message IMMEDIATELY to chat_messages table
        db.save_chat_message(
            session_id=conv_id,
            sender="user",
            message=user_message,
        )

        # 2. Update conversation title if it's the first message
        current_messages = db.get_chat_messages(conv_id)
        if len(current_messages) <= 2 and (conv.get("title") == "New chat" or not conv.get("title")):
            title = user_message[:30] + ("..." if len(user_message) > 30 else "")
            db.update_conversation_title(conv_id, title)

        # 3. Multi-Agent Orchestration via LangGraph
        tracker = conv.get("follow_up_tracker", {})
        try:
            agent_result = run_orchestrator(
                username=username,
                session_id=conv_id,
                user_input=user_message,
                follow_up_tracker=tracker,
            )
            response_text = agent_result.get("response_text", "I've noted your update.")
            categories = agent_result.get("categories", [])
            guardrail_flags = agent_result.get("guardrail_flags", [])
            guardian_alert = agent_result.get("guardian_alert", "")
        except Exception as e:
            print(f"[Chat Error in Orchestrator]: {e}", file=sys.stderr)
            response_text = "I've recorded that detail in your care record. How else can I help support you today?"
            categories = ["symptoms"]
            guardrail_flags = []
            guardian_alert = ""

        # 4. Save final LangGraph conversational response IMMEDIATELY to chat_messages table
        db.save_chat_message(
            session_id=conv_id,
            sender="assistant",
            message=response_text,
            categories=categories,
        )

        # 5. Autonomous Long-Term Memory (LTM) Consolidation
        # Triggered in a background daemon thread so it never delays user chat response
        try:
            import threading
            from memory_agent import consolidate_patient_memory
            threading.Thread(
                target=consolidate_patient_memory,
                args=(username, conv_id),
                daemon=True
            ).start()
        except Exception as e:
            print(f"[Memory Consolidation Async Notice]: {e}", file=sys.stderr)

        # 6. Fetch updated patient record from DB for the frontend
        record = db.get_full_patient_record(username, active_conv_id=conv_id)

        self.send_json({
            "record": record,
            "category": categories[0] if categories else "symptoms",
            "categories": categories,
            "guardrail_flags": guardrail_flags,
            "guardian_alert": guardian_alert,
        })

    # ------------------ GUARDRAIL AUDIT ------------------

    def handle_get_guardrail_events(self, query_string: str):
        params = parse_qs(query_string)
        username = (params.get("username") or [""])[0].strip()
        if not username:
            self.send_json({"error": "Missing username parameter."}, status=HTTPStatus.BAD_REQUEST)
            return
        self.send_json({"events": db.get_guardrail_events(username)})

    def handle_get_guardian_alerts(self, query_string: str):
        params = parse_qs(query_string)
        username = (params.get("username") or [""])[0].strip()
        if not username:
            self.send_json({"error": "Missing username parameter."}, status=HTTPStatus.BAD_REQUEST)
            return
        self.send_json({"alerts": db.get_guardian_alerts(username=username)})

    # ------------------ LONG-TERM MEMORY ENDPOINTS ------------------

    def handle_get_memory(self, query_string: str):
        params = parse_qs(query_string)
        username = (params.get("username") or [""])[0].strip()
        if not username:
            self.send_json({"error": "Missing username parameter."}, status=HTTPStatus.BAD_REQUEST)
            return
        baseline = db.get_patient_baseline(username)
        memories = db.get_patient_memories(username, status="active")
        self.send_json({
            "username": username,
            "baseline": baseline,
            "memories": memories
        })

    def handle_consolidate_memory(self):
        body = self.read_json_body()
        username = (body.get("username") or "").strip()
        conv_id = (body.get("conversation_id") or "").strip()
        if not username:
            self.send_json({"error": "Missing username parameter."}, status=HTTPStatus.BAD_REQUEST)
            return
        from memory_agent import consolidate_patient_memory
        result = consolidate_patient_memory(username, conv_id or None)
        self.send_json(result)


def run_server(port: int = PORT):
    db.init_db()
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, CareFlowHandler)
    print(f"[CareFlow] Server running at http://localhost:{port}")
    print(f"[CareFlow] Database initialized with complete chat history & structured intake.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[CareFlow] Server stopped.")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
