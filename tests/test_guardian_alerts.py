"""Tests for guardian email alerts raised by triage events (offline: SMTP and LLM are mocked)."""
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import database as db
import guardian_notifier
import orchestrator

SMTP_ENV = {"SMTP_HOST": "smtp.example.com", "SMTP_FROM": "alerts@careflow.test", "GUARDIAN_ALERT_COOLDOWN_MINUTES": "30"}


class GuardianAlertTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_patch = mock.patch.object(db, "SQLITE_PATH", Path(self.tmp.name) / "test.db")
        self.db_patch.start()
        db.init_db()
        db.create_profile("pat", "Priya Rao", guardian_name="Ravi Rao", guardian_email="ravi@example.com")
        self.session_id = db.get_conversations("pat")[0]["id"]

    def tearDown(self):
        self.db_patch.stop()
        self.tmp.cleanup()

    def test_migration_adds_guardian_columns_to_legacy_table(self):
        legacy = Path(self.tmp.name) / "legacy.db"
        conn = sqlite3.connect(legacy)
        conn.execute("CREATE TABLE patient_profiles (username TEXT PRIMARY KEY, name TEXT NOT NULL, date_of_birth TEXT, height TEXT, weight TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
        conn.execute("INSERT INTO patient_profiles VALUES ('old', 'Old User', NULL, NULL, NULL, 'x', 'x')")
        conn.commit()
        conn.close()
        with mock.patch.object(db, "SQLITE_PATH", legacy):
            db.init_db()
            db.update_guardian_contact("old", "Mum", "mum@example.com")
            self.assertEqual(db.get_profile("old")["guardian_email"], "mum@example.com")

    def test_no_guardian_means_no_email(self):
        db.update_guardian_contact("pat", None, None)
        with mock.patch.dict(os.environ, SMTP_ENV), mock.patch.object(guardian_notifier, "send_email") as send:
            self.assertEqual(guardian_notifier.notify_guardian_of_triage("pat", self.session_id, "chest pain"), "no_guardian")
        send.assert_not_called()

    def test_unconfigured_smtp_is_skipped_and_audited(self):
        with mock.patch.dict(os.environ, {"SMTP_HOST": "", "SMTP_FROM": "", "SMTP_USERNAME": ""}), mock.patch.object(guardian_notifier, "send_email") as send:
            self.assertEqual(guardian_notifier.notify_guardian_of_triage("pat", self.session_id, "chest pain"), "not_configured")
        send.assert_not_called()
        self.assertEqual(db.get_guardian_alerts(username="pat")[0]["status"], "skipped")

    def test_sends_email_then_respects_cooldown(self):
        with mock.patch.dict(os.environ, SMTP_ENV), mock.patch.object(guardian_notifier, "send_email") as send:
            status = guardian_notifier.notify_guardian_of_triage("pat", self.session_id, "Crushing chest pain spreading to my arm", background=False)
            self.assertEqual(status, "sent")
            msg = send.call_args[0][0]
            self.assertEqual(msg["To"], "ravi@example.com")
            self.assertIn("Priya Rao", msg["Subject"])
            self.assertIn("Crushing chest pain", msg.get_content())
            self.assertIn("Dear Ravi Rao", msg.get_content())

            self.assertEqual(guardian_notifier.notify_guardian_of_triage("pat", self.session_id, "still hurts", background=False), "cooldown")
            self.assertEqual(send.call_count, 1)

    def test_smtp_failure_is_audited(self):
        with mock.patch.dict(os.environ, SMTP_ENV), mock.patch.object(guardian_notifier, "send_email", side_effect=OSError("connection refused")):
            self.assertEqual(guardian_notifier.notify_guardian_of_triage("pat", self.session_id, "chest pain", background=False), "failed")
        alert = db.get_guardian_alerts(username="pat")[0]
        self.assertEqual(alert["status"], "failed")
        self.assertIn("connection refused", alert["detail"])

    def _triage_state(self, text, guard_action):
        return {"user_input": text, "messages": [], "username": "pat", "session_id": self.session_id, "guard_action": guard_action}

    def test_emergency_triage_event_triggers_guardian_alert(self):
        with mock.patch.object(orchestrator, "safe_llm_invoke", return_value="Please call 112 now."), \
             mock.patch.object(guardian_notifier, "notify_guardian_of_triage", return_value="queued") as notify:
            result = orchestrator.triage_agent_node(self._triage_state("I have crushing chest pain and can't breathe", "emergency"))
        notify.assert_called_once_with("pat", self.session_id, "I have crushing chest pain and can't breathe")
        self.assertEqual(result["guardian_alert"], "queued")

    def test_non_emergency_triage_does_not_alert(self):
        with mock.patch.object(orchestrator, "safe_llm_invoke", return_value="This sounds routine."), \
             mock.patch.object(guardian_notifier, "notify_guardian_of_triage") as notify:
            result = orchestrator.triage_agent_node(self._triage_state("How urgent is a mild sore throat?", "allow"))
        notify.assert_not_called()
        self.assertEqual(result["guardian_alert"], "")


if __name__ == "__main__":
    unittest.main()
