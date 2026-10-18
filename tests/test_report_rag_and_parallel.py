"""Unit tests for Report Branch RAG, Strict 24h Onset Constraint, and Parallel Execution."""
import unittest
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import database as db
import onset_normalizer as on
from api_pool import key_pool
from rag_report_engine import retrieve_patient_history_context, generate_rag_report_response
from orchestrator import supervisor_node, route_supervisor, run_orchestrator


class ReportRAGAndParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()

    def setUp(self):
        self.username = f"test_user_rag_{uuid.uuid4().hex[:8]}"
        db.create_profile(self.username, "Dr. John Watson", "1975-07-07", "180 cm", "78 kg")
        convs = db.get_conversations(self.username)
        self.conv_id = convs[0]["id"]

    def test_strict_24h_onset_normalizer(self):
        """Test that onset normalizer hard-constrains any input to YYYY-MM-DD HH:MM."""
        self.assertTrue(on.is_strict_24h("2026-10-18 14:30"))
        self.assertTrue(on.is_strict_24h("2026-05-01 08:00"))
        self.assertFalse(on.is_strict_24h("2 hours ago"))
        self.assertFalse(on.is_strict_24h("yesterday"))

        # Conversions
        norm1 = on.normalize_onset_24h("2 hours ago")
        self.assertTrue(on.is_strict_24h(norm1))

        norm2 = on.normalize_onset_24h("this morning")
        self.assertTrue(on.is_strict_24h(norm2))
        self.assertTrue(norm2.endswith("08:00"))

        norm3 = on.normalize_onset_24h("yesterday 3pm")
        self.assertTrue(on.is_strict_24h(norm3))
        self.assertTrue(norm3.endswith("15:00"))

    def test_database_hard_constrains_onset_to_24h(self):
        """Verify upsert_condition normalizes onset before writing to DB."""
        cond = db.upsert_condition(self.username, "Gastritis", {
            "symptoms": {
                "chief_complaint": "Burning stomach",
                "onset": "yesterday evening"
            }
        })
        # Check DB column
        self.assertTrue(on.is_strict_24h(cond["onset"]))
        self.assertTrue(cond["onset"].endswith("20:00"))
        # Check raw_data
        self.assertTrue(on.is_strict_24h(cond["data"]["symptoms"]["onset"]))

    def test_supervisor_routes_report_query(self):
        """Test that questions about medical history or reports route to the report branch."""
        state = {
            "user_input": "Can you summarize my past medical history and test results?",
            "messages": [],
            "guard_action": "allow",
        }
        routing = supervisor_node(state)
        self.assertEqual(routing["next_agent"], "report")

    def test_parallel_route_supervisor_fan_out(self):
        """Test that route_supervisor returns parallel branches when user reports both emergency and symptoms."""
        # 1. Triage + Symptoms -> parallel
        state1 = {
            "next_agent": "triage",
            "user_input": "I have crushing chest pain and burning stomach ache since 2 hours ago."
        }
        branches1 = route_supervisor(state1)
        self.assertIn("triage", branches1)
        self.assertIn("intake", branches1)

        # 2. Report query + Symptoms -> parallel
        state2 = {
            "next_agent": "report",
            "user_input": "What were my last scan results? Also my headache is getting worse."
        }
        branches2 = route_supervisor(state2)
        self.assertIn("report", branches2)
        self.assertIn("intake", branches2)

    def test_api_pool_key_management(self):
        """Test that APIKeyPool handles rotation and agent distribution."""
        env_patch = {
            "GROQ_API_KEYS": "key_alpha,key_beta,key_gamma",
            "GROQ_API_KEY": "",
            "TRIAGE_API": "",
            "REPORT_API": "",
            "EVENT_API": "",
            "INTAKE_API": "",
        }
        with patch.dict("os.environ", env_patch):
            key_pool.reload_keys()
            keys = key_pool.get_keys()
            self.assertEqual(len(keys), 3)
            self.assertIn("key_alpha", keys)
            self.assertIn("key_beta", keys)

            # Round robin
            k1 = key_pool.get_next_key()
            k2 = key_pool.get_next_key()
            self.assertNotEqual(k1, k2)

            # Agent hashing
            agent_k = key_pool.get_key_for_agent("triage")
            self.assertIn(agent_k, keys)

    def test_report_rag_retrieval(self):
        """Test that retrieve_patient_history_context gathers and ranks baseline, conditions, and memories."""
        # Setup test data
        db.upsert_patient_baseline(
            self.username,
            chronic_conditions=["Hypertension"],
            active_medications=[{"name": "Lisinopril", "dosage_frequency": "10mg"}],
            allergies=["Penicillin"],
            general_summary="Stable hypertension patient."
        )
        db.save_patient_memory(
            self.username,
            memory_type="symptom_trigger",
            content="Heavy coffee consumption triggers palpitations",
            confidence_score=0.9
        )
        db.upsert_condition(self.username, "Hypertension", {
            "symptoms": {"chief_complaint": "High blood pressure", "onset": "2026-09-01 10:00"}
        })

        retrieved = retrieve_patient_history_context(self.username, "What medications and allergies do I have?")
        self.assertIn("baseline", retrieved)
        self.assertEqual(retrieved["baseline"]["allergies"], ["Penicillin"])
        self.assertTrue(len(retrieved["conditions"]) > 0)
        self.assertTrue(len(retrieved["memories"]) > 0)


if __name__ == "__main__":
    unittest.main()
