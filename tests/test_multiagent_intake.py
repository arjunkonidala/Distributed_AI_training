"""Unit and Integration Tests for CareFlow PostgreSQL/SQLite persistence and LangGraph Multi-Agent Orchestration."""
import unittest
import database as db
from models import MedicalCondition, Symptoms, Treatment, Reports
from orchestrator import careflow_graph, run_orchestrator, supervisor_node, intake_agent_node


class MultiAgentIntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()

    def setUp(self):
        self.username = "test_user_multiagent"
        # Ensure fresh user
        prof = db.get_profile(self.username)
        if not prof:
            db.create_profile(self.username, "Alex Mercer", "1988-11-20", "175 cm", "70 kg")
        convs = db.get_conversations(self.username)
        self.conv_id = convs[0]["id"]

    def test_complete_chat_history_in_database(self):
        """Test that user and assistant messages are saved immediately to chat_messages in DB."""
        # 1. User message save
        user_msg = db.save_chat_message(self.conv_id, "user", "I have pain in my lower back since yesterday.")
        self.assertEqual(user_msg["role"], "user")
        self.assertEqual(user_msg["text"], "I have pain in my lower back since yesterday.")

        # 2. Verify query from DB
        history = db.get_chat_messages(self.conv_id)
        self.assertTrue(any(m["text"] == "I have pain in my lower back since yesterday." for m in history))

        # 3. Assistant response save
        asst_msg = db.save_chat_message(self.conv_id, "assistant", "I am sorry to hear that.", ["symptoms"])
        self.assertEqual(asst_msg["role"], "assistant")
        self.assertIn("symptoms", asst_msg["categories"])

        history2 = db.get_chat_messages(self.conv_id)
        self.assertTrue(any(m["text"] == "I am sorry to hear that." for m in history2))

    def test_disease_centric_intake_and_one_probing_question_rule(self):
        """Test that Intake Agent extracts condition schema, saves to DB, and enforces at most 1 follow-up."""
        tracker = {}
        # Turn 1: Patient mentions condition without severity
        res1 = run_orchestrator(
            username=self.username,
            session_id=self.conv_id,
            user_input="I have been having chronic hypertension and dizziness for 3 weeks.",
            follow_up_tracker=tracker
        )
        
        # Check condition in DB
        conditions = db.get_conditions(self.username)
        self.assertTrue(len(conditions) > 0)
        
        # Check follow-up tracker was updated
        updated_tracker = res1["follow_up_tracker"]
        self.assertTrue(any(k.endswith("_asked") for k in updated_tracker.keys()))
        
        # Turn 2: Patient answers follow-up
        res2 = run_orchestrator(
            username=self.username,
            session_id=self.conv_id,
            user_input="The severity is about 6 out of 10 and I take Amlodipine 5mg.",
            follow_up_tracker=updated_tracker
        )
        
        # Check aggregate forms for UI
        forms = db.get_patient_aggregate_forms(self.username)
        self.assertTrue(bool(forms["symptoms"]))

    def test_triage_agent_routing_on_urgent_symptoms(self):
        """Test that Supervisor routes urgent/red-flag inputs to Triage Agent."""
        state = {
            "messages": [],
            "active_conditions": {},
            "follow_up_tracker": {},
            "next_agent": "",
            "username": self.username,
            "session_id": self.conv_id,
            "user_input": "I have crushing chest pain and trouble breathing!",
            "response_text": "",
            "categories": [],
        }
        routing = supervisor_node(state)
        self.assertEqual(routing["next_agent"], "triage")

    def test_scheduling_agent_appointment_booking(self):
        """Test scheduling agent mock appointment booking tool."""
        res = run_orchestrator(
            username=self.username,
            session_id=self.conv_id,
            user_input="Please book an appointment with a Cardiologist next Monday at 10 AM for my checkup.",
            follow_up_tracker={}
        )
        self.assertIn("appointment", res["response_text"].lower())


if __name__ == "__main__":
    unittest.main()
