"""Unit tests for CareFlow Autonomous Long-Term Memory (LTM) System."""
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import database as db
import memory_agent


class LongTermMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()

    def setUp(self):
        import uuid
        self.username = f"test_user_ltm_{uuid.uuid4().hex[:8]}"
        prof = db.get_profile(self.username)
        if not prof:
            db.create_profile(self.username, "Sarah Connor", "1985-05-12", "168 cm", "62 kg")
        convs = db.get_conversations(self.username)
        self.conv_id = convs[0]["id"]

    def test_save_and_retrieve_memories(self):
        """Test saving atomic clinical memories and querying by status."""
        mem1 = db.save_patient_memory(
            username=self.username,
            memory_type="allergy",
            content="Severe anaphylactic reaction to Penicillin",
            confidence_score=0.98,
            status="active",
            source_session_id=self.conv_id
        )
        self.assertIsNotNone(mem1["id"])
        self.assertEqual(mem1["memory_type"], "allergy")

        mem2 = db.save_patient_memory(
            username=self.username,
            memory_type="symptom_trigger",
            content="Spicy Mexican food causes acute acid reflux",
            confidence_score=0.95,
            status="active",
            source_session_id=self.conv_id
        )

        active = db.get_patient_memories(self.username, status="active")
        contents = [m["content"] for m in active]
        self.assertIn("Severe anaphylactic reaction to Penicillin", contents)
        self.assertIn("Spicy Mexican food causes acute acid reflux", contents)

        # Invalidate mem2
        db.update_patient_memory_status(mem2["id"], "contradicted")
        active_after = db.get_patient_memories(self.username, status="active")
        active_ids = [m["id"] for m in active_after]
        self.assertIn(mem1["id"], active_ids)
        self.assertNotIn(mem2["id"], active_ids)

    def test_upsert_and_version_patient_baseline(self):
        """Test creating and updating the consolidated clinical baseline profile."""
        b1 = db.upsert_patient_baseline(
            username=self.username,
            chronic_conditions=["Migraine"],
            active_medications=[{"name": "Sumatriptan", "dosage_frequency": "50mg as needed"}],
            allergies=["Penicillin"],
            general_summary="Patient has recurrent hormonal migraines, managed with triptans."
        )
        self.assertEqual(b1["version"], 1)

        retrieved1 = db.get_patient_baseline(self.username)
        self.assertIsNotNone(retrieved1)
        self.assertEqual(retrieved1["chronic_conditions"], ["Migraine"])
        self.assertEqual(retrieved1["allergies"], ["Penicillin"])

        # Second update should increment version
        b2 = db.upsert_patient_baseline(
            username=self.username,
            chronic_conditions=["Migraine", "Mild Gastritis"],
            active_medications=[{"name": "Sumatriptan", "dosage_frequency": "50mg as needed"}, {"name": "Omeprazole", "dosage_frequency": "20mg daily"}],
            allergies=["Penicillin"],
            general_summary="Patient manages migraines and newly diagnosed mild gastritis."
        )
        self.assertEqual(b2["version"], 2)

        retrieved2 = db.get_patient_baseline(self.username)
        self.assertEqual(len(retrieved2["chronic_conditions"]), 2)
        self.assertEqual(retrieved2["version"], 2)

    def test_full_patient_record_includes_ltm(self):
        """Verify that get_full_patient_record includes baseline and active memories."""
        full_rec = db.get_full_patient_record(self.username)
        self.assertIn("baseline", full_rec)
        self.assertIn("memories", full_rec)
        self.assertTrue(isinstance(full_rec["memories"], list))

    def test_markdown_summary_export(self):
        """Test generating an agent-maintained Markdown memory summary report."""
        md_file = memory_agent.export_patient_summary_md(self.username)
        self.assertTrue(Path(md_file).exists())
        content = Path(md_file).read_text(encoding="utf-8")
        self.assertIn("# CareFlow Long-Term Clinical Memory Profile", content)
        self.assertIn(self.username, content)

    @patch("memory_agent.get_llm")
    def test_consolidation_agent_pipeline(self, mock_get_llm):
        """Test autonomous consolidation logic when LLM returns structured updates."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        mock_llm.invoke.return_value = MagicMock(
            content='''{
                "new_memories": [
                    {
                        "memory_type": "lifestyle_habit",
                        "content": "Works night shifts as an ICU nurse, reporting chronic sleep deficit",
                        "confidence_score": 0.95
                    }
                ],
                "contradicted_memory_ids": [],
                "chronic_conditions": ["Migraine", "Mild Gastritis"],
                "active_medications": [{"name": "Omeprazole", "dosage_frequency": "20mg daily"}],
                "allergies": ["Penicillin"],
                "general_summary": "Night-shift nurse managing migraine and gastritis with irregular sleep patterns."
            }'''
        )

        # Save a sample message
        db.save_chat_message(self.conv_id, "user", "I work night shifts at the hospital and haven't slept properly.")
        db.save_chat_message(self.conv_id, "assistant", "That sounds very demanding. Sleep deficit can trigger migraines.")

        result = memory_agent.consolidate_patient_memory(self.username, self.conv_id)
        self.assertEqual(result["status"], "success")
        self.assertGreaterEqual(result["new_memories_count"], 1)

        # Check that the new memory is in DB
        memories = db.get_patient_memories(self.username, status="active")
        nurse_mem = any("night shifts as an ICU nurse" in m["content"] for m in memories)
        self.assertTrue(nurse_mem)


if __name__ == "__main__":
    unittest.main()
