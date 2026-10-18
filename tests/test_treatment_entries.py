"""Unit tests for treatment entries and condition schema extraction in database & models."""
import unittest
import database as db
from models import MedicalCondition, Symptoms, Treatment, Reports


class TreatmentEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()

    def setUp(self):
        self.username = "test_treatment_patient"
        if not db.get_profile(self.username):
            db.create_profile(self.username, "Test Patient")
        convs = db.get_conversations(self.username)
        self.conv_id = convs[0]["id"]

    def test_separate_procedure_does_not_overwrite_pending_medicine(self):
        db.upsert_condition(self.username, "Allergy", {
            "treatments": {"intervention_name": "MONTEC LC", "status": "active"}
        })
        db.upsert_condition(self.username, "Appendicitis", {
            "treatments": {"intervention_name": "Appendectomy", "status": "completed"}
        })

        forms = db.get_patient_aggregate_forms(self.username)
        treatments = forms["treatment"]
        names = [t.get("intervention_name") for t in treatments]
        self.assertIn("MONTEC LC", names)
        self.assertIn("Appendectomy", names)

    def test_condition_pydantic_validation(self):
        cond = MedicalCondition(
            disease_name="Migraine",
            symptoms=Symptoms(chief_complaint="Throbbing headache", onset="3 days", severity_score=8),
            treatments=Treatment(intervention_name="Sumatriptan", status="active"),
            reports=Reports(report_type="Brain MRI")
        )
        self.assertEqual(cond.disease_name, "Migraine")
        self.assertEqual(cond.symptoms.severity_score, 8)
        self.assertEqual(cond.treatments.intervention_name, "Sumatriptan")
        self.assertEqual(cond.reports.report_type, "Brain MRI")


if __name__ == "__main__":
    unittest.main()
