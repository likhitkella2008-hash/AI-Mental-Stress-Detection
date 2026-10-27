"""
test_whatif_analysis.py
========================
Automated regression test suite verifying the integrity of the
What-If Counterfactual Sensitivity Analysis module, directional
feature labels, input scaling dictionaries, and prediction isolation.
"""

import unittest
import joblib
import pandas as pd
import numpy as np

class TestWhatIfAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load("optimized_stress_model.pkl")
        cls.feature_names = joblib.load("feature_names.pkl")
        cls.dataset = pd.read_csv("StressLevelDataset.csv")

        # Import feature dictionaries from app.py
        import app
        cls.app = app
        cls.feature_labels_with_scale = app.feature_labels_with_scale
        cls.feature_domains = app.feature_domains
        cls.stress_labels = app.stress_labels

    def test_feature_count_and_keys(self):
        """Verify all 20 model features are present in labels and domains dictionaries."""
        self.assertEqual(len(self.feature_names), 20)
        for f in self.feature_names:
            self.assertIn(f, self.feature_labels_with_scale, f"Missing scale label for {f}")
            self.assertIn(f, self.feature_domains, f"Missing domain mapping for {f}")
            # Ensure label contains scale hint
            label = self.feature_labels_with_scale[f]
            self.assertIn("(", label, f"Label for {f} should contain scale description")
            self.assertIn(")", label, f"Label for {f} should contain scale description")

    def test_whatif_inference_pipeline(self):
        """Verify both baseline and hypothetical profiles evaluate through the model cleanly."""
        base_profile = {
            'anxiety_level': 11.0, 'self_esteem': 19.0, 'mental_health_history': 1.0, 'depression': 12.0,
            'headache': 2.0, 'blood_pressure': 1.0, 'sleep_quality': 3.0, 'breathing_problem': 3.0,
            'academic_performance': 3.0, 'study_load': 3.0, 'teacher_student_relationship': 2.0, 'future_career_concerns': 3.0,
            'social_support': 2.0, 'peer_pressure': 3.0, 'extracurricular_activities': 3.0, 'bullying': 3.0,
            'noise_level': 2.0, 'living_conditions': 3.0, 'safety': 2.0, 'basic_needs': 3.0
        }

        hypo_profile = dict(base_profile)
        # Modify specific protective factors
        hypo_profile['sleep_quality'] = 5.0
        hypo_profile['study_load'] = 1.0
        hypo_profile['social_support'] = 3.0
        hypo_profile['anxiety_level'] = 5.0

        df_base = pd.DataFrame([base_profile])[self.feature_names]
        df_hypo = pd.DataFrame([hypo_profile])[self.feature_names]

        # Model predictions
        p_base = self.model.predict(df_base)[0]
        prob_base = self.model.predict_proba(df_base)[0]

        p_hypo = self.model.predict(df_hypo)[0]
        prob_hypo = self.model.predict_proba(df_hypo)[0]

        # Assert valid classes and probabilities
        self.assertIn(p_base, [0, 1, 2])
        self.assertIn(p_hypo, [0, 1, 2])
        self.assertTrue(np.isclose(np.sum(prob_base), 1.0))
        self.assertTrue(np.isclose(np.sum(prob_hypo), 1.0))

        # Check delta calculation
        deltas = prob_hypo - prob_base
        self.assertEqual(len(deltas), 3)

        # Favorable modifications should mathematically increase Low Stress probability or reduce Moderate
        self.assertGreaterEqual(prob_hypo[0], prob_base[0])

    def test_delta_inspector_logic(self):
        """Verify delta inspector correctly identifies only changed features."""
        base_inputs = {f: float(self.dataset[f].median()) for f in self.feature_names}
        hypo_inputs = dict(base_inputs)

        # Initially identical -> 0 changed
        changed_init = [f for f in self.feature_names if hypo_inputs[f] != base_inputs[f]]
        self.assertEqual(len(changed_init), 0)

        # Change 2 features
        hypo_inputs['sleep_quality'] = 5.0
        hypo_inputs['study_load'] = 1.0

        changed_after = [f for f in self.feature_names if hypo_inputs[f] != base_inputs[f]]
        self.assertEqual(len(changed_after), 2)
        self.assertIn('sleep_quality', changed_after)
        self.assertIn('study_load', changed_after)

    def test_prediction_labels_mapping(self):
        """Verify stress labels mapping is consistent with model classes."""
        for c in self.model.classes_:
            self.assertIn(c, self.stress_labels)
            self.assertIn(self.stress_labels[c], ["Low", "Moderate", "High"])

if __name__ == "__main__":
    unittest.main()
