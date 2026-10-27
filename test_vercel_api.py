"""
test_vercel_api.py
===================
Automated regression test suite for the Vercel Python Serverless API.
Validates:
- Health check
- Prediction & class-to-index integrity
- Mathematical additivity of TreeSHAP
- What-If counterfactual sensitivity & attribution shifts
- Model performance benchmarks alignment with eval_artifacts.pkl
- Dataset summary statistics
- Global SHAP feature importances
- Recommendations engine
"""

import unittest
from api.index import (
    health_check,
    predict_endpoint,
    whatif_endpoint,
    performance_endpoint,
    dataset_endpoint,
    global_shap_endpoint,
    recommendations_endpoint
)
from api.inference import get_feature_names

class TestVercelAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.feature_names = get_feature_names()
        
        # Standard test sample
        cls.test_sample = {
            'anxiety_level': 11.0, 'self_esteem': 19.0, 'mental_health_history': 1.0, 'depression': 12.0,
            'headache': 2.0, 'blood_pressure': 1.0, 'sleep_quality': 3.0, 'breathing_problem': 3.0,
            'academic_performance': 3.0, 'study_load': 3.0, 'teacher_student_relationship': 2.0, 'future_career_concerns': 3.0,
            'social_support': 2.0, 'peer_pressure': 3.0, 'extracurricular_activities': 3.0, 'bullying': 3.0,
            'noise_level': 2.0, 'living_conditions': 3.0, 'safety': 2.0, 'basic_needs': 3.0
        }

    def test_health_check(self):
        """Verify health_check() returns healthy status."""
        data = health_check()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_ready"])
        self.assertEqual(data["feature_count"], 20)

    def test_predict_endpoint(self):
        """Verify predict_endpoint returns valid class, probabilities, SHAP, and recommendations."""
        data = predict_endpoint(self.test_sample)
        self.assertTrue(data["success"])
        
        pred = data["prediction"]
        self.assertIn(pred["predicted_class"], [0, 1, 2])
        self.assertIn(pred["predicted_label"], ["Low", "Moderate", "High"])
        
        # Probabilities sum to 1.0
        probs = pred["probabilities"]
        prob_sum = probs["Low"] + probs["Moderate"] + probs["High"]
        self.assertAlmostEqual(prob_sum, 1.0, places=4)
        
        # SHAP additivity
        exp = data["explanation"]
        self.assertTrue(exp["is_additive"])
        self.assertLess(exp["error"], 1e-3)
        self.assertEqual(len(exp["top_features"]), 10)
        
        # Recommendations
        rec = data["recommendations"]
        self.assertIn("focus_areas", rec)
        self.assertIn("positive_buffers", rec)

    def test_whatif_endpoint(self):
        """Verify whatif_endpoint computes probability shifts and SHAP deltas."""
        hypo = dict(self.test_sample)
        hypo['sleep_quality'] = 5.0
        hypo['study_load'] = 1.0
        hypo['social_support'] = 3.0

        data = whatif_endpoint({
            "baseline": self.test_sample,
            "hypothetical": hypo
        })
        self.assertTrue(data["success"])
        self.assertIn(data["decision_state"], ["transition", "stability"])
        self.assertEqual(data["changed_features_count"], 3)
        
        # Check SHAP delta shifts
        shifts = data["shap_shifts"]
        self.assertIn("top_shifts", shifts)
        self.assertGreater(len(shifts["top_shifts"]), 0)

    def test_performance_endpoint(self):
        """Verify performance_endpoint returns verified benchmarks."""
        res = performance_endpoint()
        self.assertTrue(res["success"])
        
        perf = res["data"]
        # Check that Random Forest baseline accuracy is 88.64%
        rf_row = next(r for r in perf["baseline_comparison"] if r["model"] == "Random Forest")
        self.assertEqual(rf_row["accuracy"], 88.64)
        self.assertIn("0.9831", str(perf["roc_auc_scores"].get("Random Forest")))
        self.assertIn("Random Forest", perf["confusion_matrices"])

    def test_dataset_endpoint(self):
        """Verify dataset_endpoint returns dataset distributions and 1100 records."""
        res = dataset_endpoint()
        self.assertTrue(res["success"])
        d = res["data"]
        self.assertEqual(d["total_records"], 1100)
        self.assertEqual(d["total_features"], 20)
        self.assertEqual(d["missing_values"], 0)
        self.assertEqual(d["class_distribution"], {"Low": 373, "Moderate": 358, "High": 369})

    def test_global_shap_endpoint(self):
        """Verify global_shap_endpoint returns ranked importance items."""
        res = global_shap_endpoint()
        self.assertTrue(res["success"])
        shap_data = res["data"]
        self.assertGreater(len(shap_data["global"]), 0)
        self.assertGreater(len(shap_data["low_stress"]), 0)
        self.assertGreater(len(shap_data["high_stress"]), 0)

    def test_input_validation_error(self):
        """Verify invalid or missing inputs raise HTTPException(400)."""
        from fastapi import HTTPException
        invalid_sample = dict(self.test_sample)
        del invalid_sample["anxiety_level"]
        
        with self.assertRaises(HTTPException) as ctx:
            predict_endpoint(invalid_sample)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Missing required input feature", str(ctx.exception.detail))


if __name__ == "__main__":
    unittest.main()
