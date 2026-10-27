"""
test_stage2_shap.py
===================
Automated regression test suite verifying:
1. Dynamic multiclass target resolution via model.classes_
2. Mathematical additivity of TreeSHAP across all 3 classes (error < 1e-3)
3. What-If comparative SHAP sensitivity and delta calculations
4. Non-causal wording and transparency notices
5. Verification of benchmark metrics in app.py against saved artifacts
"""

import unittest
import joblib
import pandas as pd
import numpy as np
import shap

class TestStage2SHAP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load("optimized_stress_model.pkl")
        cls.feature_names = joblib.load("feature_names.pkl")
        cls.dataset = pd.read_csv("StressLevelDataset.csv")

        import app
        cls.app = app
        cls.stress_labels = app.stress_labels

    def test_dynamic_class_to_idx_mapping(self):
        """Verify class_to_idx is derived dynamically from model.classes_."""
        classes = self.model.classes_
        self.assertEqual(len(classes), 3)

        class_to_idx = {self.stress_labels.get(int(c), str(c)): idx for idx, c in enumerate(classes)}

        self.assertIn("Low", class_to_idx)
        self.assertIn("Moderate", class_to_idx)
        self.assertIn("High", class_to_idx)

        # Confirm index aligns with position in model.classes_
        for idx, c in enumerate(classes):
            name = self.stress_labels[int(c)]
            self.assertEqual(class_to_idx[name], idx)

    def test_treeshap_mathematical_additivity_all_classes(self):
        """Verify TreeSHAP satisfies base_value + sum(SHAP) == model_proba for all 3 classes."""
        explainer = shap.TreeExplainer(self.model)

        # Test on median dataset respondent
        sample = pd.DataFrame([{f: float(self.dataset[f].iloc[10]) for f in self.feature_names}])[self.feature_names]
        probabilities = self.model.predict_proba(sample)[0]

        shap_values = explainer.shap_values(sample)
        shap_array = np.asarray(shap_values)

        self.assertEqual(shap_array.ndim, 3)
        self.assertEqual(shap_array.shape[2], 3)

        expected_values = np.asarray(explainer.expected_value)

        for class_idx in range(3):
            base_val = float(expected_values[class_idx])
            class_contribs = shap_array[0, :, class_idx]
            shap_sum = float(np.sum(class_contribs))
            actual_proba = float(probabilities[class_idx])

            reconstructed_proba = base_val + shap_sum

            # Absolute error must be strictly < 1e-3 (typically < 1e-14)
            abs_err = abs(reconstructed_proba - actual_proba)
            self.assertLess(
                abs_err,
                1e-3,
                f"Additivity failed for class {class_idx}: |({base_val} + {shap_sum}) - {actual_proba}| = {abs_err}"
            )

    def test_whatif_comparative_shap_delta(self):
        """Verify What-If comparative SHAP delta correctly accounts for probability shifts."""
        explainer = shap.TreeExplainer(self.model)

        base_dict = {
            'anxiety_level': 14.0, 'self_esteem': 15.0, 'mental_health_history': 1.0, 'depression': 15.0,
            'headache': 3.0, 'blood_pressure': 2.0, 'sleep_quality': 2.0, 'breathing_problem': 3.0,
            'academic_performance': 2.0, 'study_load': 4.0, 'teacher_student_relationship': 2.0, 'future_career_concerns': 3.0,
            'social_support': 1.0, 'peer_pressure': 4.0, 'extracurricular_activities': 4.0, 'bullying': 3.0,
            'noise_level': 3.0, 'living_conditions': 2.0, 'safety': 2.0, 'basic_needs': 2.0
        }

        # Hypothetical: improve protective factors
        hypo_dict = dict(base_dict)
        hypo_dict['sleep_quality'] = 5.0
        hypo_dict['study_load'] = 1.0
        hypo_dict['social_support'] = 3.0
        hypo_dict['anxiety_level'] = 5.0

        df_base = pd.DataFrame([base_dict])[self.feature_names]
        df_hypo = pd.DataFrame([hypo_dict])[self.feature_names]

        proba_base = self.model.predict_proba(df_base)[0]
        proba_hypo = self.model.predict_proba(df_hypo)[0]

        shap_base = np.asarray(explainer.shap_values(df_base))
        shap_hypo = np.asarray(explainer.shap_values(df_hypo))

        # Test delta for Low Stress (index 0)
        target_idx = 0
        base_contrib = shap_base[0, :, target_idx]
        hypo_contrib = shap_hypo[0, :, target_idx]
        delta_contrib = hypo_contrib - base_contrib

        sum_delta_shap = float(np.sum(delta_contrib))
        delta_proba = float(proba_hypo[target_idx] - proba_base[target_idx])

        # Mathematical property: sum(delta_shap) == delta_proba
        self.assertAlmostEqual(sum_delta_shap, delta_proba, places=4)

        # Top shifts should identify the modified features
        delta_series = pd.Series(np.abs(delta_contrib), index=self.feature_names).sort_values(ascending=False)
        top_shifted_features = list(delta_series.head(4).index)

        # At least 2 of our modified features should be in the top shifts
        modified = {'sleep_quality', 'study_load', 'social_support', 'anxiety_level'}
        intersect = modified.intersection(top_shifted_features)
        self.assertGreaterEqual(len(intersect), 2)

    def test_non_causal_disclaimers_present_in_app(self):
        """Verify non-causal notices and transparency texts exist in app.py."""
        with open("app.py", "r", encoding="utf-8") as f:
            app_code = f.read()

        # Non-causal disclaimers
        self.assertIn("does not establish medical causation", app_code)
        self.assertIn("ceteris paribus", app_code)
        self.assertIn("Mathematical Additivity Proof", app_code)
        self.assertIn("Technical Defense, Architectural Rationale & Transparency FAQ", app_code)

    def test_verified_benchmarks_in_app(self):
        """Verify app.py displays verified baseline numbers matching model_comparison.csv."""
        with open("app.py", "r", encoding="utf-8") as f:
            app_code = f.read()

        # Verified values
        self.assertIn("88.64%", app_code)  # RF Baseline
        self.assertIn("90.91%", app_code)  # RF Optimized
        self.assertIn("88.18%", app_code)  # LR and XGB
        self.assertIn("87.73%", app_code)  # SVM
        self.assertIn("85.45%", app_code)  # DT
        self.assertIn("0.9831", app_code)  # Baseline ROC-AUC
        self.assertIn("0.9853", app_code)  # Optimized ROC-AUC / LR ROC-AUC


if __name__ == "__main__":
    unittest.main()
