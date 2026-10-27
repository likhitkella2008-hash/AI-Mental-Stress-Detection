"""
test_app_pipeline.py
====================
Automated test suite verifying the integrity of the ML pipeline,
trained model inference, confidence calculation, evaluation artifacts,
and baseline result preservation.
"""

import os
import joblib
import pandas as pd
import numpy as np

def run_tests():
    print("=" * 60)
    print("RUNNING AI MENTAL STRESS DETECTION TEST SUITE")
    print("=" * 60)

    # Test 1: Baseline file existence
    print("Test 1: Checking required baseline files...")
    required_files = [
        "StressLevelDataset.csv",
        "best_stress_model.pkl",
        "feature_names.pkl",
        "model_comparison.csv",
        "eval_artifacts.pkl",
        "app.py",
        "requirements.txt"
    ]
    for rf in required_files:
        assert os.path.exists(rf), f"Missing required file: {rf}"
        print(f"  [PASS] Found {rf}")

    # Test 2: Dataset integrity
    print("\nTest 2: Verifying dataset statistics...")
    df = pd.read_csv("StressLevelDataset.csv")
    assert df.shape == (1100, 21), f"Unexpected shape: {df.shape}"
    assert df.isna().sum().sum() == 0, "Missing values detected!"
    counts = df["stress_level"].value_counts().to_dict()
    assert counts[0] == 373, f"Low count mismatch: {counts[0]}"
    assert counts[1] == 358, f"Moderate count mismatch: {counts[1]}"
    assert counts[2] == 369, f"High count mismatch: {counts[2]}"
    print("  [PASS] Shape (1100, 21), 0 missing values, distribution 373/358/369 verified.")

    # Test 3: Model and feature order
    print("\nTest 3: Checking model and feature names...")
    model = joblib.load("best_stress_model.pkl")
    features = joblib.load("feature_names.pkl")
    assert len(features) == 20, f"Expected 20 features, got {len(features)}"
    print(f"  [PASS] Model loaded: {type(model).__name__} with 20 features.")

    # Test 4: Baseline model_comparison.csv integrity
    print("\nTest 4: Verifying baseline model_comparison.csv values...")
    comp = pd.read_csv("model_comparison.csv", index_col=0)
    rf_acc = comp.loc["Random Forest", "Accuracy"]
    assert round(rf_acc, 4) == 0.8864, f"Baseline Random Forest accuracy altered: {rf_acc}"
    print(f"  [PASS] Random Forest baseline accuracy confirmed: {rf_acc*100:.2f}%")

    # Test 5: End-to-end inference across all 3 classes
    print("\nTest 5: Testing multi-class inference (Low, Moderate, High)...")
    low_profile = {
        'anxiety_level': 5, 'self_esteem': 25, 'mental_health_history': 0, 'depression': 6,
        'headache': 1, 'blood_pressure': 2, 'sleep_quality': 4, 'breathing_problem': 2,
        'academic_performance': 4, 'study_load': 2, 'teacher_student_relationship': 4, 'future_career_concerns': 1,
        'social_support': 3, 'peer_pressure': 2, 'extracurricular_activities': 2, 'bullying': 1,
        'noise_level': 2, 'living_conditions': 3, 'safety': 4, 'basic_needs': 4
    }
    mod_profile = {
        'anxiety_level': 11, 'self_esteem': 19, 'mental_health_history': 1, 'depression': 12,
        'headache': 2, 'blood_pressure': 1, 'sleep_quality': 3, 'breathing_problem': 3,
        'academic_performance': 3, 'study_load': 3, 'teacher_student_relationship': 2, 'future_career_concerns': 3,
        'social_support': 2, 'peer_pressure': 3, 'extracurricular_activities': 3, 'bullying': 3,
        'noise_level': 2, 'living_conditions': 3, 'safety': 2, 'basic_needs': 3
    }
    high_profile = {
        'anxiety_level': 16, 'self_esteem': 9, 'mental_health_history': 1, 'depression': 20,
        'headache': 4, 'blood_pressure': 3, 'sleep_quality': 1, 'breathing_problem': 4,
        'academic_performance': 2, 'study_load': 4, 'teacher_student_relationship': 2, 'future_career_concerns': 4,
        'social_support': 1, 'peer_pressure': 4, 'extracurricular_activities': 4, 'bullying': 4,
        'noise_level': 4, 'living_conditions': 2, 'safety': 2, 'basic_needs': 2
    }

    test_cases = [
        ("Low Stress Profile", low_profile, 0),
        ("Moderate Stress Profile", mod_profile, 1),
        ("High Stress Profile", high_profile, 2)
    ]

    for label, profile, expected_class in test_cases:
        df_in = pd.DataFrame([profile])[features]
        pred = model.predict(df_in)[0]
        proba = model.predict_proba(df_in)[0]
        conf = float(np.max(proba) * 100)
        assert pred == expected_class, f"Expected {expected_class} for {label}, got {pred}"
        assert 0.0 <= conf <= 100.0, f"Confidence out of range: {conf}"
        assert np.isclose(np.sum(proba), 1.0), "Probabilities do not sum to 1.0"
        print(f"  [PASS] {label:25s} -> Predicted Class {pred} | Confidence: {conf:.2f}% | Probabilities: {proba.round(2)}")

    # Test 6: Evaluation artifacts check
    print("\nTest 6: Checking evaluation artifacts (Confusion Matrices, ROC-AUC, Feature Importance)...")
    eval_art = joblib.load("eval_artifacts.pkl")
    assert "confusion_matrices" in eval_art
    assert "roc_curves" in eval_art
    assert "roc_auc_scores" in eval_art
    assert "feature_importances" in eval_art
    assert "cv_results" in eval_art

    # Check Random Forest confusion matrix
    rf_cm = np.array(eval_art["confusion_matrices"]["Random Forest"])
    assert rf_cm.shape == (3, 3), f"Expected (3,3) confusion matrix, got {rf_cm.shape}"
    assert rf_cm.sum() == 220, f"Expected 220 test samples, got {rf_cm.sum()}"
    print(f"  [PASS] Random Forest test confusion matrix verified (220 samples, diagonal sum={np.trace(rf_cm)})")

    # Check ROC-AUC
    rf_auc = eval_art["roc_auc_scores"]["Random Forest"]
    assert 0.90 < rf_auc < 1.0, f"Unexpected RF ROC-AUC: {rf_auc}"
    print(f"  [PASS] Random Forest One-vs-Rest ROC-AUC verified: {rf_auc:.4f}")

    # Check Feature Importance
    fi = eval_art["feature_importances"]
    assert len(fi) == 20, f"Expected 20 feature importances, got {len(fi)}"
    total_fi = sum(item["Importance"] for item in fi)
    assert np.isclose(total_fi, 1.0), f"Feature importances do not sum to 1.0: {total_fi}"
    print(f"  [PASS] 20 Feature importances verified (Top feature: {fi[0]['Feature']} with {fi[0]['Importance']*100:.2f}%)")

    # Check 5-Fold Cross-Validation
    cv_res = eval_art["cv_results"]
    assert len(cv_res) == 5, f"Expected 5 models in CV results, got {len(cv_res)}"
    for m, stats in cv_res.items():
        print(f"  [PASS] 5-Fold CV for {m:20s}: Mean={stats['mean_accuracy']*100:.2f}% (±{stats['std_accuracy']*100:.2f}%)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! APPLICATION PIPELINE IS COMPLETELY VERIFIED.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
