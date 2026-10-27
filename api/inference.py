"""
api/inference.py
================
Inference module for AI-Based Mental Stress Detection.
Provides cached model loading, input validation for all 20 features,
and multiclass probability prediction.
"""

import os
import joblib
import numpy as np
import pandas as pd

# Resolve base directories safely for both local execution and Vercel serverless
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)

_MODEL_PATH = os.path.join(_PROJECT_ROOT, "optimized_stress_model.pkl")
if not os.path.exists(_MODEL_PATH):
    _MODEL_PATH = os.path.join(_CURRENT_DIR, "optimized_stress_model.pkl")

_FEATURES_PATH = os.path.join(_PROJECT_ROOT, "feature_names.pkl")
if not os.path.exists(_FEATURES_PATH):
    _FEATURES_PATH = os.path.join(_CURRENT_DIR, "feature_names.pkl")

# Cached singletons
_MODEL = None
_FEATURE_NAMES = None

# Verified class mapping
STRESS_LABELS = {0: "Low", 1: "Moderate", 2: "High"}

# Domain mappings and scale limits
FEATURE_DOMAINS = {
    'anxiety_level': 'Mental & Emotional',
    'self_esteem': 'Mental & Emotional',
    'mental_health_history': 'Mental & Emotional',
    'depression': 'Mental & Emotional',
    'headache': 'Physical Health',
    'blood_pressure': 'Physical Health',
    'sleep_quality': 'Physical Health',
    'breathing_problem': 'Physical Health',
    'noise_level': 'Environmental',
    'living_conditions': 'Environmental',
    'safety': 'Environmental',
    'basic_needs': 'Environmental',
    'academic_performance': 'Academic Factors',
    'study_load': 'Academic Factors',
    'teacher_student_relationship': 'Academic Factors',
    'future_career_concerns': 'Academic Factors',
    'social_support': 'Social Factors',
    'peer_pressure': 'Social Factors',
    'extracurricular_activities': 'Social Factors',
    'bullying': 'Social Factors'
}

FEATURE_RANGES = {
    'anxiety_level': (0, 21),
    'self_esteem': (0, 30),
    'mental_health_history': (0, 1),
    'depression': (0, 27),
    'headache': (0, 5),
    'blood_pressure': (1, 3),
    'sleep_quality': (0, 5),
    'breathing_problem': (0, 5),
    'noise_level': (0, 5),
    'living_conditions': (0, 5),
    'safety': (0, 5),
    'basic_needs': (0, 5),
    'academic_performance': (0, 5),
    'study_load': (0, 5),
    'teacher_student_relationship': (0, 5),
    'future_career_concerns': (0, 5),
    'social_support': (0, 3),
    'peer_pressure': (0, 5),
    'extracurricular_activities': (0, 5),
    'bullying': (0, 5)
}


def get_model():
    """Returns the cached Random Forest model."""
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {_MODEL_PATH}")
        _MODEL = joblib.load(_MODEL_PATH)
    return _MODEL


def get_feature_names():
    """Returns the cached list of 20 feature names in exact order."""
    global _FEATURE_NAMES
    if _FEATURE_NAMES is None:
        if not os.path.exists(_FEATURES_PATH):
            raise FileNotFoundError(f"Feature names file not found at {_FEATURES_PATH}")
        _FEATURE_NAMES = joblib.load(_FEATURES_PATH)
    return _FEATURE_NAMES


def validate_and_vectorize(input_dict: dict) -> np.ndarray:
    """
    Validates that input_dict contains all 20 required features,
    clips them to valid ranges, and returns a 2D numpy array of shape (1, 20).
    """
    features = get_feature_names()
    values = []
    
    for f in features:
        if f not in input_dict:
            raise ValueError(f"Missing required input feature: '{f}'")
        try:
            val = float(input_dict[f])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid numeric value for feature '{f}': {input_dict[f]}")
        
        # Enforce range constraints
        min_v, max_v = FEATURE_RANGES[f]
        val = max(min_v, min(max_v, val))
        values.append(val)
        
    return pd.DataFrame([values], columns=features)


def predict(input_dict: dict) -> dict:
    """
    Executes inference using the trained Random Forest model.
    Returns predicted class, label, confidence, and three-class probabilities.
    """
    model = get_model()
    x = validate_and_vectorize(input_dict)
    
    pred_class = int(model.predict(x)[0])
    probabilities = model.predict_proba(x)[0] # shape (3,)
    
    # Class-to-index alignment derived dynamically from model.classes_
    class_to_idx = {STRESS_LABELS.get(int(c), str(c)): idx for idx, c in enumerate(model.classes_)}
    
    proba_dict = {
        "Low": float(probabilities[class_to_idx.get("Low", 0)]),
        "Moderate": float(probabilities[class_to_idx.get("Moderate", 1)]),
        "High": float(probabilities[class_to_idx.get("High", 2)])
    }
    
    conf = float(np.max(probabilities))
    label = STRESS_LABELS.get(pred_class, str(pred_class))
    
    return {
        "predicted_class": pred_class,
        "predicted_label": label,
        "confidence": conf,
        "confidence_pct": round(conf * 100, 2),
        "probabilities": proba_dict,
        "model_classes": [int(c) for c in model.classes_]
    }
