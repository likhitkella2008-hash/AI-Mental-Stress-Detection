"""
api/index.py
============
FastAPI serverless entrypoint for Vercel deployment.
Routes all /api/* requests to model inference, what-if counterfactual sensitivity,
Explainable AI (TreeSHAP), performance benchmarks, and educational recommendations.
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from .inference import predict, get_feature_names, STRESS_LABELS, FEATURE_DOMAINS
from .treeshap_fast import explain_prediction, compare_whatif_shap
from .recommendations import generate_personalized_guidance
from .artifacts import get_model_performance, get_dataset_summary, get_global_shap

app = FastAPI(
    title="AI-Based Mental Stress Detection API",
    description="Vercel Serverless API for Stress Prediction, TreeSHAP Explainability, and What-If Analysis",
    version="2.0.0"
)

# Enable CORS for Next.js frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    """Health check endpoint confirming API and model readiness."""
    try:
        features = get_feature_names()
        return {
            "status": "healthy",
            "model_ready": True,
            "feature_count": len(features),
            "engine": "Scikit-Learn RandomForestClassifier",
            "version": "2.0.0"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "model_ready": False,
            "error": str(e)
        }


@app.post("/api/predict")
def predict_endpoint(payload: Dict[str, Any] = Body(...)):
    """
    Stress Prediction & Local TreeSHAP Explanation Endpoint.
    Expects dictionary containing all 20 survey features.
    Optional key: 'target_category' ("Low", "Moderate", or "High") to decompose.
    """
    try:
        target_cat = payload.get("target_category", None)
        
        # Filter inputs to only feature fields
        feature_names = get_feature_names()
        inputs = {f: payload[f] for f in feature_names if f in payload}
        
        # 1. Classification & Probabilities
        prediction_res = predict(inputs)
        pred_class = prediction_res["predicted_class"]
        pred_label = prediction_res["predicted_label"]
        
        # 2. Local TreeSHAP decomposition
        shap_res = explain_prediction(inputs, target_category=target_cat)
        
        # 3. Personalized Educational Recommendations
        guidance = generate_personalized_guidance(inputs, pred_class, pred_label)
        
        return {
            "success": True,
            "prediction": prediction_res,
            "explanation": shap_res,
            "recommendations": guidance
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/whatif")
def whatif_endpoint(payload: Dict[str, Any] = Body(...)):
    """
    What-If Counterfactual Sensitivity Analysis Endpoint.
    Compares baseline and hypothetical profiles.
    Returns probability shifts, changed factors, and comparative SHAP delta attributions.
    """
    try:
        baseline = payload.get("baseline")
        hypothetical = payload.get("hypothetical")
        target_cat = payload.get("target_category", None)
        
        if not baseline or not hypothetical:
            raise ValueError("Payload must contain both 'baseline' and 'hypothetical' profiles.")
            
        feature_names = get_feature_names()
        base_inputs = {f: baseline[f] for f in feature_names if f in baseline}
        hypo_inputs = {f: hypothetical[f] for f in feature_names if f in hypothetical}
        
        # Run inference on both
        p_base = predict(base_inputs)
        p_hypo = predict(hypo_inputs)
        
        # Detect changed features
        changed_features = []
        for f in feature_names:
            bv = float(base_inputs[f])
            hv = float(hypo_inputs[f])
            if bv != hv:
                diff = hv - bv
                direction = f"Increased (+{diff:.0f})" if diff > 0 else f"Decreased ({diff:.0f})"
                changed_features.append({
                    "feature": f,
                    "name": f.replace("_", " ").title(),
                    "domain": FEATURE_DOMAINS.get(f, "General"),
                    "baseline_value": bv,
                    "hypothetical_value": hv,
                    "shift": diff,
                    "direction": direction
                })
                
        # Comparative probability shifts
        proba_shifts = {}
        for c in ["Low", "Moderate", "High"]:
            bp = p_base["probabilities"][c]
            hp = p_hypo["probabilities"][c]
            proba_shifts[c] = {
                "baseline_prob": bp,
                "hypo_prob": hp,
                "delta": hp - bp,
                "delta_pct": round((hp - bp) * 100, 2)
            }
            
        # Decision state assessment
        if p_hypo["predicted_class"] != p_base["predicted_class"]:
            decision_state = "transition"
            summary_text = (
                f"Model Decision Transition: The hypothetical modifications shifted the dominant category "
                f"from {p_base['predicted_label']} Stress to {p_hypo['predicted_label']} Stress."
            )
        else:
            decision_state = "stability"
            summary_text = (
                f"Model Decision Stability: The predicted category remained {p_base['predicted_label']} Stress. "
                f"However, the internal confidence margin shifted by {p_hypo['confidence_pct'] - p_base['confidence_pct']:+.1f}%."
            )
            
        # Comparative SHAP shifts
        shap_shifts = compare_whatif_shap(base_inputs, hypo_inputs, target_category=target_cat)
        
        return {
            "success": True,
            "decision_state": decision_state,
            "summary": summary_text,
            "baseline_prediction": p_base,
            "hypothetical_prediction": p_hypo,
            "probability_shifts": proba_shifts,
            "changed_features_count": len(changed_features),
            "changed_features": changed_features,
            "shap_shifts": shap_shifts
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"What-If computation error: {str(e)}")


@app.get("/api/performance")
def performance_endpoint():
    """Returns verified model benchmarks, confusion matrices, and ROC curves."""
    try:
        return {
            "success": True,
            "data": get_model_performance()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load evaluation data: {str(e)}")


@app.get("/api/dataset")
def dataset_endpoint():
    """Returns dataset summary statistics and feature distributions."""
    try:
        return {
            "success": True,
            "data": get_dataset_summary()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset summary: {str(e)}")


@app.get("/api/shap/global")
def global_shap_endpoint():
    """Returns global and class-specific feature importance rankings."""
    try:
        return {
            "success": True,
            "data": get_global_shap()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load SHAP importance: {str(e)}")


@app.post("/api/recommendations")
def recommendations_endpoint(payload: Dict[str, Any] = Body(...)):
    """Generates personalized educational recommendations for given inputs."""
    try:
        inputs = payload.get("inputs", {})
        pred = payload.get("prediction", 1)
        label = payload.get("label", "Moderate")
        return {
            "success": True,
            "data": generate_personalized_guidance(inputs, pred, label)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
