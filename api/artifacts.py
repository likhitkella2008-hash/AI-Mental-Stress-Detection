"""
api/artifacts.py
================
Serves verified evaluation artifacts, benchmark comparisons,
dataset distributions, and global SHAP importance tables.
"""

import os
import joblib
import pandas as pd
import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)

def _find_file(filename):
    p1 = os.path.join(_PROJECT_ROOT, filename)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(_CURRENT_DIR, filename)
    if os.path.exists(p2):
        return p2
    return filename


def get_model_performance():
    """Returns verified model comparison table, confusion matrices, ROC curves, and CV scores."""
    # 1. Comparison table
    comp_path = _find_file("model_comparison.csv")
    comparison_data = []
    if os.path.exists(comp_path):
        df_comp = pd.read_csv(comp_path, index_col=0)
        for model_name, row in df_comp.iterrows():
            comparison_data.append({
                "model": model_name,
                "accuracy": round(float(row["Accuracy"]) * 100, 2),
                "precision": round(float(row["Precision"]) * 100, 2),
                "recall": round(float(row["Recall"]) * 100, 2),
                "f1_score": round(float(row["F1-Score"]) * 100, 2)
            })

    # 2. Optimized comparison
    opt_comp_path = _find_file("optimized_model_comparison.csv")
    opt_data = {}
    if os.path.exists(opt_comp_path):
        df_opt = pd.read_csv(opt_comp_path, index_col=0)
        for m_name, row in df_opt.iterrows():
            opt_data[str(m_name)] = {
                "accuracy": round(float(row["Accuracy"]) * 100, 2),
                "precision": round(float(row["Precision"]) * 100, 2),
                "recall": round(float(row["Recall"]) * 100, 2),
                "f1_score": round(float(row["F1-Score"]) * 100, 2)
            }

    # 3. Eval artifacts (confusion matrices, ROC curves, CV)
    eval_path = _find_file("eval_artifacts.pkl")
    confusion_matrices = {}
    roc_curves_data = {}
    roc_auc_scores = {}
    cv_results_data = {}

    if os.path.exists(eval_path):
        artifacts = joblib.load(eval_path)
        
        # Format confusion matrices
        for m_name, cm in artifacts.get("confusion_matrices", {}).items():
            if isinstance(cm, np.ndarray):
                confusion_matrices[m_name] = cm.tolist()
            else:
                confusion_matrices[m_name] = cm

        # Format ROC curves: downsample points if needed for fast JSON serialization
        for m_name, classes_dict in artifacts.get("roc_curves", {}).items():
            roc_curves_data[m_name] = {}
            for c_idx, curve in classes_dict.items():
                fpr = [round(float(v), 4) for v in curve["fpr"]]
                tpr = [round(float(v), 4) for v in curve["tpr"]]
                # Downsample to max 50 points if too large
                if len(fpr) > 50:
                    indices = np.linspace(0, len(fpr) - 1, 50, dtype=int)
                    fpr = [fpr[idx] for idx in indices]
                    tpr = [tpr[idx] for idx in indices]
                roc_curves_data[m_name][str(c_idx)] = {
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": round(float(curve["auc"]), 4)
                }

        roc_auc_scores = {
            m: round(float(s), 4) for m, s in artifacts.get("roc_auc_scores", {}).items()
        }
        
        for m_name, cv_info in artifacts.get("cv_results", {}).items():
            cv_results_data[m_name] = {
                "mean_accuracy": round(float(cv_info["mean_accuracy"]) * 100, 2),
                "std_accuracy": round(float(cv_info["std_accuracy"]) * 100, 2),
                "fold_scores": [round(float(s) * 100, 2) for s in cv_info.get("fold_scores", [])]
            }

    return {
        "baseline_comparison": comparison_data,
        "optimized_comparison": opt_data,
        "confusion_matrices": confusion_matrices,
        "roc_curves": roc_curves_data,
        "roc_auc_scores": roc_auc_scores,
        "cv_results": cv_results_data
    }


def get_dataset_summary():
    """Returns dataset summary statistics and feature distributions."""
    csv_path = _find_file("StressLevelDataset.csv")
    if not os.path.exists(csv_path):
        return {"error": "Dataset file not found"}

    df = pd.read_csv(csv_path)
    total_records = len(df)
    features_count = len(df.columns) - 1
    
    # Class distribution
    class_counts = df["stress_level"].value_counts().to_dict()
    distribution = {
        "Low": int(class_counts.get(0, 0)),
        "Moderate": int(class_counts.get(1, 0)),
        "High": int(class_counts.get(2, 0))
    }
    
    # Numerical summaries per feature
    stats = {}
    desc = df.describe().T
    for feat in df.columns:
        if feat == "stress_level":
            continue
        stats[feat] = {
            "name": feat.replace("_", " ").title(),
            "mean": round(float(desc.loc[feat, "mean"]), 2),
            "std": round(float(desc.loc[feat, "std"]), 2),
            "min": round(float(desc.loc[feat, "min"]), 1),
            "p25": round(float(desc.loc[feat, "25%"]), 1),
            "median": round(float(desc.loc[feat, "50%"]), 1),
            "p75": round(float(desc.loc[feat, "75%"]), 1),
            "max": round(float(desc.loc[feat, "max"]), 1)
        }

    return {
        "total_records": total_records,
        "total_features": features_count,
        "missing_values": int(df.isna().sum().sum()),
        "class_distribution": distribution,
        "feature_statistics": stats
    }


def get_global_shap():
    """Returns global and class-specific SHAP importance rankings."""
    res = {
        "global": [],
        "low_stress": [],
        "moderate_stress": [],
        "high_stress": []
    }

    def load_shap_csv(filename):
        p = _find_file(filename)
        if os.path.exists(p):
            df_s = pd.read_csv(p)
            items = []
            for _, r in df_s.iterrows():
                val_col = [c for c in r.index if c != "Feature"][0]
                items.append({
                    "feature": r["Feature"],
                    "feature_name": str(r["Feature"]).replace("_", " ").title(),
                    "importance": round(float(r[val_col]), 4)
                })
            return items
        return []

    res["global"] = load_shap_csv("shap_feature_importance.csv")
    res["low_stress"] = load_shap_csv("shap_low_importance.csv")
    res["moderate_stress"] = load_shap_csv("shap_moderate_importance.csv")
    res["high_stress"] = load_shap_csv("shap_high_importance.csv")

    return res
