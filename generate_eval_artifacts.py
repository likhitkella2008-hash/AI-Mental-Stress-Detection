"""
generate_eval_artifacts.py
==========================
Generates authentic evaluation artifacts for all 5 machine learning models
using the EXACT configuration from train_model.py:
- Dataset: StressLevelDataset.csv (1100 rows, 20 features)
- Split: 80% train (880), 20% test (220), random_state=42, stratify=y
- Models: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost

Outputs:
- eval_artifacts.pkl (Confusion matrices, multiclass ROC curves, ROC-AUC scores,
  Random Forest feature importances, 5-fold cross-validation scores)

NOTE: Does NOT overwrite model_comparison.csv or best_stress_model.pkl.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)

def main():
    print("=" * 60)
    print("EVALUATION ARTIFACT GENERATOR (AUTHENTIC ML METRICS)")
    print("=" * 60)

    # 1. Load dataset & features
    print("\nLoading dataset and feature names...")
    df = pd.read_csv("StressLevelDataset.csv")
    feature_names = joblib.load("feature_names.pkl")
    
    X = df[feature_names]
    y = df["stress_level"]
    print(f"Data shape: {df.shape} | Features: {len(feature_names)}")

    # 2. Train-test split (exact same configuration: 80/20, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3. Define models (exact same definitions from train_model.py)
    models = {
        "Random Forest": joblib.load("best_stress_model.pkl"),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    # Verify baseline model accuracy matches baseline
    rf_pred = models["Random Forest"].predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Random Forest baseline verification: {rf_acc:.4f} ({rf_acc*100:.2f}%)")

    # Fit other models
    for name, model in models.items():
        if name != "Random Forest":
            print(f"Fitting {name} on training set...")
            model.fit(X_train, y_train)

    # 4. Generate Confusion Matrices, Predictions & ROC-AUC data
    print("\nGenerating confusion matrices and ROC curves...")
    eval_data = {
        "confusion_matrices": {},
        "roc_curves": {},
        "roc_auc_scores": {},
        "class_names": ["Low", "Moderate", "High"]
    }

    # Binarize targets for multiclass ROC curves (One-vs-Rest)
    y_test_bin = pd.get_dummies(y_test).values

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        eval_data["confusion_matrices"][name] = cm.tolist()

        # Weighted One-vs-Rest ROC-AUC
        ovr_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        eval_data["roc_auc_scores"][name] = float(ovr_auc)

        # ROC Curve per class
        model_roc = {}
        for c in [0, 1, 2]:
            fpr, tpr, _ = roc_curve(y_test_bin[:, c], y_proba[:, c])
            class_auc = auc(fpr, tpr)
            model_roc[c] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(class_auc)
            }
        eval_data["roc_curves"][name] = model_roc

        print(f"  {name:20s} | Test Acc: {accuracy_score(y_test, y_pred):.4f} | ROC-AUC: {ovr_auc:.4f}")

    # 5. Feature Importance from actual Random Forest model
    print("\nComputing actual feature importances from Random Forest model...")
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    eval_data["feature_importances"] = fi_df.to_dict(orient="records")
    print("Top 5 features by actual importance:")
    for idx, row in fi_df.head(5).iterrows():
        print(f"  {row['Feature']:30s}: {row['Importance']:.4f}")

    # 6. Additional Experiment: 5-Fold Stratified Cross-Validation
    print("\nRunning 5-Fold Stratified Cross-Validation (Additional Experiment)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        cv_results[name] = {
            "mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std()),
            "fold_scores": scores.tolist()
        }
        print(f"  {name:20s} | 5-Fold CV: {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f}%)")
    eval_data["cv_results"] = cv_results

    # 7. Save evaluation artifacts
    joblib.dump(eval_data, "eval_artifacts.pkl")
    print("\nSaved evaluation artifacts to 'eval_artifacts.pkl' successfully.")
    print("=" * 60)
    print("DONE! Baseline files preserved.")
    print("=" * 60)

if __name__ == "__main__":
    main()
