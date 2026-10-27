import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, auc


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("StressLevelDataset.csv")

X = df.drop("stress_level", axis=1)
y = df["stress_level"]


# ==========================================
# 2. SAME TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. DEFINE MODELS
# ==========================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            probability=True,
            random_state=42
        ))
    ]),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42
    )
}


# ==========================================
# 4. CONVERT TARGET TO BINARY FORMAT
# ==========================================

classes = [0, 1, 2]

y_test_binary = label_binarize(
    y_test,
    classes=classes
)


# ==========================================
# 5. CREATE ROC CURVES
# ==========================================

roc_data = {}

for name, model in models.items():

    print(f"Processing {name}...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)

    model_curves = {}

    for i, class_name in enumerate(
        ["Low", "Moderate", "High"]
    ):

        fpr, tpr, _ = roc_curve(
            y_test_binary[:, i],
            probabilities[:, i]
        )

        roc_auc = auc(fpr, tpr)

        model_curves[class_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc
        }

    roc_data[name] = model_curves


# ==========================================
# 6. SAVE ROC DATA
# ==========================================

import joblib

joblib.dump(
    roc_data,
    "roc_curve_data.pkl"
)

print("\nROC curve data saved as:")
print("roc_curve_data.pkl")


# ==========================================
# 7. CREATE INDIVIDUAL MODEL CURVES
# ==========================================

for model_name, curves in roc_data.items():

    plt.figure(figsize=(8, 6))

    for class_name, data in curves.items():

        plt.plot(
            data["fpr"],
            data["tpr"],
            label=f"{class_name} (AUC = {data['auc']:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"ROC Curve - {model_name}"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_roc_curve.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {filename}")


print("\n==========================================")
print("ROC CURVE ANALYSIS COMPLETE!")
print("==========================================")