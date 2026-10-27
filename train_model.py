import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
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
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv("StressLevelDataset.csv")

print("Dataset shape:", df.shape)
print("Missing values:", df.isna().sum().sum())


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("stress_level", axis=1)
y = df["stress_level"]

print("\nNumber of input features:", X.shape[1])
print("Target variable: stress_level")


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. DEFINE MODELS
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
# 5. TRAIN AND EVALUATE MODELS
# ==========================================

results = {}
trained_models = {}

print("\n==========================================")
print("TRAINING 5 MACHINE LEARNING MODELS")
print("==========================================")

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    # Store results
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Confusion Matrix": cm
    }

    trained_models[name] = model

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("Confusion Matrix:")
    print(cm)


# ==========================================
# 6. MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame({
    name: {
        "Accuracy": results[name]["Accuracy"],
        "Precision": results[name]["Precision"],
        "Recall": results[name]["Recall"],
        "F1-Score": results[name]["F1-Score"]
    }
    for name in results
}).T

comparison = comparison.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(comparison)


# ==========================================
# 7. FIND BEST MODEL
# ==========================================

best_model_name = comparison.index[0]

best_model = trained_models[best_model_name]

print("\n==========================================")
print("BEST MODEL")
print("==========================================")

print("Best Model:", best_model_name)
print(
    "Best Accuracy:",
    f"{comparison.loc[best_model_name, 'Accuracy']:.4f}"
)


# ==========================================
# 8. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    "best_stress_model.pkl"
)

print("\nBest model saved as:")
print("best_stress_model.pkl")


# ==========================================
# 9. SAVE MODEL COMPARISON
# ==========================================

comparison.to_csv(
    "model_comparison.csv"
)

print("Model comparison saved as:")
print("model_comparison.csv")


# ==========================================
# 10. SAVE FEATURE NAMES
# ==========================================

joblib.dump(
    list(X.columns),
    "feature_names.pkl"
)

print("Feature names saved as:")
print("feature_names.pkl")


print("\n==========================================")
print("TRAINING COMPLETE!")
print("==========================================")