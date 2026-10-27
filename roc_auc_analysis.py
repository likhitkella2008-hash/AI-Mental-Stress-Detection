import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv("StressLevelDataset.csv")

X = df.drop("stress_level", axis=1)
y = df["stress_level"]

print("Dataset shape:", df.shape)


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

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 3. DEFINE THE FIVE MODELS
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
# 4. CONVERT TARGET TO ONE-HOT FORMAT
# ==========================================

# Classes:
# 0 = Low
# 1 = Moderate
# 2 = High

classes = [0, 1, 2]

y_test_binary = label_binarize(
    y_test,
    classes=classes
)


# ==========================================
# 5. CALCULATE ROC-AUC
# ==========================================

roc_results = {}

print("\n==========================================")
print("MULTI-CLASS ROC-AUC ANALYSIS")
print("==========================================")


for name, model in models.items():

    print(f"\nEvaluating {name}...")

    # Train model
    model.fit(X_train, y_train)

    # Get class probabilities
    y_probability = model.predict_proba(X_test)

    # Calculate One-vs-Rest ROC-AUC
    auc = roc_auc_score(
        y_test_binary,
        y_probability,
        multi_class="ovr",
        average="weighted"
    )

    roc_results[name] = auc

    print(f"ROC-AUC: {auc:.4f}")


# ==========================================
# 6. CREATE RESULTS TABLE
# ==========================================

roc_df = pd.DataFrame(
    list(roc_results.items()),
    columns=["Model", "ROC-AUC"]
)

roc_df = roc_df.sort_values(
    by="ROC-AUC",
    ascending=False
)


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("ROC-AUC RESULTS")
print("==========================================")

print(roc_df.to_string(index=False))


# ==========================================
# 8. SAVE RESULTS
# ==========================================

roc_df.to_csv(
    "roc_auc_results.csv",
    index=False
)

print("\nROC-AUC results saved as:")
print("roc_auc_results.csv")


# ==========================================
# 9. BEST ROC-AUC MODEL
# ==========================================

best_model = roc_df.iloc[0]["Model"]
best_auc = roc_df.iloc[0]["ROC-AUC"]

print("\n==========================================")
print("BEST ROC-AUC MODEL")
print("==========================================")

print("Best Model:", best_model)
print(f"ROC-AUC: {best_auc:.4f}")

print("\n==========================================")
print("ROC-AUC ANALYSIS COMPLETE!")
print("==========================================")