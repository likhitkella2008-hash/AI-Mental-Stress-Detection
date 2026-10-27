import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("stress_level", axis=1)
y = df["stress_level"]

print("Number of input features:", X.shape[1])
print("Target variable: stress_level")


# ==========================================
# 3. SAME TRAIN-TEST SPLIT AS BASELINE
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
# 4. BASELINE RANDOM FOREST
# ==========================================

baseline_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

baseline_model.fit(X_train, y_train)

baseline_pred = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)

baseline_precision = precision_score(
    y_test,
    baseline_pred,
    average="weighted",
    zero_division=0
)

baseline_recall = recall_score(
    y_test,
    baseline_pred,
    average="weighted",
    zero_division=0
)

baseline_f1 = f1_score(
    y_test,
    baseline_pred,
    average="weighted",
    zero_division=0
)


print("\n==========================================")
print("BASELINE RANDOM FOREST")
print("==========================================")

print(f"Accuracy : {baseline_accuracy:.4f}")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall   : {baseline_recall:.4f}")
print(f"F1-Score : {baseline_f1:.4f}")


# ==========================================
# 5. DEFINE RANDOM FOREST
# ==========================================

rf = RandomForestClassifier(
    random_state=42
)


# ==========================================
# 6. DEFINE HYPERPARAMETER GRID
# ==========================================

param_grid = {

    "n_estimators": [100, 200, 300],

    "max_depth": [None, 10, 20],

    "min_samples_split": [2, 5],

    "min_samples_leaf": [1, 2],

    "max_features": ["sqrt", "log2"]
}


# ==========================================
# 7. GRID SEARCH + 5-FOLD CV
# ==========================================

print("\n==========================================")
print("STARTING HYPERPARAMETER OPTIMIZATION")
print("==========================================")

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)


# ==========================================
# 8. BEST PARAMETERS
# ==========================================

print("\n==========================================")
print("BEST HYPERPARAMETERS")
print("==========================================")

print(grid_search.best_params_)

print(
    "\nBest Cross-Validation Accuracy:",
    f"{grid_search.best_score_:.4f}"
)


# ==========================================
# 9. GET OPTIMIZED MODEL
# ==========================================

optimized_model = grid_search.best_estimator_


# ==========================================
# 10. TEST OPTIMIZED MODEL
# ==========================================

optimized_pred = optimized_model.predict(X_test)


optimized_accuracy = accuracy_score(
    y_test,
    optimized_pred
)

optimized_precision = precision_score(
    y_test,
    optimized_pred,
    average="weighted",
    zero_division=0
)

optimized_recall = recall_score(
    y_test,
    optimized_pred,
    average="weighted",
    zero_division=0
)

optimized_f1 = f1_score(
    y_test,
    optimized_pred,
    average="weighted",
    zero_division=0
)

optimized_cm = confusion_matrix(
    y_test,
    optimized_pred
)


# ==========================================
# 11. PRINT OPTIMIZED RESULTS
# ==========================================

print("\n==========================================")
print("OPTIMIZED RANDOM FOREST RESULTS")
print("==========================================")

print(f"Accuracy : {optimized_accuracy:.4f}")
print(f"Precision: {optimized_precision:.4f}")
print(f"Recall   : {optimized_recall:.4f}")
print(f"F1-Score : {optimized_f1:.4f}")

print("\nConfusion Matrix:")
print(optimized_cm)


# ==========================================
# 12. COMPARE BASELINE VS OPTIMIZED
# ==========================================

comparison = pd.DataFrame({

    "Baseline Random Forest": {
        "Accuracy": baseline_accuracy,
        "Precision": baseline_precision,
        "Recall": baseline_recall,
        "F1-Score": baseline_f1
    },

    "Optimized Random Forest": {
        "Accuracy": optimized_accuracy,
        "Precision": optimized_precision,
        "Recall": optimized_recall,
        "F1-Score": optimized_f1
    }

}).T


print("\n==========================================")
print("BASELINE VS OPTIMIZED")
print("==========================================")

print(comparison)


# ==========================================
# 13. SAVE OPTIMIZED MODEL
# ==========================================

joblib.dump(
    optimized_model,
    "optimized_stress_model.pkl"
)

print("\nOptimized model saved as:")
print("optimized_stress_model.pkl")


# ==========================================
# 14. SAVE OPTIMIZATION RESULTS
# ==========================================

comparison.to_csv(
    "optimized_model_comparison.csv"
)

print("Comparison saved as:")
print("optimized_model_comparison.csv")


# ==========================================
# 15. SAVE BEST PARAMETERS
# ==========================================

best_params_df = pd.DataFrame(
    [grid_search.best_params_]
)

best_params_df["CV_Accuracy"] = grid_search.best_score_

best_params_df.to_csv(
    "best_rf_parameters.csv",
    index=False
)

print("Best parameters saved as:")
print("best_rf_parameters.csv")


# ==========================================
# 16. COMPLETE
# ==========================================

print("\n==========================================")
print("HYPERPARAMETER OPTIMIZATION COMPLETE!")
print("==========================================")