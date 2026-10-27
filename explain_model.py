import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n==========================================")
print("SHAP EXPLAINABLE AI ANALYSIS")
print("==========================================")

print("\nLoading dataset...")

df = pd.read_csv("StressLevelDataset.csv")

X = df.drop("stress_level", axis=1)
y = df["stress_level"]

print("Dataset shape:", df.shape)
print("Number of features:", X.shape[1])


# ==========================================
# 2. LOAD OPTIMIZED RANDOM FOREST
# ==========================================

print("\nLoading optimized Random Forest...")

model = joblib.load("optimized_stress_model.pkl")

print("Optimized Random Forest loaded successfully.")


# ==========================================
# 3. CREATE SHAP EXPLAINER
# ==========================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

shap_array = np.asarray(shap_values)

print("SHAP values calculated successfully.")
print("SHAP values shape:", shap_array.shape)


# ==========================================
# 4. CALCULATE GLOBAL FEATURE IMPORTANCE
# ==========================================

print("\nCalculating global feature importance...")


# SHAP 0.52.0 for multiclass Random Forest
# generally returns:
#
# (number_of_samples, number_of_features, number_of_classes)
#
# Example:
# (1100, 20, 3)

if shap_array.ndim == 3:

    # Average absolute SHAP values across:
    # samples AND classes

    importance = np.mean(
        np.abs(shap_array),
        axis=(0, 2)
    )

elif shap_array.ndim == 2:

    # Binary/single-output case

    importance = np.mean(
        np.abs(shap_array),
        axis=0
    )

else:

    raise ValueError(
        f"Unexpected SHAP output shape: {shap_array.shape}"
    )


# ==========================================
# 5. VERIFY FEATURE IMPORTANCE
# ==========================================

print("\nNumber of features:", len(X.columns))
print("Number of importance values:", len(importance))


if len(X.columns) != len(importance):

    raise ValueError(
        "Number of features does not match "
        "number of SHAP importance values."
    )


# ==========================================
# 6. CREATE FEATURE IMPORTANCE TABLE
# ==========================================

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Mean_ABS_SHAP": importance

})


feature_importance = feature_importance.sort_values(
    by="Mean_ABS_SHAP",
    ascending=False
)


# ==========================================
# 7. DISPLAY FEATURE IMPORTANCE
# ==========================================

print("\n==========================================")
print("GLOBAL SHAP FEATURE IMPORTANCE")
print("==========================================")

print(
    feature_importance.to_string(index=False)
)


# ==========================================
# 8. SAVE FEATURE IMPORTANCE
# ==========================================

feature_importance.to_csv(
    "shap_feature_importance.csv",
    index=False
)

print("\nSaved:")
print("shap_feature_importance.csv")


# ==========================================
# 9. CREATE GLOBAL FEATURE IMPORTANCE PLOT
# ==========================================

print("\nCreating global SHAP feature importance plot...")

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Mean_ABS_SHAP"][::-1]
)

plt.xlabel("Mean Absolute SHAP Value")

plt.ylabel("Feature")

plt.title(
    "Top 15 Features Influencing Stress Prediction"
)

plt.tight_layout()

plt.savefig(
    "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print("shap_feature_importance.png")


# ==========================================
# 10. CLASS-SPECIFIC SHAP IMPORTANCE
# ==========================================

if shap_array.ndim == 3:

    print("\nCreating class-specific SHAP analysis...")

    class_names = [
        "Low",
        "Moderate",
        "High"
    ]

    for class_index, class_name in enumerate(class_names):

        class_shap = shap_array[:, :, class_index]

        class_importance = np.mean(
            np.abs(class_shap),
            axis=0
        )

        class_df = pd.DataFrame({

            "Feature": X.columns,

            "Mean_ABS_SHAP": class_importance

        })

        class_df = class_df.sort_values(
            by="Mean_ABS_SHAP",
            ascending=False
        )

        filename = (
            "shap_"
            + class_name.lower()
            + "_importance.csv"
        )

        class_df.to_csv(
            filename,
            index=False
        )

        print("Saved:", filename)


# ==========================================
# 11. CREATE CLASS-SPECIFIC SHAP PLOTS
# ==========================================

if shap_array.ndim == 3:

    print("\nCreating class-specific SHAP plots...")

    class_names = [
        "Low",
        "Moderate",
        "High"
    ]

    for class_index, class_name in enumerate(class_names):

        class_shap = shap_array[:, :, class_index]

        class_importance = np.mean(
            np.abs(class_shap),
            axis=0
        )

        class_df = pd.DataFrame({

            "Feature": X.columns,

            "Importance": class_importance

        })

        class_df = class_df.sort_values(
            by="Importance",
            ascending=False
        )

        top_class_features = class_df.head(15)

        plt.figure(figsize=(10, 8))

        plt.barh(
            top_class_features["Feature"][::-1],
            top_class_features["Importance"][::-1]
        )

        plt.xlabel("Mean Absolute SHAP Value")

        plt.ylabel("Feature")

        plt.title(
            f"SHAP Feature Importance - {class_name} Stress"
        )

        plt.tight_layout()

        filename = (
            "shap_"
            + class_name.lower()
            + "_importance.png"
        )

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("Saved:", filename)


# ==========================================
# 12. INDIVIDUAL PREDICTION
# ==========================================

print("\n==========================================")
print("INDIVIDUAL PREDICTION EXPLANATION")
print("==========================================")


sample_index = 0

sample = X.iloc[[sample_index]]

prediction = model.predict(sample)[0]

probabilities = model.predict_proba(sample)[0]


stress_labels = {

    0: "Low",

    1: "Moderate",

    2: "High"

}


print(
    "Sample Index:",
    sample_index
)

print(
    "Predicted Stress:",
    stress_labels[prediction]
)

print("\nPrediction Probabilities:")

for class_index, probability in enumerate(probabilities):

    print(
        f"{stress_labels[class_index]}: "
        f"{probability:.4f}"
    )


# ==========================================
# 13. INDIVIDUAL SHAP CONTRIBUTIONS
# ==========================================

if shap_array.ndim == 3:

    predicted_class = int(prediction)

    sample_shap = shap_array[
        sample_index,
        :,
        predicted_class
    ]

    individual_explanation = pd.DataFrame({

        "Feature": X.columns,

        "SHAP_Value": sample_shap,

        "Absolute_SHAP": np.abs(sample_shap)

    })

    individual_explanation = (
        individual_explanation
        .sort_values(
            by="Absolute_SHAP",
            ascending=False
        )
    )


    print(
        "\nTop factors influencing this prediction:"
    )

    print(
        individual_explanation.head(10)
        .to_string(index=False)
    )


    # Save individual explanation

    individual_explanation.to_csv(
        "individual_shap_explanation.csv",
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        "individual_shap_explanation.csv"
    )


# ==========================================
# 14. SUMMARY
# ==========================================

print("\n==========================================")
print("SHAP ANALYSIS COMPLETE!")
print("==========================================")

print("\nGenerated files:")

print("1. shap_feature_importance.csv")

print("2. shap_feature_importance.png")

print("3. shap_low_importance.csv")

print("4. shap_moderate_importance.csv")

print("5. shap_high_importance.csv")

print("6. shap_low_importance.png")

print("7. shap_moderate_importance.png")

print("8. shap_high_importance.png")

print("9. individual_shap_explanation.csv")

print("\n==========================================")