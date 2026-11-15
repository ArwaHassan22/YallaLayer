
import json
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "yallalayer_model.joblib"
TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"

FIGURES_DIR = ROOT / "reports" / "figures"
METRICS_DIR = ROOT / "reports" / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SAMPLE_SIZE = 2000
MAX_DISPLAY = 20


# ============================================================
# 2. LOAD MODEL AND TRAINING DATA
# ============================================================

print("\n========== LOADING MODEL ==========")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}")

artifact = joblib.load(MODEL_PATH)

pipeline = artifact["model"]
features = artifact["features"]
target = artifact["target"]

print("Model:", artifact["model_name"])
print("PageValues included:", artifact["page_values_included"])
print("Selected threshold:", artifact["threshold"])

train_df = pd.read_csv(TRAIN_PATH)

X_train = train_df[features].copy()

print("\nTraining data shape:", X_train.shape)


# ============================================================
# 3. SAMPLE TRAINING DATA
# ============================================================

print("\n========== SAMPLING DATA ==========")

X_sample = X_train.sample(
    n=min(SAMPLE_SIZE, len(X_train)),
    random_state=RANDOM_STATE
).copy()

print("SHAP sample shape:", X_sample.shape)


# ============================================================
# 4. TRANSFORM FEATURES
# ============================================================

print("\n========== PREPROCESSING ==========")

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

X_transformed = preprocessor.transform(X_sample)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_transformed = np.asarray(X_transformed)

encoded_features = preprocessor.get_feature_names_out()

print("Original features:", len(features))
print("Encoded features:", len(encoded_features))

if X_transformed.shape[1] != len(encoded_features):
    raise ValueError(
        "Transformed feature count does not match feature names."
    )


# ============================================================
# 5. CALCULATE SHAP VALUES
# ============================================================

print("\n========== CALCULATING SHAP VALUES ==========")

explainer = shap.TreeExplainer(classifier)

raw_shap_values = explainer.shap_values(X_transformed)

positive_class_index = list(classifier.classes_).index(1)

# Handle different SHAP output formats.
if isinstance(raw_shap_values, list):

    if positive_class_index >= len(raw_shap_values):
        raise ValueError(
            "Positive class SHAP values were not found."
        )

    shap_values = np.asarray(
        raw_shap_values[positive_class_index]
    )

else:

    shap_values = np.asarray(raw_shap_values)

    if shap_values.ndim == 3:

        if (
            shap_values.shape[0] == len(X_sample)
            and shap_values.shape[1] == len(encoded_features)
        ):
            # Shape: samples x features x classes
            shap_values = shap_values[
                :, :, positive_class_index
            ]

        elif (
            shap_values.shape[0] == len(X_sample)
            and shap_values.shape[2] == len(encoded_features)
        ):
            # Shape: samples x classes x features
            shap_values = shap_values[
                :, positive_class_index, :
            ]

        else:
            raise ValueError(
                f"Unexpected 3D SHAP shape: {shap_values.shape}"
            )

if shap_values.ndim != 2:
    raise ValueError(
        f"Unexpected SHAP shape: {shap_values.shape}"
    )

if shap_values.shape != X_transformed.shape:
    raise ValueError(
        f"SHAP shape {shap_values.shape} does not match "
        f"transformed data shape {X_transformed.shape}."
    )

if not np.isfinite(shap_values).all():
    raise ValueError("SHAP values contain non-finite values.")

print("SHAP values shape:", shap_values.shape)


# ============================================================
# 6. GROUP ENCODED FEATURES
# ============================================================

print("\n========== GROUPING FEATURES ==========")

n_samples = len(X_sample)
n_features = len(features)

# Signed grouped SHAP:
# Sum signed encoded contributions within each original feature.
grouped_signed_shap = np.zeros(
    (n_samples, n_features)
)

# Absolute grouped SHAP:
# Sum absolute encoded contributions within each original feature.
grouped_absolute_shap = np.zeros(
    (n_samples, n_features)
)

feature_groups = {}

numeric_features = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]

for i, feature in enumerate(features):

    if feature in numeric_features:

        matching_indices = [
            j
            for j, name in enumerate(encoded_features)
            if name == f"num__{feature}"
        ]

    else:

        matching_indices = [
            j
            for j, name in enumerate(encoded_features)
            if name.startswith(f"cat__{feature}_")
        ]

    if not matching_indices:
        raise ValueError(
            f"No encoded columns found for feature: {feature}"
        )

    feature_groups[feature] = matching_indices

    feature_shap = shap_values[:, matching_indices]

    # Preserve signed contributions.
    grouped_signed_shap[:, i] = feature_shap.sum(axis=1)

    # Correct global importance aggregation.
    grouped_absolute_shap[:, i] = np.abs(
        feature_shap
    ).sum(axis=1)

print("Grouped features:", len(feature_groups))


# ============================================================
# 7. GLOBAL FEATURE IMPORTANCE
# ============================================================

print("\n========== GLOBAL FEATURE IMPORTANCE ==========")

importance = pd.DataFrame({
    "Feature": features,

    # Canonical grouped Mean Absolute SHAP.
    "Mean_Absolute_SHAP": (
        grouped_absolute_shap.mean(axis=0)
    ),

    # Average signed contribution.
    "Mean_Signed_SHAP": (
        grouped_signed_shap.mean(axis=0)
    ),

    # Absolute value after signed contributions are grouped.
    # Diagnostic measure, not the primary importance metric.
    "Mean_Absolute_Net_Grouped_SHAP": (
        np.abs(grouped_signed_shap).mean(axis=0)
    )
})

importance = importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
).reset_index(drop=True)

importance.to_csv(
    METRICS_DIR / "shap_feature_importance.csv",
    index=False
)

print(
    importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 8. PLOT 1: GROUPED FEATURE IMPORTANCE
# ============================================================

print("\n========== GENERATING GLOBAL IMPORTANCE PLOT ==========")

plt.figure(figsize=(11, 7))

plot_data = importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=True
)

plt.barh(
    plot_data["Feature"],
    plot_data["Mean_Absolute_SHAP"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Original Feature")
plt.title("Global Feature Importance - Grouped SHAP")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "shap_global_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. PLOT 2: SHAP SUMMARY PLOT
# ============================================================

print("\n========== GENERATING SHAP SUMMARY PLOT ==========")

plt.figure(figsize=(12, 9))

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=encoded_features,
    max_display=MAX_DISPLAY,
    show=False
)

plt.title("SHAP Summary - Encoded Features")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. INDIVIDUAL PREDICTION EXPLANATION
# ============================================================

print("\n========== INDIVIDUAL EXPLANATION ==========")

base_value = np.asarray(
    explainer.expected_value
).reshape(-1)

if len(base_value) > 1:

    base_value = float(
        base_value[positive_class_index]
    )

else:

    base_value = float(base_value[0])


individual_shap_values = shap_values[0]

individual_explanation = shap.Explanation(
    values=individual_shap_values,
    base_values=base_value,
    data=X_transformed[0],
    feature_names=encoded_features
)

# Prediction probability for the positive class.
sample_probability = float(
    classifier.predict_proba(
        X_transformed[0].reshape(1, -1)
    )[0, positive_class_index]
)

selected_threshold = float(artifact["threshold"])

predicted_class = int(
    sample_probability >= selected_threshold
)

print("Sample index:", X_sample.index[0])
print(f"Purchase probability: {sample_probability:.4f}")
print(f"Decision threshold: {selected_threshold:.4f}")

print(
    "Predicted class:",
    "Purchase" if predicted_class == 1 else "No Purchase"
)

print(
    "Actual class:",
    int(train_df.loc[X_sample.index[0], target])
)

plt.figure(figsize=(12, 8))

shap.plots.waterfall(
    individual_explanation,
    max_display=15,
    show=False
)

plt.title(
    "Individual Session Prediction Explanation "
    "(Positive Class)"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "shap_individual_explanation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. SAVE SUMMARY
# ============================================================

print("\n========== SAVING SUMMARY ==========")

summary = {
    "model_name": artifact["model_name"],

    "page_values_included": artifact["page_values_included"],

    "selected_threshold": selected_threshold,

    "sample_size": len(X_sample),

    "original_feature_count": len(features),

    "encoded_feature_count": len(encoded_features),

    "positive_class": 1,

    "shap_method": "TreeExplainer",

    "global_importance_method": (
        "Mean of per-sample sums of absolute encoded SHAP values "
        "grouped by original feature"
    ),

    "signed_contribution_method": (
        "Mean of per-sample sums of signed encoded SHAP values "
        "grouped by original feature"
    ),

    "shap_output_space": (
        "Raw model output; SHAP values are not probabilities "
        "or percentage changes"
    ),

    "top_5_features": importance.head(5).to_dict(
        orient="records"
    ),

    "individual_explanation": {
        "sample_index": int(X_sample.index[0]),
        "purchase_probability": sample_probability,
        "predicted_class": predicted_class,
        "actual_class": int(
            train_df.loc[X_sample.index[0], target]
        ),
        "decision_threshold": selected_threshold
    },

    "data_source": "Training set only",

    "random_state": RANDOM_STATE
}

with open(
    METRICS_DIR / "shap_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 12. COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("EXPLAINABILITY COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nFeature importance:")
print(METRICS_DIR / "shap_feature_importance.csv")

print("\nFigures:")
print(FIGURES_DIR / "shap_global_importance.png")
print(FIGURES_DIR / "shap_summary.png")
print(FIGURES_DIR / "shap_individual_explanation.png")

print("\nSummary:")
print(METRICS_DIR / "shap_summary.json")

print("\nTop 5 Features:")

print(
    importance.head(5)[
        ["Feature", "Mean_Absolute_SHAP", "Mean_Signed_SHAP"]
    ].to_string(index=False)
)

print("\nAll explainability outputs generated successfully.")
