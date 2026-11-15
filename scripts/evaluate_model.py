
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ==========================================
# 1. PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "data" / "splits" / "test.csv"
MODEL_PATH = ROOT / "models" / "yallalayer_model.joblib"

REPORT_DIR = ROOT / "reports" / "metrics"
FIGURE_DIR = ROOT / "reports" / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD MODEL AND TEST DATA
# ==========================================

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

df = pd.read_csv(TEST_PATH)

if "Revenue" not in df.columns:
    raise ValueError("Revenue target is missing.")

missing = [col for col in features if col not in df.columns]

if missing:
    raise ValueError(f"Missing test features: {missing}")

X_test = df[features].copy()
y_test = df["Revenue"].astype(int)

if y_test.nunique() != 2:
    raise ValueError("Test target must contain both classes.")

print("\n========== TEST DATA ==========")
print(f"Test shape: {df.shape}")
print(f"Test sessions: {len(df)}")
print(f"Purchasing sessions: {y_test.sum()}")
print(f"Selected model: {artifact['model_name']}")
print(f"PageValues included: {artifact['page_values_included']}")
print(f"Selected threshold: {threshold:.4f}")


# ==========================================
# 3. GENERATE PREDICTIONS
# ==========================================

probabilities = model.predict_proba(X_test)[:, 1]

predictions_default = (probabilities >= 0.5).astype(int)
predictions_optimized = (
    probabilities >= threshold
).astype(int)


# ==========================================
# 4. EVALUATION FUNCTION
# ==========================================

def calculate_metrics(y_true, predictions, probabilities):

    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "F2": fbeta_score(
            y_true, predictions, beta=2, zero_division=0
        ),
        "PR_AUC_Average_Precision": average_precision_score(
            y_true, probabilities
        ),
        "ROC_AUC": roc_auc_score(
            y_true, probabilities
        )
    }


default_metrics = calculate_metrics(
    y_test,
    predictions_default,
    probabilities
)

optimized_metrics = calculate_metrics(
    y_test,
    predictions_optimized,
    probabilities
)


# ==========================================
# 5. PRINT RESULTS
# ==========================================

comparison = pd.DataFrame(
    [
        {"Threshold": 0.5, **default_metrics},
        {
            "Threshold": threshold,
            **optimized_metrics
        }
    ]
)

print("\n========== TEST METRICS ==========")
print(comparison.round(4).to_string(index=False))


# ==========================================
# 6. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions_optimized
)

tn, fp, fn, tp = cm.ravel()

print("\n========== CONFUSION MATRIX ==========")
print(cm)

print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")


# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

report = classification_report(
    y_test,
    predictions_optimized,
    target_names=["No Purchase", "Purchase"],
    output_dict=True,
    zero_division=0
)

print(
    classification_report(
        y_test,
        predictions_optimized,
        target_names=["No Purchase", "Purchase"],
        zero_division=0
    )
)


# ==========================================
# 8. SAVE CONFUSION MATRIX
# ==========================================

fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Purchase", "Purchase"]
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

ax.set_title("YallaLayer - Test Confusion Matrix")
plt.tight_layout()

figure_path = FIGURE_DIR / "test_confusion_matrix.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.close()


# ==========================================
# 9. SAVE TEST RESULTS
# ==========================================

results = {
    "model_name": artifact["model_name"],
    "page_values_included": artifact["page_values_included"],
    "threshold": float(threshold),
    "test_samples": len(X_test),
    "default_threshold_metrics": {
        key: float(value)
        for key, value in default_metrics.items()
    },
    "optimized_threshold_metrics": {
        key: float(value)
        for key, value in optimized_metrics.items()
    },
    "confusion_matrix": {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    },
    "classification_report": report
}

results_path = REPORT_DIR / "test_evaluation.json"

with open(results_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\n========== EVALUATION COMPLETED ==========")
print(f"Metrics saved: {results_path}")
print(f"Confusion matrix saved: {figure_path}")
