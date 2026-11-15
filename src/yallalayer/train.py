
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Revenue"
RANDOM_STATE = 42
N_SPLITS = 5

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

NUMERICAL_FEATURES = [
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


# ==========================================
# 2. LOAD AND VALIDATE TRAINING DATA
# ==========================================

def load_training_data():

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_PATH}"
        )

    df = pd.read_csv(TRAIN_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    expected_features = (
        CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    )

    missing_columns = [
        col for col in expected_features
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    X = df[expected_features].copy()
    y = df[TARGET].astype(int)

    if y.nunique() != 2:
        raise ValueError("Expected a binary target.")

    print("\n========== DATA LOADED ==========")
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Purchasing sessions: {y.sum()}")
    print(f"Non-purchasing sessions: {(y == 0).sum()}")

    return X, y


# ==========================================
# 3. PREPROCESSING PIPELINE
# ==========================================

def build_preprocessor(numerical_features):

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor


# ==========================================
# 4. DEFINE CANDIDATE MODELS
# ==========================================

def get_models():

    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.8,
            random_state=RANDOM_STATE
        )
    }


# ==========================================
# 5. CROSS-VALIDATION MODEL COMPARISON
# ==========================================

def compare_models(X, y, include_page_values):

    numerical_features = [
        col for col in NUMERICAL_FEATURES
        if include_page_values or col != "PageValues"
    ]

    feature_columns = (
        numerical_features + CATEGORICAL_FEATURES
    )

    X_subset = X[feature_columns].copy()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1"
    }

    results = []
    pipelines = {}

    print("\n======================================")
    print(
        "EXPERIMENT:",
        "WITH PageValues" if include_page_values
        else "WITHOUT PageValues"
    )
    print("======================================")

    for model_name, estimator in get_models().items():

        pipeline = Pipeline([
            (
                "preprocessor",
                build_preprocessor(numerical_features)
            ),
            ("classifier", estimator)
        ])

        scores = cross_validate(
            pipeline,
            X_subset,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False
        )

        row = {
            "Model": model_name,
            "PageValues": include_page_values
        }

        for metric in scoring:
            values = scores[f"test_{metric}"]

            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()

        results.append(row)

        pipelines[model_name] = (
            pipeline,
            X_subset,
            numerical_features
        )

        print(
            f"\n{model_name}: "
            f"PR-AUC={row['average_precision_mean']:.4f} "
            f"(+/- {row['average_precision_std']:.4f})"
        )

    return results, pipelines


# ==========================================
# 6. THRESHOLD OPTIMIZATION
# ==========================================

def find_best_threshold(y_true, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    # F2 gives Recall more importance than Precision.
    f2_scores = (
        5 * precision[:-1] * recall[:-1]
        / (4 * precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = np.argmax(f2_scores)

    return (
        float(thresholds[best_index]),
        float(f2_scores[best_index])
    )


# ==========================================
# 7. MAIN TRAINING WORKFLOW
# ==========================================

def main():

    X, y = load_training_data()

    all_results = []
    experiment_pipelines = {}

    # Compare three models with and without PageValues.
    for include_page_values in [True, False]:

        results, pipelines = compare_models(
            X,
            y,
            include_page_values
        )

        all_results.extend(results)

        for model_name, pipeline_info in pipelines.items():

            key = (
                f"{model_name} | "
                f"{'With' if include_page_values else 'Without'} PageValues"
            )

            experiment_pipelines[key] = pipeline_info

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by="average_precision_mean",
        ascending=False
    ).reset_index(drop=True)

    results_path = REPORT_DIR / "cv_model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\n========== MODEL COMPARISON ==========")

    display_columns = [
        "Model",
        "PageValues",
        "average_precision_mean",
        "roc_auc_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean"
    ]

    print(
        results_df[display_columns].round(4).to_string(
            index=False
        )
    )

    # Select the candidate with the highest CV Average Precision.
    best_row = results_df.iloc[0]

    best_key = (
        f"{best_row['Model']} | "
        f"{'With' if best_row['PageValues'] else 'Without'} PageValues"
    )

    best_pipeline, X_best, numerical_features = (
        experiment_pipelines[best_key]
    )

    print("\n========== SELECTED MODEL ==========")
    print(best_key)

    # Generate out-of-fold probabilities for threshold selection.
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    oof_probabilities = cross_val_predict(
        best_pipeline,
        X_best,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    best_threshold, best_f2 = find_best_threshold(
        y,
        oof_probabilities
    )

    print(f"Selected threshold: {best_threshold:.4f}")
    print(f"OOF F2-score: {best_f2:.4f}")

    # Fit the selected pipeline on all training data.
    best_pipeline.fit(X_best, y)

    # Save the complete fitted pipeline and threshold.
    artifact = {
        "model": best_pipeline,
        "model_name": best_row["Model"],
        "page_values_included": bool(best_row["PageValues"]),
        "threshold": best_threshold,
        "features": list(X_best.columns),
        "target": TARGET,
        "random_state": RANDOM_STATE
    }

    model_path = MODEL_DIR / "yallalayer_model.joblib"

    joblib.dump(artifact, model_path)

    # Save experiment metadata.
    summary = {
        "selected_model": best_row["Model"],
        "page_values_included": bool(best_row["PageValues"]),
        "cv_average_precision": float(
            best_row["average_precision_mean"]
        ),
        "cv_roc_auc": float(best_row["roc_auc_mean"]),
        "cv_recall": float(best_row["recall_mean"]),
        "cv_f1": float(best_row["f1_mean"]),
        "oof_selected_threshold": best_threshold,
        "oof_f2_score": best_f2,
        "training_rows": len(X),
        "cross_validation_folds": N_SPLITS
    }

    with open(
        REPORT_DIR / "training_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(summary, file, indent=4)

    print("\n========== TRAINING COMPLETED ==========")
    print(f"Model saved to: {model_path}")
    print(f"Comparison saved to: {results_path}")
    print("\nTest data was NOT used.")


if __name__ == "__main__":
    main()
