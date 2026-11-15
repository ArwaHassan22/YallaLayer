
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "reports" / "marketing"
OUTPUT_DIR = INPUT_DIR / "validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

Z_SCORE = 1.959963984540054  # 95% confidence level
MIN_TRAFFIC_SESSIONS = 100

FILES_TO_VALIDATE = [
    "monthly_performance.csv",
    "visitor_type_performance.csv",
    "traffic_source_performance.csv",
    "weekend_performance.csv",
    "productrelated_level_performance.csv",
    "productrelated_duration_level_performance.csv",
    "bounce_rate_performance.csv",
    "exit_rate_performance.csv",
]


# ============================================================
# 2. WILSON CONFIDENCE INTERVAL
# ============================================================

def wilson_interval(purchases, sessions, z=Z_SCORE):
    """
    Calculate the 95% Wilson confidence interval
    for a binomial proportion.

    Returns lower and upper bounds as percentages.
    """

    if sessions <= 0:
        return np.nan, np.nan

    p = purchases / sessions

    denominator = 1 + (z ** 2 / sessions)

    center = (
        p + (z ** 2 / (2 * sessions))
    ) / denominator

    margin = (
        z
        * np.sqrt(
            (p * (1 - p) / sessions)
            + (z ** 2 / (4 * sessions ** 2))
        )
        / denominator
    )

    lower = max(0, center - margin)
    upper = min(1, center + margin)

    return lower * 100, upper * 100


# ============================================================
# 3. VALIDATE ONE SEGMENT FILE
# ============================================================

def validate_file(filename):

    input_path = INPUT_DIR / filename

    if not input_path.exists():
        print(f"SKIPPED - File not found: {filename}")
        return None

    df = pd.read_csv(input_path)

    required_columns = [
        "Sessions",
        "Purchases",
        "Conversion_Rate"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        print(f"SKIPPED - Missing columns in {filename}: {missing}")
        return None

    df["Sessions"] = pd.to_numeric(
        df["Sessions"], errors="coerce"
    )

    df["Purchases"] = pd.to_numeric(
        df["Purchases"], errors="coerce"
    )

    if df[["Sessions", "Purchases"]].isna().any().any():
        raise ValueError(
            f"Invalid Sessions/Purchases values in {filename}"
        )

    if (
        (df["Sessions"] <= 0).any()
        or (df["Purchases"] < 0).any()
        or (df["Purchases"] > df["Sessions"]).any()
    ):
        raise ValueError(
            f"Invalid counts detected in {filename}"
        )

    intervals = df.apply(
        lambda row: wilson_interval(
            row["Purchases"],
            row["Sessions"]
        ),
        axis=1
    )

    df["CI_95_Lower"] = [item[0] for item in intervals]
    df["CI_95_Upper"] = [item[1] for item in intervals]

    df["CI_95_Width"] = (
        df["CI_95_Upper"] - df["CI_95_Lower"]
    )

    # Traffic source eligibility is kept explicit.
    if "Traffic" in filename:
        df["Eligible_For_Comparison"] = (
            df["Sessions"] >= MIN_TRAFFIC_SESSIONS
        )

    output_path = OUTPUT_DIR / filename.replace(
        ".csv",
        "_validated.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"\n--- {filename} ---")

    display_columns = [
        col for col in df.columns
        if col in [
            "Month",
            "VisitorType",
            "TrafficType",
            "Day_Type",
            "ProductRelated_Level",
            "ProductRelated_Duration_Level",
            "BounceRates_Level",
            "ExitRates_Level",
            "Sessions",
            "Purchases",
            "Conversion_Rate",
            "CI_95_Lower",
            "CI_95_Upper",
            "CI_95_Width",
            "Eligible_For_Comparison"
        ]
    ]

    print(
        df[display_columns].to_string(index=False)
    )

    print(f"Saved: {output_path}")

    return df


# ============================================================
# 4. RUN VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("MARKETING STATISTICAL VALIDATION")
print("95% WILSON CONFIDENCE INTERVALS")
print("=" * 65)

validated_results = {}

for filename in FILES_TO_VALIDATE:

    result = validate_file(filename)

    if result is not None:
        validated_results[filename] = result


# ============================================================
# 5. FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("VALIDATION SUMMARY")
print("=" * 65)

print(f"Files validated: {len(validated_results)}")
print(f"Output directory: {OUTPUT_DIR}")

print(
    "\nInterpretation:"
    "\n- CI_95_Lower: Lower confidence bound."
    "\n- CI_95_Upper: Upper confidence bound."
    "\n- CI_95_Width: Width of the confidence interval."
    "\n- Narrower intervals generally indicate more precise estimates."
)

print(
    "\nImportant:"
    "\nConfidence intervals quantify uncertainty in each segment's"
    "\nconversion rate. Overlapping intervals alone do not establish"
    "\nthat two segments are statistically equal or different."
)

print("\nStatistical validation completed successfully.")
