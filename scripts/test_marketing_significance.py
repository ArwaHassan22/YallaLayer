
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "reports" / "marketing" / "validation"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "marketing" / "statistical_tests"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
MIN_TRAFFIC_SESSIONS = 100

SEGMENTS = {
    "Month": (
        "monthly_performance_validated.csv",
        "Month"
    ),
    "VisitorType": (
        "visitor_type_performance_validated.csv",
        "VisitorType"
    ),
    "TrafficType": (
        "traffic_source_performance_validated.csv",
        "TrafficType"
    ),
    "DayType": (
        "weekend_performance_validated.csv",
        "Day_Type"
    ),
    "ProductRelated": (
        "productrelated_level_performance_validated.csv",
        "ProductRelated_Level"
    ),
    "ProductDuration": (
        "productrelated_duration_level_performance_validated.csv",
        "ProductRelated_Duration_Level"
    ),
    "BounceRates": (
        "bounce_rate_performance_validated.csv",
        "BounceRates_Level"
    ),
    "ExitRates": (
        "exit_rate_performance_validated.csv",
        "ExitRates_Level"
    )
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def holm_correction(p_values):
    """
    Apply Holm-Bonferroni correction.

    Returns adjusted p-values in their original order.
    Controls the family-wise error rate.
    """

    p_values = np.asarray(p_values, dtype=float)

    if len(p_values) == 0:
        return np.array([], dtype=float)

    if (
        not np.isfinite(p_values).all()
        or ((p_values < 0) | (p_values > 1)).any()
    ):
        raise ValueError("P-values must be finite and between 0 and 1.")

    n = len(p_values)

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.maximum.accumulate(
        (n - np.arange(n)) * sorted_p
    )

    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_sorted

    return adjusted


def cohen_h(p1, p2):
    """Calculate Cohen's h effect size for two proportions."""

    p1 = np.clip(p1, 0, 1)
    p2 = np.clip(p2, 0, 1)

    return 2 * (
        np.arcsin(np.sqrt(p1))
        - np.arcsin(np.sqrt(p2))
    )


# ============================================================
# 3. LOAD AND VALIDATE DATA
# ============================================================

def load_segment_data(segment_name, filename, group_column):

    path = INPUT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required validated file not found: {path}"
        )

    df = pd.read_csv(path)

    required = [
        group_column,
        "Sessions",
        "Purchases"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{filename} is missing columns: {missing}"
        )

    df = df[required].copy()

    df["Sessions"] = pd.to_numeric(
        df["Sessions"], errors="raise"
    )

    df["Purchases"] = pd.to_numeric(
        df["Purchases"], errors="raise"
    )

    if (
        df[["Sessions", "Purchases"]].isna().any().any()
        or not np.isfinite(
            df[["Sessions", "Purchases"]].to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError(
            f"Missing or non-finite counts in {filename}"
        )

    if (
        (df["Sessions"] <= 0).any()
        or (df["Purchases"] < 0).any()
        or (df["Purchases"] > df["Sessions"]).any()
        or (df["Sessions"] % 1 != 0).any()
        or (df["Purchases"] % 1 != 0).any()
    ):
        raise ValueError(
            f"Invalid session or purchase counts in {filename}"
        )

    if df[group_column].isna().any():
        raise ValueError(
            f"Missing group labels in {filename}"
        )

    # Exclude low-volume traffic sources.
    if segment_name == "TrafficType":
        df = df[
            df["Sessions"] >= MIN_TRAFFIC_SESSIONS
        ].copy()

    df = df.reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(
            f"Not enough groups to test: {segment_name}"
        )

    return df


# ============================================================
# 4. OMNIBUS CHI-SQUARE TEST
# ============================================================

def run_omnibus_test(df, segment_name):

    non_purchases = (
        df["Sessions"] - df["Purchases"]
    )

    contingency_table = np.column_stack([
        df["Purchases"].to_numpy(),
        non_purchases.to_numpy()
    ])

    chi2, p_value, degrees_freedom, expected = (
        chi2_contingency(contingency_table)
    )

    total_sessions = df["Sessions"].sum()

    minimum_expected = float(expected.min())

    denominator = (
        total_sessions
        * min(contingency_table.shape[0] - 1, 1)
    )

    cramers_v = (
        np.sqrt(chi2 / denominator)
        if denominator > 0 else np.nan
    )

    return {
        "Segment": segment_name,
        "Groups": len(df),
        "Total_Sessions": int(total_sessions),
        "Chi_Square": float(chi2),
        "Degrees_of_Freedom": int(degrees_freedom),
        "P_Value": float(p_value),
        "Cramers_V": float(cramers_v),
        "Minimum_Expected_Count": minimum_expected,
        "Expected_Counts_At_Least_5": bool(
            minimum_expected >= 5
        ),
        "Statistically_Significant": bool(
            p_value < ALPHA
        )
    }


# ============================================================
# 5. PAIRWISE TWO-PROPORTION Z-TEST
# ============================================================

def run_pairwise_tests(df, segment_name, group_column):

    results = []

    group_records = df.to_dict("records")

    for first, second in combinations(group_records, 2):

        n1 = int(first["Sessions"])
        n2 = int(second["Sessions"])

        x1 = int(first["Purchases"])
        x2 = int(second["Purchases"])

        p1 = x1 / n1
        p2 = x2 / n2

        pooled_p = (x1 + x2) / (n1 + n2)

        standard_error = np.sqrt(
            pooled_p
            * (1 - pooled_p)
            * (1 / n1 + 1 / n2)
        )

        if standard_error == 0:
            z_statistic = 0.0
            p_value = 1.0

        else:
            z_statistic = (
                (p1 - p2) / standard_error
            )

            p_value = 2 * norm.sf(
                abs(z_statistic)
            )

        difference = (p1 - p2) * 100

        results.append({
            "Segment": segment_name,
            "Group_1": str(first[group_column]),
            "Group_2": str(second[group_column]),
            "Sessions_1": n1,
            "Sessions_2": n2,
            "Purchases_1": x1,
            "Purchases_2": x2,
            "Conversion_Rate_1": p1 * 100,
            "Conversion_Rate_2": p2 * 100,
            "Difference_Percentage_Points": difference,
            "Z_Statistic": z_statistic,
            "P_Value": p_value,
            "Cohens_H": cohen_h(p1, p2)
        })

    result_df = pd.DataFrame(results)

    # Holm correction within each segment's pairwise family.
    result_df["Holm_Adjusted_P_Value"] = holm_correction(
        result_df["P_Value"].to_numpy()
    )

    result_df["Statistically_Significant"] = (
        result_df["Holm_Adjusted_P_Value"] < ALPHA
    )

    return result_df


# ============================================================
# 6. RUN ALL TESTS
# ============================================================

print("\n" + "=" * 70)
print("MARKETING HYPOTHESIS TESTING")
print("CHI-SQUARE + TWO-PROPORTION Z-TEST + HOLM CORRECTION")
print("=" * 70)

omnibus_results = []
pairwise_results = []

for segment_name, config in SEGMENTS.items():

    filename, group_column = config

    print(f"\n{'-' * 60}")
    print(f"Analyzing: {segment_name}")
    print("-" * 60)

    df = load_segment_data(
        segment_name,
        filename,
        group_column
    )

    omnibus = run_omnibus_test(
        df,
        segment_name
    )

    pairwise = run_pairwise_tests(
        df,
        segment_name,
        group_column
    )

    omnibus_results.append(omnibus)
    pairwise_results.append(pairwise)

    print(f"Chi-Square: {omnibus['Chi_Square']:.4f}")
    print(f"P-Value: {omnibus['P_Value']:.8e}")
    print(f"Cramer's V: {omnibus['Cramers_V']:.4f}")

    print(
        "Minimum Expected Count:",
        f"{omnibus['Minimum_Expected_Count']:.2f}"
    )

    print(
        "Significant before global correction:",
        omnibus["Statistically_Significant"]
    )

    print("\nPairwise Results:")

    print(
        pairwise[
            [
                "Group_1",
                "Group_2",
                "Difference_Percentage_Points",
                "P_Value",
                "Holm_Adjusted_P_Value",
                "Statistically_Significant"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 7. GLOBAL OMNIBUS HOLM CORRECTION
# ============================================================

omnibus_df = pd.DataFrame(omnibus_results)

omnibus_df["Holm_Adjusted_P_Value"] = holm_correction(
    omnibus_df["P_Value"].to_numpy()
)

omnibus_df["Statistically_Significant_After_Holm"] = (
    omnibus_df["Holm_Adjusted_P_Value"] < ALPHA
)


# ============================================================
# 8. COMBINE PAIRWISE RESULTS
# ============================================================

pairwise_df = pd.concat(
    pairwise_results,
    ignore_index=True
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

omnibus_path = OUTPUT_DIR / "omnibus_tests.csv"
pairwise_path = OUTPUT_DIR / "pairwise_tests.csv"

omnibus_df.to_csv(
    omnibus_path,
    index=False
)

pairwise_df.to_csv(
    pairwise_path,
    index=False
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OMNIBUS TEST SUMMARY AFTER GLOBAL HOLM CORRECTION")
print("=" * 70)

print(
    omnibus_df[
        [
            "Segment",
            "P_Value",
            "Holm_Adjusted_P_Value",
            "Cramers_V",
            "Statistically_Significant_After_Holm"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("PAIRWISE SIGNIFICANT COMPARISONS")
print("=" * 70)

significant_pairs = pairwise_df[
    pairwise_df["Statistically_Significant"]
].copy()

if significant_pairs.empty:
    print(
        "No significant pairwise differences "
        "after Holm correction."
    )

else:
    print(
        significant_pairs[
            [
                "Segment",
                "Group_1",
                "Group_2",
                "Difference_Percentage_Points",
                "Holm_Adjusted_P_Value",
                "Cohens_H"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 11. FINAL OUTPUT
# ============================================================

print("\n========== OUTPUT FILES ==========")
print(omnibus_path)
print(pairwise_path)

print("\nStatistical hypothesis testing completed successfully.")
