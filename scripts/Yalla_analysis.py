
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "marketing"
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Revenue"

MIN_TRAFFIC_SESSIONS = 100

sns.set_theme(style="whitegrid", context="notebook")

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def save_figure(filename):
    """Save the current matplotlib figure."""
    output_path = FIGURES_DIR / filename

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved: {output_path}")


def analyze_segments(data, group_column):
    """
    Calculate marketing performance metrics for each segment.

    Metrics:
    - Sessions
    - Purchases
    - Conversion Rate
    - Session Share
    - Purchase Contribution
    """

    total_sessions = len(data)
    total_purchases = data[TARGET].sum()

    result = (
        data.groupby(group_column, observed=False)
        .agg(
            Sessions=(TARGET, "count"),
            Purchases=(TARGET, "sum")
        )
        .reset_index()
    )

    result["Conversion_Rate"] = (
        result["Purchases"] / result["Sessions"] * 100
    )

    result["Session_Share"] = (
        result["Sessions"] / total_sessions * 100
    )

    if total_purchases > 0:
        result["Purchase_Contribution"] = (
            result["Purchases"] / total_purchases * 100
        )
    else:
        result["Purchase_Contribution"] = 0.0

    result = result.sort_values(
        by="Conversion_Rate",
        ascending=False
    ).reset_index(drop=True)

    return result


def create_quantile_groups(data, column):
    """
    Create quartile-based groups without splitting tied values.

    If duplicate quantile boundaries exist, they are dropped.
    Therefore, the function may return fewer than four groups.

    This is intentional: preserving equal values in the same
    group is more important than forcing exactly four groups.
    """

    values = data[column]

    # Handle constant or nearly constant columns.
    if values.nunique(dropna=True) <= 1:
        return pd.Series(
            "No Variation",
            index=data.index,
            name=f"{column}_Level"
        )

    try:
        _, bin_edges = pd.qcut(
            values,
            q=4,
            duplicates="drop",
            retbins=True
        )

        number_of_bins = len(bin_edges) - 1

        if number_of_bins < 2:
            return pd.Series(
                "No Variation",
                index=data.index,
                name=f"{column}_Level"
            )

        labels = [
            "Low",
            "Medium",
            "High",
            "Very High"
        ][:number_of_bins]

        grouped_values = pd.cut(
            values,
            bins=bin_edges,
            labels=labels,
            include_lowest=True
        )

        return grouped_values

    except ValueError:
        return pd.Series(
            "No Variation",
            index=data.index,
            name=f"{column}_Level"
        )


def plot_bar_chart(
    data,
    x_column,
    y_column,
    title,
    xlabel,
    ylabel,
    filename,
    rotate_labels=False
):
    """Generate and save a bar chart."""

    plt.figure(figsize=(11, 6))

    sns.barplot(
        data=data,
        x=x_column,
        y=y_column,
        hue=x_column,
        legend=False
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if rotate_labels:
        plt.xticks(rotation=45, ha="right")

    save_figure(filename)


def save_segment_analysis(data, group_column, filename):
    """Run segment analysis and save its CSV output."""

    result = analyze_segments(data, group_column)

    output_path = OUTPUT_DIR / filename
    result.to_csv(output_path, index=False)

    print(f"\n--- {group_column} Analysis ---")
    print(result.to_string(index=False))

    return result


# ============================================================
# 3. LOAD AND VALIDATE DATA
# ============================================================

print("\n" + "=" * 65)
print("MARKETING INTELLIGENCE ANALYSIS")
print("=" * 65)

if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Training data not found: {TRAIN_PATH}"
    )

data = pd.read_csv(TRAIN_PATH)

required_columns = [
    TARGET,
    "Month",
    "VisitorType",
    "TrafficType",
    "Weekend",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates"
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if data[TARGET].isna().any():
    raise ValueError(
        "Target column contains missing values."
    )

print("\n========== DATA VALIDATION ==========")
print(f"Dataset shape: {data.shape}")
print(f"Total sessions: {len(data)}")
print(f"Missing values: {data.isnull().sum().sum()}")
print(f"Purchases: {int(data[TARGET].sum())}")

data[TARGET] = data[TARGET].astype(int)


# ============================================================
# 4. OVERALL MARKETING PERFORMANCE
# ============================================================

total_sessions = len(data)
total_purchases = int(data[TARGET].sum())
total_non_purchases = total_sessions - total_purchases

overall_conversion_rate = (
    total_purchases / total_sessions * 100
    if total_sessions > 0 else 0
)

print("\n========== OVERALL PERFORMANCE ==========")
print(f"Total Sessions: {total_sessions}")
print(f"Purchases: {total_purchases}")
print(f"Non-Purchase Sessions: {total_non_purchases}")
print(f"Overall Conversion Rate: {overall_conversion_rate:.2f}%")


# ============================================================
# 5. MONTHLY PERFORMANCE ANALYSIS
# ============================================================

monthly = save_segment_analysis(
    data,
    "Month",
    "monthly_performance.csv"
)

plot_bar_chart(
    monthly,
    "Month",
    "Conversion_Rate",
    "Monthly Conversion Rate",
    "Month",
    "Conversion Rate (%)",
    "monthly_conversion_rate.png",
    rotate_labels=True
)


# ============================================================
# 6. VISITOR TYPE ANALYSIS
# ============================================================

visitor = save_segment_analysis(
    data,
    "VisitorType",
    "visitor_type_performance.csv"
)

plot_bar_chart(
    visitor,
    "VisitorType",
    "Conversion_Rate",
    "Conversion Rate by Visitor Type",
    "Visitor Type",
    "Conversion Rate (%)",
    "visitor_type_conversion_rate.png"
)


# ============================================================
# 7. TRAFFIC SOURCE ANALYSIS
# ============================================================

traffic = analyze_segments(
    data,
    "TrafficType"
)

# TrafficType represents dataset codes.
# Do not interpret these codes as platform names without
# a reliable data dictionary.

traffic["TrafficType"] = traffic["TrafficType"].astype(int)

# A source needs at least 100 sessions to qualify
# for the comparative ranking.

traffic["Eligible_For_Comparison"] = (
    traffic["Sessions"] >= MIN_TRAFFIC_SESSIONS
)

traffic.to_csv(
    OUTPUT_DIR / "traffic_source_performance.csv",
    index=False
)

qualified_traffic = (
    traffic[
        traffic["Eligible_For_Comparison"]
    ]
    .sort_values(
        by="Conversion_Rate",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\n--- All Traffic Sources ---")
print(traffic.to_string(index=False))

print(
    f"\n--- Qualified Traffic Sources "
    f"(Minimum {MIN_TRAFFIC_SESSIONS} Sessions) ---"
)

if qualified_traffic.empty:
    print("No traffic sources meet the minimum session requirement.")

else:
    print(
        qualified_traffic[
            [
                "TrafficType",
                "Sessions",
                "Purchases",
                "Conversion_Rate",
                "Purchase_Contribution"
            ]
        ].to_string(index=False)
    )

if not qualified_traffic.empty:
    plot_bar_chart(
        qualified_traffic,
        "TrafficType",
        "Conversion_Rate",
        "Qualified Traffic Sources - Conversion Rate",
        "Traffic Type Code",
        "Conversion Rate (%)",
        "traffic_source_conversion_rate.png"
    )


# ============================================================
# 8. WEEKEND VS WEEKDAY ANALYSIS
# ============================================================

data["Day_Type"] = np.where(
    data["Weekend"].astype(bool),
    "Weekend",
    "Weekday"
)

weekend = save_segment_analysis(
    data,
    "Day_Type",
    "weekend_performance.csv"
)

plot_bar_chart(
    weekend,
    "Day_Type",
    "Conversion_Rate",
    "Weekend vs Weekday Conversion Rate",
    "Day Type",
    "Conversion Rate (%)",
    "weekend_conversion_rate.png"
)


# ============================================================
# 9. PRODUCT ENGAGEMENT ANALYSIS
# ============================================================

engagement_columns = [
    "ProductRelated",
    "ProductRelated_Duration"
]

engagement_results = {}

for column in engagement_columns:

    level_column = f"{column}_Level"

    data[level_column] = create_quantile_groups(
        data,
        column
    )

    result = save_segment_analysis(
        data,
        level_column,
        f"{column.lower()}_level_performance.csv"
    )

    engagement_results[column] = result

    plot_bar_chart(
        result,
        level_column,
        "Conversion_Rate",
        f"Conversion Rate by {column} Level",
        f"{column} Level",
        "Conversion Rate (%)",
        f"{column.lower()}_level_conversion_rate.png"
    )


# ============================================================
# 10. BOUNCE RATE ANALYSIS
# ============================================================

data["BounceRates_Level"] = create_quantile_groups(
    data,
    "BounceRates"
)

bounce = save_segment_analysis(
    data,
    "BounceRates_Level",
    "bounce_rate_performance.csv"
)

plot_bar_chart(
    bounce,
    "BounceRates_Level",
    "Conversion_Rate",
    "Conversion Rate by Bounce Rate Level",
    "Bounce Rate Level",
    "Conversion Rate (%)",
    "bounce_rate_conversion_rate.png"
)


# ============================================================
# 11. EXIT RATE ANALYSIS
# ============================================================

data["ExitRates_Level"] = create_quantile_groups(
    data,
    "ExitRates"
)

exit_analysis = save_segment_analysis(
    data,
    "ExitRates_Level",
    "exit_rate_performance.csv"
)

plot_bar_chart(
    exit_analysis,
    "ExitRates_Level",
    "Conversion_Rate",
    "Conversion Rate by Exit Rate Level",
    "Exit Rate Level",
    "Conversion Rate (%)",
    "exit_rate_conversion_rate.png"
)


# ============================================================
# 12. EXECUTIVE SUMMARY
# ============================================================

best_month = (
    str(monthly.iloc[0]["Month"])
    if not monthly.empty else None
)

best_visitor_type = (
    str(visitor.iloc[0]["VisitorType"])
    if not visitor.empty else None
)

best_traffic_type = (
    int(qualified_traffic.iloc[0]["TrafficType"])
    if not qualified_traffic.empty else None
)

best_day_type = (
    str(weekend.iloc[0]["Day_Type"])
    if not weekend.empty else None
)

summary = {
    "dataset": {
        "source": "UCI Online Shoppers Purchasing Intention",
        "analysis_split": "Training Set",
        "total_sessions": int(total_sessions),
        "total_purchases": int(total_purchases),
        "non_purchase_sessions": int(total_non_purchases)
    },

    "overall_performance": {
        "conversion_rate": round(
            float(overall_conversion_rate), 2
        )
    },

    "best_segments_by_conversion": {
        "best_month": best_month,
        "best_visitor_type": best_visitor_type,

        "best_traffic_type_by_conversion_min_100_sessions": (
            best_traffic_type
        ),

        "best_day_type": best_day_type
    },

    "traffic_comparison_rule": {
        "minimum_sessions": MIN_TRAFFIC_SESSIONS,
        "eligible_sources": int(len(qualified_traffic)),
        "ranking_metric": "Conversion_Rate"
    },

    "interpretation_notes": [
        "Each row represents a browsing session, not necessarily a unique customer.",
        "TrafficType values are dataset codes, not verified platform names.",
        "Conversion rates describe observed associations, not causal effects.",
        "Quantile groups preserve tied values and may contain fewer than four levels.",
        "The analysis uses the training split only."
    ]
}

summary_path = OUTPUT_DIR / "marketing_summary.json"

with open(
    summary_path,
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
# 13. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("MARKETING INTELLIGENCE EXECUTIVE SUMMARY")
print("=" * 65)

print(
    f"Overall Conversion Rate: "
    f"{overall_conversion_rate:.2f}%"
)

print(f"Best Month: {best_month}")
print(f"Best Visitor Type: {best_visitor_type}")
print(f"Best Qualified Traffic Type: {best_traffic_type}")
print(f"Best Day Type: {best_day_type}")

print("\n========== OUTPUT FILES ==========")

for file_path in sorted(OUTPUT_DIR.glob("*.csv")):
    print(file_path.name)

print("\nFigures Directory:")
print(FIGURES_DIR)

print("\nExecutive Summary:")
print(summary_path)

print("\nMarketing Intelligence Analysis completed successfully.")
