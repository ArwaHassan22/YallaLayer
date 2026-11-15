
from pathlib import Path
import json
from datetime import date

import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import StratifiedGroupKFold


# Project paths
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
SPLITS_DIR = ROOT / "data" / "splits"
REPORTS_DIR = ROOT / "reports"

for folder in [RAW_DIR, SPLITS_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def get_data():

    print("Fetching dataset from UCI...")

    # Official UCI import code
    dataset = fetch_ucirepo(id=468)

    # Get features and target separately
    X = dataset.data.features
    y = dataset.data.targets

    # Combine them into one DataFrame
    df = pd.concat([X, y], axis=1)

    print("Dataset downloaded successfully.")

    return df, dataset


def validate_data(df):

    print("\n  DATA VALIDATION  ")

    print("Shape:", df.shape)

    if df.shape != (12330, 18):
        raise ValueError(f"Unexpected shape: {df.shape}")

    if df.isnull().sum().sum() != 0:
        raise ValueError("Missing values detected.")

    if set(df["Revenue"].unique()) != {True, False}:
        raise ValueError("Unexpected Revenue values.")

    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

    print("\nTarget distribution:")
    print(df["Revenue"].value_counts())

    print("\nData types:")
    print(df.dtypes)

    print("\nValidation passed.")


def save_raw_data(df):

    raw_path = RAW_DIR / "online_shoppers_intention.csv"

    df.to_csv(raw_path, index=False)

    print("\nRaw dataset saved:", raw_path)

    return raw_path



def create_train_test_split(df):

    print("\n DATA SPLITTING ")

    # Assign the same group ID to completely identical rows.
    groups = pd.factorize(
        pd.MultiIndex.from_frame(df)
    )[0]

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_indices, test_indices = next(
        splitter.split(
            df,
            df["Revenue"],
            groups=groups
        )
    )

    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    # Verify that identical rows cannot cross the split.
    train_groups = set(groups[train_indices])
    test_groups = set(groups[test_indices])

    if train_groups.intersection(test_groups):
        raise ValueError(
            "Duplicate groups found across training and test sets."
        )

    train_path = SPLITS_DIR / "train.csv"
    test_path = SPLITS_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Training rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("Split method: StratifiedGroupKFold")
    print("Duplicate groups overlap: 0")

    print(
        "\nTest set saved. Do not inspect it during development."
    )

    return len(train_df), len(test_df)




def save_report(df, train_rows, test_rows, dataset):

    report = {
        "dataset_name": dataset.metadata.name,
        "dataset_id": dataset.metadata.uci_id,
        "download_date": date.today().isoformat(),

        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),

        "target_distribution": {
            str(key): int(value)
            for key, value in df["Revenue"].value_counts().items()
        },

        "split": {
            "train_rows": train_rows,
            "test_rows": test_rows,
            "test_size": 0.20,
            "split_method": "StratifiedGroupKFold",
            "n_splits": 5,
            "grouped_by": "Exact duplicate rows",
            "stratified_by": "Revenue",
            "random_state": 42
        },

        "license": "CC BY 4.0",
        "doi": "10.24432/C5F88Q"
    }

    report_path = REPORTS_DIR / "data_validation.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\nValidation report saved:", report_path)


def main():

    df, dataset = get_data()

    validate_data(df)

    save_raw_data(df)

    train_rows, test_rows = create_train_test_split(df)

    save_report(df, train_rows, test_rows, dataset)

    print("\nSTEP 2 DATA ACQUISITION COMPLETED!")


if __name__ == "__main__":
    main()
