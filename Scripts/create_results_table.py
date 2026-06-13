from pathlib import Path
import pandas as pd

# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    PROJECT_ROOT
    / "report_assets"
    / "results"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "report_assets"
    / "tables"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# LOAD RESULTS
# =====================================================

rows = []

for attribute in [
    "glasses",
    "hat",
    "young"
]:

    metrics_file = (
        RESULTS_DIR
        / f"{attribute}_metrics.csv"
    )

    if not metrics_file.exists():
        continue

    df = pd.read_csv(
        metrics_file
    )

    row = df.iloc[0].to_dict()

    row["model"] = attribute

    rows.append(row)

# =====================================================
# CREATE COMPARISON TABLE
# =====================================================

results = pd.DataFrame(rows)

results = results[
    [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "auc",
        "tp",
        "tn",
        "fp",
        "fn"
    ]
]

results.to_csv(
    TABLE_DIR / "model_comparison.csv",
    index=False
)

print("\n===================================")
print("MODEL COMPARISON TABLE")
print("===================================")

print(results)

print(
    "\nSaved to report_assets/tables/model_comparison.csv"
)