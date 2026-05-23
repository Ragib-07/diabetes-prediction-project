"""
model_training.py

Train and compare machine learning models for diabetes prediction.
The script saves the best model pipeline as diabetes_model.pkl.

Run:
    python model_training.py

Optional:
    python model_training.py --download
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    import requests
except ImportError:  # requests is listed in requirements.txt, but keep this beginner-safe.
    requests = None


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetes.csv"
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"
SCREENSHOT_DIR = BASE_DIR / "screenshots"

# Public GitHub copy of the Pima-style diabetes CSV used by many tutorials.
DATASET_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COLUMN = "Outcome"

# In this dataset, zeros in these medical measurements usually represent missing values.
ZERO_AS_MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def download_dataset(output_path: Path = DATASET_PATH) -> None:
    """Download the diabetes CSV if internet access is available."""
    if requests is None:
        raise ImportError("The requests package is required to download the dataset.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()
    output_path.write_text(response.text, encoding="utf-8")
    print(f"Dataset downloaded to: {output_path}")


def create_demo_dataset(output_path: Path = DATASET_PATH, rows: int = 768, random_state: int = 42) -> None:
    """
    Create a demo dataset with the same columns as the Pima dataset.
    This fallback keeps the project runnable without internet access.
    For final submission, prefer the real Pima dataset.
    """
    rng = np.random.default_rng(random_state)
    pregnancies = np.clip(rng.poisson(3.8, rows), 0, 17)
    age = np.clip(rng.normal(34, 12, rows).round(), 21, 81).astype(int)
    bmi = np.clip(rng.normal(31.8, 7.2, rows), 16, 60).round(1)
    glucose = np.clip(rng.normal(121, 31, rows), 45, 199).round().astype(int)
    blood_pressure = np.clip(rng.normal(72, 12, rows), 38, 122).round().astype(int)
    skin_thickness = np.clip(rng.normal(25, 10, rows), 7, 60).round().astype(int)
    insulin = np.clip(rng.lognormal(mean=4.55, sigma=0.72, size=rows), 15, 650).round().astype(int)
    pedigree = np.clip(rng.gamma(2.0, 0.22, rows), 0.08, 2.5).round(3)

    risk_score = (
        -8.25
        + 0.035 * glucose
        + 0.065 * bmi
        + 0.025 * age
        + 0.13 * pregnancies
        + 0.60 * pedigree
        + rng.normal(0, 0.85, rows)
    )
    probability = 1 / (1 + np.exp(-risk_score))
    outcome = (probability > rng.random(rows)).astype(int)

    df = pd.DataFrame(
        {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age,
            "Outcome": outcome,
        }
    )

    # Add a few zero placeholders to imitate missing medical measurements.
    zero_rates = {"Glucose": 0.01, "BloodPressure": 0.03, "SkinThickness": 0.18, "Insulin": 0.34, "BMI": 0.02}
    for column, rate in zero_rates.items():
        df.loc[rng.random(rows) < rate, column] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Demo dataset created at: {output_path}")


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load CSV and verify that all required columns exist."""
    if not dataset_path.exists():
        print("Dataset not found. Trying to download the public CSV...")
        try:
            download_dataset(dataset_path)
        except Exception as error:
            print(f"Download failed: {error}")
            print("Creating a demo dataset instead. Replace it with the real Pima dataset for final use.")
            create_demo_dataset(dataset_path)

    df = pd.read_csv(dataset_path)
    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean numeric columns, remove duplicates, and mark impossible zero values as missing."""
    df = df.copy()
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]]

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    for column in ZERO_AS_MISSING_COLUMNS:
        df[column] = df[column].replace(0, np.nan)

    return df


def build_models() -> Dict[str, Pipeline]:
    """Create pipelines for each model. Pipelines prevent preprocessing mistakes."""
    numeric_preprocessor_with_scaling = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    numeric_preprocessor_no_scaling = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    scaled_preprocessor = ColumnTransformer(
        transformers=[("num", numeric_preprocessor_with_scaling, FEATURE_COLUMNS)]
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[("num", numeric_preprocessor_no_scaling, FEATURE_COLUMNS)]
    )

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", scaled_preprocessor),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
            ]
        ),
        "Support Vector Machine": Pipeline(
            steps=[
                ("preprocessor", scaled_preprocessor),
                ("model", SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced")),
            ]
        ),
        "K-Nearest Neighbors": Pipeline(
            steps=[
                ("preprocessor", scaled_preprocessor),
                ("model", KNeighborsClassifier(n_neighbors=9)),
            ]
        ),
    }
    return models


def evaluate_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """Train every model, evaluate it, and choose the best one by F1-score then accuracy."""
    results = []
    trained_models: Dict[str, Pipeline] = {}
    confusion_matrices: Dict[str, list] = {}
    reports: Dict[str, dict] = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        result = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        }
        results.append(result)
        trained_models[name] = pipeline
        confusion_matrices[name] = confusion_matrix(y_test, y_pred).tolist()
        reports[name] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    results_df = pd.DataFrame(results).sort_values(by=["F1-Score", "Accuracy"], ascending=False)
    best_model_name = results_df.iloc[0]["Model"]

    details = {
        "trained_models": trained_models,
        "confusion_matrices": confusion_matrices,
        "classification_reports": reports,
    }
    return results_df, details, best_model_name


def save_plots(df: pd.DataFrame, results_df: pd.DataFrame, best_model_name: str, best_cm: list) -> None:
    """Save useful charts for the README/report/screenshots folder."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # Model comparison chart
    plt.figure(figsize=(10, 6))
    plot_df = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]]
    plot_df.plot(kind="bar", figsize=(11, 6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / "model_comparison.png", dpi=160)
    plt.close()

    # Confusion matrix chart for the best model
    plt.figure(figsize=(5, 4))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / "confusion_matrix_best_model.png", dpi=160)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()


def save_model_artifact(
    best_model: Pipeline,
    best_model_name: str,
    results_df: pd.DataFrame,
    details: Dict[str, Any],
    dataset_path: Path,
    output_path: Path,
) -> None:
    """Save the selected model and metadata in one Joblib file."""
    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "metrics": results_df.to_dict(orient="records"),
        "confusion_matrices": details["confusion_matrices"],
        "classification_reports": details["classification_reports"],
        "dataset_path": str(dataset_path),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "This is a prediction support tool, not a medical diagnosis system.",
    }
    joblib.dump(artifact, output_path)

    # A JSON metrics file is useful for quick review on GitHub.
    metrics_json_path = output_path.with_name("model_metrics.json")
    metrics_json_path.write_text(json.dumps(artifact["metrics"], indent=4), encoding="utf-8")


def train(dataset_path: Path = DATASET_PATH, model_output_path: Path = MODEL_PATH, force_download: bool = False) -> None:
    """Main training workflow."""
    if force_download:
        download_dataset(dataset_path)

    raw_df = load_dataset(dataset_path)
    cleaned_df = clean_dataset(raw_df)

    X = cleaned_df[FEATURE_COLUMNS]
    y = cleaned_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = build_models()
    results_df, details, best_model_name = evaluate_models(models, X_train, X_test, y_train, y_test)
    best_model = details["trained_models"][best_model_name]
    best_cm = details["confusion_matrices"][best_model_name]

    save_plots(cleaned_df, results_df, best_model_name, best_cm)
    save_model_artifact(best_model, best_model_name, results_df, details, dataset_path, model_output_path)

    print("\nTraining completed successfully!")
    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(f"Saved model artifact to: {model_output_path}")
    print(f"Saved charts to: {SCREENSHOT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train diabetes prediction models.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Path to diabetes.csv")
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH, help="Where to save diabetes_model.pkl")
    parser.add_argument("--download", action="store_true", help="Download the public diabetes CSV before training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(dataset_path=args.dataset, model_output_path=args.model_output, force_download=args.download)
