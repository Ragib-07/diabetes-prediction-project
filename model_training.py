"""
model_training.py

Train, tune, compare, and save a diabetes prediction model.

Main improvements over the beginner version:
- Uses cross-validation on the training set for model selection.
- Uses GridSearchCV/RandomizedSearchCV for hyperparameter tuning.
- Keeps the test set untouched until the final evaluation.
- Treats impossible zero medical values as missing values.
- Saves the full best pipeline, metrics, and plots.
- Optimizes by accuracy by default, while still reporting precision, recall, F1, and ROC-AUC.

Run:
    python model_training.py

Optional:
    python model_training.py --download
    python model_training.py --scoring accuracy
    python model_training.py --scoring f1
    python model_training.py --n-iter 40
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import requests
except ImportError:
    requests = None


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetes.csv"
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"
SCREENSHOT_DIR = BASE_DIR / "screenshots"
METRICS_PATH = BASE_DIR / "model_metrics.json"

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

# In the Pima-style diabetes dataset, zero is not medically valid for these columns.
# We convert those zeros to NaN and let the imputer handle them.
ZERO_AS_MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

RANDOM_STATE = 42


@dataclass
class SearchResult:
    name: str
    best_estimator: Pipeline
    best_params: Dict[str, Any]
    best_cv_score: float
    cv_std: float
    search_type: str
    test_metrics: Dict[str, Any]
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]


def download_dataset(output_path: Path = DATASET_PATH) -> None:
    """Download the diabetes CSV if internet access is available."""
    if requests is None:
        raise ImportError("The requests package is required to download the dataset.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()
    output_path.write_text(response.text, encoding="utf-8")
    print(f"Dataset downloaded to: {output_path}")


def create_demo_dataset(output_path: Path = DATASET_PATH, rows: int = 768, random_state: int = RANDOM_STATE) -> None:
    """
    Create a demo dataset with the same columns as the Pima dataset.
    Prefer the real dataset for real project results.
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

    zero_rates = {"Glucose": 0.01, "BloodPressure": 0.03, "SkinThickness": 0.18, "Insulin": 0.34, "BMI": 0.02}
    for column, rate in zero_rates.items():
        df.loc[rng.random(rows) < rate, column] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Demo dataset created at: {output_path}")


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load the CSV and verify required columns."""
    if not dataset_path.exists():
        print("Dataset not found. Trying to download the public CSV...")
        try:
            download_dataset(dataset_path)
        except Exception as error:
            print(f"Download failed: {error}")
            print("Creating a demo dataset instead. Replace it with the real dataset for final use.")
            create_demo_dataset(dataset_path)

    df = pd.read_csv(dataset_path)
    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")

    return df


def clean_dataset(df: pd.DataFrame, treat_zero_as_missing: bool = True) -> pd.DataFrame:
    """Clean numeric columns, remove duplicates, and optionally convert impossible zeros to missing values."""
    df = df.copy()
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]]

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    invalid_targets = sorted(set(df[TARGET_COLUMN].unique()) - {0, 1})
    if invalid_targets:
        raise ValueError(f"Target column must contain only 0 and 1. Found: {invalid_targets}")

    if treat_zero_as_missing:
        for column in ZERO_AS_MISSING_COLUMNS:
            df[column] = df[column].replace(0, np.nan)

    return df


def make_preprocessors() -> Tuple[ColumnTransformer, ColumnTransformer]:
    """Create preprocessors for scaled models and tree models."""
    # add_indicator=True gives models extra columns saying whether a value was originally missing.
    # This can improve performance because missingness itself may contain useful signal.
    scaled_numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    unscaled_numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )

    scaled_preprocessor = ColumnTransformer(
        transformers=[("num", scaled_numeric, FEATURE_COLUMNS)],
        remainder="drop",
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[("num", unscaled_numeric, FEATURE_COLUMNS)],
        remainder="drop",
    )

    return scaled_preprocessor, tree_preprocessor


def build_model_search_spaces(random_state: int = RANDOM_STATE) -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]]:
    """Create candidate models and hyperparameter search spaces."""
    scaled_preprocessor, tree_preprocessor = make_preprocessors()

    return {
        "Logistic Regression": (
            Pipeline(
                steps=[
                    ("preprocessor", scaled_preprocessor),
                    ("model", LogisticRegression(max_iter=3000, random_state=random_state, solver="liblinear")),
                ]
            ),
            {
                "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                "model__penalty": ["l1", "l2"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            Pipeline(
                steps=[
                    ("preprocessor", tree_preprocessor),
                    ("model", RandomForestClassifier(random_state=random_state, n_jobs=-1)),
                ]
            ),
            {
                "model__n_estimators": [200, 400, 700],
                "model__max_depth": [3, 5, 7, 10, None],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", None],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Extra Trees": (
            Pipeline(
                steps=[
                    ("preprocessor", tree_preprocessor),
                    ("model", ExtraTreesClassifier(random_state=random_state, n_jobs=-1)),
                ]
            ),
            {
                "model__n_estimators": [200, 400, 700],
                "model__max_depth": [3, 5, 7, 10, None],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", None],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Gradient Boosting": (
            Pipeline(
                steps=[
                    ("preprocessor", tree_preprocessor),
                    ("model", GradientBoostingClassifier(random_state=random_state)),
                ]
            ),
            {
                "model__n_estimators": [50, 100, 150, 250],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__min_samples_leaf": [1, 3, 5, 10],
            },
        ),
        "Hist Gradient Boosting": (
            Pipeline(
                steps=[
                    ("preprocessor", tree_preprocessor),
                    ("model", HistGradientBoostingClassifier(random_state=random_state)),
                ]
            ),
            {
                "model__max_iter": [100, 200, 300],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__max_leaf_nodes": [7, 15, 31],
                "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
            },
        ),
        "Support Vector Machine": (
            Pipeline(
                steps=[
                    ("preprocessor", scaled_preprocessor),
                    ("model", SVC(probability=True, random_state=random_state)),
                ]
            ),
            {
                "model__C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
                "model__gamma": ["scale", 0.01, 0.03, 0.1, 0.3],
                "model__kernel": ["rbf"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "K-Nearest Neighbors": (
            Pipeline(
                steps=[
                    ("preprocessor", scaled_preprocessor),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            {
                "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
        "Gaussian Naive Bayes": (
            Pipeline(
                steps=[
                    ("preprocessor", tree_preprocessor),
                    ("model", GaussianNB()),
                ]
            ),
            {
                "model__var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
            },
        ),
    }


def count_grid_combinations(param_grid: Dict[str, Iterable[Any]]) -> int:
    """Count parameter combinations when all values are finite lists."""
    total = 1
    for values in param_grid.values():
        total *= len(list(values))
    return total


def predict_scores(estimator: Pipeline, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Return probability-like scores for ROC-AUC when available."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]

    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)

    return None


def evaluate_on_test(estimator: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict[str, Any], List[List[int]], Dict[str, Any]]:
    """Evaluate the fitted estimator on the untouched test set."""
    y_pred = estimator.predict(X_test)
    y_score = predict_scores(estimator, X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": None,
    }

    if y_score is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_test, y_score)
        except ValueError:
            metrics["ROC-AUC"] = None

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return metrics, cm, report


def tune_and_evaluate_models(
    model_spaces: Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scoring: str,
    cv_splits: int,
    n_iter: int,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, SearchResult]]:
    """Tune each model using training-only CV, then evaluate each best model once on the test set."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    results: Dict[str, SearchResult] = {}
    rows = []

    for name, (pipeline, param_grid) in model_spaces.items():
        combinations = count_grid_combinations(param_grid)

        if combinations <= n_iter:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            search_type = "GridSearchCV"
        else:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                refit=True,
                random_state=random_state,
            )
            search_type = "RandomizedSearchCV"

        print(f"\nTuning {name} with {search_type}...")
        search.fit(X_train, y_train)

        test_metrics, cm, report = evaluate_on_test(search.best_estimator_, X_test, y_test)

        cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
        best_cv_score = float(search.best_score_)

        result = SearchResult(
            name=name,
            best_estimator=search.best_estimator_,
            best_params=dict(search.best_params_),
            best_cv_score=best_cv_score,
            cv_std=cv_std,
            search_type=search_type,
            test_metrics=test_metrics,
            confusion_matrix=cm,
            classification_report=report,
        )
        results[name] = result

        row = {
            "Model": name,
            "Search": search_type,
            "Best CV Score": best_cv_score,
            "CV Std": cv_std,
            **test_metrics,
        }
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values(
        by=["Best CV Score", "Accuracy", "F1-Score"],
        ascending=False,
    )

    return results_df, results


def save_plots(results_df: pd.DataFrame, best_model_name: str, best_cm: List[List[int]], df: pd.DataFrame) -> None:
    """Save model comparison, confusion matrix, and correlation charts."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    score_columns = ["Accuracy", "Precision", "Recall", "F1-Score"]
    plot_df = results_df.set_index("Model")[score_columns]

    ax = plot_df.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Test Set Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / "model_comparison.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    cm_array = np.array(best_cm)
    image = ax.imshow(cm_array)
    ax.set_title(f"Confusion Matrix - {best_model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["No Diabetes", "Diabetes"])
    ax.set_yticks([0, 1], labels=["No Diabetes", "Diabetes"])

    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(j, i, str(cm_array[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / "confusion_matrix_best_model.png", dpi=160)
    plt.close()

    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr.values)
    ax.set_title("Feature Correlation Heatmap")
    ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), labels=corr.index)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()


def save_artifacts(
    best_result: SearchResult,
    all_results: Dict[str, SearchResult],
    results_df: pd.DataFrame,
    dataset_path: Path,
    model_output_path: Path,
    scoring: str,
    cv_splits: int,
    test_size: float,
    train_rows: int,
    test_rows: int,
) -> None:
    """Save the best model pipeline and metadata."""
    artifact = {
        "model": best_result.best_estimator,
        "model_name": best_result.name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "primary_scoring": scoring,
        "cv_splits": cv_splits,
        "test_size": test_size,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "best_params": best_result.best_params,
        "best_cv_score": best_result.best_cv_score,
        "test_metrics": best_result.test_metrics,
        "confusion_matrix": best_result.confusion_matrix,
        "classification_report": best_result.classification_report,
        "all_model_results": results_df.to_dict(orient="records"),
        "all_best_params": {name: result.best_params for name, result in all_results.items()},
        "dataset_path": str(dataset_path),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Prediction support only. This is not a medical diagnosis system.",
    }

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_output_path)

    json_safe_artifact = {key: value for key, value in artifact.items() if key != "model"}
    METRICS_PATH.write_text(json.dumps(json_safe_artifact, indent=4), encoding="utf-8")


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a short dataset summary."""
    print("\nDataset summary")
    print("-" * 60)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nClass balance:")
    print(df[TARGET_COLUMN].value_counts(normalize=True).rename("ratio"))
    print("\nMissing values after cleaning:")
    print(df[FEATURE_COLUMNS].isna().sum())


def train(
    dataset_path: Path = DATASET_PATH,
    model_output_path: Path = MODEL_PATH,
    force_download: bool = False,
    scoring: str = "accuracy",
    cv_splits: int = 5,
    n_iter: int = 40,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    treat_zero_as_missing: bool = True,
    no_plots: bool = False,
) -> None:
    """Main training workflow."""
    if force_download:
        download_dataset(dataset_path)

    raw_df = load_dataset(dataset_path)
    cleaned_df = clean_dataset(raw_df, treat_zero_as_missing=treat_zero_as_missing)
    print_dataset_summary(cleaned_df)

    X = cleaned_df[FEATURE_COLUMNS]
    y = cleaned_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model_spaces = build_model_search_spaces(random_state=random_state)
    results_df, all_results = tune_and_evaluate_models(
        model_spaces=model_spaces,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scoring=scoring,
        cv_splits=cv_splits,
        n_iter=n_iter,
        random_state=random_state,
    )

    best_model_name = str(results_df.iloc[0]["Model"])
    best_result = all_results[best_model_name]

    if not no_plots:
        save_plots(results_df, best_model_name, best_result.confusion_matrix, cleaned_df)

    save_artifacts(
        best_result=best_result,
        all_results=all_results,
        results_df=results_df,
        dataset_path=dataset_path,
        model_output_path=model_output_path,
        scoring=scoring,
        cv_splits=cv_splits,
        test_size=test_size,
        train_rows=len(X_train),
        test_rows=len(X_test),
    )

    print("\nTraining completed successfully!")
    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    print(f"\nBest model by CV {scoring}: {best_model_name}")
    print("Best parameters:")
    print(json.dumps(best_result.best_params, indent=4))

    print("\nFinal test metrics for best model:")
    for metric, value in best_result.test_metrics.items():
        print(f"{metric}: {value}")

    print("\nConfusion matrix for best model:")
    print(np.array(best_result.confusion_matrix))

    print(f"\nSaved model artifact to: {model_output_path}")
    print(f"Saved metrics JSON to: {METRICS_PATH}")
    if not no_plots:
        print(f"Saved charts to: {SCREENSHOT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and tune diabetes prediction models.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Path to diabetes.csv")
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH, help="Where to save diabetes_model.pkl")
    parser.add_argument("--download", action="store_true", help="Download the public diabetes CSV before training")
    parser.add_argument(
        "--scoring",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1", "recall", "precision", "roc_auc"],
        help="Primary metric used during cross-validation model selection",
    )
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--n-iter", type=int, default=40, help="Max random search iterations per model")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data reserved for final testing")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed for reproducibility")
    parser.add_argument(
        "--keep-zeros",
        action="store_true",
        help="Do not convert medically impossible zero values to missing values",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip saving charts")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_path=args.dataset,
        model_output_path=args.model_output,
        force_download=args.download,
        scoring=args.scoring,
        cv_splits=args.cv,
        n_iter=args.n_iter,
        test_size=args.test_size,
        random_state=args.random_state,
        treat_zero_as_missing=not args.keep_zeros,
        no_plots=args.no_plots,
    )