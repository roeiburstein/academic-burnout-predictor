"""Model Training, Validation, Tuning, and Asset Export Pipeline.

This module loads the cleaned data, encodes categorical features, sets up
a rigorous ML pipeline with data scaling inside cross-validation loops,
tunes three classifiers (Logistic Regression, Decision Tree, and Random Forest),
computes permutation feature importances, generates diagnostic curves (ROC & PR),
and exports all model assets to JSON.
"""

from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants for paths
BASE_DIR = Path(__file__).parent.parent
CLEANED_DATA_PATH = BASE_DIR / "data" / "cleaned_burnout_data.csv"
MODEL_ASSETS_PATH = BASE_DIR / "web" / "model_assets.json"
PLOTS_DIR = BASE_DIR / "web" / "plots"


def load_cleaned_data(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads cleaned data and splits into features and binary target.

    Args:
        path: Path to the cleaned data CSV.

    Returns:
        A tuple containing the features DataFrame and target Series.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {path}. Please run eda.py first."
        )

    df = pd.read_csv(path)
    logger.info("Loaded cleaned dataset: %s rows, %s cols", df.shape[0], df.shape[1])

    # Predict binary high burnout
    df["target"] = (df["burnout_level"] == "High").astype(int)
    target = df["target"]

    # Drop non-predictors and targets
    features = df.drop(columns=["student_id", "burnout_level", "target"])
    return features, target


def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Encodes categorical features.

    Args:
        df: Raw feature DataFrame.

    Returns:
        A tuple of (encoded DataFrame, ordinal mappings dict).
    """
    df_encoded = df.copy()

    # Ordinal mappings
    ordinal_mappings = {
        "year": {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4},
        "stress_level": {"Low": 1, "Medium": 2, "High": 3},
        "sleep_quality": {"Poor": 1, "Average": 2, "Good": 3},
        "internet_quality": {"Poor": 1, "Average": 2, "Good": 3},
    }

    for col, mapping in ordinal_mappings.items():
        df_encoded[col] = df_encoded[col].map(mapping)

    # Nominal variables (One-Hot dummy encoding)
    nominal_cols = ["gender", "course"]
    df_encoded = pd.get_dummies(
        df_encoded, columns=nominal_cols, drop_first=True
    )

    # Convert all columns to float
    for col in df_encoded.columns:
        df_encoded[col] = df_encoded[col].astype(float)

    return df_encoded, ordinal_mappings


def run_cross_validation(
    pipeline: Pipeline, X: np.ndarray, y: np.ndarray, cv: int = 5
) -> Dict[str, float]:
    """Performs Stratified K-Fold Cross-Validation safely inside a Pipeline.

    Args:
        pipeline: Evaluated estimator Pipeline.
        X: Unscaled feature matrix.
        y: Target vector.
        cv: Number of cross-validation splits.

    Returns:
        Dictionary of mean and standard deviation of accuracy and F1 metrics.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    acc_scores = []
    f1_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # Clone and fit pipeline to prevent data leakage
        from sklearn.base import clone
        cloned_pipeline = clone(pipeline)
        cloned_pipeline.fit(X_tr, y_tr)
        preds = cloned_pipeline.predict(X_va)

        acc_scores.append(accuracy_score(y_va, preds))
        f1_scores.append(f1_score(y_va, preds, average="macro"))

    return {
        "cv_accuracy_mean": float(np.mean(acc_scores)),
        "cv_accuracy_std": float(np.std(acc_scores)),
        "cv_f1_macro_mean": float(np.mean(f1_scores)),
        "cv_f1_macro_std": float(np.std(f1_scores)),
    }


def perform_hyperparameter_tuning(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[Pipeline, Pipeline, Pipeline]:
    """Runs GridSearchCV for hyperparameter optimization inside Pipelines.

    Args:
        X_train: Unscaled training features.
        y_train: Training labels.

    Returns:
        Tuple of tuned pipelines: (best_lr, best_dt, best_rf).
    """
    logger.info("Initializing hyperparameter tuning with Data Leakage prevention...")

    # 1. Logistic Regression Pipeline
    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ])
    lr_grid = {"classifier__C": [0.1, 1.0, 10.0]}
    lr_cv = GridSearchCV(pipe_lr, lr_grid, cv=3, scoring="f1_macro", n_jobs=-1)
    lr_cv.fit(X_train, y_train)
    logger.info("Tuned LR Best Params: %s", lr_cv.best_params_)

    # 2. Decision Tree Pipeline
    pipe_dt = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(random_state=42, class_weight="balanced"))
    ])
    dt_grid = {"classifier__max_depth": [5, 8, 12]}
    dt_cv = GridSearchCV(pipe_dt, dt_grid, cv=3, scoring="f1_macro", n_jobs=-1)
    dt_cv.fit(X_train, y_train)
    logger.info("Tuned DT Best Params: %s", dt_cv.best_params_)

    # 3. Random Forest Pipeline
    pipe_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42, class_weight="balanced"))
    ])
    rf_grid = {"classifier__n_estimators": [50, 100], "classifier__max_depth": [6, 10]}
    rf_cv = GridSearchCV(pipe_rf, rf_grid, cv=3, scoring="f1_macro", n_jobs=-1)
    rf_cv.fit(X_train, y_train)
    logger.info("Tuned RF Best Params: %s", rf_cv.best_params_)

    return lr_cv.best_estimator_, dt_cv.best_estimator_, rf_cv.best_estimator_


def calculate_permutation_importance(
    model_pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str]
) -> Dict[str, float]:
    """Computes Permutation Feature Importance for a pipeline on test dataset.

    Args:
        model_pipeline: Trained estimator Pipeline.
        X_test: Unscaled test feature matrix.
        y_test: Test labels.
        feature_names: List of feature label strings.

    Returns:
        Dictionary mapping feature name to permutation importance score.
    """
    logger.info("Computing permutation importances...")
    r = permutation_importance(
        model_pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
    )
    importances = {}
    for i in r.importances_mean.argsort()[::-1]:
        importances[feature_names[i]] = float(r.importances_mean[i])
    return importances


def plot_diagnostic_curves(
    lr_pipeline: Pipeline,
    dt_pipeline: Pipeline,
    rf_pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: Path,
) -> None:
    """Generates and saves ROC and Precision-Recall curves.

    Args:
        lr_pipeline: Tuned Logistic Regression Pipeline.
        dt_pipeline: Tuned Decision Tree Pipeline.
        rf_pipeline: Tuned Random Forest Pipeline.
        X_test: Unscaled test features.
        y_test: Test labels.
        save_dir: Destination folder.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating ROC and Precision-Recall curves...")

    # ROC Curves
    plt.figure(figsize=(8, 6))

    for pipeline, label in [
        (lr_pipeline, "Logistic Regression"),
        (dt_pipeline, "Decision Tree"),
        (rf_pipeline, "Random Forest"),
    ]:
        probs = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_score = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_score:.4f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_dir / "roc_auc_curve.png", dpi=150)
    plt.close()

    # Precision-Recall Curves
    plt.figure(figsize=(8, 6))

    for pipeline, label in [
        (lr_pipeline, "Logistic Regression"),
        (dt_pipeline, "Decision Tree"),
        (rf_pipeline, "Random Forest"),
    ]:
        probs = pipeline.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, probs)
        plt.plot(recall, precision, label=label)

    plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_dir / "precision_recall_curve.png", dpi=150)
    plt.close()


def main() -> None:
    """Primary execution driver for the train pipeline."""
    logger.info("--- Starting Strict Model Training & Export Pipeline ---")

    try:
        features_df, target = load_cleaned_data(CLEANED_DATA_PATH)
        feature_names = list(features_df.columns)

        X_processed, ordinal_mappings = preprocess_features(features_df)
        feature_columns = list(X_processed.columns)

        X = X_processed.values
        y = target.values

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(
            "Train set size: %d, Test set size: %d", len(X_train), len(X_test)
        )

        # Train & Tune Pipelines (Scaling happens strictly within CV)
        lr_best, dt_best, rf_best = perform_hyperparameter_tuning(
            X_train, y_train
        )

        # Evaluate Models on Test Data
        logger.info("Evaluating tuned pipelines on test dataset...")
        for name, pipeline in [
            ("Logistic Regression", lr_best),
            ("Decision Tree", dt_best),
            ("Random Forest", rf_best),
        ]:
            preds = pipeline.predict(X_test)
            probs = pipeline.predict_proba(X_test)[:, 1]
            logger.info("Evaluating: %s", name)
            logger.info("Accuracy: %.4f", accuracy_score(y_test, preds))
            logger.info("F1 Score (macro): %.4f", f1_score(y_test, preds, average="macro"))
            logger.info("Precision (macro): %.4f", precision_score(y_test, preds, average="macro"))
            logger.info("Recall (macro): %.4f", recall_score(y_test, preds, average="macro"))
            logger.info("ROC AUC: %.4f", roc_auc_score(y_test, probs))

        # Perform Cross Validation on training set
        logger.info("Evaluating generalization metrics using Strict 5-fold Stratified CV...")
        lr_cv_metrics = run_cross_validation(lr_best, X_train, y_train, cv=5)
        dt_cv_metrics = run_cross_validation(dt_best, X_train, y_train, cv=5)
        rf_cv_metrics = run_cross_validation(rf_best, X_train, y_train, cv=5)

        # Permutation feature importances
        perm_importances = calculate_permutation_importance(
            rf_best, X_test, y_test, feature_columns
        )

        # Generate ROC/PR Curves
        plot_diagnostic_curves(
            lr_best, dt_best, rf_best, X_test, y_test, PLOTS_DIR
        )

        # Extract scaling parameters and coefficients from the fitted Logistic Regression pipeline
        lr_classifier = lr_best.named_steps["classifier"]
        lr_scaler = lr_best.named_steps["scaler"]
        
        coefficients = lr_classifier.coef_[0]
        intercept = float(lr_classifier.intercept_[0])
        
        scaling_params = {}
        for i, col_name in enumerate(feature_columns):
            scaling_params[col_name] = {
                "mean": float(lr_scaler.mean_[i]),
                "std": float(lr_scaler.scale_[i]),
            }

        # Extract RF parameters
        rf_classifier = rf_best.named_steps["classifier"]
        dt_classifier = dt_best.named_steps["classifier"]

        # Evaluate final Logistic Regression test metrics
        lr_preds = lr_best.predict(X_test)
        lr_probs = lr_best.predict_proba(X_test)[:, 1]

        model_assets = {
            "model_name": "Tuned Logistic Regression (Strict Pipeline)",
            "intercept": intercept,
            "coefficients": {
                col_name: float(coefficients[i])
                for i, col_name in enumerate(feature_columns)
            },
            "scaling": scaling_params,
            "categorical_mappings": ordinal_mappings,
            "metrics": {
                "test_accuracy": float(accuracy_score(y_test, lr_preds)),
                "test_f1_macro": float(f1_score(y_test, lr_preds, average="macro")),
                "test_precision_macro": float(
                    precision_score(y_test, lr_preds, average="macro")
                ),
                "test_recall_macro": float(
                    recall_score(y_test, lr_preds, average="macro")
                ),
                "test_roc_auc": float(roc_auc_score(y_test, lr_probs)),
            },
            "cross_validation": lr_cv_metrics,
            "random_forest_importance": {
                col_name: float(rf_classifier.feature_importances_[i])
                for i, col_name in enumerate(feature_columns)
            },
            "permutation_importance": perm_importances,
            "hyperparameters": {
                "lr_C": float(lr_classifier.C),
                "dt_max_depth": int(dt_classifier.max_depth) if dt_classifier.max_depth else None,
                "rf_n_estimators": int(rf_classifier.n_estimators),
                "rf_max_depth": int(rf_classifier.max_depth) if rf_classifier.max_depth else None,
            },
        }

        logger.info("Saving strict model assets to %s", MODEL_ASSETS_PATH)
        MODEL_ASSETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_ASSETS_PATH, "w", encoding="utf-8") as f:
            json.dump(model_assets, f, indent=4)

        logger.info("--- Model Training & Export Completed Successfully ---")

    except Exception as e:
        logger.error("Model training process failed.", exc_info=True)
        raise e


if __name__ == "__main__":
    main()
