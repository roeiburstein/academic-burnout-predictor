"""Model Training, Validation, Tuning, and Asset Export Pipeline.

This module loads the cleaned data, encodes categorical features, scales
features, trains and tunes three classifiers (Logistic Regression, Decision
Tree, and Random Forest), performs cross-validation, computes permutation
feature importances, generates diagnostic curves (ROC & PR), and exports all
model assets to JSON.
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
    model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5
) -> Dict[str, float]:
    """Performs Stratified K-Fold Cross-Validation and collects stats.

    Args:
        model: Evaluated estimator.
        X: Scaled feature matrix.
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

        # Clone and fit
        cloned_model = model.__class__(**model.get_params())
        cloned_model.fit(X_tr, y_tr)
        preds = cloned_model.predict(X_va)

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
) -> Tuple[LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]:
    """Runs GridSearchCV for hyperparameter optimization of the three classifiers.

    Args:
        X_train: Scaled training features.
        y_train: Training labels.

    Returns:
        Tuple of tuned models: (best_lr, best_dt, best_rf).
    """
    logger.info("Initializing hyperparameter tuning...")

    # 1. Logistic Regression
    lr = LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )
    lr_grid = {"C": [0.1, 1.0, 10.0]}
    lr_cv = GridSearchCV(lr, lr_grid, cv=3, scoring="f1_macro", n_jobs=-1)
    lr_cv.fit(X_train, y_train)
    logger.info("Tuned Logistic Regression Best Params: %s", lr_cv.best_params_)

    # 2. Decision Tree
    dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    dt_grid = {"max_depth": [5, 8, 12]}
    dt_cv = GridSearchCV(dt, dt_grid, cv=3, scoring="f1_macro", n_jobs=-1)
    dt_cv.fit(X_train, y_train)
    logger.info("Tuned Decision Tree Best Params: %s", dt_cv.best_params_)

    # 3. Random Forest
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf_grid = {"n_estimators": [50, 100], "max_depth": [6, 10]}
    rf_cv = GridSearchCV(rf, rf_grid, cv=3, scoring="f1_macro", n_jobs=-1)
    rf_cv.fit(X_train, y_train)
    logger.info("Tuned Random Forest Best Params: %s", rf_cv.best_params_)

    return lr_cv.best_estimator_, dt_cv.best_estimator_, rf_cv.best_estimator_


def calculate_permutation_importance(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str]
) -> Dict[str, float]:
    """Computes Permutation Feature Importance for a model on test dataset.

    Args:
        model: Trained estimator.
        X_test: Scaled test feature matrix.
        y_test: Test labels.
        feature_names: List of feature label strings.

    Returns:
        Dictionary mapping feature name to permutation importance score.
    """
    logger.info("Computing permutation importances...")
    r = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
    )
    importances = {}
    for i in r.importances_mean.argsort()[::-1]:
        importances[feature_names[i]] = float(r.importances_mean[i])
    return importances


def plot_diagnostic_curves(
    lr_model: LogisticRegression,
    dt_model: DecisionTreeClassifier,
    rf_model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: Path,
) -> None:
    """Generates and saves ROC and Precision-Recall curves.

    Args:
        lr_model: Tuned Logistic Regression.
        dt_model: Tuned Decision Tree.
        rf_model: Tuned Random Forest.
        X_test: Scaled test features.
        y_test: Test labels.
        save_dir: Destination folder.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating ROC and Precision-Recall curves...")

    # ROC Curves
    plt.figure(figsize=(8, 6))

    for model, label in [
        (lr_model, "Logistic Regression"),
        (dt_model, "Decision Tree"),
        (rf_model, "Random Forest"),
    ]:
        probs = model.predict_proba(X_test)[:, 1]
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

    for model, label in [
        (lr_model, "Logistic Regression"),
        (dt_model, "Decision Tree"),
        (rf_model, "Random Forest"),
    ]:
        probs = model.predict_proba(X_test)[:, 1]
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
    logger.info("--- Starting Model Training & Export Pipeline ---")

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

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        scaling_params = {}
        for i, col_name in enumerate(feature_columns):
            scaling_params[col_name] = {
                "mean": float(scaler.mean_[i]),
                "std": float(scaler.scale_[i]),
            }

        # Train & Tune
        lr_best, dt_best, rf_best = perform_hyperparameter_tuning(
            X_train_scaled, y_train
        )

        # Evaluate Models on Test Data
        logger.info("Evaluating tuned models on test dataset...")
        for name, model in [
            ("Logistic Regression", lr_best),
            ("Decision Tree", dt_best),
            ("Random Forest", rf_best),
        ]:
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
            logger.info("Evaluating: %s", name)
            logger.info("Accuracy: %.4f", accuracy_score(y_test, preds))
            logger.info("F1 Score (macro): %.4f", f1_score(y_test, preds, average="macro"))
            logger.info("Precision (macro): %.4f", precision_score(y_test, preds, average="macro"))
            logger.info("Recall (macro): %.4f", recall_score(y_test, preds, average="macro"))
            logger.info("ROC AUC: %.4f", roc_auc_score(y_test, probs))
            logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, preds))
            logger.info("Classification Report:\n%s", classification_report(y_test, preds))

        # Perform Cross Validation on training set
        logger.info("Evaluating generalization metrics using 5-fold Stratified CV...")
        lr_cv_metrics = run_cross_validation(lr_best, X_train_scaled, y_train, cv=5)
        dt_cv_metrics = run_cross_validation(dt_best, X_train_scaled, y_train, cv=5)
        rf_cv_metrics = run_cross_validation(rf_best, X_train_scaled, y_train, cv=5)

        # Permutation feature importances
        perm_importances = calculate_permutation_importance(
            rf_best, X_test_scaled, y_test, feature_columns
        )

        # Generate ROC/PR Curves
        plot_diagnostic_curves(
            lr_best, dt_best, rf_best, X_test_scaled, y_test, PLOTS_DIR
        )

        # Save model assets (primarily LR coefficients for Javascript Calculator)
        coefficients = lr_best.coef_[0]
        intercept = float(lr_best.intercept_[0])

        # Evaluate final Logistic Regression test metrics
        lr_preds = lr_best.predict(X_test_scaled)
        lr_probs = lr_best.predict_proba(X_test_scaled)[:, 1]

        model_assets = {
            "model_name": "Tuned Logistic Regression (Binary High Burnout)",
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
                col_name: float(rf_best.feature_importances_[i])
                for i, col_name in enumerate(feature_columns)
            },
            "permutation_importance": perm_importances,
            "hyperparameters": {
                "lr_C": float(lr_best.C),
                "dt_max_depth": int(dt_best.max_depth)
                if dt_best.max_depth
                else None,
                "rf_n_estimators": int(rf_best.n_estimators),
                "rf_max_depth": int(rf_best.max_depth)
                if rf_best.max_depth
                else None,
            },
        }

        logger.info("Saving model assets to %s", MODEL_ASSETS_PATH)
        MODEL_ASSETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_ASSETS_PATH, "w", encoding="utf-8") as f:
            json.dump(model_assets, f, indent=4)

        logger.info("--- Model Training & Export Completed Successfully ---")

    except Exception as e:
        logger.error("Model training process failed.", exc_info=True)
        raise e


if __name__ == "__main__":
    main()
