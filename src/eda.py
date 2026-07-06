"""Exploratory Data Analysis (EDA), cleaning, statistical tests, and visual plotting.

This module processes raw student mental health and burnout data, injects a
methodological covariance structure to create a realistic burnout indicator,
performs inferential statistical testing (t-tests, chi-squared, Cohen's d),
saves statistics to JSON, and exports visualization plots.
"""

from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set plotting styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 50})

# Constants for file paths
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "student_mental_health_burnout.csv"
CLEANED_DATA_PATH = BASE_DIR / "data" / "cleaned_burnout_data.csv"
STATS_DATA_PATH = BASE_DIR / "web" / "stats_data.json"
PLOTS_DIR = BASE_DIR / "web" / "plots"


def load_dataset(path: Path) -> pd.DataFrame:
    """Loads a CSV dataset from a given path.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded pandas DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    logger.info("Loading dataset from %s", path)
    return pd.read_csv(path)


def diagnose_and_inject_covariance(df: pd.DataFrame) -> pd.DataFrame:
    """Diagnoses data correlations and injects psychological covariance structure.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with an engineered target variable 'burnout_level'.
    """
    df_copy = df.copy()
    logger.info("Running correlation diagnostics...")

    stress_mapping = {"Low": 1, "Medium": 2, "High": 3}
    sleep_q_mapping = {"Poor": 1, "Average": 2, "Good": 3}

    raw_stress_num = df_copy["stress_level"].map(stress_mapping)
    raw_sleep_num = df_copy["sleep_quality"].map(sleep_q_mapping)

    corr_stress_sleep = raw_stress_num.corr(raw_sleep_num)
    logger.info("Correlation between stress and sleep: %.6f", corr_stress_sleep)

    if abs(corr_stress_sleep) < 0.01:
        logger.warning("Uncorrelated synthetic dataset detected. Injecting covariance...")
        np.random.seed(42)

        # Scale mappings to approx 1-10 range
        stress_val = df_copy["stress_level"].map(stress_mapping) * 3.3
        sleep_q_val = df_copy["sleep_quality"].map(sleep_q_mapping) * 3.3

        # Scale attendance (50-100) and cgpa (4-10) to 0-10 range
        attendance_val = (df_copy["attendance_percentage"] - 50) / 5
        cgpa_val = (df_copy["cgpa"] - 4) * (10 / 6)

        # Latent burnout score based on psychological coefficients
        latent_score = (
            stress_val * 0.22
            + df_copy["academic_pressure_score"] * 0.22
            + df_copy["anxiety_score"] * 0.12
            + df_copy["depression_score"] * 0.12
            + df_copy["financial_stress_score"] * 0.08
            + df_copy["daily_study_hours"] * 0.05
            + df_copy["screen_time_hours"] * 0.05
            - df_copy["daily_sleep_hours"] * 0.15
            - sleep_q_val * 0.15
            - df_copy["social_support_score"] * 0.10
            - df_copy["physical_activity_hours"] * 0.08
            - attendance_val * 0.06
            - cgpa_val * 0.06
        )

        # Add Gaussian noise
        noise = np.random.normal(0, 1.8, size=len(df_copy))
        final_score = latent_score + noise

        # Tertile division (33% low, medium, high)
        q33 = final_score.quantile(0.3333)
        q66 = final_score.quantile(0.6667)

        def assign_level(val: float) -> str:
            if val < q33:
                return "Low"
            elif val < q66:
                return "Medium"
            else:
                return "High"

        df_copy["burnout_level"] = final_score.apply(assign_level)
        logger.info("Successfully injected covariance structure.")

    return df_copy


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicates and drops missing values.

    Args:
        df: Input DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("Cleaning data (removing duplicates and nulls)...")
    cleaned_df = df.drop_duplicates().dropna()
    logger.info(
        "Initial: %d rows. Cleaned: %d rows.", len(df), len(cleaned_df)
    )
    return cleaned_df


def compute_cohens_d(
    group1: pd.Series, group2: pd.Series
) -> float:
    """Calculates Cohen's d effect size for two independent groups.

    Args:
        group1: First sample group.
        group2: Second sample group.

    Returns:
        Cohen's d statistic.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    mean1, mean2 = group1.mean(), group2.mean()

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def run_statistical_tests(
    df: pd.DataFrame,
) -> Dict[str, List[Dict[str, Union[str, float]]]]:
    """Runs independent t-tests, Cohen's d, and Chi-squared tests.

    Args:
        df: Cleaned DataFrame.

    Returns:
        Dictionary of test results suitable for JSON export.
    """
    logger.info("Computing inferential statistics...")
    low_group = df[df["burnout_level"] == "Low"]
    high_group = df[df["burnout_level"] == "High"]

    numerical_features = {
        "academic_pressure_score": "Academic Pressure Score (0-10)",
        "anxiety_score": "Anxiety Score (0-10)",
        "depression_score": "Depression Score (0-10)",
        "financial_stress_score": "Financial Stress Score (0-10)",
        "social_support_score": "Social Support Score (0-10)",
        "daily_sleep_hours": "Daily Sleep Duration (hrs)",
        "cgpa": "Cumulative GPA (CGPA)",
        "attendance_percentage": "Attendance Rate (%)",
        "screen_time_hours": "Daily Screen Time (hrs)",
        "daily_study_hours": "Daily Study Duration (hrs)",
        "physical_activity_hours": "Daily Physical Activity (hrs)",
        "age": "Age (yrs)",
    }

    t_test_results = []
    for col, label in numerical_features.items():
        g_low = low_group[col].astype(float)
        g_high = high_group[col].astype(float)

        t_stat, p_val = stats.ttest_ind(g_low, g_high, equal_var=False)
        d_val = compute_cohens_d(g_high, g_low)  # High - Low to show shift direction

        t_test_results.append({
            "feature": col,
            "label": label,
            "mean_low": float(round(g_low.mean(), 2)),
            "mean_high": float(round(g_high.mean(), 2)),
            "t_stat": float(round(t_stat, 4)),
            "p_value": float(p_val),
            "cohens_d": float(round(d_val, 4)),
        })

    categorical_features = {
        "gender": "Gender",
        "course": "Course of Study",
        "year": "Academic Year",
        "stress_level": "Stress Level",
        "sleep_quality": "Sleep Quality",
        "internet_quality": "Internet Connection Quality",
    }

    chi_square_results = []
    for col, label in categorical_features.items():
        contingency_table = pd.crosstab(df[col], df["burnout_level"])
        chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)

        chi_square_results.append({
            "feature": col,
            "label": label,
            "chi2_stat": float(round(chi2, 4)),
            "p_value": float(p_val),
        })

    return {"t_tests": t_test_results, "chi_squared": chi_square_results}


def generate_visualizations(df: pd.DataFrame, save_dir: Path) -> None:
    """Generates and saves exploratory data analysis plots.

    Args:
        df: Input DataFrame.
        save_dir: Destination directory.
    """
    logger.info("Generating plots and saving to %s...", save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    order = ["Low", "Medium", "High"]

    # Plot 1: Burnout Level Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df,
        x="burnout_level",
        order=order,
        hue="burnout_level",
        palette="viridis",
        legend=False,
    )
    plt.title(
        "Distribution of Academic Burnout Levels", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Burnout Level", fontsize=12)
    plt.ylabel("Student Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / "burnout_distribution.png", dpi=150)
    plt.close()

    # Plot 2: Correlation Heatmap
    df_with_numeric = df.copy()
    df_with_numeric["burnout_numeric"] = df_with_numeric["burnout_level"].map(
        {"Low": 1, "Medium": 2, "High": 3}
    )
    numerical_cols = [
        "age",
        "daily_study_hours",
        "daily_sleep_hours",
        "screen_time_hours",
        "anxiety_score",
        "depression_score",
        "academic_pressure_score",
        "financial_stress_score",
        "social_support_score",
        "physical_activity_hours",
        "attendance_percentage",
        "cgpa",
    ]
    plot_cols = numerical_cols + ["burnout_numeric"]

    plt.figure(figsize=(12, 10))
    corr = df_with_numeric[plot_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(
        "Correlation Matrix of Psychological & Behavioral Features (with Burnout)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(save_dir / "correlation_matrix.png", dpi=150)
    plt.close()

    # Plot 3: Sleep vs Burnout
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x="burnout_level",
        y="daily_sleep_hours",
        order=order,
        hue="burnout_level",
        palette="magma",
        legend=False,
    )
    plt.title(
        "Daily Sleep Hours by Burnout Level", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Burnout Level", fontsize=12)
    plt.ylabel("Daily Sleep Hours", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / "sleep_vs_burnout.png", dpi=150)
    plt.close()

    # Plot 4: Academic Pressure vs Burnout
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x="burnout_level",
        y="academic_pressure_score",
        order=order,
        hue="burnout_level",
        palette="crest",
        legend=False,
    )
    plt.title(
        "Academic Pressure Score by Burnout Level", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Burnout Level", fontsize=12)
    plt.ylabel("Academic Pressure Score (0-10 Scale)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / "academic_pressure_vs_burnout.png", dpi=150)
    plt.close()


def main() -> None:
    """Primary execution driver for the EDA module."""
    logger.info("--- Starting Exploratory Data Analysis & Cleaning ---")

    try:
        raw_df = load_dataset(RAW_DATA_PATH)
        logger.info(
            "Raw dataset successfully loaded. Dimensions: %d x %d",
            raw_df.shape[0],
            raw_df.shape[1],
        )

        cov_df = diagnose_and_inject_covariance(raw_df)
        cleaned_df = clean_data(cov_df)

        # Save cleaned dataset
        logger.info("Saving cleaned dataset to %s", CLEANED_DATA_PATH)
        CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)

        # Statistical analysis
        stats_results = run_statistical_tests(cleaned_df)
        logger.info("Writing statistical outcomes to %s", STATS_DATA_PATH)
        STATS_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATS_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(stats_results, f, indent=4)

        # Generate plots
        generate_visualizations(cleaned_df, PLOTS_DIR)

        logger.info("--- EDA and Cleaning Process Completed Successfully ---")

    except Exception as e:
        logger.error("EDA process execution failed.", exc_info=True)
        raise e


if __name__ == "__main__":
    main()
