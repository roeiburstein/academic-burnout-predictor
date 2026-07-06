import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add src to python path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from eda import diagnose_and_inject_covariance

def test_diagnose_and_inject_covariance():
    # Setup dummy dataframe with all required columns
    np.random.seed(42)
    data = {
        "student_id": range(1, 101),
        "age": np.random.randint(18, 25, 100),
        "stress_level": np.random.choice(["Low", "Medium", "High"], 100),
        "sleep_quality": np.random.choice(["Poor", "Average", "Good"], 100),
        "daily_study_hours": np.random.uniform(1.0, 10.0, 100),
        "daily_sleep_hours": np.random.uniform(4.0, 10.0, 100),
        "screen_time_hours": np.random.uniform(1.0, 12.0, 100),
        "academic_pressure_score": np.random.randint(0, 11, 100),
        "anxiety_score": np.random.randint(0, 11, 100),
        "depression_score": np.random.randint(0, 11, 100),
        "financial_stress_score": np.random.randint(0, 11, 100),
        "social_support_score": np.random.randint(0, 11, 100),
        "physical_activity_hours": np.random.uniform(0, 3, 100),
        "attendance_percentage": np.random.randint(50, 101, 100),
        "cgpa": np.random.uniform(4.0, 10.0, 100),
        "burnout_level": np.random.choice(["Low", "Medium", "High"], 100) # Pre-existing target
    }
    df = pd.DataFrame(data)

    df_modified = diagnose_and_inject_covariance(df)
    
    # Assert burnout level was re-engineered and remains categorical
    assert "burnout_level" in df_modified.columns
    assert set(df_modified["burnout_level"].unique()).issubset({"Low", "Medium", "High"})
    
    # Check that data wasn't accidentally dropped
    assert len(df_modified) == 100
