import sys
from pathlib import Path

# Add src to python path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

import pandas as pd
import pytest

from train import preprocess_features

def test_preprocess_features():
    data = {
        "year": ["1st", "2nd", "3rd", "4th"],
        "stress_level": ["Low", "Medium", "High", "Low"],
        "sleep_quality": ["Poor", "Average", "Good", "Good"],
        "internet_quality": ["Poor", "Average", "Good", "Average"],
        "gender": ["Male", "Female", "Other", "Male"],
        "course": ["BSc", "BTech", "BCA", "MCA"]
    }
    df = pd.DataFrame(data)
    
    encoded_df, mappings = preprocess_features(df)
    
    # Assert ordinal mappings were applied
    assert encoded_df["year"].tolist() == [1, 2, 3, 4]
    assert encoded_df["stress_level"].tolist() == [1, 2, 3, 1]
    
    # Assert nominal categorical dummies were created (drop_first=True)
    # gender (Female is dropped reference usually if sorted alphanumerically, but let's just check dummy generation)
    assert "gender_Male" in encoded_df.columns
    assert "course_BTech" in encoded_df.columns

    # Assert everything is float type
    for col in encoded_df.columns:
        assert pd.api.types.is_float_dtype(encoded_df[col])
