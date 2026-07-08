"""Unit test suite for the UrbanNest ML and preprocessing pipeline.

Tests verify the correctness of category encoding, feature formatting,
prediction intervals, and benchmarking helpers.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder


import config
import utils


def test_city_coordinates_exist():
    """Verifies that all required cities have coordinate presets."""
    for city in ["Delhi", "Hisar", "Mumbai", "Pune"]:
        assert city in config.CITY_COORDINATES
        assert "latitude" in config.CITY_COORDINATES[city]
        assert "longitude" in config.CITY_COORDINATES[city]


def test_build_row_schema_alignment():
    """Ensures build_row produces a DataFrame aligned with FEATURE_NAMES."""
    payload = {
        "city": 1,
        "location": 5,
        "latitude": 19.163,
        "longitude": 72.956,
        "Size_ft²": 1200.0,
        "BHK": 2,
    }
    row = utils.build_row(payload)
    
    assert isinstance(row, pd.DataFrame)
    assert len(row) == 1
    # Check that all features in config.FEATURE_NAMES are present
    assert list(row.columns) == config.FEATURE_NAMES
    # Check that filled features match
    assert row["latitude"].iloc[0] == 19.163
    # Check that missing features in payload default to 0
    assert row["SecurityDeposit"].iloc[0] == 0


def test_safe_encode_known_and_unknown_classes():
    """Tests safe_encode with exact classes, case mismatches, and unknown fallback values."""
    # Setup dummy encoder
    le = LabelEncoder()
    le.fit(["Apartment", "Villa", "penthouse"])
    encoders = {"property_type": le}
    
    # 1. Test exact match
    val = utils.safe_encode(encoders, "property_type", "Villa")
    assert val == le.transform(["Villa"])[0]

    # 2. Test case-insensitive match
    val = utils.safe_encode(encoders, "property_type", "APARTMENT")
    assert val == le.transform(["Apartment"])[0]

    # 3. Test unknown fallback (should default to first class 'Apartment' encoding)
    val = utils.safe_encode(encoders, "property_type", "Studio")
    assert val == le.transform(["Apartment"])[0]


def test_safe_encode_contextual_fallback():
    """Tests safe_encode contextual city fallback on unseen locations."""
    le = LabelEncoder()
    le.fit(["Thane West", "Airoli", "Jasola"])
    encoders = {"location": le}
    
    # Setup historical dataset with most common location for Mumbai
    market_data = pd.DataFrame([
        {"city": "Mumbai", "location": "Thane West"},
        {"city": "Mumbai", "location": "Thane West"},
        {"city": "Mumbai", "location": "Airoli"},
        {"city": "Delhi", "location": "Jasola"}
    ])

    # Ask to encode an unseen location 'Dombivli' under Mumbai city context
    # It should fall back to 'Thane West' (Mumbai's mode location)
    val = utils.safe_encode(
        encoders=encoders,
        col="location",
        value="Dombivli",
        city_context="Mumbai",
        market_df=market_data
    )
    assert val == le.transform(["Thane West"])[0]


def test_market_average_calculation():
    """Verifies that historical averages are computed accurately at different scales."""
    # Define a mock dataset
    test_df = pd.DataFrame([
        {"city": "Pune", "location": "Kothrud", "price": 10000},
        {"city": "Pune", "location": "Kothrud", "price": 12000},
        {"city": "Pune", "location": "Kothrud", "price": 11000},
        {"city": "Pune", "location": "Kothrud", "price": 13000},
        {"city": "Pune", "location": "Kothrud", "price": 14000},  # 5 listings for Kothrud
        {"city": "Pune", "location": "Baner", "price": 20000},
        {"city": "Pune", "location": "Baner", "price": 22000},
        {"city": "Pune", "location": "Baner", "price": 21000},
        {"city": "Pune", "location": "Hinjewadi", "price": 15000},
        {"city": "Pune", "location": "Hinjewadi", "price": 16000},
        {"city": "Pune", "location": "Hinjewadi", "price": 17000},
        {"city": "Pune", "location": "Hinjewadi", "price": 18000},
        {"city": "Pune", "location": "Hinjewadi", "price": 19000},
        {"city": "Pune", "location": "Wakad", "price": 13000},
        {"city": "Pune", "location": "Wakad", "price": 14000},
        {"city": "Pune", "location": "Wakad", "price": 15000},  # Total 16 listings for Pune (>=15)
        {"city": "Mumbai", "location": "Bandra", "price": 40000}
    ])
    
    # 1. Area average (>= 5 listings matches Kothrud)
    mean_val, label = utils.market_average(test_df, "Pune", "Kothrud")
    assert mean_val == pytest.approx(12000.0)
    assert "Area" in label
    
    # 2. City fallback average (< 5 listings for Baner, falls back to Pune mean)
    mean_val, label = utils.market_average(test_df, "Pune", "Baner")
    # Pune mean: (10k+12k+11k+13k+14k + 20k+22k+21k + 15k+16k+17k+18k+19k + 13k+14k+15k) / 16
    # Sum: 60k + 63k + 85k + 42k = 250k / 16 = 15625.0
    assert mean_val == pytest.approx(15625.0)

    assert "City" in label


def test_prediction_interval_bounds():
    """Validates that prediction_interval correctly computes bounds from tree variance."""
    class MockTree:
        def __init__(self, val: float):
            self.val = val
        def predict(self, X):
            return np.array([self.val])

    class MockForest:
        def __init__(self, pred: float, tree_vals: list[float]):
            self.pred = pred
            self.estimators_ = [MockTree(v) for v in tree_vals]
        def predict(self, X):
            return np.array([self.pred])


    # Instantiate mock forest: point prediction is 100.0, tree predictions range from 80.0 to 120.0
    mock_model = MockForest(100.0, [80.0, 90.0, 100.0, 110.0, 120.0])
    dummy_row = pd.DataFrame([[0]*len(config.FEATURE_NAMES)], columns=config.FEATURE_NAMES)
    
    # 1. Check with 80% confidence
    low, pred, high = utils.prediction_interval(mock_model, dummy_row, confidence=80)
    assert pred == 100.0
    assert low <= pred <= high
    # Percentiles: 10th percentile of [80, 90, 100, 110, 120] is 84.0, 90th percentile is 116.0
    assert low == pytest.approx(84.0)
    assert high == pytest.approx(116.0)

    # 2. Check fallback if estimators are missing (should apply static 15% margin)
    class MockFlatModel:
        def predict(self, X):
            return np.array([200.0])
            
    flat_model = MockFlatModel()
    low, pred, high = utils.prediction_interval(flat_model, dummy_row, confidence=85)
    assert pred == 200.0
    assert low == pytest.approx(170.0) # 200 - 15%
    assert high == pytest.approx(230.0) # 200 + 15%


def test_generate_json_report_format():
    """Validates that generate_json_report produces correct structures and values."""
    payload = {
        "city_name": "Delhi",
        "location_name": "Jasola",
        "property_type_name": "Apartment",
        "furnishing_status_name": "Unfurnished",
        "size_sqft": 1000.0,
        "BHK": 2,
        "numBathrooms": 2,
        "numBalconies": 1,
        "rooms_num": 3,
        "SecurityDeposit": 40000.0,
        "isNegotiable": True,
        "verification_days": 15,
        "latitude": 28.529,
        "longitude": 77.293
    }
    
    import json
    report_str = utils.generate_json_report(payload, 25000.0, 22000.0, 28000.0, confidence=85)
    report = json.loads(report_str)
    
    assert report["report_title"] == "UrbanNest Rent Estimate Report"
    assert report["estimated_rent_inr"] == 25000.0
    assert report["prediction_interval"]["confidence_level_percentage"] == 85
    assert report["prediction_interval"]["lower_bound_inr"] == 22000.0
    assert report["prediction_interval"]["upper_bound_inr"] == 28000.0
    assert report["property_details"]["city"] == "Delhi"
    assert report["property_details"]["size_sqft"] == 1000.0
    assert report["property_details"]["price_negotiable"] is True

