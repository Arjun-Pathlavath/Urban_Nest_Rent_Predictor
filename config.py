"""Global configuration settings for the UrbanNest Rent Predictor project.

This file centralizes all file paths, column names, coordinate mapping presets,
and hyperparameter tuning configurations for clean, modular code.
"""

from pathlib import Path
from typing import Dict, List, Any

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

TRAIN_DATA_PATH = DATASET_DIR / "train.csv"
TEST_DATA_PATH = DATASET_DIR / "test.csv"

# Model and preprocessor pickle paths
MODEL_PATH = MODELS_DIR / "best_rf_model.pkl"
ENCODERS_PATH = MODELS_DIR / "label_encoders.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# -----------------------------------------------------------------------------
# Data Schema
# -----------------------------------------------------------------------------
CATEGORICAL_COLS = ["location", "city", "Status", "property_type"]
TARGET_COL = "price"

# Features used in training the RandomForest model
FEATURE_NAMES = [
    "location",
    "city",
    "latitude",
    "longitude",
    "numBathrooms",
    "numBalconies",
    "isNegotiable",
    "SecurityDeposit",
    "Status",
    "Size_ft²",
    "BHK",
    "rooms_num",
    "property_type",
    "verification_days"
]

# -----------------------------------------------------------------------------
# City Defaults (Geographical coordinates from dataset averages)
# -----------------------------------------------------------------------------
CITY_COORDINATES: Dict[str, Dict[str, float]] = {
    "Delhi": {"latitude": 28.577319, "longitude": 77.173373},
    "Hisar": {"latitude": 19.135377, "longitude": 72.822156}, # Matches dataset averages
    "Mumbai": {"latitude": 19.163118, "longitude": 72.956616},
    "Pune": {"latitude": 18.574060, "longitude": 73.825543},
}

# Default list of cities if dataset is not loaded
DEFAULT_CITIES = list(CITY_COORDINATES.keys())

# Default categories for fallback/initial UI render
DEFAULT_STATUSES = ["Unfurnished", "Semi-Furnished", "Furnished"]
DEFAULT_PROPERTY_TYPES = [
    "Apartment",
    "Studio Apartment",
    "Independent Floor",
    "Independent House",
    "Villa",
    "penthouse"
]

# -----------------------------------------------------------------------------
# Model Tuning Hyperparameters
# -----------------------------------------------------------------------------
RANDOM_STATE = 42

GRID_SEARCH_SPACE = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [10, 15, 20, 25, 30],
    "min_samples_split": [2, 5, 8]
}

# Used for Random Search & Bayesian (Optuna)
HYPERPARAMETER_BOUNDS = {
    "n_estimators": (50, 200),
    "max_depth": (10, 30),
    "min_samples_split": (2, 10)
}
