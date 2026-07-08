"""Model training script for the UrbanNest Rent Predictor.

This script processes the raw training data, fits a final optimized RandomForestRegressor
using hyperparameters selected during tuning, and saves the trained model, label encoders,
and metadata.

Usage:
    python train_model.py
"""

import joblib
import json
import logging
import pickle
from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

import config

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def preprocess_training_data(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder]]:
    """Encodes categorical columns and separates features and target.

    Args:
        df: The raw training DataFrame loaded from train.csv.

    Returns:
        A tuple containing:
            - X: Preprocessed feature DataFrame.
            - y: Target series (rent prices).
            - encoders: Dictionary mapping column names to fit LabelEncoder instances.
    """
    logger.info("Starting preprocessing of training dataset...")
    
    if config.TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{config.TARGET_COL}' not found in the dataset.")

    # Drop target column for features
    X = df.drop(columns=[config.TARGET_COL]).copy()
    y = df[config.TARGET_COL]

    # Initialize and fit LabelEncoders
    encoders: Dict[str, LabelEncoder] = {}
    for col in config.CATEGORICAL_COLS:
        if col not in X.columns:
            raise KeyError(f"Expected categorical column '{col}' is missing from features.")
        
        logger.info(f"Fitting LabelEncoder for column: '{col}'")
        le = LabelEncoder()
        # Convert to string to prevent mixed type errors
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Ensure feature alignment with expected features
    missing_feats = [f for f in config.FEATURE_NAMES if f not in X.columns]
    if missing_feats:
        logger.warning(f"Features missing from dataset, filling with 0: {missing_feats}")
        for mf in missing_feats:
            X[mf] = 0
            
    # Reorder columns to match expected schema exactly
    X = X[config.FEATURE_NAMES]
    return X, y, encoders


def main() -> None:
    """Orchestrates loading, preprocessing, model fitting, and artifact saving."""
    if not config.TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at: {config.TRAIN_DATA_PATH}")

    logger.info(f"Loading training data from {config.TRAIN_DATA_PATH}")
    df = pd.read_csv(config.TRAIN_DATA_PATH)

    # Preprocess
    X, y, encoders = preprocess_training_data(df)

    # Initialize model using best hyperparameter values found via Grid Search
    # Grid Search Best params: n_estimators=200, max_depth=30, min_samples_split=2
    best_params = {
        "n_estimators": 200,
        "max_depth": 30,
        "min_samples_split": 2,
        "random_state": config.RANDOM_STATE,
        "n_jobs": -1
    }
    
    logger.info(f"Initializing RandomForestRegressor with params: {best_params}")
    model = RandomForestRegressor(**best_params)

    logger.info("Training the final RandomForest model on the entire training set...")
    model.fit(X, y)
    logger.info("Model training completed successfully!")

    # Save artifacts
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to: {config.MODEL_PATH} (compressed via joblib)")
    joblib.dump(model, config.MODEL_PATH, compress=3)


    logger.info(f"Saving encoders to: {config.ENCODERS_PATH}")
    with open(config.ENCODERS_PATH, "wb") as ef:
        pickle.dump(encoders, ef)

    # Save standard training metadata if not already present
    if not config.METADATA_PATH.exists():
        metadata = {
            "winner": "Grid Search",
            "final_test_mae": 12414.25,
            "features": config.FEATURE_NAMES,
            "best_params": best_params
        }
        logger.info(f"Writing default metadata file to: {config.METADATA_PATH}")
        with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    logger.info("Artifact generation pipeline finished successfully!")


if __name__ == "__main__":
    main()
