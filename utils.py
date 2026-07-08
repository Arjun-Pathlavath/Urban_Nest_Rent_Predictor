"""Utility and helper functions for the UrbanNest Rent Predictor application.

This module encapsulates all core data loading, processing, inference, plotting,
and report generation logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[Any | None, Dict[str, Any] | None, Dict[str, Any] | None, str | None]:
    """Loads model, label encoders, and optimization metadata from disk.

    Returns:
        A tuple containing:
            - model: The loaded Scikit-Learn RandomForestRegressor.
            - encoders: Dictionary of fit LabelEncoder instances.
            - metadata: Dictionary containing model training and search metadata.
            - error_message: Error string if loading fails, otherwise None.
    """
    model_candidates = [
        config.MODEL_PATH,
        config.MODELS_DIR / "best_rf_model.joblib",
        config.MODELS_DIR / "rf_model.pkl",
        config.MODELS_DIR / "model.pkl",
    ]
    existing_models = [p for p in model_candidates if p.exists()]
    
    if not existing_models:
        return None, None, None, f"Model not found in {config.MODELS_DIR}. Run training first."
    if not config.ENCODERS_PATH.exists():
        return None, None, None, f"Label encoders not found at {config.ENCODERS_PATH}."

    model = None
    loaded_model_path: Path | None = None
    load_errors: List[str] = []

    for candidate in existing_models:
        try:
            # Using joblib.load is standard for scikit-learn models and transparently supports compression
            model = joblib.load(candidate)
            loaded_model_path = candidate
            break
        except Exception as exc:  # noqa: BLE001
            load_errors.append(f"{candidate.name}: {exc}")

    if model is None or loaded_model_path is None:
        return None, None, None, f"Failed loading model artifacts: {', '.join(load_errors[:2])}"

    try:
        with open(config.ENCODERS_PATH, "rb") as ef:
            encoders = pickle.load(ef)
    except Exception as exc:  # noqa: BLE001
        return None, None, None, f"Failed loading label encoders: {exc}"

    metadata = {}
    if config.METADATA_PATH.exists():
        try:
            with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Failed to load metadata: {exc}")

    return model, encoders, metadata, None


@st.cache_data(show_spinner=False)
def load_market_df() -> pd.DataFrame:
    """Loads and concatenates the training and test CSVs for market comparisons.

    Returns:
        A combined pandas DataFrame of historical data.
    """
    dfs: List[pd.DataFrame] = []
    for p in [config.TRAIN_DATA_PATH, config.TEST_DATA_PATH]:
        if p.exists():
            try:
                dfs.append(pd.read_csv(p))
            except Exception as e:  # noqa: BLE001
                st.warning(f"Error loading {p.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def encoder_values(encoders: Dict[str, Any] | None, col: str) -> List[str]:
    """Retrieves list of fitted categories/classes for a column.

    Args:
        encoders: Loaded encoders dictionary.
        col: Target categorical column.

    Returns:
        List of class label strings.
    """
    if encoders is None:
        return []
    enc = encoders.get(col)
    if enc is None:
        return []
    return list(enc.classes_)


def safe_encode(encoders: Dict[str, Any] | None, col: str, value: str, city_context: str | None = None, market_df: pd.DataFrame = pd.DataFrame()) -> int:
    """Safely encodes a categorical value, with robust fallback for unseen categories.

    If the category is unseen:
      1. Tries to match case-insensitively.
      2. Falls back to the most common category for the selected city.
      3. Falls back to the global mode category.
      4. Defaults to the first encoder class.

    Args:
        encoders: Loaded encoders dictionary.
        col: Target column name.
        value: Input category string.
        city_context: The selected city to refine fallback matching.
        market_df: Dataset to lookup mode categories.

    Returns:
        The integer encoding for the categorical feature.
    """
    if encoders is None:
        return 0
    enc = encoders.get(col)
    if enc is None:
        return 0

    classes = list(enc.classes_)
    
    # 1. Exact match check
    if value in classes:
        return int(enc.transform([value])[0])

    # 2. Case-insensitive match check
    value_lower = value.lower()
    for c in classes:
        if str(c).lower() == value_lower:
            return int(enc.transform([c])[0])

    # 3. Contextual fallback for locations
    if col == "location" and city_context and not market_df.empty:
        city_subset = market_df[market_df["city"].astype(str) == city_context]
        if not city_subset.empty and "location" in city_subset.columns:
            most_common = city_subset["location"].mode()
            if not most_common.empty and most_common.iloc[0] in classes:
                return int(enc.transform([most_common.iloc[0]])[0])

    # 4. Global fallback to mode
    if not market_df.empty and col in market_df.columns:
        global_mode = market_df[col].mode()
        if not global_mode.empty and global_mode.iloc[0] in classes:
            return int(enc.transform([global_mode.iloc[0]])[0])

    # 5. Last resort fallback
    return int(enc.transform([classes[0]])[0])


def city_location_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Builds a map linking cities to their specific locations/neighborhoods.

    Args:
        df: Combined market DataFrame.

    Returns:
        A dictionary mapping city names to sorted lists of locations.
    """
    if df.empty or "city" not in df.columns or "location" not in df.columns:
        return {}
    tmp = df[["city", "location"]].dropna().astype(str)
    out: Dict[str, List[str]] = {}
    for city, g in tmp.groupby("city"):
        out[city] = sorted(g["location"].unique().tolist())
    return out


def market_average(df: pd.DataFrame, city: str, location: str) -> Tuple[float, str]:
    """Calculates historical benchmarks at local, city-wide, and global levels.

    Args:
        df: Combined historical DataFrame.
        city: Property city.
        location: Property neighborhood.

    Returns:
        A tuple containing:
            - average_price: Benchmark monthly rent (INR).
            - label: String description of benchmark scale (e.g. Area, City).
    """
    if df.empty or "price" not in df.columns:
        return 0.0, "No historical data available"

    city_df = df[df["city"].astype(str) == city] if "city" in df.columns else df
    if "location" in df.columns:
        area_df = city_df[city_df["location"].astype(str) == location]
        if len(area_df) >= 5: # Lowered minimum threshold to 5 for granular neighborhood matching
            return float(area_df["price"].mean()), f"Area benchmark ({len(area_df)} listings)"

    if len(city_df) >= 15:
        return float(city_df["price"].mean()), f"City benchmark ({len(city_df)} listings)"

    return float(df["price"].mean()), "Overall market benchmark"


def prediction_interval(rf_model: Any, row: pd.DataFrame, confidence: int) -> Tuple[float, float, float]:
    """Calculates point prediction and dynamic confidence interval using tree variance.

    Args:
        rf_model: Loaded RandomForestRegressor.
        row: Input features formatted as a single-row DataFrame.
        confidence: Confidence level percentage (e.g. 85).

    Returns:
        A tuple containing (lower_bound, estimate, upper_bound).
    """
    row_values = row.to_numpy(dtype=float, copy=False)
    pred = float(rf_model.predict(row)[0])

    if hasattr(rf_model, "estimators_") and rf_model.estimators_:
        # Compute predictions across all individual trees in the random forest
        tree_preds = np.array([float(t.predict(row_values)[0]) for t in rf_model.estimators_])
        alpha = (100 - confidence) / 2
        low = float(np.percentile(tree_preds, alpha))
        high = float(np.percentile(tree_preds, 100 - alpha))
        return max(0.0, low), max(0.0, pred), max(0.0, high)

    # Standard fallback if trees are inaccessible
    margin = 0.15 * pred
    return max(0.0, pred - margin), max(0.0, pred), max(0.0, pred + margin)


def build_row(payload: Dict[str, float | int]) -> pd.DataFrame:
    """Formats inputs into a aligned DataFrame matching training feature columns.

    Args:
        payload: Dictionary of input features and values.

    Returns:
        Aligned DataFrame matching configuration feature schema.
    """
    row = pd.DataFrame([payload])
    for c in config.FEATURE_NAMES:
        if c not in row.columns:
            row[c] = 0
    return row[config.FEATURE_NAMES]


def generate_feature_importance_chart(rf_model: Any) -> Any:
    """Generates an interactive Plotly horizontal bar chart of top feature importances.

    Args:
        rf_model: Trained RandomForestRegressor model.

    Returns:
        Plotly Figure object or None if unavailable.
    """
    if not hasattr(rf_model, "feature_importances_"):
        return None
    names = list(getattr(rf_model, "feature_names_in_", []))
    if not names:
        names = config.FEATURE_NAMES

    imp = pd.DataFrame({"feature": names, "importance": rf_model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).head(7)

    # Clean display labels
    alias = {
        "city": "City",
        "location": "Location",
        "Status": "Furnishing Status",
        "property_type": "Property Type",
        "BHK": "Bedrooms (BHK)",
        "numBathrooms": "Bathrooms",
        "numBalconies": "Balconies",
        "rooms_num": "Total Rooms",
        "verification_days": "Listing Age (Days)",
        "SecurityDeposit": "Security Deposit",
        "isNegotiable": "Price Negotiable",
        "latitude": "Latitude Coordinate",
        "longitude": "Longitude Coordinate",
        "Size_ft²": "Size (sqft)"
    }
    imp["feature"] = imp["feature"].map(lambda x: alias.get(x, x))

    fig = px.bar(
        imp.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=[[0, "#0EA5E9"], [1, "#F43F5E"]], # Sleek modern gradient
        title="Top Rent Price Drivers (SHAP Style)",
    )
    
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB", size=11),
        title_font=dict(size=14, color="#E5E7EB"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def generate_json_report(payload: Dict[str, Any], pred: float, low: float, high: float, confidence: int) -> str:
    """Formats model outputs and input details into a JSON report for export.

    Args:
        payload: Input properties dictionary.
        pred: Estimated rent.
        low: Prediction interval lower bound.
        high: Prediction interval upper bound.
        confidence: Prediction interval confidence level.

    Returns:
        JSON string representation of the report.
    """
    report = {
        "report_title": "UrbanNest Rent Estimate Report",
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "estimated_rent_inr": round(pred, 2),
        "prediction_interval": {
            "confidence_level_percentage": confidence,
            "lower_bound_inr": round(low, 2),
            "upper_bound_inr": round(high, 2)
        },
        "property_details": {
            "city": payload.get("city_name"),
            "location": payload.get("location_name"),
            "property_type": payload.get("property_type_name"),
            "furnishing_status": payload.get("furnishing_status_name"),
            "size_sqft": payload.get("size_sqft"),
            "bedrooms_bhk": payload.get("BHK"),
            "bathrooms": payload.get("numBathrooms"),
            "balconies": payload.get("numBalconies"),
            "total_rooms": payload.get("rooms_num"),
            "security_deposit_inr": payload.get("SecurityDeposit"),
            "price_negotiable": bool(payload.get("isNegotiable")),
            "listing_age_days": payload.get("verification_days"),
            "coordinates": {
                "latitude": payload.get("latitude"),
                "longitude": payload.get("longitude")
            }
        }
    }
    return json.dumps(report, indent=2)


def generate_location_map(df: pd.DataFrame, city: str, lat: float, lon: float) -> go.Figure:
    """Generates an interactive Mapbox plot overlaying historical city listings.

    Colors points based on historical rent prices and overlays a solid red marker
    pinpointing the user's current selected property location.

    Args:
        df: Combined historical listings DataFrame.
        city: Selected city context.
        lat: Latitude of the current property.
        lon: Longitude of the current property.

    Returns:
        Plotly Figure object representing the map.
    """
    fig = go.Figure()

    # 1. Plot historical city listings as a reference layer
    if not df.empty and "city" in df.columns:
        city_df = df[
            (df["city"].astype(str) == city) & 
            (df["latitude"].notna()) & 
            (df["longitude"].notna()) & 
            (df["price"].notna())
        ].copy()
        
        # Downsample if city size is extremely large to preserve rendering speeds
        if len(city_df) > 600:
            city_df = city_df.sample(600, random_state=config.RANDOM_STATE)

        if not city_df.empty:
            fig.add_trace(
                go.Scattermapbox(
                    lat=city_df["latitude"],
                    lon=city_df["longitude"],
                    mode="markers",
                    marker=go.scattermapbox.Marker(
                        size=6,
                        color=city_df["price"],
                        colorscale="Viridis",
                        showscale=True,
                        cmin=float(city_df["price"].min()),
                        cmax=float(city_df["price"].quantile(0.96)),
                        colorbar=dict(
                            title="Rent (₹)",
                            thickness=14,
                            x=1.02,
                            len=0.9
                        ),
                        opacity=0.55
                    ),
                    text=city_df.apply(
                        lambda r: f"Area: {r.get('location', 'N/A')}<br>Rent: ₹{r['price']:,}<br>{r['BHK']} BHK", 
                        axis=1
                    ),
                    hoverinfo="text",
                    name="Market Listings"
                )
            )

    # 2. Plot the user's selected property as a distinct red marker
    fig.add_trace(
        go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=14,
                color="#EF4444",  # Tailwind red-500
                symbol="circle"
            ),
            text=["Selected Property Location"],
            hoverinfo="text",
            name="Selected Property"
        )
    )

    # Set map configuration defaults
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",  # Open source, token-less style
            center=dict(lat=lat, lon=lon),
            zoom=11.2
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

