"""Streamlit Web Application for the UrbanNest Rent Predictor.

This script implements the user interface, incorporating a multi-tab portfolio layout:
1. Rent Predictor: Property configurations, interactive map, price estimation, and export.
2. Analytics & Optimization: Hyperparameter comparison (Grid vs. Random vs. Optuna) and plots.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

import config
import utils

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="UrbanNest Rent Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Premium custom styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;700;800&family=Manrope:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero-banner {
        border-radius: 18px;
        border: 1px solid rgba(14, 165, 233, 0.25);
        background: radial-gradient(120% 140% at 80% -10%, rgba(14, 165, 233, 0.15), rgba(15, 23, 42, 0.95)),
                    linear-gradient(135deg, #0F172A 0%, #111827 100%);
        color: #F3F4F6;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    .hero-banner h1 {
        margin: 0.2rem 0;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.2rem;
        background: linear-gradient(90deg, #E0F2FE 0%, #0EA5E9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-banner p {
        margin: 0.4rem 0 0 0;
        color: #9CA3AF;
        font-size: 1rem;
    }

    .status-badge {
        display: inline-block;
        font-size: 0.72rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-weight: 700;
        color: #38BDF8;
        background: rgba(14, 165, 233, 0.15);
        border: 1px solid rgba(14, 165, 233, 0.4);
        border-radius: 999px;
        padding: 0.2rem 0.7rem;
    }

    .card-panel {
        border: 1px solid rgba(75, 85, 99, 0.25);
        border-radius: 14px;
        padding: 1.2rem;
        background: linear-gradient(180deg, #0F172A 0%, #111827 100%);
        margin-bottom: 1rem;
    }

    .result-container {
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 14px;
        padding: 1.3rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(14, 165, 233, 0.08) 100%);
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .result-value {
        margin: 0;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        color: #10B981;
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    }

    .stButton > button {
        height: 3rem;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: 1px solid rgba(14, 165, 233, 0.6) !important;
        background: linear-gradient(90deg, #0EA5E9, #0284C7) !important;
        color: #FFFFFF !important;
        transition: all 0.2s ease-in-out;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 18px rgba(14, 165, 233, 0.3);
        transform: translateY(-1px);
    }
    
    .metrics-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #F3F4F6;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Load ML artifacts & Historical dataset
# -----------------------------------------------------------------------------
model, encoders, metadata, load_error = utils.load_artifacts()
market_df = utils.load_market_df()

if load_error:
    st.error(f"⚠️ Artifact Load Failure: {load_error}")
    st.stop()

# Auto-detect features
feature_names = config.FEATURE_NAMES
size_col = "Size_ft²"

# -----------------------------------------------------------------------------
# Sidebar: UI / Display Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/real-estate.png", width=70)
    st.title("Settings")
    
    st.subheader("Inference Settings")
    confidence_lvl = st.slider("Prediction Interval (Confidence)", 70, 95, 85, step=5, help="Controls the width of the predicted rent range.")
    live_prediction = st.toggle("Real-Time Inference", value=True, help="Update estimations automatically as inputs change.")

    st.divider()
    
    st.subheader("System Info")
    st.markdown(
        f"""
        - **Model Family**: `Random Forest Regressor`
        - **Features Logged**: `{len(feature_names)}` columns
        - **Categorical Columns**: `Status, property_type, city, location`
        """
    )
    if config.MODEL_PATH.exists():
        m_time = pd.Timestamp(config.MODEL_PATH.stat().st_mtime, unit='s', tz='UTC').tz_convert('Asia/Kolkata')
        st.caption(f"Model updated: {m_time.strftime('%Y-%m-%d %H:%M %Z')}")

# -----------------------------------------------------------------------------
# Hero Section
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <span class="status-badge">MLOps Portfolio Showcase</span>
        <h1>UrbanNest Rent Predictor</h1>
        <p>A machine learning engine evaluating properties across Mumbai, Delhi, Pune, and Hisar using optimized random forests.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize tabs
pred_tab, opt_tab = st.tabs(["🔮 Rent Predictor", "📊 Optimization & Analytics"])

# -----------------------------------------------------------------------------
# Tab 1: Rent Predictor
# -----------------------------------------------------------------------------
with pred_tab:
    city_map = utils.city_location_map(market_df)
    city_choices = sorted(list(city_map.keys())) if city_map else config.DEFAULT_CITIES
    
    # Grid Layout for main inputs
    col_left, col_right = st.columns([1.1, 0.9])
    
    with col_left:
        st.markdown('<div class="card-panel">', unsafe_allow_html=True)
        st.markdown('<div class="metrics-header">Property Characteristics</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            city_sel = st.selectbox("City", city_choices, key="prop_city")
            
            # Determine locations based on chosen city
            loc_choices = city_map.get(city_sel, []) if city_map else utils.encoder_values(encoders, "location")
            if not loc_choices:
                loc_choices = ["Default Location"]
            loc_sel = st.selectbox("Neighborhood / Location", loc_choices)
            
            status_choices = utils.encoder_values(encoders, "Status") or config.DEFAULT_STATUSES
            furnishing_sel = st.selectbox("Furnishing Status", status_choices)
            
            ptype_choices = utils.encoder_values(encoders, "property_type") or config.DEFAULT_PROPERTY_TYPES
            ptype_sel = st.selectbox("Property Type", ptype_choices)

        with c2:
            size_unit = st.radio("Area Unit", ["Square Feet (sqft)", "Square Meters (sqm)"], horizontal=True)
            size_input = st.number_input("Property Size", min_value=150.0, max_value=15000.0, value=1000.0, step=50.0)
            bhk_sel = st.slider("Bedrooms (BHK)", 1, 6, 2)
            bathrooms_input = st.number_input("Bathrooms", min_value=1, max_value=6, value=2, step=1)
            balconies_input = st.number_input("Balconies", min_value=0, max_value=4, value=1, step=1)
            
        c3, c4, c5 = st.columns(3)
        with c3:
            total_rooms_input = st.number_input("Total Rooms", min_value=1, max_value=12, value=int(bhk_sel + 1), step=1)
        with c4:
            listing_age_input = st.slider("Listing Age (Days)", min_value=0, max_value=365, value=30)
        with c5:
            negotiable_sel = st.selectbox("Rent Negotiable?", ["No", "Yes"]) == "Yes"

        security_deposit_input = st.number_input("Security Deposit (₹)", min_value=0.0, max_value=2000000.0, value=50000.0, step=5000.0)

        # Dynamic Coordinates Section (presets linked to cities)
        default_coords = config.CITY_COORDINATES.get(city_sel, {"latitude": 19.1631, "longitude": 72.9566})
        
        with st.expander("🌐 Geographical Location Details"):
            c_lat, c_lon = st.columns(2)
            with c_lat:
                lat_input = st.number_input(
                    "Latitude", 
                    min_value=8.0, 
                    max_value=38.0, 
                    value=default_coords["latitude"], 
                    format="%.5f", 
                    key=f"lat_input_{city_sel}",
                    help="Linked automatically to city average."
                )
            with c_lon:
                lon_input = st.number_input(
                    "Longitude", 
                    min_value=68.0, 
                    max_value=98.0, 
                    value=default_coords["longitude"], 
                    format="%.5f", 
                    key=f"lon_input_{city_sel}",
                    help="Linked automatically to city average."
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        predict_btn = st.button("Calculate Rent Prediction", use_container_width=True)

    with col_right:
        # Custom Mapbox visualization overlaying surrounding market listings
        st.markdown('<div class="card-panel">', unsafe_allow_html=True)
        st.markdown('<div class="metrics-header">Property Geographical Context</div>', unsafe_allow_html=True)
        fig_map = utils.generate_location_map(market_df, city_sel, lat_input, lon_input)
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Inference Trigger
    # -------------------------------------------------------------------------
    if predict_btn or live_prediction:
        # Convert size to sqft if sqm is selected
        size_sqft = float(size_input * 10.7639) if "sqm" in size_unit.lower() else float(size_input)

        # Encode categorical variables using safe fallbacks
        encoded_payload = {
            "location": utils.safe_encode(encoders, "location", str(loc_sel), city_sel, market_df),
            "city": utils.safe_encode(encoders, "city", str(city_sel), city_sel, market_df),
            "latitude": float(lat_input),
            "longitude": float(lon_input),
            "numBathrooms": int(bathrooms_input),
            "numBalconies": int(balconies_input),
            "isNegotiable": 1 if negotiable_sel else 0,
            "SecurityDeposit": float(security_deposit_input),
            "Status": utils.safe_encode(encoders, "Status", str(furnishing_sel), city_sel, market_df),
            size_col: float(size_sqft),
            "BHK": int(bhk_sel),
            "rooms_num": int(total_rooms_input),
            "property_type": utils.safe_encode(encoders, "property_type", str(ptype_sel), city_sel, market_df),
            "verification_days": float(listing_age_input),
        }

        # Build single row for prediction
        input_row = utils.build_row(encoded_payload)
        low_bound, pred_val, high_bound = utils.prediction_interval(model, input_row, confidence_lvl)
        
        # Benchmarking against market historical data
        m_avg_price, benchmark_lbl = utils.market_average(market_df, city_sel, loc_sel)
        pct_diff = ((pred_val - m_avg_price) / m_avg_price * 100) if m_avg_price > 0 else 0.0

        st.divider()

        # Display results panel
        col_res1, col_res2 = st.columns([1.1, 0.9])
        
        with col_res1:
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.markdown(f"<p class='result-value'>₹{pred_val:,.0f}/mo</p>", unsafe_allow_html=True)
            st.markdown(f"**Estimated Fair Market Rent** ({confidence_lvl}% confidence interval)")
            st.markdown(f"<div style='color: #6B7280; font-size: 0.9rem;'>Bounds: ₹{low_bound:,.0f} — ₹{high_bound:,.0f}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Metrics Row
            m1, m2, m3 = st.columns(3)
            m1.metric("Historical Benchmark", f"₹{m_avg_price:,.0f}" if m_avg_price > 0 else "N/A", benchmark_lbl)
            m2.metric("Variance vs. Benchmark", f"{pct_diff:+.1f}%", help="Compares predicted rent with historical listings.")
            m3.metric("Inputs Overview", f"{bhk_sel} BHK • {size_sqft:,.0f} sqft")

            # Report Export Downloader
            raw_payload = {
                "city_name": city_sel,
                "location_name": loc_sel,
                "property_type_name": ptype_sel,
                "furnishing_status_name": furnishing_sel,
                "size_sqft": size_sqft,
                "BHK": bhk_sel,
                "numBathrooms": bathrooms_input,
                "numBalconies": balconies_input,
                "rooms_num": total_rooms_input,
                "SecurityDeposit": security_deposit_input,
                "isNegotiable": negotiable_sel,
                "verification_days": listing_age_input,
                "latitude": lat_input,
                "longitude": lon_input
            }
            json_report = utils.generate_json_report(raw_payload, pred_val, low_bound, high_bound, confidence_lvl)
            
            st.download_button(
                label="📥 Download Structured JSON Rent Report",
                data=json_report,
                file_name=f"urbannest_report_{city_sel}_{loc_sel.replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )

        with col_res2:
            st.markdown('<div class="card-panel" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown('<div class="metrics-header">Simulated What-If Comparisons</div>', unsafe_allow_html=True)
            
            # What-If scenario 1: Size Increase
            size_inc_row = input_row.copy()
            size_inc_row[size_col] = size_inc_row[size_col] * 1.15
            size_inc_pred = float(model.predict(size_inc_row.to_numpy(dtype=float, copy=False))[0])
            
            # What-If scenario 2: Adding a Bedroom
            bhk_inc_row = input_row.copy()
            bhk_inc_row["BHK"] = min(int(bhk_inc_row["BHK"].iloc[0]) + 1, 6)
            bhk_inc_pred = float(model.predict(bhk_inc_row.to_numpy(dtype=float, copy=False))[0])
            
            w1, w2 = st.columns(2)
            w1.metric("+15% Property Size", f"₹{size_inc_pred:,.0f}", f"{size_inc_pred - pred_val:+,.0f}")
            w2.metric("+1 Bedroom (BHK)", f"₹{bhk_inc_pred:,.0f}", f"{bhk_inc_pred - pred_val:+,.0f}")
            
            # Feature driver chart
            fig_importance = utils.generate_feature_importance_chart(model)
            if fig_importance is not None:
                st.plotly_chart(fig_importance, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tab 2: Analytics & Optimization Dashboard
# -----------------------------------------------------------------------------
with opt_tab:
    st.subheader("Model Hyperparameter Tuning Analysis")
    st.markdown(
        """
        To identify the most compute-efficient hyperparameter optimization (HPO) technique for high-accuracy estimation,
        we evaluated and compared three search frameworks over a strict **60-trial budget** using 5-fold cross-validation.
        """
    )
    
    # Read metadata dynamically
    if metadata and "methods" in metadata:
        methods_data = metadata["methods"]
        
        # Display cards
        c_grid, c_rand, c_bayes = st.columns(3)
        
        with c_grid:
            grid_info = methods_data.get("Grid Search", {})
            st.metric("Grid Search MAE", f"₹{grid_info.get('best_cv_mae', 0):,.2f}", help=grid_info.get("description"))
            st.markdown(f"**Runtime**: `{grid_info.get('time_seconds', 0):.1f} seconds`")
            st.caption(f"Best Params: `{grid_info.get('best_params', {})}`")
            
        with c_rand:
            rand_info = methods_data.get("Random Search", {})
            st.metric("Random Search MAE", f"₹{rand_info.get('best_cv_mae', 0):,.2f}", help=rand_info.get("description"))
            st.markdown(f"**Runtime**: `{rand_info.get('time_seconds', 0):.1f} seconds`")
            st.caption(f"Best Params: `{rand_info.get('best_params', {})}`")
            
        with c_bayes:
            bayes_info = methods_data.get("Bayesian (Optuna)", {})
            st.metric("Bayesian (Optuna) MAE", f"₹{bayes_info.get('best_cv_mae', 0):,.2f}", help=bayes_info.get("description"))
            st.markdown(f"**Runtime**: `{bayes_info.get('time_seconds', 0):.1f} seconds`")
            st.caption(f"Best Params: `{bayes_info.get('best_params', {})}`")
            
        # Summary Box
        st.info(
            f"🏆 **Tuning Winner**: `{metadata.get('winner', 'Grid Search')}` achieved the lowest validation error. "
            f"🎯 **Generalization Error (Untouched Test set MAE)**: `₹{metadata.get('final_test_mae', 0):,.2f}`"
        )
    else:
        st.warning(" Tunning summary metadata not available. Displaying static performance records.")

    # Visualization Columns
    st.subheader("Convergence & Hyperparameter Exploration Plots")
    p1, p2 = st.columns(2)
    
    with p1:
        st.markdown("**HPO Convergence History (`trials_vs_error.png`)**")
        trials_plot = config.PLOTS_DIR / "trials_vs_error.png"
        if trials_plot.exists():
            st.image(str(trials_plot), use_container_width=True)
        else:
            st.caption("Plot `plots/trials_vs_error.png` not found.")
            
    with p2:
        st.markdown("**Bayesian Hyperparameter Search Space (`optuna_hyperparameter_space.png`)**")
        space_plot = config.PLOTS_DIR / "optuna_hyperparameter_space.png"
        if space_plot.exists():
            st.image(str(space_plot), use_container_width=True)
        else:
            st.caption("Plot `plots/optuna_hyperparameter_space.png` not found.")

st.divider()
st.caption("UrbanNest Rent Predictor • Architected by Lead MLOps Engineer")
