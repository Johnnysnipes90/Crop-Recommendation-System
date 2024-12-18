import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# Load configurations
CONFIG_DIR = "config"
MODEL_PATH = "model/best_rf_model.joblib"
TRAINING_COLUMNS_PATH = f"{CONFIG_DIR}/training_columns.json"
LABEL_MAPPING_PATH = f"{CONFIG_DIR}/label_mapping.json"

with open(TRAINING_COLUMNS_PATH, 'r') as file:
    training_columns = json.load(file)['training_columns']

with open(LABEL_MAPPING_PATH, 'r') as file:
    label_mapping = json.load(file)

# Load model
model = joblib.load(MODEL_PATH)

# Feature Descriptions
feature_descriptions = {
    "Nitrogen": "Nitrogen content in the soil (integer value).",
    "Phosphorus": "Phosphorus content in the soil (integer value).",
    "Potassium": "Potassium content in the soil (integer value).",
    "Temperature": "Soil temperature in degrees Celsius.",
    "Humidity": "Relative humidity percentage in the field.",
    "pH_Value": "Measure of soil acidity/alkalinity.",
    "Rainfall": "Amount of rainfall in millimeters.",
}

# Sidebar Theme Toggle
theme = st.sidebar.radio("Choose Theme", options=["🌞 Light Theme", "🌙 Dark Theme"])
if theme == "🌙 Dark Theme":
    st.markdown(
        """
        <style>
        .stApp { background-color: #1E1E1E; color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Sidebar Instructions
st.sidebar.title("🌿 Crop Recommendation System")
st.sidebar.markdown(
    """
    **How It Works**:
    - Input soil and environmental parameters.
    - View the best crop recommendation.
    - Download a prediction report.
    """
)

# Header
st.markdown(
    '<h1 style="text-align: center; color: #4CAF50;">🌾 Crop Recommendation System</h1>',
    unsafe_allow_html=True,
)

# Input Form
st.subheader("Enter Soil and Environmental Parameters")
input_features = {}

with st.form("prediction_form", clear_on_submit=False):
    for col in training_columns:
        if col in ["Nitrogen", "Phosphorus", "Potassium"]:
            input_features[col] = st.number_input(
                f"{col} (integer)", value=0, step=1, help=feature_descriptions[col]
            )
        elif col == "pH_Value":
            input_features[col] = st.slider(
                "pH Value (1-14)", min_value=1.0, max_value=14.0, value=7.0, step=0.1
            )
        else:
            input_features[col] = st.number_input(f"{col}", value=0.0, help=feature_descriptions[col])

    submit = st.form_submit_button("🌾 Predict")

# Prediction and Output
if submit:
    try:
        # Prepare input features
        features = np.array([list(input_features.values())])
        prediction = model.predict(features)[0]
        recommended_crop = label_mapping[str(prediction)]

        # Display prediction
        st.success(f"🌱 Recommended Crop: **{recommended_crop}**")
        st.balloons()

        # Show Input Data
        st.subheader("Your Input Data")
        input_df = pd.DataFrame([input_features])
        st.table(input_df)

        # Feature Importance Visualization
        if hasattr(model, "feature_importances_"):
            st.subheader("🔍 Feature Importance")
            feature_importances = model.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": training_columns,
                "Importance": feature_importances
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(importance_df.set_index("Feature")["Importance"])

        # Downloadable Prediction Report
        report = input_features.copy()
        report["Recommended Crop"] = recommended_crop
        report_df = pd.DataFrame([report])
        csv = report_df.to_csv(index=False)

        st.download_button(
            label="📥 Download Report",
            data=csv,
            file_name="crop_recommendation.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    Developed by **John Olalemi**  
    🌾 Empowering Farmers with AI Solutions.
    """
)
