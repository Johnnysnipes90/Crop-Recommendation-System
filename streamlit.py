import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# Load configurations
CONFIG_DIR = "config"
MODEL_PATH = "model/best_rf_model.joblib"
TRAINING_COLUMNS_PATH = f"{CONFIG_DIR}/training_columns.json"

with open(TRAINING_COLUMNS_PATH, 'r') as file:
    training_columns = json.load(file)['training_columns']

# Load model
model = joblib.load(MODEL_PATH)

# Extract feature importance (if available)
if hasattr(model, "feature_importances_"):
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": training_columns,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

# Background and custom styling with a green theme
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #a8d08d, #3a6d40);  /* Green gradient background */
        color: #ffffff;  /* White text for contrast */
    }
    .header {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #2c4f2c;
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from {
            text-shadow: 0 0 10px #58d68d, 0 0 20px #2c4f2c;
        }
        to {
            text-shadow: 0 0 20px #58d68d, 0 0 30px #2c4f2c;
        }
    }
    .form-style {
        background-color: rgba(255, 255, 255, 0.9);  /* Lighter background for form */
        padding: 20px;
        border-radius: 15px;
    }
    .button-style:hover {
        background-color: #3a6d40;  /* Dark green button hover effect */
        color: white;
    }

    /* Sidebar Styling */
    .sidebar .sidebar-content {
        color: #ffffff;
        background-color: #2c6e33;  /* Darker green sidebar background */
        font-size: 18px;
    }
    .sidebar .sidebar-content h1 {
        color: #ffffff;
        font-size: 22px;
        font-weight: bold;
    }
    .sidebar .sidebar-content p {
        color: #ffffff;
        font-size: 16px;
    }
    .sidebar .sidebar-content li {
        font-size: 16px;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("🌿 Crop Recommendation System")
st.sidebar.markdown(
    """
    **Instructions**:
    - Input soil and weather parameters.
    - Get the best crop recommendation.
    - Download a prediction report for your records.
    """
)

# Header
st.markdown('<div class="header">Crop Recommendation System</div>', unsafe_allow_html=True)

# Descriptions for features
feature_descriptions = {
    "Nitrogen": "Nitrogen content ratio in the soil (integer value).",
    "Phosphorus": "Phosphorous content ratio in the soil (integer value).",
    "Potassium": "Potassium content ratio in the soil (integer value).",
    "Temperature": "Soil temperature in degrees Celsius.",
    "Humidity": "Relative humidity percentage in the field.",
    "pH_Value": "Measure of soil acidity/alkalinity.",
    "Rainfall": "Amount of rainfall in millimeters."
}

# Input Form
input_features = {}
with st.form("prediction_form", clear_on_submit=False):
    st.markdown('<div class="form-style">', unsafe_allow_html=True)
    st.header("Enter Soil and Environmental Parameters:")

    for col in training_columns:
        if col in ["Nitrogen", "Phosphorus", "Potassium"]:
            input_features[col] = st.number_input(
                f"{col} (integer)", value=0, step=1, help=feature_descriptions[col]
            )
        else:
            input_features[col] = st.number_input(f"{col}", value=0.0, help=feature_descriptions[col])
    
    submit = st.form_submit_button("🌾 Predict")
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction Logic
if submit:
    try:
        # Prepare input and predict
        features = np.array([list(input_features.values())])
        prediction = model.predict(features)
        recommended_crop = prediction[0]  # Extract prediction
        
        # Display prediction
        st.success(f"🌱 Recommended Crop: **{recommended_crop}**")

        # Visualize input data
        st.subheader("Your Input Data:")
        input_df = pd.DataFrame([input_features])
        st.table(input_df)

        # Feature Importance Visualization
        if "feature_importances_" in dir(model):
            st.subheader("🔍 Feature Importance:")
            st.bar_chart(feature_importance_df.set_index("Feature")["Importance"])

        # Generate downloadable report
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
    Developed by **John Olalemi** | Contact: [johnolalemi90@gmail.com]  
    🌾 Empowering Farmers with AI Solutions.
    """
)
