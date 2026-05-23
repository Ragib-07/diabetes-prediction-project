"""
app.py

Streamlit web app for Diabetes Prediction and Health Monitoring System.

Run:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"


@st.cache_resource
def load_model_artifact(model_path: Path = MODEL_PATH) -> Dict:
    """Load the saved model artifact only once for faster Streamlit performance."""
    return joblib.load(model_path)


def get_risk_level(probability: float) -> Tuple[str, str]:
    """Convert model probability into an easy-to-understand risk level."""
    if probability < 0.35:
        return "Low", "🟢"
    if probability < 0.65:
        return "Medium", "🟡"
    return "High", "🔴"


def bmi_advice(bmi: float) -> str:
    if bmi < 18.5:
        return "BMI is below the usual healthy range. Consider speaking with a healthcare professional about balanced nutrition."
    if bmi < 25:
        return "BMI is within the usual healthy range. Maintain regular physical activity and balanced meals."
    if bmi < 30:
        return "BMI is in the overweight range. Small changes in diet and activity may help reduce long-term risk."
    return "BMI is in the obesity range. Consider a structured weight-management plan with professional guidance."


def glucose_advice(glucose: float) -> str:
    if glucose < 70:
        return "Glucose value is low. If this is a real reading and you feel unwell, seek medical advice."
    if glucose < 140:
        return "Glucose value is in a lower-risk range for this simple monitoring view. Keep tracking healthy habits."
    if glucose < 200:
        return "Glucose value is elevated. Reduce sugary foods and consider checking with a clinician."
    return "Glucose value is very high. Please consult a healthcare professional for proper testing and advice."


def blood_pressure_advice(bp: float) -> str:
    # The Pima dataset uses diastolic blood pressure in mm Hg.
    if bp < 60:
        return "Blood pressure value is low. Monitor symptoms such as dizziness and consult a professional if needed."
    if bp <= 80:
        return "Blood pressure value looks within a common target range for diastolic pressure."
    if bp <= 90:
        return "Blood pressure value is slightly elevated. Reduce salt intake, manage stress, and stay active."
    return "Blood pressure value is high. Regular monitoring and medical advice are recommended."


def age_advice(age: int) -> str:
    if age < 35:
        return "At a younger age, prevention habits matter: regular exercise, balanced diet, and routine checkups."
    if age < 50:
        return "Risk can increase with age. Keep monitoring glucose, BMI, blood pressure, and family history."
    return "Age-related risk can be higher. Regular screening and professional checkups are strongly recommended."


def collect_user_inputs() -> Dict[str, float]:
    """Create sidebar input widgets and return values in model feature order."""
    st.sidebar.header("Enter Health Details")
    st.sidebar.caption("Use realistic values for better prediction quality.")

    return {
        "Pregnancies": st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1),
        "Glucose": st.sidebar.number_input("Glucose Level", min_value=0, max_value=300, value=120, step=1),
        "BloodPressure": st.sidebar.number_input("Blood Pressure (diastolic mm Hg)", min_value=0, max_value=180, value=72, step=1),
        "SkinThickness": st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1),
        "Insulin": st.sidebar.number_input("Insulin Level", min_value=0, max_value=900, value=80, step=1),
        "BMI": st.sidebar.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0, step=0.1),
        "DiabetesPedigreeFunction": st.sidebar.number_input(
            "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01
        ),
        "Age": st.sidebar.number_input("Age", min_value=1, max_value=120, value=30, step=1),
    }


def show_indicator_chart(inputs: Dict[str, float]) -> None:
    """Show a simple chart comparing key user indicators with reference values."""
    chart_data = pd.DataFrame(
        {
            "Indicator": ["Glucose", "BMI", "Blood Pressure"],
            "Your Value": [inputs["Glucose"], inputs["BMI"], inputs["BloodPressure"]],
            "Reference Value": [140, 25, 80],
        }
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    chart_data.set_index("Indicator").plot(kind="bar", ax=ax)
    ax.set_title("Basic Health Indicator Comparison")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)


def show_health_suggestions(inputs: Dict[str, float]) -> None:
    """Display personalized but general health monitoring suggestions."""
    suggestions: List[str] = [
        bmi_advice(float(inputs["BMI"])),
        glucose_advice(float(inputs["Glucose"])),
        blood_pressure_advice(float(inputs["BloodPressure"])),
        age_advice(int(inputs["Age"])),
    ]

    st.subheader("Health Monitoring Suggestions")
    for suggestion in suggestions:
        st.write(f"• {suggestion}")

    st.info(
        "These suggestions are general educational guidance only. This app cannot diagnose diabetes. "
        "Please consult a qualified healthcare professional for medical decisions."
    )


def main() -> None:
    st.set_page_config(
        page_title="Diabetes Prediction System",
        page_icon="🩺",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .main-title {font-size: 2.4rem; font-weight: 800; margin-bottom: 0.2rem;}
        .subtitle {font-size: 1.05rem; color: #555; margin-bottom: 1.5rem;}
        .metric-card {padding: 1rem; border-radius: 0.8rem; border: 1px solid #e6e6e6; background: #fafafa;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">Diabetes Prediction and Health Monitoring System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">A machine learning web app for portfolio, final-year, and GitHub projects.</div>',
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
        st.error("Model file not found. Please run `python model_training.py` first to create diabetes_model.pkl.")
        st.stop()

    artifact = load_model_artifact(MODEL_PATH)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    inputs = collect_user_inputs()
    input_df = pd.DataFrame([inputs], columns=feature_columns)

    tab_prediction, tab_monitoring, tab_model = st.tabs(["Prediction", "Health Monitoring", "Model Details"])

    with tab_prediction:
        left, right = st.columns([1.1, 1])

        with left:
            st.subheader("Input Summary")
            st.dataframe(input_df, use_container_width=True)

            predict_button = st.button("Predict Diabetes Risk", type="primary", use_container_width=True)

            if predict_button:
                prediction = int(model.predict(input_df)[0])

                if hasattr(model, "predict_proba"):
                    probability = float(model.predict_proba(input_df)[0][1])
                else:
                    probability = 0.75 if prediction == 1 else 0.25

                risk_level, icon = get_risk_level(probability)
                result_text = "Likely Diabetic" if prediction == 1 else "Likely Not Diabetic"

                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("Prediction", result_text)
                col2.metric("Risk Level", f"{icon} {risk_level}")
                col3.metric("Estimated Probability", f"{probability * 100:.1f}%")

                st.progress(min(max(probability, 0.0), 1.0))

                if risk_level == "High":
                    st.error("High risk detected. Please consider professional medical testing and guidance.")
                elif risk_level == "Medium":
                    st.warning("Medium risk detected. Improving lifestyle habits and monitoring readings may help.")
                else:
                    st.success("Low risk detected. Keep maintaining healthy habits and routine checkups.")

        with right:
            st.subheader("Basic Indicator Chart")
            show_indicator_chart(inputs)

    with tab_monitoring:
        show_health_suggestions(inputs)

    with tab_model:
        st.subheader("Training Summary")
        st.write(f"Best model: **{artifact.get('model_name', 'Unknown')}**")
        st.write(f"Trained at: **{artifact.get('trained_at', 'Unknown')}**")

        metrics_df = pd.DataFrame(artifact.get("metrics", []))
        if not metrics_df.empty:
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.write("No metrics found in the model artifact.")

        st.caption("Model selection is based mainly on F1-score and then accuracy.")

    st.markdown("---")
    st.caption("Disclaimer: This application is for educational purposes only and is not a medical diagnosis tool.")


if __name__ == "__main__":
    main()
