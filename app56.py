# ✅ Install necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import xgboost as xgb

# ✅ Load trained model & scaler
try:
    with open("xgboost_heart_disease (1).pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model or Scaler Not Found! Ensure 'xgboost_heart_disease.pkl' and 'scaler.pkl' exist.")
    st.stop()

# ✅ Define required features for prediction
required_features = [
    "Age", "Gender", "Systolic_BP", "Diastolic_BP", "Cholesterol",
    "Glucose", "Smoker", "Alcohol", "Physical_Activity", "BMI",
    "BP_Diff", "Pulse_Pressure", "Heart_Rate_Ratio", "BP_Variation"
]

# ✅ Streamlit App UI
st.title("🔍 AI-Powered Heart Disease Prediction System")
st.sidebar.header("📋 Enter Patient Details")

# 📌 **User Input Fields**
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=40, step=1)
gender = st.sidebar.radio("Gender", ["Male", "Female"])
systolic_bp = st.sidebar.number_input("Systolic BP (ap_hi)", min_value=80, max_value=200, value=120, step=1)
diastolic_bp = st.sidebar.number_input("Diastolic BP (ap_lo)", min_value=50, max_value=130, value=80, step=1)
cholesterol = st.sidebar.selectbox("Cholesterol Level", [1, 2, 3])  # 1: Normal, 2: Above Normal, 3: High
glucose = st.sidebar.selectbox("Glucose Level", [1, 2, 3])  # 1: Normal, 2: Above Normal, 3: High
smoker = st.sidebar.radio("Smoker?", ["No", "Yes"])
alcohol = st.sidebar.radio("Alcohol Consumption?", ["No", "Yes"])
physical_activity = st.sidebar.radio("Physically Active?", ["No", "Yes"])
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170, step=1)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70, step=1)

# ✅ Convert Inputs to DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [1 if gender == "Male" else 0],
    "Systolic_BP": [systolic_bp],
    "Diastolic_BP": [diastolic_bp],
    "Cholesterol": [cholesterol],
    "Glucose": [glucose],
    "Smoker": [1 if smoker == "Yes" else 0],
    "Alcohol": [1 if alcohol == "Yes" else 0],
    "Physical_Activity": [1 if physical_activity == "Yes" else 0],
    "BMI": [weight / ((height / 100) ** 2)],
    "BP_Diff": [systolic_bp - diastolic_bp],
    "Pulse_Pressure": [systolic_bp - diastolic_bp],
    "Heart_Rate_Ratio": [systolic_bp / diastolic_bp],
    "BP_Variation": [(systolic_bp - diastolic_bp) / diastolic_bp]
})

# ✅ Ensure input_data has all required columns
input_data = input_data.reindex(columns=required_features, fill_value=0)

# ✅ Prediction Logic
if st.button("🔍 Predict"):
    try:
        # ✅ Scale input data
        input_data_scaled = scaler.transform(input_data)

        # ✅ Make prediction
        prediction = model.predict(input_data_scaled)[0]
        prediction_proba = model.predict_proba(input_data_scaled)[:, 1][0] * 100

        # ✅ Display results
        if prediction == 1:
            st.error(f"⚠️ High Risk of Heart Disease ({prediction_proba:.2f}%)")
        else:
            st.success(f"✅ Low Risk of Heart Disease ({prediction_proba:.2f}%)")

        # ✅ SHAP Explanation
        explainer = shap.Explainer(model)
        shap_values = explainer(input_data_scaled)
        st.subheader("🔍 Feature Importance")
        st.write("Higher values indicate stronger influence on heart disease risk.")
        st.pyplot(shap.force_plot(explainer.expected_value, shap_values.values, input_data, matplotlib=True))

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")