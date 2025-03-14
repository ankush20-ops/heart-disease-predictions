import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import base64
import os
from io import BytesIO
from sklearn.preprocessing import StandardScaler

# Load Model & Scaler
MODEL_PATH = "xgboost_heart_disease (1).pkl"
SCALER_PATH = "scaler.pkl"

# Load trained model and scaler
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
    
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Expected Features (Ensuring consistency)
EXPECTED_FEATURES = ['Age', 'Gender', 'Systolic_BP', 'Diastolic_BP', 'Cholesterol', 'Glucose',
                     'Smoker', 'Alcohol', 'Physical_Activity', 'BMI', 'BP_Ratio']

# Heart Disease Types
HEART_DISEASE_TYPES = {
    0: "No Heart Disease",
    1: "Possible Cardiovascular Disease"
}

# Function to predict heart disease
def predict_heart_disease(input_data):
    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)[0]
    return prediction, HEART_DISEASE_TYPES[prediction]

# Function to generate SHAP explanation
def generate_shap_explanation(input_data):
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)
    return shap_values

# Function to process PDF and predict automatically
def process_pdf(file):
    # Implement PDF extraction logic (For now, dummy extraction)
    extracted_data = {
        "Age": 50, "Gender": 1, "Systolic_BP": 140, "Diastolic_BP": 90, "Cholesterol": 2,
        "Glucose": 1, "Smoker": 0, "Alcohol": 0, "Physical_Activity": 1,
        "BMI": 27.5, "BP_Ratio": 1.56
    }
    return pd.DataFrame([extracted_data])

# Function to download report
def generate_pdf_report(input_data, prediction):
    buffer = BytesIO()
    with open("report.txt", "w") as f:
        f.write(f"Heart Disease Prediction Report\n\n")
        for key, value in input_data.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nPrediction: {prediction}")
    with open("report.txt", "rb") as f:
        buffer.write(f.read())
    buffer.seek(0)
    return buffer

# Streamlit UI
st.set_page_config(page_title="AI Heart Disease Predictor", layout="wide")
st.markdown("<h1 style='text-align: center; color: red;'>❤️ AI Heart Disease Prediction System</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("User Input Features")
age = st.sidebar.slider("Age", 20, 90, 50)
gender = st.sidebar.radio("Gender", ["Male", "Female"])
systolic_bp = st.sidebar.slider("Systolic BP", 80, 200, 120)
diastolic_bp = st.sidebar.slider("Diastolic BP", 50, 130, 80)
cholesterol = st.sidebar.selectbox("Cholesterol", sorted([1, 2, 3]))
glucose = st.sidebar.selectbox("Glucose", sorted([1, 2, 3]))
smoker = st.sidebar.selectbox("Smoker", sorted([0, 1]))
alcohol = st.sidebar.selectbox("Alcohol", sorted([0, 1]))
physical_activity = st.sidebar.selectbox("Physical Activity", sorted([0, 1]))
height = st.sidebar.number_input("Height (cm)", 100, 220, 170)
weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)

# Feature Engineering
bmi = weight / ((height / 100) ** 2)
bp_ratio = systolic_bp / diastolic_bp

# Convert categorical values
gender = 1 if gender == "Male" else 0

# Organize input data
input_data = [age, gender, systolic_bp, diastolic_bp, cholesterol, glucose,
              smoker, alcohol, physical_activity, bmi, bp_ratio]

# Prediction Button
if st.sidebar.button("Predict"):
    prediction, disease_name = predict_heart_disease(input_data)
    st.success(f"Prediction: {disease_name}")

    # SHAP Explanation
    st.subheader("Feature Importance (SHAP Values)")
    shap_values = generate_shap_explanation(np.array(input_data).reshape(1, -1))
    shap.initjs()
    st.write(shap.force_plot(shap_values))

    # Download Report
    pdf = generate_pdf_report(dict(zip(EXPECTED_FEATURES, input_data)), disease_name)
    st.download_button("Download Report", pdf, "Heart_Disease_Report.pdf")

# PDF Upload for Auto-Prediction
uploaded_file = st.file_uploader("Upload Patient PDF Report", type=["pdf"])
if uploaded_file:
    df = process_pdf(uploaded_file)
    st.dataframe(df)
    prediction, disease_name = predict_heart_disease(df.iloc[0])
    st.success(f"Prediction: {disease_name}")