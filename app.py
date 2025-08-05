import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# App layout settings
st.set_page_config(page_title="Claims Attorney Predictor", layout="centered")
st.title("⚖️ Claims Data Analysis & Attorney Prediction")

# Load model
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    else:
        st.error("❌ Model file not found. Please upload 'model.pkl'")
        return None

model = load_model()

if model:
    # Input form
    with st.form("input_form"):
        st.write("### 📝 Enter Claim Details")

        CLMSEX = st.selectbox("Claimant Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        CLMINSUR = st.selectbox("Has Insurance", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        SEATBELT = st.selectbox("Wearing Seatbelt", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        CLMAGE = st.slider("Claimant Age", 18, 100, 30)
        LOSS = st.number_input("Loss Amount ($)", min_value=0)
        Accident_Severity = st.slider("Accident Severity (1-10)", 1, 10, 5)
        Claim_Amount_Requested = st.number_input("Claim Amount Requested ($)", min_value=0)
        Claim_Approval_Status = st.selectbox("Claim Approved", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        Settlement_Amount = st.number_input("Settlement Amount ($)", min_value=0)
        Policy_Type = st.selectbox("Policy Type", [0, 1], format_func=lambda x: "Basic" if x == 0 else "Comprehensive")
        Driving_Record = st.slider("Driving Record Score", 1, 10, 5)

        submitted = st.form_submit_button("🚀 Predict Attorney")

    if submitted:
        input_df = pd.DataFrame([{
            'CLMSEX': CLMSEX,
            'CLMINSUR': CLMINSUR,
            'SEATBELT': SEATBELT,
            'CLMAGE': CLMAGE,
            'LOSS': LOSS,
            'Accident_Severity': Accident_Severity,
            'Claim_Amount_Requested': Claim_Amount_Requested,
            'Claim_Approval_Status': Claim_Approval_Status,
            'Settlement_Amount': Settlement_Amount,
            'Policy_Type': Policy_Type,
            'Driving_Record': Driving_Record
        }])

        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.success("✅ Prediction: Attorney Present")
        else:
            st.info("❌ Prediction: No Attorney")
