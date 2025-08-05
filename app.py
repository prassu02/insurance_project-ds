import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page settings
st.set_page_config(page_title="Attorney Involvement Predictor", layout="centered")
st.title("⚖️ Insurance Claim: Attorney Involvement Predictor")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("model_attorney.pkl")

model = load_model()

# User input form
with st.form("claim_form"):
    st.subheader("📋 Enter Claim Details")

    CLMSEX = st.selectbox("Claimant Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    CLMINSUR = st.selectbox("Is the Claimant Insured?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    SEATBELT = st.selectbox("Was Seatbelt Worn?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    CLMAGE = st.slider("Claimant Age", min_value=18, max_value=100, value=30)
    LOSS = st.number_input("Loss Amount ($)", min_value=0.0, step=100.0)

    Accident_Severity = st.selectbox("Accident Severity", ["Minor", "Moderate", "Severe"])
    Claim_Amount_Requested = st.number_input("Claim Amount Requested ($)", min_value=0.0, step=100.0)
    Claim_Approval_Status = st.selectbox("Was Claim Approved?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    Settlement_Amount = st.number_input("Settlement Amount ($)", min_value=0.0, step=100.0)

    Policy_Type = st.selectbox("Policy Type", ["Comprehensive", "Third-Party"])
    Driving_Record = st.selectbox("Driving Record", ["Clean", "Minor Offenses", "Major Offenses"])

    submitted = st.form_submit_button("🔍 Predict Attorney Involvement")

# Prediction logic
if submitted:
    # Prepare input as DataFrame
    input_data = pd.DataFrame([{
        "CLMSEX": CLMSEX,
        "CLMINSUR": CLMINSUR,
        "SEATBELT": SEATBELT,
        "CLMAGE": CLMAGE,
        "LOSS": LOSS,
        "Accident_Severity": Accident_Severity,
        "Claim_Amount_Requested": Claim_Amount_Requested,
        "Claim_Approval_Status": Claim_Approval_Status,
        "Settlement_Amount": Settlement_Amount,
        "Policy_Type": Policy_Type,
        "Driving_Record": Driving_Record
    }])

    # Map categorical to numeric
    encode_map = {
        'Accident_Severity': {'Minor': 0, 'Moderate': 1, 'Severe': 2},
        'Policy_Type': {'Comprehensive': 0, 'Third-Party': 1},
        'Driving_Record': {'Clean': 0, 'Minor Offenses': 1, 'Major Offenses': 2}
    }

    for col, mapping in encode_map.items():
        input_data[col] = input_data[col].map(mapping)

    # Predict
    prediction = model.predict(input_data)[0]
    result = "✅ Attorney Involved" if prediction == 1 else "❌ No Attorney Involved"

    st.success(f"### Prediction Result: {result}")







