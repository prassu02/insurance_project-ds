import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Attorney Prediction", layout="centered")
st.title("⚖️ Attorney Prediction for Insurance Claims")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("model_attorney.pkl")

model = load_model()

# Input Form
with st.form("input_form"):
    st.subheader("📝 Enter Claim Details")

    CLMSEX = st.selectbox("Claimant Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    CLMINSUR = st.selectbox("Has Insurance?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    SEATBELT = st.selectbox("Wearing Seatbelt?", [0, 1])
    CLMAGE = st.slider("Claimant Age", 18, 100, 30)
    LOSS = st.number_input("Loss Amount", min_value=0.0, step=100.0)
    Accident_Severity = st.selectbox("Accident Severity", ['Minor', 'Moderate', 'Severe'])
    Claim_Amount_Requested = st.number_input("Claim Amount Requested", min_value=0.0, step=100.0)
    Claim_Approval_Status = st.selectbox("Claim Approved?", [0, 1])
    Settlement_Amount = st.number_input("Settlement Amount", min_value=0.0, step=100.0)
    Policy_Type = st.selectbox("Policy Type", ['Comprehensive', 'Third-Party'])
    Driving_Record = st.selectbox("Driving Record", ['Clean', 'Minor Offenses', 'Major Offenses'])

    submitted = st.form_submit_button("🚀 Predict Attorney")

# Run prediction
if submitted:
    input_df = pd.DataFrame([{
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

    # Encoding
    encode_map = {
        'Accident_Severity': {'Minor': 0, 'Moderate': 1, 'Severe': 2},
        'Policy_Type': {'Comprehensive': 0, 'Third-Party': 1},
        'Driving_Record': {'Clean': 0, 'Minor Offenses': 1, 'Major Offenses': 2}
    }
    for col, mapping in encode_map.items():
        input_df[col] = input_df[col].map(mapping)

    # Predict
    prediction = model.predict(input_df)[0]
    result = "✅ Attorney Involved" if prediction == 1 else "❌ No Attorney"

    st.success(f"### Prediction: {result}")






