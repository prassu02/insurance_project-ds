import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Claims Data Analysis and Prediction")

def load_model():
    model = pickle.load(open('model.pkl', 'rb'))
    return model

model = load_model()

# User input for prediction
st.write("### Enter Claim Details for Prediction")
user_input = {}
columns = ['CLMSEX', 'CLMINSUR', 'SEATBELT', 'CLMAGE',
       'LOSS', 'Accident_Severity', 'Claim_Amount_Requested',
       'Claim_Approval_Status', 'Settlement_Amount', 'Policy_Type',
       'Driving_Record']

for col in columns:
    user_input[col] = int(st.number_input(f"Enter {col}", step=1))

if st.button("Predict Attorney"):  
    user_df = pd.DataFrame([user_input])
    user_df = user_df.astype(int)  # Ensure all values are integers
    
    prediction = model.predict(user_df)
    st.write(f"### Prediction: {'Attorney Present' if prediction[0] == 1 else 'No Attorney'}")
