import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Customer Churn Prediction App")
st.write("Enter customer details below")

tenure = st.number_input("Tenure (Months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

if st.button("Predict"):
    input_data = np.array([[tenure, monthly_charges, total_charges]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("❌ Customer is likely to churn")
    else:
        st.success("✅ Customer will NOT churn")