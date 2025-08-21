import streamlit as st
import pickle
import numpy as np

# Load saved model
with open("naive.pkl", "rb") as file:
    model = pickle.load(file)

st.title("💳 Loan Approval Prediction App")
st.write("Fill in the details below to check loan eligibility.")

# User inputs
income_annum = st.number_input("Annual Income", min_value=200000, max_value=10000000, step=10000)
no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=5, step=1)
loan_amount = st.number_input("Loan Amount", min_value=300000, max_value=10000000, step=50000)
loan_term = st.slider("Loan Term (in years)", min_value=2, max_value=20, step=1)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0, max_value=3000000, step=50000)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, max_value=2000000, step=50000)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, max_value=4000000, step=50000)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0, max_value=2000000, step=50000)

education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

# Encode categorical variables (make sure same mapping as training dataset)
education_map = {"Graduate": 1, "Not Graduate": 0}
self_employed_map = {"Yes": 1, "No": 0}

education_val = education_map[education]
self_employed_val = self_employed_map[self_employed]

# Prepare input data
input_data = np.array([[income_annum, no_of_dependents, loan_amount, loan_term, 
                        cibil_score, residential_assets_value, commercial_assets_value, 
                        luxury_assets_value, bank_asset_value, 
                        education_val, self_employed_val]])

# Prediction
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
