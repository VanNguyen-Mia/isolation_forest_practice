import streamlit as st
from module_1 import classify_new_applicant
from pathlib import Path

# Dynamically resolve paths relative to the project root
BASE_DIR = Path(__file__).resolve().parent.parent  # Parent of 'src'
MODEL_PATH = BASE_DIR / 'saved_model' / 'rf_model.joblib'
TRAIN_COLUMNS_PATH = BASE_DIR / 'saved_model' / 'trained_columns.joblib'

model_path = str(MODEL_PATH)
train_columns_path = str(TRAIN_COLUMNS_PATH)

st.title("Loan Appplicant Classifier")
st.write("Provide the details below to determine if the applicant qualifies for a loan.")

with st.form("applicant_form"):
    st.subheader("Applicant Information")
    
    person_age = st.number_input("Age", min_value=18, max_value=100, value=25, step=1)
    person_gender = st.selectbox("Gender", ['male', 'female'])
    person_education = st.selectbox("Education Level", [
        "High School", "Associate", "Bachelor", "Master", "Doctorate"
    ])
    person_income = st.number_input("Income (EUR)", min_value=0, value=50000, step=1000)
    person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=100, value=5, step=1)
    person_home_ownership = st.selectbox("Home Ownership", [
        "MORTGAGE", "OWN", "RENT", "OTHER"
    ])
    loan_amnt = st.number_input("Loan Amount (EUR)", min_value=0, value=5000, step=500)
    loan_intent = st.selectbox("Loan Intent", [
        "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
        "MEDICAL", "PERSONAL", "VENTURE"
    ])
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    cb_person_cred_hist_length = st.number_input("Client's Credit History Length (years)", min_value=0, max_value=100, value=10, step=1)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["No", "Yes"])
    
    submitted = st.form_submit_button("Classify Applicant")


if submitted:
    applicant_data = {
        "person_age": person_age,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }

try:
    result = classify_new_applicant(applicant_data, model_path, train_columns_path)
    st.success(f"Loan Application Status: {result}")
except Exception as e:
        st.error(f"An error occurred: {e}")