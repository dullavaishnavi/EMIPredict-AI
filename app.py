import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💳",
    layout="wide"
)

MODEL_DIR = "models"
CLS_PATH = os.path.join(MODEL_DIR, "best_classification_model.joblib")
REG_PATH = os.path.join(MODEL_DIR, "best_regression_model.joblib")

@st.cache_resource
def load_models():
    if not os.path.exists(CLS_PATH) or not os.path.exists(REG_PATH):
        return None, None
    return joblib.load(CLS_PATH), joblib.load(REG_PATH)

def add_features(data):
    data = data.copy()
    eps = 1e-6

    expense_cols = [
        "monthly_rent", "school_fees", "college_fees",
        "travel_expenses", "groceries_utilities",
        "other_monthly_expenses"
    ]
    existing = [c for c in expense_cols if c in data.columns]
    data["total_monthly_expenses"] = data[existing].sum(axis=1, skipna=True)

    data["expense_to_income"] = (
        data["total_monthly_expenses"] / (data["monthly_salary"] + eps)
    )
    data["current_emi_to_income"] = (
        data["current_emi_amount"] / (data["monthly_salary"] + eps)
    )
    data["bank_balance_to_income"] = (
        data["bank_balance"] / (data["monthly_salary"] + eps)
    )
    data["emergency_fund_to_income"] = (
        data["emergency_fund"] / (data["monthly_salary"] + eps)
    )
    data["estimated_disposable_income"] = (
        data["monthly_salary"]
        - data["total_monthly_expenses"]
        - data["current_emi_amount"]
    )
    data["requested_amount_per_month"] = (
        data["requested_amount"] / (data["requested_tenure"] + eps)
    )
    data["dependents_per_family_member"] = (
        data["dependents"] / (data["family_size"] + eps)
    )
    return data

def get_customer_input():
    st.subheader("Customer Financial Profile")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 25, 60, 32)
        gender = st.selectbox("Gender", ["Female", "Male"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        education = st.selectbox(
            "Education",
            ["High School", "Graduate", "Post Graduate", "Professional"]
        )
        monthly_salary = st.number_input("Monthly Salary (₹)", 15000, 200000, 60000, step=1000)
        employment_type = st.selectbox(
            "Employment Type", ["Private", "Government", "Self-employed"]
        )
        years_of_employment = st.number_input("Years of Employment", 0, 40, 5)

    with c2:
        company_type = st.selectbox("Company Type", ["Private", "Government", "MNC", "Startup"])
        house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
        monthly_rent = st.number_input("Monthly Rent (₹)", 0, 100000, 12000, step=1000)
        family_size = st.number_input("Family Size", 1, 15, 3)
        dependents = st.number_input("Dependents", 0, 15, 1)
        school_fees = st.number_input("School Fees (₹)", 0, 100000, 4000, step=500)
        college_fees = st.number_input("College Fees (₹)", 0, 100000, 0, step=500)
        travel_expenses = st.number_input("Travel Expenses (₹)", 0, 100000, 3000, step=500)

    with c3:
        groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0, 100000, 10000, step=500)
        other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", 0, 100000, 3000, step=500)
        existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
        current_emi_amount = st.number_input("Current EMI Amount (₹)", 0, 100000, 0, step=500)
        credit_score = st.number_input("Credit Score", 300, 850, 750)
        bank_balance = st.number_input("Bank Balance (₹)", 0, 5000000, 100000, step=5000)
        emergency_fund = st.number_input("Emergency Fund (₹)", 0, 5000000, 80000, step=5000)

    st.subheader("Loan Application")

    l1, l2, l3 = st.columns(3)
    with l1:
        emi_scenario = st.selectbox(
            "EMI Scenario",
            [
                "E-commerce Shopping EMI",
                "Home Appliances EMI",
                "Vehicle EMI",
                "Personal Loan EMI",
                "Education EMI"
            ]
        )
    with l2:
        requested_amount = st.number_input("Requested Amount (₹)", 10000, 1500000, 300000, step=5000)
    with l3:
        requested_tenure = st.number_input("Requested Tenure (months)", 3, 84, 36)

    return {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
    }

cls_model, reg_model = load_models()

st.title("💳 EMIPredict AI")
st.caption("Intelligent Financial Risk Assessment Platform")

if cls_model is None or reg_model is None:
    st.error(
        "Trained models are missing. Add these two files to the repository under models/: "
        "best_classification_model.joblib and best_regression_model.joblib"
    )
    st.info(
        "Run the Google Colab notebook first, download the EMIPredict_models ZIP, "
        "extract it, and upload the two .joblib files into the models folder."
    )
    st.stop()

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "🔮 EMI Prediction", "📊 About the Model"]
)

if page == "🏠 Dashboard":
    st.header("Project Dashboard")

    a, b, c, d = st.columns(4)
    a.metric("Problem Type", "Classification + Regression")
    b.metric("Classification", "EMI Eligibility")
    c.metric("Regression", "Maximum EMI")
    d.metric("Models", "6")

    st.markdown("""
    ### What this application does

    **EMI Eligibility Prediction**
    - Eligible
    - High_Risk
    - Not_Eligible

    **Maximum Safe EMI Prediction**
    - Predicts the maximum monthly EMI amount for the applicant.

    The application uses the best-performing trained pipelines produced by the
    Google Colab project.
    """)

elif page == "🔮 EMI Prediction":
    customer = get_customer_input()

    if st.button("🚀 Assess EMI Risk", type="primary", use_container_width=True):
        input_df = pd.DataFrame([customer])
        input_df = add_features(input_df)

        predicted_class = cls_model.predict(input_df)[0]
        predicted_emi = float(reg_model.predict(input_df)[0])

        st.divider()
        st.subheader("Prediction Result")

        r1, r2 = st.columns(2)
        r1.metric("EMI Eligibility", str(predicted_class))
        r2.metric("Maximum Safe Monthly EMI", f"₹{predicted_emi:,.2f}")

        if str(predicted_class).lower() == "eligible":
            st.success("The applicant is predicted to be eligible.")
        elif "risk" in str(predicted_class).lower():
            st.warning("The applicant is predicted to be high risk.")
        else:
            st.error("The applicant is predicted to be not eligible.")

        st.subheader("Derived Financial Indicators")
        total_expenses = float(input_df["total_monthly_expenses"].iloc[0])
        disposable = float(input_df["estimated_disposable_income"].iloc[0])
        ratio = float(input_df["expense_to_income"].iloc[0])

        x1, x2, x3 = st.columns(3)
        x1.metric("Monthly Expenses", f"₹{total_expenses:,.0f}")
        x2.metric("Estimated Disposable Income", f"₹{disposable:,.0f}")
        x3.metric("Expense / Income", f"{ratio:.2%}")

        with st.expander("View applicant data"):
            st.dataframe(pd.DataFrame([customer]), use_container_width=True)

elif page == "📊 About the Model":
    st.header("Model & Project Information")

    st.markdown("""
    ### Classification models
    1. Logistic Regression
    2. Random Forest Classifier
    3. XGBoost Classifier

    ### Regression models
    1. Linear Regression
    2. Random Forest Regressor
    3. XGBoost Regressor

    ### Feature engineering
    The model pipeline creates:
    - Total monthly expenses
    - Expense-to-income ratio
    - Current EMI-to-income ratio
    - Bank balance-to-income ratio
    - Emergency fund-to-income ratio
    - Estimated disposable income
    - Requested amount per month
    - Dependents per family member

    ### Important
    This application is an educational machine-learning project. A prediction
    should not be treated as a real financial approval or rejection decision.
    """)
