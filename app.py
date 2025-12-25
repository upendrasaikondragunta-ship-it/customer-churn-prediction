import streamlit as st
import pandas as pd
import joblib

# ===============================
# Load model and features
# ===============================
model = joblib.load("churn_model.pkl")
features = joblib.load("feature_columns.pkl")

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="centered"
)

st.title("📉 Customer Churn Prediction App")
st.markdown(
    "This app predicts whether a customer is likely to **CHURN** or **STAY** "
    "based on their service details."
)

st.divider()

# ===============================
# User Inputs
# ===============================
monthly_charges = st.number_input(
    "Monthly Charges ($)", min_value=0.0, step=1.0
)

tenure = st.number_input(
    "Tenure (Months)", min_value=0, step=1
)

latitude = st.number_input("Latitude", format="%.6f")
longitude = st.number_input("Longitude", format="%.6f")

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

contract = st.selectbox(
    "Contract Type",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

st.divider()

# ===============================
# Create input dataframe (FLOAT SAFE)
# ===============================
input_df = pd.DataFrame(0.0, index=[0], columns=features)

# Fill numeric values
if "monthly charges" in input_df.columns:
    input_df.at[0, "monthly charges"] = float(monthly_charges)

if "tenure months" in input_df.columns:
    input_df.at[0, "tenure months"] = float(tenure)

if "latitude" in input_df.columns:
    input_df.at[0, "latitude"] = float(latitude)

if "longitude" in input_df.columns:
    input_df.at[0, "longitude"] = float(longitude)

# One-hot encoding (safe)
pay_col = f"payment method_{payment_method.lower()}"
con_col = f"contract_{contract.lower()}"

if pay_col in input_df.columns:
    input_df.at[0, pay_col] = 1.0

if con_col in input_df.columns:
    input_df.at[0, con_col] = 1.0

# ===============================
# Prediction
# ===============================
if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(
            f"⚠️ **Customer is likely to CHURN**\n\n"
            f"Probability: **{probability*100:.2f}%**"
        )
    else:
        st.success(
            f"✅ **Customer is likely to STAY**\n\n"
            f"Probability: **{(1-probability)*100:.2f}%**"
        )
