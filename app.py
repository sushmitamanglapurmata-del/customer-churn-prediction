
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊"
)

st.title("Customer Churn Prediction")
st.write("Predict whether a customer may leave a service.")

try:
    model = joblib.load("churn_model.pkl")

    st.subheader("Enter Customer Details")

    tenure = st.slider("Tenure (months)", 1, 72, 12)
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=500.0,
        value=70.0
    )
    support_calls = st.slider("Support Calls", 0, 10, 2)

    if st.button("Predict Churn"):
        customer = pd.DataFrame([{
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "support_calls": support_calls
        }])

        prediction = model.predict(customer)[0]

        if prediction == 1:
            st.warning("This customer is predicted to churn.")
        else:
            st.success("This customer is predicted to stay.")

except FileNotFoundError:
    st.error("Model file missing. Run train_model.py first.")