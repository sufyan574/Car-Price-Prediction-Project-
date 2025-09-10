import pandas as pd
import pickle
import streamlit as st

# -------------------------------
# Load Model, Scaler & Feature Order
# -------------------------------
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")
st.title("🚗 Car Price Prediction App")
st.write("Enter the details below to get the estimated car price:")

try:
    with open("car_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_order.pkl", "rb") as f:
        feature_order = pickle.load(f)
    st.success("✅ Model, Scaler & Feature Order Loaded Successfully!")
except:
    st.error("❌ Required files not found! Please run the training notebook first.")
    st.stop()

# -------------------------------
# Load Dataset (for Dropdown Values)
# -------------------------------
try:
    df = pd.read_csv("quikr_car.csv")
except:
    st.error("❌ quikr_car.csv not found! Please keep it in the same folder.")
    st.stop()

companies = sorted(df['company'].dropna().unique())
fuel_types = sorted(df['fuel_type'].dropna().unique())
names = sorted(df['name'].dropna().unique())

# -------------------------------
# User Inputs
# -------------------------------
st.header("🔧 Input Car Details")

company = st.selectbox("Company", companies)
fuel_type = st.selectbox("Fuel Type", fuel_types)
name = st.selectbox("Car Name", names)
year = st.number_input("Car Manufacturing Year", min_value=2000, max_value=2025, value=2018)
kms_driven = st.number_input("Kms Driven", min_value=0, max_value=500000, value=30000)

# INR → PKR Conversion Rate
INR_TO_PKR = 3.3  # ✅ You can change if rate updates

if st.button("🔮 Predict Price"):
    try:
        input_dict = {
            "company": company,
            "fuel_type": fuel_type,
            "name": name,
            "year": year,
            "kms_driven": kms_driven
        }

        input_data = pd.DataFrame([input_dict])

        for col in ['company', 'fuel_type', 'name']:
            input_data[col] = input_data[col].astype('category').cat.codes

        input_data = input_data[feature_order]

        scaled_data = scaler.transform(input_data)
        prediction_inr = model.predict(scaled_data)[0]

        # Convert to PKR
        prediction_pkr = prediction_inr * INR_TO_PKR

        st.subheader("💰 Estimated Price")
        st.success(f"PKR {round(prediction_pkr, 2):,}")  # formatted with commas
    except Exception as e:
        st.error(f"⚠ Error: {e}")
