import streamlit as st
import pandas as pd
import joblib

# Load trained components
model = joblib.load("best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", label_encoders["education"].classes_)
occupation = st.sidebar.selectbox("Job Role", label_encoders["occupation"].classes_)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
relationship = st.sidebar.selectbox("Relationship", label_encoders["relationship"].classes_)
race = st.sidebar.selectbox("Race", label_encoders["race"].classes_)
gender = st.sidebar.selectbox("Gender", label_encoders["gender"].classes_)
native_country = st.sidebar.selectbox("Native Country", label_encoders["native-country"].classes_)
marital_status = st.sidebar.selectbox("Marital Status", label_encoders["marital-status"].classes_)
capital_gain = st.sidebar.number_input("Capital Gain", 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0)
educational_num = st.sidebar.slider("Education Number", 1, 16, 10)
workclass = st.sidebar.selectbox("Workclass", label_encoders["workclass"].classes_)

# Create input dictionary
input_dict = {
    "age": age,
    "workclass": label_encoders["workclass"].transform([workclass])[0],
    "marital-status": label_encoders["marital-status"].transform([marital_status])[0],
    "occupation": label_encoders["occupation"].transform([occupation])[0],
    "relationship": label_encoders["relationship"].transform([relationship])[0],
    "race": label_encoders["race"].transform([race])[0],
    "gender": label_encoders["gender"].transform([gender])[0],
    "native-country": label_encoders["native-country"].transform([native_country])[0],
    "hours-per-week": hours_per_week,
    "capital-gain": capital_gain,
    "capital-loss": capital_loss,
    "educational-num": educational_num,
    "education": label_encoders["education"].transform([education])[0]
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=feature_columns, fill_value=0)
input_scaled = scaler.transform(input_df)

# Show input
st.write("### 🔎 Model Input Preview")
st.write(input_df)

# Predict
if st.button("Predict Salary Class"):
    prediction = model.predict(input_scaled)[0]
    label = ">50K" if prediction == 1 else "≤50K"
    st.success(f"✅ Predicted Salary Class: **{label}**")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV for prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    for col in label_encoders:
        if col in batch_data.columns:
            batch_data[col] = label_encoders[col].transform(batch_data[col])
    batch_data = batch_data.reindex(columns=feature_columns, fill_value=0)
    batch_scaled = scaler.transform(batch_data)
    batch_preds = model.predict(batch_scaled)
    batch_data["PredictedClass"] = [">50K" if p == 1 else "≤50K" for p in batch_preds]
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results", csv, "predicted_classes.csv", mime="text/csv")
