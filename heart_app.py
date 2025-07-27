
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    return df

df = load_data()

st.title("🫀 Heart Disease Prediction App")
st.markdown("Enter the details below to check your heart disease risk.")

age = st.sidebar.slider("Age", 20, 80, 45)
sex = st.sidebar.radio("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.slider("Cholesterol", 100, 600, 200)
fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.sidebar.selectbox("Rest ECG Results", [0, 1, 2])
thalach = st.sidebar.slider("Max Heart Rate (thalach)", 60, 220, 150)
exang = st.sidebar.radio("Exercise-Induced Angina (exang)", [0, 1])

df["sex"] = df["sex"].map({1: "Male", 0: "Female"})
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})

X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

user_data = pd.DataFrame({
    "age": [age],
    "sex": [1 if sex == "Male" else 0],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang]
})

user_scaled = scaler.transform(user_data)

if st.button("Check Heart Risk"):
    prediction = model.predict(user_scaled)
    if prediction[0] == 1:
        st.error("⚠️ You are at risk of Heart Disease.")
    else:
        st.success("✅ You are not at risk of Heart Disease.")

acc = accuracy_score(y_test, model.predict(X_test))
st.caption(f"🔍 Model Accuracy: {round(acc * 100, 2)}%")

with st.expander("See Raw Data"):
    st.dataframe(df.head())
