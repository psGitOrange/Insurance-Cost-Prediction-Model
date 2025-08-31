import joblib
import streamlit as st
# import pickle
import pandas as pd
import numpy as np

# Load your trained model
model_rf = joblib.load('cost_pred_rf.pkl')
print(model_rf)
scaler_transform = joblib.load('scaler_transform.pkl')
print(scaler_transform)
#     model = pickle.load(f)
# with open("cost_pred_rf.pkl", "rb") as f:
#     print(model)

st.title("Predict Your Health Insurance Premium")  # Title

# Row 1: Age, Height, Weight
col1, col2, col3 = st.columns(3)
age = col1.number_input("Age", min_value=0, max_value=120, value=30)
height = col2.number_input("Height (cm)", min_value=50, max_value=250, value=170)
weight = col3.number_input("Weight (kg)", min_value=10, max_value=200, value=70)

# Row 2: Diabetes, Blood Pressure Problems, Known Allergies
col4, col5, col6 = st.columns(3)
diabetes = col4.selectbox("Diabetes", [0, 1])
bp_problems = col5.selectbox("Blood Pressure Problems", [0, 1])
allergies = col6.selectbox("Known Allergies", [0, 1])

# Row 3: Any Transplants, Chronic Diseases, Cancer History, Number of Surgeries
col7, col8, col9, col10 = st.columns(4)
transplants = col7.selectbox("Any Transplants", [0, 1])
chronic = col8.selectbox("Any Chronic Diseases", [0, 1])
cancer_history = col9.selectbox("History of Cancer in Family", [0, 1])
surgeries = col10.number_input("Number of Major Surgeries", min_value=0, max_value=3, value=0)

# data preprocessing
features = np.array([[age, diabetes, bp_problems, transplants, chronic,
                           height, weight, allergies, cancer_history, surgeries]])
columns = ['Age', 'Diabetes', 'BloodPressureProblems', 'AnyTransplants',
       'AnyChronicDiseases', 'Height', 'Weight', 'KnownAllergies',
       'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries']
df = pd.DataFrame(features, columns=columns)

df['bmi'] = df['Weight'] / (df['Height'] / 100) ** 2
df['bmi_category'] = pd.cut(
    df['bmi'], bins=[0, 18.5, 25, 30, float('inf')],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)
df['health_risk_score'] = (
    df['Diabetes'] + df['BloodPressureProblems'] +
    df['AnyTransplants'] + df['AnyChronicDiseases'] +
    df['KnownAllergies'] + df['HistoryOfCancerInFamily']
)
df['high_risk_score'] = (
    df['AnyTransplants'] + df['AnyChronicDiseases'] +
    df['HistoryOfCancerInFamily'] + df['NumberOfMajorSurgeries']
)

bmi_order = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
df['bmi_category'] = df['bmi_category'].map(bmi_order)

numeric_cols = ['Age', 'Height', 'Weight', 'bmi']
df[numeric_cols] = scaler_transform.transform(df[numeric_cols])
print(df.columns)
print(df)

# Predict button
if st.button("Predict Premium Price"):
    prediction = model_rf.predict(df)
    st.success(f"Predicted Premium Price: ₹{prediction[0]:,.2f}")
