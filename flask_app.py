from flask import Flask, request, jsonify
import pandas as pd
import joblib
# import numpy as np

model_rf = joblib.load('cost_pred_rf.pkl')
scaler_transform = joblib.load('scaler_transform.pkl')

app = Flask(__name__)

def preprocess_input(data):
    df = pd.DataFrame([data])
    df['bmi'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, float('inf')],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    df['health_risk_score'] = (df['Diabetes'] + df['BloodPressureProblems'] +
        df['AnyTransplants'] + df['AnyChronicDiseases'] +
        df['KnownAllergies'] + df['HistoryOfCancerInFamily'])
    df['high_risk_score'] = (df['AnyTransplants'] + df['AnyChronicDiseases'] +
        df['HistoryOfCancerInFamily'] + df['NumberOfMajorSurgeries'])

    bmi_order = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
    df['bmi_category'] = df['bmi_category'].map(bmi_order)

    numeric_cols = ['Age', 'Height', 'Weight', 'bmi']
    df[numeric_cols] = scaler_transform.transform(df[numeric_cols])
    print(df)
    return df

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = preprocess_input(data)
        prediction = model_rf.predict(features)

        return jsonify({
            "estimated_premium": int(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)