from flask import Flask, request, jsonify
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model and scaler
model = load_model("fraud_model.h5")
scaler = joblib.load("scaler.save")

@app.route("/")
def home():
    return "Credit Card Fraud Detection API is running."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # expects JSON input
    df = pd.DataFrame([data])
    
    # Scale features
    if 'Time' in df.columns:
        df = df.drop('Time', axis=1)
    df_scaled = scaler.transform(df)
    
    # Predict
    pred_prob = model.predict(df_scaled)[0][0]
    pred_class = int(pred_prob > 0.5)
    
    return jsonify({"fraud_probability": float(pred_prob), "predicted_class": pred_class})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
