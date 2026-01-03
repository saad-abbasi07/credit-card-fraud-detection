from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler safely
MODEL_PATH = "fraud_model.h5"
SCALER_PATH = "scaler.save"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    scaler = None
    print(f"Error loading scaler: {e}")

@app.route("/")
def index():
    return jsonify({"message": "Credit Card Fraud Detection API is running ..."})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        # Convert incoming values to numpy array
        features = np.array([list(data.values())], dtype=float)

        # Scale
        features_scaled = scaler.transform(features)

        # Predict
        pred_prob = model.predict(features_scaled)[0][0]
        pred_class = int(pred_prob > 0.5)

        return jsonify({
            "prediction": pred_class,
            "probability": float(pred_prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
