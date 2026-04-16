from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import numpy as np

from federated import train_client_model, federated_averaging
from model import predict_transaction, update_global_model

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route("/")
def home():
    return jsonify({
        "message": "Federated Fraud Detection Backend Running"
    })


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health")
def health():
    return jsonify({
        "status": "running"
    })


# --------------------------------------------------
# TRAIN CLIENT MODEL
# --------------------------------------------------

@app.route("/train_client", methods=["POST"])
def train_client():

    try:
        data = request.json
        client_id = data.get("client_id")

        if not client_id:
            return jsonify({"error": "client_id required"}), 400

        logging.info(f"Client training started: {client_id}")

        weights = train_client_model(client_id)

        logging.info(f"Client training completed: {client_id}")

        return jsonify({
            "client_id": client_id,
            "weights": weights.tolist()
        })

    except Exception as e:
        logging.error(f"Client training error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# FEDERATED AGGREGATION
# --------------------------------------------------

@app.route("/aggregate", methods=["POST"])
def aggregate():

    try:

        data = request.json
        weights_list = data.get("weights_list")
        round_number = data.get("round")

        # -------- Validation --------

        if not isinstance(weights_list, list):
            return jsonify({"error": "weights_list must be a list"}), 400

        if len(weights_list) < 2:
            return jsonify({"error": "At least 2 client models required"}), 400

        weight_shapes = [len(w) for w in weights_list]

        if len(set(weight_shapes)) != 1:
            return jsonify({"error": "Inconsistent weight dimensions"}), 400

        logging.info(f"Federated Round: {round_number}")
        logging.info(f"Aggregating {len(weights_list)} client models")

        weights_array = np.array(weights_list)

        global_weights = federated_averaging(weights_array)

        update_global_model(global_weights)

        logging.info("Global model updated successfully")

        return jsonify({
            "message": "Global model updated",
            "round": round_number,
            "global_weights": global_weights.tolist()
        })

    except Exception as e:
        logging.error(f"Aggregation error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# FRAUD PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        amount = data["amount"]
        time = data["time"]
        location = data["location"]
        device = data["device"]

        logging.info("Prediction request received")

        probability, label = predict_transaction(
            amount,
            time,
            location,
            device
        )

        confidence = f"{probability*100:.2f}%"

        logging.info(f"Prediction result: {label} ({confidence})")

        return jsonify({
            "fraud_probability": float(probability),
            "result": label,
            "confidence": confidence
        })

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)