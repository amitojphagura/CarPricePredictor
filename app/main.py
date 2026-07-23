from flask import Flask, jsonify, request

from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

REQUIRED_FIELDS = [
    "make",
    "year",
    "kilometer",
    "fuel_type",
    "transmission",
    "location",
    "color",
    "owner",
    "seller_type",
    "drivetrain",
    "seating_capacity",
    "fuel_tank_capacity",
    "volume",
    "power_bhp",
    "power_rpm",
    "torque_nm",
    "torque_rpm",
]


@app.route("/")
def index():
    return jsonify(
        status="ok",
        message="Used car price prediction API.",
        usage=f"POST JSON to /predict with fields: {REQUIRED_FIELDS}",
    )


@app.route("/predict", methods=["POST"])
def predict_datapoint():
    payload = request.get_json(silent=True) or {}

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        return jsonify(error=f"Missing fields: {missing}"), 400

    try:
        data = CustomData(
            make=payload["make"],
            year=int(payload["year"]),
            kilometer=int(payload["kilometer"]),
            fuel_type=payload["fuel_type"],
            transmission=payload["transmission"],
            location=payload["location"],
            color=payload["color"],
            owner=payload["owner"],
            seller_type=payload["seller_type"],
            drivetrain=payload["drivetrain"],
            seating_capacity=float(payload["seating_capacity"]),
            fuel_tank_capacity=float(payload["fuel_tank_capacity"]),
            volume=float(payload["volume"]),
            power_bhp=float(payload["power_bhp"]),
            power_rpm=float(payload["power_rpm"]),
            torque_nm=float(payload["torque_nm"]),
            torque_rpm=float(payload["torque_rpm"]),
        )
        pred_df = data.get_data_as_data_frame()
        prediction = PredictPipeline().predict(pred_df)
        return jsonify(predicted_price=float(prediction[0]))
    except (ValueError, TypeError) as e:
        return jsonify(error=f"Invalid field value: {e}"), 400
    except CustomException as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
