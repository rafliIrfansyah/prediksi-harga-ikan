from app import app, model
from flask import jsonify
import numpy as np
from app.models.scaler import get_scaled_data, inverse_transform

@app.route("/predict", methods=["GET"])
def getData():
    feature = get_scaled_data()
    prediction = model.predict(feature)
    prediction = inverse_transform(prediction)

    data = {
        'prediction': prediction.tolist()
    }
    return jsonify(data)
