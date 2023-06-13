from app import app, model
from flask import jsonify, request
import numpy as np
from app.models.fish import get_scaled_data, inverse_transform, get_index_column

@app.route("/fishprice", methods=["POST"])
def fishPricePrediction():
    try:
        data = request.get_json()
        namaIkan = data['ikan']
        namaDaerah = data['daerah']
        namaKolom = namaIkan + "_" + namaDaerah
        feature = get_scaled_data()
        prediction = model.predict(feature)
        prediction = inverse_transform(prediction)
        index = get_index_column(namaKolom)
        data = {
            'namaKolom': namaKolom,
            'prediction': prediction[:,:,index].tolist()
        }
        # return jsonify(data)
        return jsonify(data)
    except Exception as e:
        error_message = str(e)
        return jsonify({"error": error_message}), 500
    
@app.route("/fishprice", methods=["GET"])
def getAllFishPricePrediction():
    try:
        # data = request.get_json()
        # namaIkan = data['ikan']
        # namaDaerah = data['daerah']
        # namaKolom = namaIkan + "_" + namaDaerah
        feature = get_scaled_data()
        prediction = model.predict(feature)
        prediction = inverse_transform(prediction)
        # index = get_index_column(namaKolom)
        data = {
            'prediction': prediction[:,:,:].tolist()
        }
        # return jsonify(data)
        return jsonify(data)
    except Exception as e:
        error_message = str(e)
        return jsonify({"error": error_message}), 500
    
@app.route("/weather", methods=["POST"])
def weatherPrediction():
    try:
        data = request.get_json()
        namaIkan = data['ikan']
        namaDaerah = data['daerah']
        namaKolom = namaIkan + "_" + namaDaerah
        feature = get_scaled_data()
        prediction = model.predict(feature)
        prediction = inverse_transform(prediction)
        index = get_index_column(namaKolom)
        data = {
            'namaKolom': namaKolom,
            'prediction': prediction[:,:,index].tolist()
        }
        # return jsonify(data)
        return jsonify(data)
    except Exception as e:
        error_message = str(e)
        return jsonify({"error": error_message}), 500
