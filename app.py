from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    year = int(data['year'])
    Present_Price = float(data['presentPrice'])
    Kms_Driven = int(data['kmsDriven'])
    Seller_Type = int(data['sellerType'])
    Fuel_Type = int(data['fuelType'])
    Transmission = int(data['transmission'])
    Owner = int(data['owner'])

    features = np.array([[year, Present_Price, Kms_Driven, Seller_Type, Fuel_Type, Transmission, Owner]])

    predicted_price = model.predict(features)

    return jsonify({'predictedPrice': predicted_price[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
