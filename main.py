from flask import Flask, jsonify, request

import Autism_Predict

app = Flask(__name__)


@app.route('/')
def hello_world():
    return jsonify({"message": "Autism-Prediction-App (https://flight-prediction62.herokuapp.com/)"})

@app.route('/predict', methods=['POST'])
def predict_autism():
    person_info = request.get_json()
    prediction = Autism_Predict.predict_data(person_info)
    return jsonify(prediction)

