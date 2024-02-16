from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# TODO: Import trained model
MODEL_PATH = 'iris_model.pkl'

# Load the model at the start of the app to avoid loading it for each request
model = joblib.load(MODEL_PATH)  

from flask import request

@app.route('/predict', methods=['GET'])
def predict():
    # Get the input array from the request
    get_json = request.get_json()
    iris_input = get_json['input']
    
    #model = MODEL_PATH
    
    # TODO: Make prediction using the model 
    # HINT: use np.array().reshape(1, -1) to convert input to 2D array
    # Convert input to 2D array as the model expects 2D array inputs
    input_array = np.array(iris_input).reshape(1, -1)
    
    # Make prediction using the model
    prediction = model.predict(input_array)
    #prediction = ...
    
    # TODO: Return the prediction as a response
    #return ...
    return jsonify({'prediction': prediction.tolist()})

@app.route('/')
def hello():
    return 'Welcome to Docker Lab'

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
