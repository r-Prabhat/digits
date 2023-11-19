import matplotlib.pyplot as plt
import sys
# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import pdb
from joblib import dump,load
import numpy as np
# import skimage
# from skimage.transform import resize
import pandas as pd
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__)

model = load('./models/svm_gamma:0.0005_C:10.joblib')

# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"

@app.route('/predict', methods=['POST'])
def compare_digits():
    try:
        # Get the two image files from the request
        data = request.json  # Parse JSON data from the request body
        image1 = data.get('image', [])

        # Preprocess the images and make predictions
        digit1 = predict_digit(image1)

        # Compare the predicted digits and return the result
        result = digit1

        return jsonify({"label" : result})
    except Exception as e:
        return jsonify({"label" : "Error"})
    
def predict_digit(image):
    try:
       # Convert the input list to a numpy array and preprocess for prediction
        img_array = np.array(image, dtype=np.float32).reshape(1, -1)

        # Assuming model is defined somewhere in your code
        prediction = model.predict(img_array)
        digit = int(prediction[0])  # Assuming the prediction is a single digit

        return digit
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run()