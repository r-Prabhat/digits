
from flask import Flask, request, jsonify
import numpy as np
from sklearn import svm
from joblib import load

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def compare_digits():
    try:
        # Get the two image files from the request
        model = load('./models/svm_gamma:0.0005_C:10.joblib')
        data = request.get_json()  # Parse JSON data from the request body
        image1 = data.get('image1', [])
        # image2 = data.get('image2', [])

        # Preprocess the images and make predictions
        digit1 = predict_digit(image1)
        # digit2 = predict_digit(image2)

        # Compare the predicted digits and return the result
        result == digit1 

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})
    
def predict_digit(image):
    try:
        # Convert the input list to a numpy array and preprocess for prediction
        img_array = np.array(image, dtype=np.float32).reshape(1, -1)

        prediction = model.predict(img_array)
        digit = int(prediction[0])

        return digit
    except Exception as e:
        return str(e)

@app.route("/compare", methods=["POST"])
def digits_prediction():
    js = request.get_json()
    Image_a = [float(i) for i in js["input1"]]
    Image_b = [float(i) for i in js["input2"]]
    model = load("models/svm_gamma:0.0005_C:10.joblib")
    pred_1 = model.predict(np.array(Image_a).reshape(-1, 64))
    pred_2 = model.predict(np.array(Image_b).reshape(-1, 64))
    return str(pred_1 == pred_2):
        
