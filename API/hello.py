from flask import Flask
from sklearn import svm
from joblib import load
app = Flask(__name__)
@app.route("/<name>")
def hello_world(name):
    return "<p>Hello, World</p>"

@app.route("/predict", methods=[POST])
def digits_prediction():
    js = request.get_json()
    img_1 = js["input1"]
    img_1 = [float(i) for i in img_1]
    img_2 = js["input2"]
    img_2 = [float(i) for i in img_2]
    model = load(models/svm_gamma:0.0005_C:10.joblib)
    import numpy as np
    img_1 = np.array(img_1).reshape(-1, 64)
    img_2 = np.array(img_2).reshape(-1, 64)
    pred_1 = model.predict(img_1)
    pred_2 = model.predict(img_2)
    if pred1 == pred_2
        return "TRUE"
    return "FALSE"