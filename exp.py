"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from utils import data_preprocess, train_model, read_digits, split_train_dev_test, predict_and_eval, get_hyperparameter_combinations 
import pdb

gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]


## Split data 
X, y = read_digits()
# X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.3)
X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(X, y, test_size=0.3, dev_size=0.3)

## Use the preprocessed datas
X_train = data_preprocess(X_train)
X_test = data_preprocess(X_test)
X_dev = data_preprocess(X_dev)


h_params={}
h_params['gamma'] = gamma_ranges
h_params['C'] = C_ranges

h_params_combinations = get_hyperparameter_combinations(h_params)
print("h_params_combinations ", len(h_params_combinations))

#Model training
model = train_model(X_train, y_train, {'gamma': 0.001}, model_type="svm")

#Evaluation
predict_and_eval(model, X_test, y_test)


