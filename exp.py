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
from utils import *
import pdb

gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]


## Split data 
X, y = read_digits()
# X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.3)


## Use the preprocessed datas

h_params={}
h_params['gamma'] = gamma_ranges
h_params['C'] = C_ranges

h_params_combinations = get_hyperparameter_combinations(h_params)
print("h_params_combinations ", len(h_params_combinations))

test_sizes =  [0.1, 0.2, 0.3, 0.45]
dev_sizes  =  [0.1, 0.2, 0.3, 0.45]
for test_size in test_sizes:
    for dev_size in dev_sizes:
        train_size = 1- test_size - dev_size
        # Data splitting -- to create train and test sets                
        X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(X, y, test_size=0.3, dev_size=0.3)
        # Data preprocessing
        X_train = data_preprocess(X_train)
        X_test = data_preprocess(X_test)
        X_dev = data_preprocess(X_dev)
    
        best_hparams, best_model, best_accuracy  = tune_hparams(X_train, y_train, X_dev, 
        y_dev, h_params_combinations)

        test_acc = predict_and_eval(best_model, X_test, y_test)
        train_acc = predict_and_eval(best_model, X_train, y_train)
        dev_acc = best_accuracy

        print("test_size={:.2f} dev_size={:.2f} train_size={:.2f} train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}".format(test_size, dev_size, train_size, train_acc, dev_acc, test_acc))