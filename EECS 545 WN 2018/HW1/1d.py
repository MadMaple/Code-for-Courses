import numpy as np
from sklearn import datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Load dataset
dataset = datasets.load_boston()

features = dataset.data
labels = dataset.target

Nsplit = 50
# Training set
X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
# Test set
X_test, y_test = features[-Nsplit:], labels[-Nsplit:]

# 1.(a) Normalization
normal_features = preprocessing.normalize(features)

# 1.(d) Closed form solution
# Training
def cfs(X,Y):
    X = preprocessing.normalize(X)  # normalization
    n, m = X.shape
    bias = np.ones(n)  # generate bias[n, 1]
    X_bias = np.insert(X, 0, bias, 1)  # insert bias before the first column in X
    mp_pseudoinverse = inv(np.dot(X_bias.T,X_bias))
    w = np.dot(np.dot(mp_pseudoinverse,X_bias.T),Y)
    error = abs(np.dot(X_bias, w) - Y)
    mse = np.dot(error, error) / n
    print("Learned weight vector = ", w, "\n", "Bias terms = ", w[0], "\n", "Training Error = ", mse)
    return w

# Testing
def cfs_predi(X,Y):
    w = cfs(X_train, y_train)
    n, m = X.shape
    X = preprocessing.normalize(X)  # normalization
    bias = np.ones(n)  # generate bias[n, 1]
    X_bias = np.insert(X, 0, bias, 1)  # insert bias before the first column in X
    error = np.dot(X_bias, w) - Y
    mse = np.dot(error, error) / n
    print("Testing Errors =", mse)

# main
cfs_predi(X_test, y_test)