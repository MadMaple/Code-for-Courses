import numpy as np
from sklearn import datasets
from numpy.linalg import inv

# Load dataset
dataset = datasets.load_boston()

features = dataset.data
labels = dataset.target

Nsplit = 50
# Training set
X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
num_row, num_col = X_train.shape
# Validation set
X_train_new, y_train_new = X_train[: -int((0.1 * num_row))],y_train[: -int((0.1 * num_row))]
X_valid, y_valid = X_train[-int((0.1 * num_row)):],y_train[-int((0.1 * num_row)):]
# Test set
X_test, y_test = features[-Nsplit:], labels[-Nsplit:]

# Normalization
def normalization(X):
    n, m = X.shape
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    for i in range(1, m): # Normalize after adding "1"
        if sigma[i] != 0:
            X[:, i] = (X[:, i] - mu[i]) / sigma[i]
        else:
            X[:, i] = X[:, i] - mu[i]
    return X

# 3(b)
def cfs(X,Y,lambda_):
    n, m = X.shape
    bias = np.ones(n)  # generate bias[n, 1]
    X_bias = np.insert(X, 0, bias, 1)  # insert bias before the first column in X
    X_bias = normalization(X_bias)
    mp_pseudoinverse = inv(n * lambda_ * np.eye(m + 1) + np.dot(X_bias.T,X_bias))
    w = np.dot(np.dot(mp_pseudoinverse,X_bias.T),Y)
    return w

# Testing
def cfs_predi(X,Y, lambda_):
    w = cfs(X_train_new, y_train_new, lambda_)
    n, m = X.shape
    mu = np.mean(X_train_new, axis=0)
    sigma = np.std(X_train_new, axis=0)
    for i in range(m):  # Normalize after adding "1"
        if sigma[i] != 0:
            X[:, i] = (X[:, i] - mu[i]) / sigma[i]
        else:
            X[:, i] = (X[:, i] - mu[i]) / 1
    bias = np.ones(n)  # generate bias[n, 1]
    X_bias = np.insert(X, 0, bias, 1)  # insert bias before the first column in X
    error = np.dot(X_bias, w) - Y
    rmse_test = (np.dot(error, error) / n) ** 0.5
    return rmse_test

# main
print("When lambda is ", 0.3, ", the RMSE is ", cfs_predi(X_valid,y_valid,0.3))
print("When lambda is ", 0.3,", Test error is ", cfs_predi(X_test,y_test,0.3))
