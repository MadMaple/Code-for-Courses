import numpy as np
from sklearn import datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt

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

# 1.(c) Batch gradient descent
# Training
def bgd(learning_rate, number_epoches, X, Y):
    X = preprocessing.normalize(X)  # normalization
    n, m = X.shape  # n: # of rows; m: # of columns
    bias = np.ones(n)  # generate bias[n, 1]
    X_bias = np.insert(X, 0, bias, 1)  # insert bias before the first column in X
    w = np.random.uniform(-0.1, 0.1, m + 1)  # initialize learned weight vector w0 with 14 columns
    count = 1  # count # of epoch
    converged = False
    while count <= number_epoches:
        if converged:
            break
        diff = np.dot((np.dot(X_bias,w) - Y), X_bias)
        error = abs(np.dot(X_bias,w) - Y)
        w = w - learning_rate * diff
        mse = np.dot(error, error) / n
        if abs(mse) < 1e-6:
                 converged = True
        plt.plot(count, mse, ".")  # plot errors
        count += 1
    print("Learned weight vector = ", w, "\n", "Bias terms = ", w[0], "\n", "Training Error = ", mse)
    plt.show()
    return w

# Testing
def bgd_predi(X,Y):
    w = bgd(5e-4,500, X_train, y_train)
    X = preprocessing.normalize(X)  # normalization
    n, m = X.shape  # n: # of rows; m: # of columns
    bias = np.ones(n)  # generate bias[n, 1]
    X_bias = np.insert(X, 0, bias, 1)  # insert bias before the first column in X
    error = np.dot(X_bias,w) - Y
    mse = np.dot(error,error)/n
    print("Testing Errors =",mse)

# main
bgd_predi(X_test,y_test)