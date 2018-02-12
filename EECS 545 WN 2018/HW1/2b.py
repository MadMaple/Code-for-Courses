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

def cfs(X,Y,percentage):
    if percentage == 0.2:
        cl3 = (X[:, 3] - np.mean(X[:, 3])) / 1 # because the standard error of the 3rd features is 0, we set this std as 1
        new = np.delete(X, 2, 1)
        X = np.insert(new, 2, cl3, 1)
    else:
        X = preprocessing.normalize(X)  # normalization
    n, m = X.shape
    bias = np.ones(n)  # generate bias[n, 1]
    X_bias = np.insert(X, 0, bias, 1)  # insert bias before the first column in X
    if percentage == 1:
        X_new = X_bias
        Y_new = Y
    else:
        X_new= X_bias[: -int(((1 - percentage) * n))]
        Y_new = Y[: -int(((1 - percentage) * n))]
    mp_pseudoinverse = np.linalg.pinv(np.dot(X_new.T, X_new))
    w = np.dot(np.dot(mp_pseudoinverse,X_new.T),Y_new)
    error = abs(np.dot(X_new, w) - Y_new)
    mse_train = np.dot(error, error) / n
    print("Learned weight vector = ", w, "\n", "Training Error = ", mse_train)
    return w, mse_train

# Testing
def cfs_predi(X,Y,percentage):
    w, mse_train = cfs(X_train, y_train, percentage)
    n, m = X.shape
    X = preprocessing.normalize(X)  # normalization
    bias = np.ones(n)  # generate bias[n, 1]
    X_bias = np.insert(X, 0, bias, 1)  # insert bias before the first column in X
    error = np.dot(X_bias, w) - Y
    mse_test = np.dot(error, error) / n
    print("Testing Errors =", mse_test)
    return mse_test

# main
rmse_tr = []
rmse_te  = []
percen = [0.2, 0.4, 0.6, 0.8, 1]
for percentage in percen:
    w, mse_train = cfs(X_train, y_train, percentage)
    rmse_train = mse_train ** 0.5
    rmse_tr.append(rmse_train)
    mse_test = cfs_predi(X_test, y_test, percentage)
    rmse_test = mse_test ** 0.5
    rmse_te.append(rmse_test)
per_nd = np.array(percen)
rmse_tr_nd = np.array(rmse_tr)
rmse_te_nd = np.array(rmse_te)
plt.plot(per_nd, rmse_tr_nd, color="r", label="RMSE_Train")
plt.plot(per_nd, rmse_te_nd, color="g", label="RMSE_Test")
plt.legend()
plt.show()

################################################################
train_per = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
n = len(X_train)#length of original training datahttp://yusuke.homeip.net/twitter4j/en/index.html
training_error = np.zeros(len(train_per))
testing_error = np.zeros(len(train_per))
for i in range(len(train_per)):
    nsplit = int(np.round(train_per[i] * n))
    X_new, y_new = X_train[:nsplit], y_train[:nsplit]
    ####
    mu_train = np.mean(X_new, axis=0)
    sigma_train = np.std(X_new, axis=0)
    sigma_train[sigma_train == 0] = 1
    X_norm = (X_new - mu_train) / sigma_train
    bias = np.ones(shape=X_norm[:, 0].shape)  # generate bias[n, 1]
    X_bias = np.c_[bias, X_norm]  # insert bias before the first column in X
    ##### ?why cannot use n, m = X_test.shape; bias = np.ones(n)
    X_test_norm = (X_test - mu_train) / sigma_train
    bias_test = np.ones(shape=X_test_norm[:, 0].shape)
    X_test_bias = np.c_[bias_test.T, X_test_norm]
    ####
    mp_pseudoinverse = np.linalg.pinv(np.dot(X_bias.T, X_bias))
    w = np.dot(np.dot(mp_pseudoinverse, X_bias.T), y_new)
    training_error[i] = (np.mean((np.dot(X_bias, w) - y_new) ** 2)) ** 0.5
    testing_error[i] = (np.mean((np.dot(X_test_bias, w) - y_test) ** 2)) ** 0.5
plt.plot(train_per,training_error,color="r", label="Training RMSE")
plt.plot(train_per,testing_error, color="g", label='Testing RMSE')
plt.legend()
plt.show()