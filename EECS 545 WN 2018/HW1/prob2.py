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


# Normalization
def normalization(X):
    mu_train = np.mean(X, axis=0)
    sigma_train = np.std(X, axis=0)
    sigma_train[sigma_train == 0] = 1
    X_norm = (X - mu_train) / sigma_train
    return  X_norm

# 2.(a)
#  Construct Phi(X)
def phi(X, deg):
    n, m = X.shape
    c = 1
    X_poly = np.ones((n, 1))
    while c <= deg:
        if deg == 0:
            break
        else:
            X_poly = np.c_[X_poly,X ** c]
            c += 1
    return X_poly

# Closed form solution
def cfs(X,Y,deg):
    X_poly = phi(X, deg)
    X_new = normalization(X_poly)  # normalization
    n, m = X.shape
    if deg == 0:
        mp_pseudoinverse = 1 / (np.dot(X_new.T,X_new))
    else:
        mp_pseudoinverse = np.linalg.pinv(np.dot(X_new.T,X_new))
    w = np.dot(np.dot(mp_pseudoinverse,X_new.T),Y)
    error = abs(np.dot(X_new, w) - Y)
    rmse_train = (np.dot(error.T, error) / n) ** 0.5
    print("Learned weight vector = ", w, "\n", "Training Error = ", rmse_train)
    return w, rmse_train

# Testing
def cfs_predi(X,Y,deg):
    w, mse_train = cfs(X_train, y_train, deg)
    X_new = phi(X, deg)
    X_train_new = phi(X_train, deg)
    n, m = X_new.shape
    mu = np.mean(X_train_new, axis=0)
    sigma = np.std(X_train_new, axis=0)
    for i in range(1,m):  # Normalize after adding "1"
        if sigma[i] != 0:
            X_new[:, i] = (X_new[:, i] - mu[i]) / sigma[i]
        else:
            X_new[:, i] = (X_new[:, i] - mu[i]) / 1
    error = Y - X_new.dot(w)
    rmse_test = (np.mean(error ** 2)) ** 0.5
    print("Testing Errors =", rmse_test)
    return rmse_test

# main
rmse_tr = []
rmse_te  = []
deg = np.array([0, 1, 2, 3, 4])
for degree in deg:
    w, rmse_train = cfs(X_train, y_train, degree)
    rmse_tr.append(rmse_train)
    rmse_test = cfs_predi(X_test, y_test, degree)
    rmse_te.append(rmse_test)
rmse_tr_nd = np.array(rmse_tr)
rmse_te_nd = np.array(rmse_te)
plt.plot(deg, rmse_tr_nd, color="r", label="RMSE_Train")
plt.plot(deg, rmse_te_nd, color="g", label="RMSE_Test")
plt.legend()
plt.show()


########################################################################################################################

# 2.(b)
train_per = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
n = len(X_train)#length of original training data
training_error = np.zeros(len(train_per))
testing_error = np.zeros(len(train_per))
for i in range(len(train_per)):
    nsplit = int(np.round(train_per[i] * n))
    X_new, y_new = X_train[:nsplit], y_train[:nsplit]
    ####
    X_norm = normalization(X_new)
    bias = np.ones(shape=X_norm[:, 0].shape)  # generate bias[n, 1]
    X_bias = np.c_[bias, X_norm]  # insert bias before the first column in X
    ##### ?why cannot use n, m = X_test.shape; bias = np.ones(n)
    mu_train = np.mean(X_new, axis=0)
    sigma_train = np.std(X_new, axis=0)
    sigma_train[sigma_train == 0] = 1
    X_test_norm = (X_test - mu_train) / sigma_train
    bias_test = np.ones(shape=X_test_norm[:, 0].shape)
    X_test_bias = np.c_[bias_test, X_test_norm]
    ####
    mp_pseudoinverse = np.linalg.pinv(np.dot(X_bias.T, X_bias))
    w = np.dot(np.dot(mp_pseudoinverse, X_bias.T), y_new)
    training_error[i] = (np.mean((np.dot(X_bias, w) - y_new) ** 2)) ** 0.5
    testing_error[i] = (np.mean((np.dot(X_test_bias, w) - y_test) ** 2)) ** 0.5
plt.plot(train_per,training_error,color="r", label="Training RMSE")
plt.plot(train_per,testing_error, color="g", label='Testing RMSE')
plt.legend()
plt.show()

