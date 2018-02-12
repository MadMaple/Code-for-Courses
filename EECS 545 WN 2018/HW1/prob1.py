import numpy as np
from sklearn import datasets
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
def normalization(X):
    n, m = X.shape
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    for i in range(1, m): # Normalize after adding "1"
        if sigma[i] != 0:
            X[:, i] = (X[:, i] - mu[i]) / sigma[i]
        else:
            X[:, i] = (X[:, i] - mu[i]) / 1
    return X

#######################################################################################################################

# 1.(b) Stochastic gradient descent
# Training
def sgd(learning_rate, number_epoches, X, Y):
    n, m = X.shape # n: # of rows; m: # of columns
    bias = np.ones(n) # generate bias[n, 1]
    X_bias = np.c_[bias, X] # insert bias before the first column in X
    X_bias = normalization(X_bias)
    w = np.random.uniform(-0.1, 0.1, m + 1) # initialize learned weight vector w0 with 14 columns
    count = 1 # count # of epoch
    errors = np.ones(n) # initialize errors
    converged = False
    dataset_train = np.c_[Y,X_bias]
    while count <= number_epoches:
        if converged:
            break
        np.random.shuffle(dataset_train)  # random shuffle
        Y_new = dataset_train[:,0]
        X_new = dataset_train[:,1:]
        for j in range(n):  # for j = 1,...,N do:
            diff = np.dot((np.dot(w, X_new[j,].T) - Y_new[j]), X_new[j,].T)
            w = w - learning_rate * diff
            error = np.dot(w, X_new[j,].T) - Y_new[j]  # error for X_j
            errors[j] = error
        mse = np.dot(errors, errors) / n
        if abs(mse) < 1e-10:
                 converged = True
        plt.plot(count, mse,".", color="b")  # plot errors
        count += 1
    print("Learned weight vector = ", w, "\n", "Bias terms = ", w[0], "\n", "Training Error = ", mse)
    plt.show()
    return w

# Testing
def sgd_predi(X,Y):
    w = sgd(5e-4,500, X_train, y_train)
    n, m = X.shape  # n: # of rows; m: # of columns
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    for i in range(m):  # Normalize after adding "1"
        if sigma[i] != 0:
            X[:, i] = (X[:, i] - mu[i]) / sigma[i]
        else:
            X[:, i] = (X[:, i] - mu[i]) / 1
    bias = np.ones(n)  # generate bias[n, 1]s
    X_bias = np.c_[bias, X]  # insert bias before the first column in X
    error = Y - X_bias.dot(w)
    mse = np.mean(error ** 2)
    print("Testing Errors =",mse)

# main
sgd_predi(X_test,y_test)

#######################################################################################################################

# 1.(c) Batch gradient descent
# Training
def bgd(learning_rate, number_epoches, X, Y):
    n, m = X.shape  # n: # of rows; m: # of columns
    bias = np.ones(n)  # generate bias[n, 1]
    X_bias = np.c_[bias, X]  # insert bias before the first column in X
    X_bias = normalization(X_bias)
    w = np.random.uniform(-0.1, 0.1, m + 1)  # initialize learned weight vector w0 with 14 columns
    count = 1  # count # of epoch
    converged = False
    while count <= number_epoches:
        if converged:
            break
        diff = np.dot((np.dot(X_bias,w) - Y), X_bias)
        error = np.dot(X_bias,w) - Y
        w = w - learning_rate * diff
        mse = np.dot(error, error) / n
        if abs(mse) < 1e-6:
                 converged = True
        plt.plot(count, mse, ".", color="b")  # plot errors
        count += 1
    print("Learned weight vector = ", w, "\n", "Bias terms = ", w[0], "\n", "Training Error = ", mse)
    plt.show()
    return w

# Testing
def bgd_predi(X,Y):
    w = bgd(5e-4,500, X_train, y_train)
    n, m = X.shape  # n: # of rows; m: # of columns
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    for i in range(m):  # Normalize after adding "1"
        if sigma[i] != 0:
            X[:, i] = (X[:, i] - mu[i]) / sigma[i]
        else:
            X[:, i] = (X[:, i] - mu[i]) / 1
    bias = np.ones(n)  # generate bias[n, 1]s
    X_bias = np.c_[bias, X]  # insert bias before the first column in X
    error = np.dot(X_bias,w) - Y
    mse = np.dot(error,error)/n
    print("Testing Errors =",mse)

# main
bgd_predi(X_test,y_test)

#######################################################################################################################

# 1.(d) Closed form solution
## This part need to run individually ##
# Training
def cfs(X,Y):
    n, m = X.shape
    bias = np.ones(n)  # generate bias[n, 1]
    X_bias = np.insert(X, 0, bias, 1)  # insert bias before the first column in X
    X_bias = normalization(X_bias)
    mp_pseudoinverse = inv(np.dot(X_bias.T,X_bias))
    w = np.dot(np.dot(mp_pseudoinverse,X_bias.T),Y)
    error = np.dot(X_bias, w) - Y
    mse = np.dot(error, error) / n
    print("Learned weight vector = ", w, "\n", "Bias terms = ", w[0], "\n", "Training Error = ", mse)
    return w

# Testing
def cfs_predi(X,Y):
    w = cfs(X_train, y_train)
    n, m = X.shape  # n: # of rows; m: # of columns
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    for i in range(m):  # Normalize after adding "1"
        if sigma[i] != 0:
            X[:, i] = (X[:, i] - mu[i]) / sigma[i]
        else:
            X[:, i] = (X[:, i] - mu[i]) / 1
    bias = np.ones(n)  # generate bias[n, 1]s
    X_bias = np.c_[bias, X]  # insert bias before the first column in X
    error = X_bias.dot(w) - Y
    mse = np.dot(error, error) / n
    print("Testing Errors =", mse)

# main
cfs_predi(X_test, y_test)

#######################################################################################################################

# 1.(e)
# Load dataset
dataset = datasets.load_boston()

# Original features
features_orig = dataset.data
labels_orig = dataset.target
Ndata = len(features_orig)

train_mse = []
test_mse = []

for k in range(100):

  # Shuffle data
  rand_perm = np.random.permutation(Ndata)
  features = [features_orig[ind] for ind in rand_perm]
  labels = [labels_orig[ind] for ind in rand_perm]

  # Train/test split
  Nsplit = 50
  X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
  X_test, y_test = features[-Nsplit:], labels[-Nsplit:]

  # Preprocess your data - Normalization, adding a constant feature
  bias_train = np.ones(len(y_train))  # generate bias
  bias_test = np.ones(len(y_test))
  X_train_b = np.c_[bias_train, X_train]  # insert bias before the first column in X_train
  X_train_b = normalization(X_train_b)
  X_test = np.array(X_test)
  n, m = X_test.shape
  mu = np.mean(X_train, axis=0)
  sigma = np.std(X_train, axis=0)
  for i in range(m):  # Normalize after adding "1"
      if sigma[i] != 0:
          X_test[:, i] = (X_test[:, i] - mu[i]) / sigma[i]
      else:
          X_test[:, i] = (X_test[:, i] - mu[i]) / 1
  bias = np.ones(n)  # generate bias[n, 1]s
  X_test_b = np.c_[bias, X_test]  # insert bias before the first column in X

  # Solve for optimal w
  # Use your solver function
  mp_pseudoinverse = inv(np.dot(X_train_b.T, X_train_b))
  w = np.dot(np.dot(mp_pseudoinverse, X_train_b.T), y_train)

  # Collect train and test errors
  # Use your implementation of the mse function
  error = np.dot(X_train_b, w) - y_train
  train_mse.append(np.dot(error, error) / len(y_train))
  error = np.dot(X_test_b, w) - y_test
  test_mse.append(np.dot(error, error) / len(y_test))

print('Mean training error: ', np.mean(train_mse))
print('Mean test error: ', np.mean(test_mse))
print("Learned weight vector = ", w, "\n", "Bias terms = ", w[0])