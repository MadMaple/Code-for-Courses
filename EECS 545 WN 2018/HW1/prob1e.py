import numpy as np
from sklearn import datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt
from numpy.linalg import inv

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
  X_train = preprocessing.normalize(X_train)
  X_test = preprocessing.normalize(X_test)
  bias_train = np.ones(len(y_train))  # generate bias
  bias_test = np.ones(len(y_test))
  X_train = np.insert(X_train, 0, bias_train, 1)  # insert bias before the first column in X_train
  X_test = np.insert(X_test, 0, bias_test, 1)  # insert bias before the first column in X_test

  # Solve for optimal w
  # Use your solver function
  mp_pseudoinverse = inv(np.dot(X_train.T, X_train))
  w = np.dot(np.dot(mp_pseudoinverse, X_train.T), y_train)

  # Collect train and test errors
  # Use your implementation of the mse function
  error = abs(np.dot(X_train, w) - y_train)
  train_mse.append(np.dot(error, error) / len(y_train))
  error = np.dot(X_test, w) - y_test
  test_mse.append(np.dot(error, error) / len(y_test))

print('Mean training error: ', np.mean(train_mse))
print('Mean test error: ', np.mean(test_mse))
print("Learned weight vector = ", w, "\n", "Bias terms = ", w[0])