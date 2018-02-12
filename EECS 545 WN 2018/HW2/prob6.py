import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# Normalized Binary dataset
# 4 features, 100 examples, 50 labeled 0 and 50 labeled 1
X, y = load_breast_cancer().data, load_breast_cancer().target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# normalize
mean_train = np.mean(X_train, axis = 0)
std_train = np.std(X_train, axis = 0)
std_train[std_train == 0]=1
X_train_norm = (X_train - mean_train) / std_train
X_test_norm = (X_test - mean_train) / std_train

def accuracy(X, Y, w):
    a = 0
    Y_w = X.dot(w)
    for i in range(len(Y)):
        if Y_w[i] >= 0:
            judge = 1
        else:
            judge = 0
        if judge == Y[i]:
            a += 1
    accu = a / len(Y)
    return accu

def sig_func(a):
    sig = np.exp(a) / (1 + np.exp(a))
    return sig

def loss(X, Y, w):
    los = []
    for i in range(len(X)):
        los.append(Y[i] * np.log(sig_func(w.T.dot(X[i,:]))) + (1-Y[i]) * np.log(1 - sig_func(w.T.dot(X[i,:]))))
    return -np.sum(los)

n, m = X_train.shape
w = np.random.uniform(-0.1, 0.1, m)
learning_rate = 1e-2
order = list(range(n))
train_accuracy = []
test_accuracy = []
train_error = []
test_error = []

dataset_train = np.c_[y_train,X_train_norm]
np.random.shuffle(dataset_train)  # random shuffle
Y_new = dataset_train[:, 0]
X_new = dataset_train[:, 1:]
for i in order:
    xi = X_train_norm[i]
    yi = y_train[i]
    gradients = xi.T.dot(sig_func(xi.dot(w)) - yi)
    w = w - learning_rate * gradients
    train_error.append(loss(X_train_norm, y_train, w) / len(y_train))
    test_error.append(loss(X_test_norm, y_test, w) / len(y_test))
    train_accuracy.append(accuracy(X_train_norm, y_train, w))
    test_accuracy.append(accuracy(X_test_norm, y_test, w))

print("w:\n", w)
print('Final train cross-entropy:', train_error[-1] * len(y_train))
print('Final test cross-entropy:', test_error[-1] * len(y_test))
print('Final train accuracy:', train_accuracy[-1])
print('Final test accuracy:', test_accuracy[-1])

plt.plot(order, train_accuracy, label = 'train accuracy')
plt.plot(order, test_accuracy, label = 'test accuracy')
plt.title('Accuracy vs # Iterations')
plt.legend()
plt.show()

plt.plot(order, train_error, label = 'Train Errors')
plt.plot(order, test_error, label = 'Test Errors')
plt.title('Errors vs # Iterations')
plt.legend()
plt.show()