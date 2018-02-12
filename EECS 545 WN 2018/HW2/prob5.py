import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab

data = np.zeros((100, 3))
val = np.random.uniform(0, 2, 100)
diff = np.random.uniform(-1, 1, 100)
data[:,0], data[:,1], data[:,2] = val - diff, val + diff, np.ones(100)
target = np.asarray(val > 1, dtype = int) * 2 - 1

def perceptron(X, Y, iteration):
    w = np.zeros(3)
    n = len(Y)
    for i in range(iteration):
        for j in range(n):
            Y_predict = w.T.dot(X[j])
            if Y[j] * Y_predict <= 0:
                w = w + Y[j] * X[j]
    return w

w = perceptron(data, target, 10)
print(w)
plt.scatter(data[target > 0][:,0], data[target > 0][:,1],label = '1')
plt.scatter(data[target < 0][:,0], data[target < 0][:,1],label = '-1')
line = []
for i in np.linspace(-1,3,100):
    a = -w[0]/w[1] * i- w[2] / w[1] # w[0]x+w[1]y+w[2]=0
    line.append(a)
plt.plot(np.linspace(-1,3,100), line)
plt.legend(loc = 'upper right')
plt.show()

# (5b)
data = np.ones((100, 3))
data[:50,0], data[50:,0] = np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)
data[:50,1], data[50:,1] = np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)
target = np.zeros(100)
target[:50], target[50:] = -1 * np.ones(50), np.ones(50)

w = perceptron(data,target,10)
print(w)
plt.scatter(data[target > 0][:,0], data[target > 0][:,1],label = '1')
plt.scatter(data[target < 0][:,0], data[target < 0][:,1],label = '-1')
line = []
for i in np.linspace(-1,3,100):
    a = -w[0]/w[1] * i- w[2] / w[1] # w[0]x+w[1]y+w[2]=0
    line.append(a)
plt.plot(np.linspace(-1,3,100), line)
plt.legend(loc = 'upper right')
plt.show()

## GDA
X1 = data[target > 0][:,0:2]
X2 = data[target < 0][:,0:2]
mean1 = np.mean(X1, axis = 0)
mean2 = np.mean(X2, axis = 0)
phi = len(X1[:,0]) / 100
sigma = ((X1 - mean1).T.dot(X1 - mean1) + (X2 - mean2).T.dot(X2 - mean2)) / 200
w = 2 * np.linalg.pinv(sigma).dot(mean2 - mean1)
b = mean2.dot(np.linalg.pinv(sigma)).dot(mean2.T) - mean1.dot(np.linalg.pinv(sigma)).dot(mean1.T) + np.log(phi) - np.log(1 - phi)
print("linear seperator is wX=b, where w is", w, ", b is", b)

plt.scatter(data[target > 0][:,0], data[target > 0][:,1],label = '1')
plt.scatter(data[target < 0][:,0], data[target < 0][:,1],label = '-1')
line = []
for i in np.linspace(-1,3,100):
    a = (-w[0] * i + b) / w[1] # w[0]x+w[1]y=b
    line.append(a)
plt.plot(np.linspace(-1,3,100), line)
plt.show()