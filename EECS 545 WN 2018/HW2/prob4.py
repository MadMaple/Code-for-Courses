import numpy as np
import matplotlib.pyplot as plt

# generate cov matrix, i.e. k(x,x')
def kernel_cov(X, Y, sigma):
    m = len(X)
    n = len(Y)
    cov = np.zeros((m,n))
    i, j =0, 0
    for x in X:
        for y in Y:
            cov[i, j] = np.exp(-0.5 * (x - y) ** 2 / (sigma ** 2))
            j += 1
        i += 1
        j = 0
    return cov

# (a)
sample_x = np.linspace(-5, 5, 100)
kernel_sigma = np.array([0.3, 0.5, 1.0])
X, Y = sample_x, sample_x
for sigma in kernel_sigma:
    mu = np.zeros(len(X))
    cov = kernel_cov(X, Y, sigma)
    rmvnorm = np.random.multivariate_normal(mu, cov, 5)
    plt.plot(X, rmvnorm.T)
    plt.title('sigma is %s' % (sigma))
    plt.show()

# (b)
X_train = np.array([-1.3, 2.4, -2.5, -3.3, 0.3])
Y_train = np.array([0, 5.2, -1.5, -0.8, 0.3])

def posterior(X_train, X_test,Y_train,sigma):
    cov_train = kernel_cov(X_train,X_train,sigma)
    cov_test = kernel_cov(X_test,X_test,sigma)
    cov_tetr = kernel_cov(X_test,X_train,sigma)
    cov_trte = kernel_cov(X_train,X_test,sigma)
    mu = cov_tetr.dot(np.linalg.inv(cov_train)).dot(Y_train)
    cov = cov_test - cov_tetr.dot(np.linalg.inv(cov_train)).dot(cov_trte)
    return mu, cov

for sigma in kernel_sigma:
        mu, cov = posterior(X_train, X, Y_train, sigma)
        plt.scatter(X_train, Y_train, c = "b", s = 100, label = "Points in D")
        plt.plot(X, mu, c = "r", lw = 4, label = "The posterior mean of the GP")
        rmvnorm = np.random.multivariate_normal(mu, cov, 5)
        plt.plot(X, rmvnorm.T, lw = 1)
        plt.title('sigma is %s' % (sigma))
        plt.legend()
        plt.show()