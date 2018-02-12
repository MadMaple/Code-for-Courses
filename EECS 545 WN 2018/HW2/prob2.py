import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# feel free to read the two examples below, try to understand them
# in this problem, we require you to generate contour plots

# generate contour plot for function z = x^2 + 2*y^2
def plot_contour():

    plot_delta = 0.025
    plot_x = np.arange(-3.0, 3.0, plot_delta)
    plot_y = np.arange(-3.0, 3.0, plot_delta)
    X, Y = np.meshgrid(plot_x, plot_y)
    x_len = plot_x.shape[0]
    y_len = plot_y.shape[0]
    Z = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            Z[j][i] = X[j][i] ** 2 + 2 * (Y[j][i] ** 2)

    plt.clf()
    # plt.axis("square")
    cs = plt.contour(X, Y, Z)
    plt.clabel(cs, inline=0.1, fontsize=10)
    plt.show()

# generate heat plot (image-like) for function z = x^2 + 2*y^2
def plot_heat():
    plot_delta = 0.025
    plot_x = np.arange(-3.0, 3.0, plot_delta)
    plot_y = np.arange(-3.0, 3.0, plot_delta)
    X, Y = np.meshgrid(plot_x, plot_y)
    x_len = plot_x.shape[0]
    y_len = plot_y.shape[0]
    Z = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            Z[j][i] = X[j][i] ** 2 + 2 * (Y[j][i] ** 2)

    plt.clf()
    img = Z
    plt.imshow(img, interpolation='none', extent=[-3.0, 3.0, -3.0, 3.0],cmap="plasma")
    plt.colorbar()
    plt.axis("square")
    plt.show()

# This function receives the parameters of a multivariate Gaussian distribution
# over variables x_1, x_2 .... x_n as input and compute the marginal
#
def marginal_for_guassian(sigma,mu,given_indices):
    # given selected indices, compute marginal distribution for them
    n = len(given_indices)
    sigma_marginal = np.zeros((n,n))
    mean_marginal = np.zeros(n)
    r, c = 0, 0
    for i in given_indices:
        mean_marginal[r] = mu[i]
        for j in given_indices:
            sigma_marginal[r, c] = sigma[i, j]
            c += 1
        r += 1
        c = 0
    return mean_marginal, sigma_marginal

def COV(index1, index2, sigma):
    m = len(index1)
    n = len(index2)
    cov = np.zeros((m,n))
    r, c = 0, 0
    for i in index1:
        for j in index2:
            cov[r, c] = sigma[i, j]
            c += 1
        r += 1
        c = 0
    return cov

def conditional_for_gaussian(sigma,mu,given_indices,given_values):
    # given some indices that have fixed value, compute the conditional distribution
    # for rest indices
    mean_given, sigma_given = marginal_for_guassian(sigma, mu, given_indices)
    mean_remain = []
    remain_index = []
    n = len(mu)
    m = len(given_indices)
    # # rest mu and sigma
    for i in range(n):
        if i in given_indices:
            pass
        else:
            mean_remain.append(mu[i])
            remain_index.append(i)
    sigma_remain = COV(remain_index, remain_index, sigma)
    sigma_gire = COV(given_indices,remain_index, sigma)
    sigma_regi = COV(remain_index,given_indices, sigma)
    sigma_conditioned = sigma_remain - sigma_regi.dot(np.linalg.inv(sigma_given)).dot(sigma_gire)
    mean_conditioned = mean_remain + sigma_regi.dot(np.linalg.inv(sigma_given)).dot(given_values - mean_given)
    return mean_conditioned,sigma_conditioned


# (2) Compute and plot marginal distribution
test_sigma_1 = np.array(
    [[1.0, 0.5],
     [0.5, 1.0]]
)

test_mu_1 = np.array(
    [0.0, 0.0]
)

indices_1 = np.array([0])

## mean and sigma of X_1
mean_test1, sigma_test1 = marginal_for_guassian(test_sigma_1,test_mu_1,indices_1)
print("Marginal mean:", mean_test1)
print("Marginal sigma:\n", sigma_test1)
## plot PDF of X_1 distribution
X = np.arange(-3.0, 3.0, 0.025)
x_len = X.shape[0]
Y = np.zeros(x_len)
### PDF function
for i in range(x_len):
    a = 1 / np.sqrt(np.pi * sigma_test1)
    b = (X[i] - mean_test1) ** 2 / (2 * sigma_test1)
    Y[i] = a * np.exp(-b)

plt.plot(X,Y)
plt.show()

# (4) Compute and plot conditional distribution
test_sigma_2 = np.array(
    [[1.0, 0.5, 0.0, 0.0],
     [0.5, 1.0, 0.0, 1.5],
     [0.0, 0.0, 2.0, 0.0],
     [0.0, 1.5, 0.0, 4.0]]
)

test_mu_2 = np.array(
    [0.5, 0.0, -0.5, 0.0]
)

indices_2 = np.array([1,2])
values_2 = np.array([0.1,-0.2])

## mean and sigma of conditioned P
mean_condition, sigma_condition = conditional_for_gaussian(test_sigma_2, test_mu_2, indices_2, values_2)
print("Conditioned mean:", mean_condition)
print("Conditioned sigma:\n", sigma_condition)
## plot PDF of conditioned distribution
plot_delta = 0.025
plot_x = np.arange(-3.0, 3.0, plot_delta)
plot_y = np.arange(-3.0, 3.0, plot_delta)
X, Y = np.meshgrid(plot_x, plot_y)
x_len = plot_x.shape[0]
y_len = plot_y.shape[0]
Z = np.zeros((x_len, y_len))
for i in range(x_len):
    for j in range(y_len):
        Z[j][i] = multivariate_normal.pdf((X[j][i], Y[j][i]), mean = mean_condition, cov = sigma_condition)

plt.clf()
cs = plt.contour(X, Y, Z)
plt.clabel(cs, inline=0.1, fontsize=10)
plt.show()