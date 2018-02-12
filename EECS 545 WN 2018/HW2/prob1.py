import numpy as np
import matplotlib.pyplot as plt


# For this problem, we use data generator instead of real dataset
def data_generator(size,noise_scale=0.05):
    xs = np.random.uniform(low=0,high=3,size=size)

    # for function y = 0.5x - 0.3 + sin(x) + epsilon, where epsilon is a gaussian noise with std dev= 0.05
    ys = xs * 0.5 - 0.3 + np.sin(3*xs) + np.random.normal(loc=0,scale=noise_scale,size=size)
    return xs, ys


def main():
    noise_scales = [0.05,0.2]

    # for example, choose the first kind of noise scale
    noise_scale = noise_scales[0]

    # generate the data form generator given noise scale
    X_train, y_train = data_generator((100,1),noise_scale=noise_scale)
    X_test, y_test = data_generator((30,1),noise_scale=noise_scale)

    # normalize data
    mean_train = np.mean(X_train, axis = 0)
    std_train = np.std(X_train, axis = 0)
    std_train[std_train == 0]=1
    X_train_norm = (X_train - mean_train) / std_train
    X_test_norm = (X_test - mean_train) / std_train

    # add intercept
    n = len(y_train)
    m = len(y_test)
    intercept_train = np.ones(n)
    intercept_test = np.ones(m)
    X_train_new = np.c_[intercept_train, X_train_norm]
    X_test_new = np.c_[intercept_test, X_test_norm]

    # # (a) closed form solution
    X_Pseudo = np.linalg.pinv(np.dot(X_train_new.T,X_train_new))
    w = np.dot(np.dot(X_Pseudo,X_train_new.T),y_train)
    predict_labels = X_test_new.dot(w)
    plt.plot(X_test, y_test,".", label = "Oringinal Data")
    plt.plot(X_test, predict_labels,".", label = "Predicted Data")
    test_error = abs(predict_labels - y_test)
    MSE = np.mean(test_error ** 2)
    print("MSE =",MSE)
    plt.legend()
    plt.show()

    # (b) local weighted linear regression
    # bandwidth parameters
    sigma_paras = [0.2,2.0]

    # choose r
    # r = np.eye(n)
    for tao in sigma_paras:
        predict_labels = []
        for i in range(m):
            r = np.exp(-np.sum((X_train - X_test[i,]) ** 2, axis = 1)/(2 * tao ** 2))
            # for j in range(n):
            #     r[j,j] = np.exp(-np.sum((X_train_new[j] - X_test[i]) ** 2) / (2 * tao ** 2))
            RX_Pseudo = np.linalg.pinv(X_train_new.T.dot(np.diag(r)).dot(X_train_new))
            w = RX_Pseudo.dot(X_train_new.T).dot(np.diag(r)).dot(y_train)
            predict_labels.append(np.dot(X_test_new[i], w))
        plt.plot(X_test, y_test, ".", label="Oringinal Data")
        plt.plot(X_test, predict_labels, ".", label="Predicted Data")
        test_error = abs(predict_labels - y_test)
        MSE = np.mean(test_error ** 2)
        print("MSE =",MSE)
        plt.legend()
        plt.show()

main()




