import numpy as np
import matplotlib.pyplot as plt
from numpy.random import sample
import seaborn as sns

from multiple_linear_and_polynomial_regression.polynomial_LR import polynomial_linear_regression

sns.set()

def make_polynomial(X, degrees):
    number_of_inputs = len(X)
    polynomial_data = [np.ones(number_of_inputs)]
    for degree in range(degrees):
        polynomial_data.append(np.power(X, degree+1))

    return np.vstack(polynomial_data).T

def fit_and_display(X, y, sample_number, degrees):
    number_of_inputs = len(X)
    train_samples_indeces = np.random.choice(number_of_inputs, sample_number)
    X_train = X[train_samples_indeces]
    y_train = y[train_samples_indeces]

    plt.figure()
    plt.scatter(X_train, y_train)
    plt.show()

    X_train_polynomial = make_polynomial(X_train, degrees)
    w, _ = polynomial_linear_regression(X_train_polynomial, y_train)

    X_polynomial = make_polynomial(X, degrees)
    y_hat = X_polynomial.dot(w)

    plt.figure()
    plt.plot(X,y, label='original data')
    plt.plot(X,y_hat, label='predicted data')
    plt.plot(X_train, y_train, 'r*', label='train data')
    plt.title("Degree = %d" %degrees)
    plt.legend()
    plt.show()

def mse(x, y):
    d = x - y
    return d.dot(d) / len(d)

def plot_train_vs_test_curves(X, y, sample_number=20, max_degrees=20):
    number_of_inputs = len(X)
    train_samples_indeces = np.random.choice(number_of_inputs, sample_number)
    X_train = X[train_samples_indeces]
    y_train = y[train_samples_indeces]

    test_indeces = [index for index in range(number_of_inputs) if index not in train_samples_indeces]
    X_test = X[test_indeces]
    y_test = y[test_indeces]

    MSEs_train = []
    MSEs_test = []

    for degree in range(max_degrees+1):
        X_train_polynomial = make_polynomial(X_train, degree)
        w, y_hat_train = polynomial_linear_regression(X_train_polynomial, y_train)
        mse_train = mse(y_train, y_hat_train)

        MSEs_train.append(mse_train)

        X_test_polynomial = make_polynomial(X_test, degree)
        y_hat_test = X_test_polynomial.dot(w)
        mse_test = mse(y_test, y_hat_test)

        MSEs_test.append(mse_test)

    plt.figure()
    plt.plot(MSEs_train, label='Train MSEs')
    plt.plot(MSEs_test, label='Test MSEs')
    plt.ylabel('MSEs')
    plt.xlabel('Polynomial Degrees')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    number_of_points = 100
    X = np.linspace(0, 6 * np.pi, number_of_points)
    y = np.sin(X)

    plt.figure()
    plt.plot(X,y)
    plt.show()

    for degrees in (5, 6, 7, 8, 9):
        fit_and_display(X, y, sample_number=10, degrees=degrees)

    plot_train_vs_test_curves(X, y)

