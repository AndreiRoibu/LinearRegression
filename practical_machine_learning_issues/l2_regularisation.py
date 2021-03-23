import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiple_linear_and_polynomial_regression.polynomial_LR import polynomial_linear_regression
from one_dimensional_linear_regression.one_dimensional_linear_regression_solution import r2_calculator


sns.set()

def max_likelyhood_regularisation(X, y):
    X = np.vstack([np.ones(len(X)), X]).T # Adding a bias term
    _, y_hat_maximum_likelyhood = polynomial_linear_regression(X, y)

    return y_hat_maximum_likelyhood

def l2_regularisation(X, y, l2_regulariser=1000):
    X = np.vstack([np.ones(len(X)), X]).T
    w_l2 = np.linalg.solve(l2_regulariser * np.eye(2) + X.T.dot(X), X.T.dot(y))
    y_hat_l2 = X.dot(w_l2)

    _ = r2_calculator(y, y_hat_l2)

    return w_l2, y_hat_l2

if __name__ == '__main__':
    
    number_of_points = 50
    X = np.linspace(0,10,number_of_points)
    y = 0.5 * X + np.random.randn(number_of_points)
    y[-1] += 30
    y[-2] += 30

    plt.figure()
    plt.scatter(X,y)
    plt.show()

    y_hat_maximum_likelyhood = max_likelyhood_regularisation(X, y)
    
    plt.figure()
    plt.scatter(X, y, label='data')
    plt.plot(X, y_hat_maximum_likelyhood, 'r-', label='max likelyhood linear regression')
    plt.legend()
    plt.show()

    _, y_hat_l2 = l2_regularisation(X, y, l2_regulariser=1000)

    plt.figure()
    plt.scatter(X, y, label='data')
    plt.plot(X, y_hat_maximum_likelyhood, 'r-', label='max likelyhood linear regression')
    plt.plot(X, y_hat_l2, 'g-', label='l2 regularised mapping')
    plt.legend()
    plt.show()