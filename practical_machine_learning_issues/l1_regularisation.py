import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiple_linear_and_polynomial_regression.polynomial_LR import polynomial_linear_regression
from one_dimensional_linear_regression.one_dimensional_linear_regression_solution import r2_calculator

sns.set()

def gradient_descent(X, y, number_data_points, dimensionality, learning_rate = 0.001, l1 = 10.0):
    costs = []
    w = np.random.randn(dimensionality) / np.sqrt(dimensionality)
    for _ in range(500):
        y_hat = X.dot(w)
        delta = y_hat - y
        w = w - learning_rate * ( X.T.dot(delta) + l1 * np.sign(w) )
        
        mse = delta.dot(delta) / number_data_points
        costs.append(mse)

    plt.plot(costs)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.show()
    print("Final w:", w)

    return w

if __name__ == '__main__':
    
    number_data_points = 50
    dimensionality = 50

    X = (np.random.random((number_data_points,dimensionality)) - 0.5) * 10 # uniformly distributed numbers between -5, +5

    true_w = np.array([1, 0.5, -0.5] + [0]*(dimensionality - 3)) # true weights - only the first 3 dimensions of X affect Y

    y = X.dot(true_w) + np.random.randn(number_data_points) * 0.5

    predicted_w = gradient_descent(X, y, number_data_points, dimensionality, learning_rate = 0.001, l1 = 10.0)

    plt.figure()
    plt.plot(true_w, label='True Weights')
    plt.plot(predicted_w, label='Predicted Weights')
    plt.legend()
    plt.show()