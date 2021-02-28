import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from one_dimensional_linear_regression.one_dimensional_linear_regression_solution import r2_calculator

sns.set()

def load_polynomial_csv_data(data_file):
    X = []
    Y = []
    for line in open(data_file):
        x, y = line.split(',') 
        x = float(x)
        X.append([1, x, np.power(x,2)]) 
        Y.append(float(y))

    return np.array(X), np.array(Y)

def polynomial_linear_regression(X, y):

    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
    y_hat = np.dot(X, w)

    _ = r2_calculator(y, y_hat)
    
    return w, y_hat

if __name__ == '__main__':

    data_file = '../large_files/data_poly.csv'
    X, y = load_polynomial_csv_data(data_file)

    fig = plt.figure()
    plt.scatter(X[:,1], y)
    plt.show()

    w, y_hat = polynomial_linear_regression(X,y)

    fig = plt.figure()
    x_line_predicted = np.linspace(X[:,1].min(), X[:,1].max())
    y_line_predicted = w[0] + w[1] * x_line_predicted + w[2] * np.power(x_line_predicted, 2)
    plt.scatter(X[:,1], y)
    plt.plot(x_line_predicted, y_line_predicted, 'r')
    plt.title("Fitted quadratic")
    plt.show()