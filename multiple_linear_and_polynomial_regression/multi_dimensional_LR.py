import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from one_dimensional_linear_regression.one_dimensional_linear_regression_solution import r2_calculator

def load_csv_data(data_file):
    X = []
    Y = []
    for line in open(data_file):
        x1, x2, y = line.split(',')
        bias = 1.0
        X.append([float(x1), float(x2), bias]) 
        Y.append(float(y))

    return np.array(X), np.array(Y)

def multi_dimensonal_linear_regression(X, y):

    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
    y_hat = np.dot(X, w)

    _ = r2_calculator(y, y_hat)
    
    return w, y_hat

if __name__ == '__main__':

    data_file = '../large_files/data_2d.csv'
    X, y = load_csv_data(data_file)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], y)
    plt.show()

    _, _ = multi_dimensonal_linear_regression(X, y)