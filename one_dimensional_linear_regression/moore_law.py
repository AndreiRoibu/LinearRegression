import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from one_dimensional_linear_regression_solution import LR_1D_calculator

sns.set()

def read_file(file_name):
    X = []
    Y = []
    non_decimal = re.compile(r'[^\d]+')
    for line in open(file_name):
        read_line = line.split('\t')
        x = int(non_decimal.sub('', read_line[2].split('[')[0]))
        y = int(non_decimal.sub('', read_line[1].split('[')[0]))
        X.append(x)
        Y.append(y)

    return np.array(X), np.array(Y)

if __name__ == '__main__':

    file_name = '../large_files/moore.csv'
    X, y = read_file(file_name)
    
    plt.figure()
    plt.scatter(X,y)
    plt.xlabel('Year')
    plt.ylabel('Number of Transistors')
    plt.show()

    y_log = np.log(y)
    plt.figure()
    plt.scatter(X,y_log)
    plt.xlabel('Year')
    plt.ylabel('Number of Transistors')
    plt.show()

    Y_hat, a, b, R2 = LR_1D_calculator(X, y_log, xlabel='Year', ylabel='Number of Transistors')

    print("Time to double:", np.log(2)/a, "years")
