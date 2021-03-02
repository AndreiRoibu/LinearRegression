# Date description:

# Systolic Blood Pressure Data

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

# Data Source: 
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from multiple_linear_and_polynomial_regression.polynomial_LR import polynomial_linear_regression
from one_dimensional_linear_regression.one_dimensional_linear_regression_solution import r2_calculator

if __name__ == '__main__':

    data_frame = pd.read_excel('../large_files/mlr02.xlsx')
    data_values = data_frame.values

    plt.figure()
    plt.scatter(data_values[:,2], data_values[:,0])
    plt.xlabel('Weight (lbs)')
    plt.ylabel('Systolic Blood Pressure')
    plt.show()

    plt.figure()
    plt.scatter(data_values[:,1], data_values[:,0])
    plt.xlabel('Age (yers)')
    plt.ylabel('Systolic Blood Pressure')
    plt.show()

    data_frame['ones'] = 1
    X = data_frame[['X2', 'X3', 'ones']]
    y = data_frame['X1']
    X2_only = data_frame[['X2', 'ones']]
    X3_only = data_frame[['X3', 'ones']]

    print("When considering both the Age and Weight:")

    _, _ = polynomial_linear_regression(X,y)

    print("When considering only the Age:")

    _, _ = polynomial_linear_regression(X2_only,y)

    print("When considering only the Weight:")

    _, _ = polynomial_linear_regression(X3_only,y)