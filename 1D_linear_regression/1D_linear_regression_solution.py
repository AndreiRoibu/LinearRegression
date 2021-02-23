import numpy as np
import matplotlib.pyplot as plt

# We first load and visualise the data:

X = []
Y = []
for line in open('../large_files/data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

plt.figure()
plt.scatter(X,Y)
plt.show()

# We now solve the equation y = aX + b

X_mean = X.mean()
X_sum = X.sum()
Y_mean = Y.mean()
Y_sum = Y.sum()

denominator = X.dot(X) - X_mean * X_sum
a = ( X.dot(Y) - Y_mean * X_sum) / denominator
b = ( Y_mean * X.dot(X) - X_mean * X.dot(Y)) / denominator

Y_hat = a * X + b

plt.figure()
plt.scatter(X,Y)
plt.plot(X, Y_hat, 'r')
plt.show()

# We also calculate the R^2 error:

d1 = Y - Y_hat
d2 = Y - Y_mean
R2 = 1 - d1.dot(d1) / d2.dot(d2)
print("The R^2 is: ", R2)