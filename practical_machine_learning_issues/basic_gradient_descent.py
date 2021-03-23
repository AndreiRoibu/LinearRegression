import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

w = 20

print(w)

for i in range(30):
    w = w - 0.1 * 2 * w
    print(w)

N = 10
D = 3
X = np.zeros((N,D))
X[:, 0] = 1.0
X[:5,1] = 1
X[:5,2] = 1
y = np.array([0] * 5 + [1] * 5)
costs = []
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 1e-3

for t in range(1000):
    y_hat = X.dot(w)
    delta = y_hat - y
    w = w - learning_rate * np.dot(X.T, delta)
    mse = np.dot(delta, delta) / N
    costs.append(mse)

plt.figure()
plt.plot(costs)
plt.show()

print(w)

plt.plot(y_hat, label='predictions')
plt.plot(y, label='targets')
plt.legend()
plt.show()