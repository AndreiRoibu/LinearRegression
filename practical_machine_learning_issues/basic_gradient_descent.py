import numpy as np

w = 20

print(w)

for i in range(100):
    w = w - 0.1 * 2 * w
    print(w)

