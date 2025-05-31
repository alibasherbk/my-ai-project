import numpy as np

def relu(x):
    return np.maximum(0, x)

def identity(x):
    return x

x = np.array([74, 5, 10, 2, 100])  # input: size, rooms, sauna, lake_dist, neighbor_dist

w0 = np.random.rand(5, 2)
b1 = np.random.rand(2)

w1 = np.random.rand(2, 2)
b2 = np.random.rand(2)

w2 = np.random.rand(2)

z1 = np.dot(x, w0) + b1
a1 = relu(z1)

z2 = np.dot(a1, w1) + b2
a2 = relu(z2)

out = identity(np.dot(a2, w2))

print("Predicted price:", out)
