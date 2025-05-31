import numpy as np

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Identity activation function
def identity(x):
    return x

# Sample input (Cabin features): size, rooms, sauna, lake_dist, neighbor_dist
x = np.array([74, 5, 10, 2, 100])

# Sample pre-defined weights and biases
w0 = np.random.rand(5, 2)
b1 = np.random.rand(2)

w1 = np.random.rand(2, 2)
b2 = np.random.rand(2)

w2 = np.random.rand(2)
# No bias in output layer for simplicity

# Forward pass
z1 = np.dot(x, w0) + b1
a1 = relu(z1)

z2 = np.dot(a1, w1) + b2
a2 = relu(z2)

output = identity(np.dot(a2, w2))

print("Predicted price (demo output):", output)
