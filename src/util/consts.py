import numpy as np


N_STATES = 3

T = 2000

# kronocker product matrices
Q_M = [[1,0], 
       [0,0]]

Q_F = [[0,1], 
       [0,0]]

Q_B = [[0,0], 
       [1,0]]

Q_N = [[0,0], 
       [0,1]]


POLAR_TABLE = np.array([
    [1, 1, -1],  # state_from 0
    [-1, 1, 1],  # state_from 1
    [1, -1, 1]   # state_from 2
])


# activation functions
def linear(x):
    return x

def stable_sigmoid(x):
    """
    Numerically stable sigmoid function.
    """
    x = np.clip(x, -500, 500)  # prevent very large positive/negative values
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )

def tanh(x):
    return np.tanh(x)


ACTIVATION_TABLE = np.array([tanh, tanh, linear])