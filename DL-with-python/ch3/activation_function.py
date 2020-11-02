import numpy as np
import matplotlib.pyplot as plt

# Step function-v1
#
# limitation : doesn't work if a user put an array input
# def step_function(x):
#     if x>0:
#         return 1
#     else:
#         return 0

# Step function-v2
#
def step_function(x):
    y = x>0
    return y.astype(np.int)

# Sigmoid function
#
def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

if __name__ == "__main__":
    # step function test cases
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = sigmoid(x)
    y2 = step_function(x)
    y3 = relu(x)
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x,y3)
    plt.savefig("compare_activation_functions.png")

    pass