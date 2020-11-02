import numpy as np

# Sigmoid function
#
def sigmoid(x):
    return 1/(1+np.exp(-x))

if __name__ == "__main__":
    #(2,0)
    X = np.array([1.0,0.5])
    #(3,2)
    W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    #(3,0)
    B1 = np.array([0.1,0.2,0.3])
    #(3,0)
    A1 = np.dot(X,W1)+B1
    Z1 = sigmoid(A1)
    print(A1)
    print(Z1)

    pass