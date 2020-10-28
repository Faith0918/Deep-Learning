import numpy as np

if __name__ == "__main__":
    # matrix multiplication
    #
    A = np.array([[1,2],[3,4]])
    print(A.shape)
    B = np.array([[5,6],[7,8]])
    print(B.shape)
    print(np.dot(A,B))
    pass