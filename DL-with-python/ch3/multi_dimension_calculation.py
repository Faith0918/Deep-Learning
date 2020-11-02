import numpy as np



if __name__ == "__main__":
    # 2 by 2 matrix multiplication
    A = np.array([[1,2],[3,4]])
    B = np.array([[5,6],[7,8]])
    print(A.shape)
    print(B.shape)
    print(np.dot(A,B))
    
    #Matrix multiplication in neural net
    X = np.array([1,2])
    W = np.array([[1,3,5],[2,4,6]])
    Y = np.dot(X,W)
    print(Y)



    pass