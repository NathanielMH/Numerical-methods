import numpy as np


# Used to solve diagonal matrix systems.
def diagonal_matrix(A, b):
    n = np.shape(b)[0]
    x = np.zeros((n, 1))
    for i in range(n):
        x[i] = b[i] / A[i, i]
    return x


# Used to solve lower triangulat matrix systems.
def down_sub(A, b):
    n = len(A)
    x = np.zeros((n, 1))
    x[0] = b[0] / A[0, 0]
    for i in range(1, n):
        s = 0
        s += A[i, 0:i] @ x[0:i]
        x[i] = (b[i] - s) / A[i, i]
    return x


# A is SPD
def Cholesky_factorization(A):
    n = len(A)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                val = A[i, i] - np.sum(np.square(L[i, :i]))
                if val < 0:
                    return 0.0
                L[i, i] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    return L


# A is PD
def Gen_Cholesky_factorization(A):
    n = np.shape(A)[0]
    L = np.eye(n)
    D = np.eye(n)
    D[0, 0] = A[0, 0]
    for k in range(1, n):
        L[:k, k] = diagonal_matrix(D[:k, :k], down_sub(L[:k, :k], A[k - 1, :k]))
        D[k, k] = A[k, k]
    return L, D
