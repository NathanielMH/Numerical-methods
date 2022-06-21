import numpy as np


# Diagonal matrix.
def diagonal_matrix(A, b):
    n = np.shape(b)[0]
    x = np.zeros((n, 1))
    for i in range(n):
        x[i] = b[i] / A[i, i]
    return x


# Upper triangular matrix.
def up_sub(A, b):
    n = len(A)
    x = np.zeros((n, 1))
    x[n - 1] = b[n - 1] / A[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        s = 0
        for j in range(i + 1, n):
            s = s + A[i, j] * x[j]
        x[i] = (b[i] - s) / A[i, i]
    return x


# Lower triangular matrix.
def down_sub(A, b):
    n = len(A)
    x = np.zeros((n, 1))
    x[0] = b[0] / A[0, 0]
    for i in range(1, n):
        s = 0
        s += A[i, 0:i] @ x[0:i]
        x[i] = (b[i] - s) / A[i, i]
    return x


# A is non singular and det of dominant minors is different than 0.
def Gaussian_elimination(A, b):
    n = len(A)
    if b.size != n:
        raise ValueError("Error: dimensions of A and b don't coincide", b.size, n)
    for k in range(n - 1):
        for i in range(k + 1, n):
            alpha = A[i, k] / A[k, k]
            A[i, k] = 0
            for j in range(k + 1, n):
                A[i, j] = A[i, j] - alpha * A[k, j]
            b[i] = b[i] - alpha * b[k]
    x = up_sub(A, b)
    return x
