import numpy as np


def LU_factorization(A):
    n = len(A)
    L = np.eye(n)
    for k in range(n - 1):
        for i in range(k + 1, n):
            alpha = A[i, k] / A[k, k]
            L[i, k] = alpha
            for j in range(k + 1, n):
                A[i, j] = A[i, j] - alpha * A[k, j]
    return L, A
