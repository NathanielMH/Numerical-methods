import numpy as np
from scipy.linalg import norm


def e_1(m):
    x = np.zeros((m, 1))
    x[0] = 1
    return x


def sgn(x):
    return int(x / abs(x))


def Householder_matrix(A, i):
    m = np.shape(A)[0]
    x = A[i - 1:m, i - 1]
    x.shape = (m - i + 1, 1)
    v = x + sgn(x[0]) * norm(x, 2) * e_1(m - i + 1)
    P = np.eye(m - i + 1) - 2 * v @ np.transpose(v) / norm(v, 2) ** 2
    H = np.block([[np.eye(i - 1), np.zeros((i - 1, m - i + 1))], [np.zeros((m - i + 1, i - 1)), P]])
    return H


def mult_Q(Q, v):  # v is already unit vector.
    k = np.shape(v)[0]
    m = np.shape(Q)[0]
    Q[m - k:, :m - k] = Q[m - k:, :m - k] - 2 * v @ (np.transpose(v) @ Q[m - k:, :m - k])
    Q[m - k:, m - k:] = Q[m - k:, m - k:] - 2 * v @ (np.transpose(v) @ Q[m - k:, m - k:])
    return Q


# returns QR decomposition using Householder matrices.
def QR_Householder(A, b):
    m, n = np.shape(A)
    Q = np.eye(m)
    for i in range(n):
        x = A[i:, i]
        x.shape = (m - i, 1)
        v = x + sgn(x[0]) * norm(x, 2) * e_1(m - i)
        v = v / norm(v, 2)
        A[i:, i:] = A[i:, i:] - 2 * v @ (np.transpose(v) @ A[i:, i:])
        b[i:] = b[i:] - 2 * v @ (np.transpose(v) @ b[i:])
        Q = mult_Q(Q, v)
    return np.transpose(Q), A, b


def QR_solve(A, b):
    n = A.shape[1]
    A_0 = A.copy()
    b_0 = b.copy()
    Q, R, f = QR_Householder(A_0, b_0)
    x = np.solve(R[:n, :n], f[:n])
    return x
