import numpy as np
from scipy.linalg import norm
import sys


# Gradient descent algorithm to solve an algorithm Ax=b.
# Pre: A is SPD, A is a 2D array (Matrix) of size n*n and b is an array of size n.
def gd(A, b):
    n = A.shape[0]
    x = np.ones(b.shape)
    y = np.zeros(b.shape)
    it = 0
    while norm(x - y) / norm(x) > 10 ** (-10) and it < 10000:
        sys.stdout.write("\rIterations: %03d" % it)
        sys.stdout.flush()
        r = A @ x - b
        alpha = - norm(r) / (np.transpose(r) @ A @ r)
        if it % n == 0:
            y = x
        x = x + alpha * r
        it += 1
    return x


# Conjugate gradient descent algorithm to solve a linear system Ax=b.
# Pre: A is SPD, A is a 2D array (Matrix) of size n*n and b is an array of size n.
def cgd(A, b):
    n = A.shape[0]
    x = np.ones(b.shape)
    y = np.zeros(b.shape)
    p = r = A @ x - b
    it = 0
    while norm(x - y) / norm(x) > 10 ** (-10) and it < n + 1:
        sys.stdout.write("\rIterations: %03d" % it)
        sys.stdout.flush()
        q = A @ p
        alpha = - np.dot(p, r) / np.dot(p, q)
        y = x
        x = x + alpha * p
        r = A @ x - b
        beta = - np.dot(r, A @ x) / np.dot(p, q)
        p = r + beta * p
        it += 1
    return x
