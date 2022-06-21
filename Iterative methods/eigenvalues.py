import numpy as np
from scipy.linalg import norm
import numpy.linalg as npl
import sys
import time

q = 4  # x.s.
epsilon = 0.5 * 10 ** (-q)


def rayleigh(A, v):
    return (np.transpose(v) @ A @ v) / norm(v) ** 2


def ivd(A):
    e = 1
    n = A.shape[0]
    v = np.ones(n)
    it = 0
    while abs(e) > epsilon:
        w = A @ v / norm(A @ v)
        e = (rayleigh(A, v) - rayleigh(A, w)) / rayleigh(A, w)
        v = w
        it += 1
    return (A @ v)[0] / v[0], it


def ivi(A):
    n = A.shape[0]
    v = np.ones(n) / n
    e = 1
    it = 0
    while abs(e) > epsilon:
        w = npl.solve(A, v)
        w = w / norm(w)
        e = (rayleigh(A, v) - rayleigh(A, w)) / rayleigh(A, w)
        v = w
        it += 1
    return (A @ v)[0] / v[0], it

# A is symmetric
def alg(A):
    n = A.shape[0]
    l = ivd(A)[0]
    d = ivd(A + l * np.eye(n))[0] - l  # vap més gran
    if d < 0:
        return "negative definite"
    s = ivi(A + l * np.eye(n))[0] - l  # vap més petit
    if s > 0:
        return "positive definite"
    return "indefinite"

