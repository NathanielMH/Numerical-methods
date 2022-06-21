import numpy as np
from scipy.linalg import norm
import numpy.linalg as npl
import sys

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


# A is symmetric and def pos
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
