import matplotlib.pyplot as plt
from scipy.linalg import lu, qr, svd, norm
import numpy as np


def show_img(A_bw, ALU, AQR, ASVD, k):
    fig = plt.figure()
    fig.add_subplot(2, 2, 1)
    plt.imshow(A_bw, cmap='gray')
    plt.axis('off')
    plt.title('Original image')
    fig.add_subplot(2, 2, 2)
    plt.imshow(ALU, cmap='gray')
    plt.axis('off')
    plt.title('LU using %d columns' % (k + 1));
    fig.add_subplot(2, 2, 3)
    plt.imshow(AQR, cmap='gray')
    plt.axis('off')
    plt.title('QR using %d columns.' % (k + 1));
    fig.add_subplot(2, 2, 4)
    plt.imshow(ASVD, cmap='gray')
    plt.axis('off')
    plt.title('SVD using %d columns.' % (k + 1));
    plt.tight_layout()
    plt.show()


def rgb2gray(A_rgb):
    return np.dot(A_rgb[..., :3], [0.2989, 0.5870, 0.1140])


def compressio(A_bw, perc):
    m = np.shape(A_bw)[0]
    n = np.shape(A_bw)[1]
    return int(m * n * perc / ((m + n + 1) * 100))


def SVD_compr(A, c):
    S, V, D = svd(A)
    ASVD = S[:, :c] @ np.diag(V[:c]) @ D[:c, :]
    error = V[c] / norm(A, 2)
    return ASVD, error


def stack_colors(A_r, A_g, A_b):
    n = np.shape(A_r)[0]
    m = np.shape(A_r)[1]
    A_compr = [[] for i in range(np.shape(A_r)[0])]
    for i in range(n):
        for j in range(m):
            A_compr[i].append([int(A_r[i][j]), int(A_g[i][j]), int(A_b[i][j])])
    return A_compr


def error_total(e_r, e_g, e_b):
    return np.sqrt(e_b ** 2 + e_r ** 2 + e_g ** 2)


def build_colors(A):
    A_r = A[:, :, 0]
    A_g = A[:, :, 1]
    A_b = A[:, :, 2]
    return A_r, A_g, A_b
