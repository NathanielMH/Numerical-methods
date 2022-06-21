from scipy.linalg import lu, qr, svd, norm
from functions import *

A = plt.imread("github.png")

# Image matrix in black and white.
A_bw = rgb2gray(A)
plt.imshow(A_bw, cmap='gray')
P, L, U = lu(A_bw)
S, V, D = svd(A_bw)
Q, R = qr(A_bw)
k = int(input())
s = np.shape(A_bw)
m = s[0]
n = s[1]

# Decomposition of the image.
ALU = P @ L[:, :k] @ U[:k, :]
AQR = Q[:, :k] @ R[:k, :]
ASVD = S[:, :k] @ np.diag(V[:k]) @ D[:k, :]

eSVD = V[k] / norm(A_bw, 2)

# Factor of compression in terms of amount of columns wanted k.
k_compr = (k * m + k * n + k) / (m * n) * 100


def print_result(A_bw, V):
    print("Compression needed for 25% reduction in size is", compressio(A_bw, 25), "with relative error",
          V[compressio(A_bw, 25)] / norm(A_bw, 2))
    print("Compression needed for 50% reduction in size is", compressio(A_bw, 50), "with relative error",
          V[compressio(A_bw, 50)] / norm(A_bw, 2))
    print("Compression needed for 75% reduction in size is", compressio(A_bw, 75), "with relative error",
          V[compressio(A_bw, 75)] / norm(A_bw, 2))


print_result(A_bw, V)
