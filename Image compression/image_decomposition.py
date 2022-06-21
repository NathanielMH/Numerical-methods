from scipy.linalg import lu, qr, svd, norm
from functions import *

A = plt.imread("github.png")

# Finding black and white image version of .jpg
A_bw = rgb2gray(A)
plt.imshow(A_bw, cmap='gray')

# Decomposition of the matrix of the image in black and white by different methods.
P, L, U = lu(A_bw)
S, V, D = svd(A_bw)
Q, R = qr(A_bw)

s = np.shape(A_bw)
n = s[0]
m = s[1]

print("Choose an integer that x belongs to [ 1 ,", min(n, m), ")")
k = int(input())

# Matrix given by decomposition for each method.
ALU = P @ L[:, :k] @ U[:k, :]
AQR = Q[:, :k] @ R[:k, :]
ASVD = S[:, :k] @ np.diag(V[:k]) @ D[:k, :]

# Computing errors for each method.
eLU = norm(A_bw - ALU, 2) / norm(A_bw, 2)
eQR = norm(A_bw - AQR, 2) / norm(A_bw, 2)
eSVD = norm(A_bw - ASVD, 2) / norm(A_bw, 2)

show_img(A_bw, ALU, AQR, ASVD, k)

# Plot of errors in terms of the amount of columns taken by our SVD decomposition.
x = np.arange(1, min(n, m), 50)
y_1 = []
y_2 = []
y_3 = []
for i in range(1, min(n, m), 50):
    ALU = P @ L[:, :i] @ U[:i, :]
    AQR = Q[:, :i] @ R[:i, :]
    ASVD = S[:, :i] @ np.diag(V[:i]) @ D[:i, :]
    eSVD_2 = V[i] / norm(A_bw, 2)
    eLU = norm(A_bw - ALU, 2) / norm(A_bw, 2)
    eQR = norm(A_bw - AQR, 2) / norm(A_bw, 2)
    y_1 = y_1 + [eSVD_2]
    y_2 = y_2 + [eQR]
    y_3 = y_3 + [eLU]

plt.plot(x, y_1)
plt.plot(x, y_2)
plt.plot(x, y_3)
