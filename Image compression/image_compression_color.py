import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu, qr, svd, norm
from functions import *

A = plt.imread("github.png")

# Decompose the colored image into its rgb components and apply previous compression algorithm.
A_r,A_g,A_b=build_colors(A)
A_compr=stack_colors(SVD_compr(A_r, compressio(A_r,50))[0],SVD_compr(A_g,compressio(A_g,50))[0],SVD_compr(A_b, compressio(A_b,50))[0])
plt.imshow(A_compr)

error=error_total(SVD_compr(A_r, compressio(A_r,50))[1], SVD_compr(A_g, compressio(A_g,50))[1], SVD_compr(A_b, compressio(A_b,50))[1])
