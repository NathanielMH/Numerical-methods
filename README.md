# Numerical-methods
These are a few algorithms regarding numerical methods, mostly regarding linear algebra problems.

## Image compression
In this mini project we explore image compression with SVD decomposition, varying the number of columns taken and studying the respective error with the original error.
To do this with colored images, we first decompose them into RGB components and apply the decomposition to each. (Colored pictures are stored as tensors aka 3D arrays)

## Iterative methods
Includes direct vectorial iterative methods, namely IVI and IVD to find the smallest and largest eigenvalue with absolute value. We also have an algorithm that returns the definition (PD, PN, Undef) of a symmetric matrix.
We also have the implementation of gradient descent and conjugate gradient descent to solve linear systems, as well as some notes on time efficiency.

## Decompositions and linear system solver
Includes implementations of methods to solve determinate linear systems, namely Gauss, Diagonal matrices and lower and upper triangular matrix.
Also includes Cholesky decomposition, general cholesky decomposition and LU decomposition. 
