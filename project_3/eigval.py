import numpy as np
import matplotlib.pyplot as plt

def exact_solution(A):
    """Calculates the eigenvalues of a square-matrix A
        by np.linalg.eigvals functionality.

    args:
        A           (np.array): Square matrix, i.e NxN

    returns:
        eigvals     (np.array): Eigenvalues of the matrix A
    """

    eigvals = np.linalg.eigvals(A)
    return eigvals


if __name__ == "__main__":
    ## The square, real, symmetric, 6x6 matrix we use for our calculations
    matrix = np.array([
    [5, 2, 3, 2, 6, 9],
    [2, 9, 8, 1, 2, 1],
    [3, 8, 7, 5, 0, 4],
    [2, 1, 5, 1, 8, 3],
    [6, 2, 0, 8, 7, 0],
    [9, 1, 4, 3, 0, 9]]
    )
    eigs = exact_solution(matrix)
    breakpoint()
