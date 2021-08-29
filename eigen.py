import numpy as np


def get_M(A, k):
    N = A.shape[0]
    M = np.identity(N)
    for i in range(N):
        if i != N - k - 2:
            M[N - k - 2][i] = - A[N - k - 1][i] / A[N - k - 1][N - k - 2]
        else:
            M[i][i] = 1 / A[N - k - 1][N - k - 2]

    return M


def get_M_inv(A, k):
    N = A.shape[0]
    M_inv = np.identity(N)

    for i in range(N):
        M_inv[N - k - 2][i] = A[N - k - 1][i]

    return M_inv


def get_frobenius(A, get_S=False):
    N = A.shape[0]
    S = np.identity(N)
    for k in range(N - 1):
        M = get_M(A, k)
        S = np.matmul(S, M)
        M_inv = get_M_inv(A, k)
        A = np.matmul(np.matmul(M_inv, A), M)

    if get_S:
        return A, S
    else:
        return A


def get_eigenvalues(A, get_S=False):
    A, S = get_frobenius(A, get_S=True)
    coef = np.insert(A[0], 0, -1, axis=0) * -1
    p = np.poly1d(coef)
    if get_S:
        return p.r, S
    else:
        return p.r


def get_eigenvectors(A):
    eigenvalues, S = get_eigenvalues(A, get_S=True)
    N = eigenvalues.size
    eigenvectors = []

    for eigenvalue in eigenvalues:
        y_vector = np.array([1.0])
        for i in range(1, N):
            y_vector = np.insert(y_vector, 0, eigenvalue ** (i), axis=0)

        eigenvectors.append(np.dot(S, y_vector))

    return eigenvectors
