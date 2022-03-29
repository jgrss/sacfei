from __future__ import division

import numpy as np
import numba

numba.config.THREADING_LAYER = 'omp'


@numba.jit(nopython=True, nogil=True)
def r(a, b, C):

    """
    Defines the compatibility coefficients
    """

    return C[a, b]


@numba.jit(nopython=True, nogil=True)
def q_s(A, i_n, j_n, label_i, C):

    """
    Defines the small q for a single iteration
    """

    return r(label_i, 1, C) * A[i_n, j_n] + r(label_i, 0, C) * (1.0 - A[i_n, j_n])


@numba.jit(nopython=True, nogil=True)
def compatibility(i, j, i_n, j_n):

    """
    Defines the compatibility function
    """

    distance = ((i - i_n)**2.0 + (j - j_n)**2.0)**0.5

    return 1 if distance > 0 else 0
    # return distance > 0


@numba.jit(nopython=True, nogil=True)
def Q_s(A, i, j, label_i, C, shape):

    """
    Calculate the big Q for a single point
    """

    Q = 0
    x, y = A.shape
    center = int(shape / 2.0)

    for i_n in range(i-center, i+center+1):

        for j_n in range(j-center, j+center+1):

            if (i_n >= 0) and (j_n >= 0) and (i_n < x) and (j_n < y):
                Q += compatibility(i, j, i_n, j_n) * q_s(A, i_n, j_n, label_i, C)

    return float(Q)


@numba.jit(nopython=True, nogil=True)
def P_s_next(A, i, j, C, shape):

    """
    Calculate the next probability at position i,j
    """

    temp = Q_s(A, i, j, 1, C, shape) * A[i, j]

    return temp / ((temp + (1.0 - A[i, j]) * Q_s(A, i, j, 0, C, shape)) + 0.000000001)


@numba.jit(nopython=True, nogil=True)
def P_iter(A, w, h, C, new_P, shape):

    for i in range(0, w):

        for j in range(0, h):
            A[i, j] = P_s_next(new_P, i, j, C, shape)

    return A


@numba.jit
def P(A, C, w, h, shape):

    new_P = A.copy()

    return P_iter(A, w, h, C, new_P, shape)


@numba.jit
def relax(A, C, iterations=3, shape=3):

    """
    Source:
        https://github.com/martinferianc/relax
    """

    A = np.float32(A)
    C = np.float32(C)

    w, h = A.shape

    for i in range(iterations):
        A = P(A, C, w, h, shape)

    return np.float32(A)
