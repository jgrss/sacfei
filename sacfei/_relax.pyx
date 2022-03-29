# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cython
cimport cython

import numpy as np
cimport numpy as np

DTYPE_float32 = np.float32
ctypedef np.float32_t DTYPE_float32_t


cdef DTYPE_float32_t _r(unsigned int a, unsigned int b, DTYPE_float32_t[:, ::1] C) nogil:

    """
    Defines the compatibility coefficients
    """

    return C[a, b]


cdef DTYPE_float32_t _q_s(DTYPE_float32_t[:, ::1] A, Py_ssize_t i_n, Py_ssize_t j_n, unsigned int label_i, DTYPE_float32_t[:, ::1] C) nogil:

    """
    Defines the small q for a single iteration
    """

    return _r(label_i, 1, C) * A[i_n, j_n] + _r(label_i, 0, C) * (1.0 - A[i_n, j_n])


cdef DTYPE_float32_t _compatibility(Py_ssize_t i, Py_ssize_t j, Py_ssize_t i_n, Py_ssize_t j_n) nogil:

    """
    Defines the compatibility function
    """

    cdef:
        DTYPE_float32_t distance = ((i - i_n)**2.0 + (j - j_n)**2.0)**0.5

    if distance > 0:
        return 1.0
    else:
        return 0.0


cdef DTYPE_float32_t _Q_s(DTYPE_float32_t[:, ::1] A, Py_ssize_t i, Py_ssize_t j, unsigned int label_i, DTYPE_float32_t[:, ::1] C, unsigned int shape) nogil:

    """
    Calculate the big Q for a single point
    """

    cdef:
        Py_ssize_t i_n, j_n
        DTYPE_float32_t Q = 0.0
        unsigned int x = A.shape[0]
        unsigned int y = A.shape[1]
        int center = <int>(shape / 2.0)

    for i_n in range(i-center, i+center+1):

        for j_n in range(j-center, j+center+1):

            if (i_n >= 0) and (j_n >= 0) and (i_n < x) and (j_n < y):
                Q += _compatibility(i, j, i_n, j_n) * _q_s(A, i_n, j_n, label_i, C)

    return Q


cdef DTYPE_float32_t _P_s_next(DTYPE_float32_t[:, ::1] A, Py_ssize_t i, Py_ssize_t j, DTYPE_float32_t[:, ::1] C, unsigned int shape) nogil:

    """
    Calculate the next probability at position i,j
    """

    cdef:
        DTYPE_float32_t temp = _Q_s(A, i, j, 1, C, shape) * A[i, j]

    return temp / ((temp + (1.0 - A[i, j]) * _Q_s(A, i, j, 0, C, shape)) + 0.000000001)


cdef void _P(DTYPE_float32_t[:, ::1] a_array, DTYPE_float32_t[:, ::1] c_array, DTYPE_float32_t[:, ::1] new_p, int shape) nogil:

    cdef:
        Py_ssize_t i, j
        unsigned int w = a_array.shape[0]
        unsigned int h = a_array.shape[1]

    new_p[...] = a_array

    for i in range(0, w):

        for j in range(0, h):
            a_array[i, j] = _P_s_next(new_p, i, j, c_array, shape)


def relax(np.ndarray[DTYPE_float32_t, ndim=2] A, np.ndarray[DTYPE_float32_t, ndim=2] C, int iterations=3, int shape=3):

    """
    Source:
        https://github.com/martinferianc/relax
    """

    cdef:
        Py_ssize_t i
        DTYPE_float32_t[:, ::1] new_p = np.empty((A.shape[0], A.shape[1]), dtype='float32')

    with nogil:

        for i in range(iterations):
            _P(A, C, new_p, shape)

    return np.float32(A)
