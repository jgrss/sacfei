# cython: language_level=3
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

"""
@author: Jordan Graesser
Date Created: 4/29/2017
"""

import cython
cimport cython

import numpy as np
cimport numpy as np

# from cython.parallel cimport parallel, prange

DTYPE_intp = np.intp
ctypedef np.intp_t DTYPE_intp_t

DTYPE_uint8 = np.uint8
ctypedef np.uint8_t DTYPE_uint8_t

DTYPE_int64 = np.int64
ctypedef np.int64_t DTYPE_int64_t

DTYPE_float32 = np.float32
ctypedef np.float32_t DTYPE_float32_t

DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t


cdef extern from 'numpy/npy_math.h':
    bint npy_isnan(double x) nogil


cdef extern from 'numpy/npy_math.h':
    bint npy_isinf(double x) nogil


cdef extern from 'stdlib.h' nogil:
    double exp(double yv)


cdef inline double _sqrt(double a) nogil:
    return a**0.5


cdef inline double _pow(double a) nogil:
    return a*a


cdef inline double _abs(double a) nogil:
    return a*-1.0 if a < 0 else a


cdef inline double _euclidean_distance(double x1, double y1, double x2, double y2) nogil:
    return _sqrt(_pow(x1 - x2) + _pow(y1 - y2))


cdef inline double _get_max(double a, double b) nogil:
    return a if a > b else b


cdef inline double _get_min(double a, double b) nogil:
    return a if a < b else b


cdef inline double _logistic(double xv, double beta, double gamma) nogil:
    return 1.0 / (1.0 + exp(beta + gamma*xv))


cdef inline double _scale_min_max(double xv, double mno, double mxo, double mni, double mxi) nogil:
    return (((mxo - mno) * (xv - mni)) / (mxi - mni)) + mno


cdef void _create_weights(double[:, ::1] dist_weights_,
                          unsigned int rcs,
                          double hw,
                          bint inverse,
                          double dist_sigma) nogil:

    """
    Creates distance weights
    
    Args:
        dist_weights_ (2d array): The array will be updated in place.
        rcs (int): The window dimension on both sides.
        hw (float): The center window coordinate.
        inverse (bool): Whether to return an inverse distance array.
        dist_sigma (float): The logistic function gamma value.
    """

    cdef:
        Py_ssize_t ri, rj, ri_, rj_
        double d, dw
        double max_dist = 0.0

    for ri in range(0, rcs):

        for rj in range(0, rcs):

            d = _euclidean_distance(hw, hw, float(rj), float(ri))

            # Get the max distance in the window.
            max_dist = _get_max(max_dist, d)

            dist_weights_[ri, rj] = d

    for ri_ in range(0, rcs):

        for rj_ in range(0, rcs):

            if inverse:

                # Get the euclidean distance, in a 0--1 range.
                # Farther distances with be closer to one.
                d = dist_weights_[ri_, rj_] / max_dist

            else:

                # Get the euclidean distance, in a 0--1 range.
                # Farther distances with be closer to zero.
                d = 1.0 - (dist_weights_[ri_, rj_] / max_dist)

            # Scale the data to a -1.0--1.0 range
            #   and get the logistic weight.
            dw = _logistic(_scale_min_max(d, -1.0, 1.0, 0.0, 1.0), 1.0, dist_sigma)

            dist_weights_[ri_, rj_] = dw


cdef double _get_weighted_sum(double[:, ::1] block,
                              double[:, ::1] weights,
                              unsigned int rc) nogil:

    cdef:
        Py_ssize_t bi, bj
        double block_sum = 0.0
        # double weights_sum = 0.0
        double bv, dv, bw

    for bi in range(0, rc):

        for bj in range(0, rc):

            # Get the block value.
            bv = block[bi, bj]

            # Get the inverse to the maximum distance
            #   in order to give closer pixels higher
            #   weights.
            bw = weights[bi, bj]

            # Weight the value by
            #   the inverse distance.
            dv = bv * bw

            block_sum += dv

    return block_sum


cdef double _get_block_std(double[:, ::1] block,
                                    unsigned int rc,
                                    double block_mean) nogil:

    cdef:
        Py_ssize_t bi, bj
        double block_std = 0.

    for bi in range(0, rc):
        for bj in range(0, rc):
            block_std += _pow(block[bi, bj] - block_mean)

    return _sqrt(block_std)


cdef double _get_block_sum(double[:, ::1] block,
                                    unsigned int rc) nogil:

    cdef:
        Py_ssize_t bi, bj
        double block_sum = 0.

    for bi in range(0, rc):
        for bj in range(0, rc):
            block_sum += block[bi, bj]

    return block_sum


cdef double _get_weighted_mean(double[:, ::1] block,
                                        double[:, ::1] weights,
                                        unsigned int rc,
                                        double weights_sum) nogil:

    return _get_weighted_sum(block, weights, rc) / weights_sum


cdef double _get_mean_c(double[:, ::1] block,
                                 unsigned int rc,
                                 int constant) nogil:

    cdef:
        double n_samples = float(rc * rc)

    return (_get_block_sum(block, rc) / n_samples) - float(constant)


cdef void _fill_otsu_hist(double[::1] otsu_hist_,
                          double[:, ::1] im_block,
                          unsigned int rc) nogil:

    cdef:
        Py_ssize_t hist_idx, bi, bj

    # Ensure zeros
    for hist_idx in range(0, 256):
        otsu_hist_[hist_idx] = 0.

    for bi in range(0, rc):
        for bj in range(0, rc):
            otsu_hist_[<int>im_block[bi, bj]] += 1


cdef double _get_otsu_threshold(double[:, ::1] im_block,
                                         unsigned int rc,
                                         double[::1] otsu_hist) nogil:

    cdef:
        Py_ssize_t t
        double sum_b = 0.
        double wb = 0
        double wf
        double mb, mf
        double var_max = 0.
        int the_threshold = 0
        double the_sum = 0.
        double var_between
        int n_samples = rc * rc

    # Fill the histogram
    _fill_otsu_hist(otsu_hist, im_block, rc)

    for t in range(0, 256):
        the_sum += t * otsu_hist[t]

    for t in range(0, 256):

        # Weight background
        wb += otsu_hist[t]

        if wb == 0:
            continue

        # Weight foreground
        wf = n_samples - wb

        if wf == 0:
            break

        sum_b += float(t) * otsu_hist[t]

        # Mean background
        mb = sum_b / wb

        # Mean foreground
        mf = (the_sum - sum_b) / wf

        # Calculate the between-class variance
        var_between = wb * wf * (mb - mf) * (mb - mf)

        # Check if new maximum found
        if var_between > var_max:

            var_max = var_between
            the_threshold = t

    return float(the_threshold)


cdef double _bernson(double[:, ::1] block_array,
                              unsigned int window_size) nogil:

    cdef:
        Py_ssize_t ii, jj
        double block_max = 0.
        double block_min = 1000000.

    for ii in range(0, window_size):

        for jj in range(0, window_size):

            block_max = _get_max(block_max, block_array[ii, jj])
            block_min = _get_min(block_min, block_array[ii, jj])

    if block_max - block_min < 15:
        return 255.
    else:
        return .5 * (block_max + block_min)


cdef double _niblack(double[:, ::1] block_array,
                              unsigned int window_size,
                              double k_param) nogil:

    cdef:
        double block_sum = _get_block_sum(block_array, window_size)
        double block_mean = block_sum / (window_size * window_size)
        double block_std = _get_block_std(block_array, window_size, block_mean)

    return block_mean + k_param * block_std


cdef double _sauvola(double[:, ::1] block_array,
                              unsigned int window_size,
                              double k_param) nogil:

    cdef:
        double block_sum = _get_block_sum(block_array, window_size)
        double block_mean = block_sum / (window_size * window_size)
        double block_std = _get_block_std(block_array, window_size, block_mean)

    return block_mean * (1. + k_param * ((block_std / 128.) - 1.))


cdef void _bradley(double[:, ::1] image_array,
                   DTYPE_uint8_t[:, ::1] out_im,
                   Py_ssize_t i,
                   Py_ssize_t j,
                   unsigned int window_size,
                   unsigned int rows,
                   unsigned int cols,
                   double t) nogil:

    cdef:
        # SxS region
        Py_ssize_t x1 = i - window_size
        Py_ssize_t x2 = i + window_size
        Py_ssize_t y1 = j - window_size
        Py_ssize_t y2 = j + window_size
        Py_ssize_t count
        double im_sum

    if x1 < 0:
        x1 = 0

    if x2 >= cols:
        x2 = cols-1

    if y1 < 0:
        y1 = 0

    if y2 >= rows:
        y2 = rows-1

    count = (y2 - y1) * (x2 - x1)
    im_sum = image_array[y2, x2] - image_array[y1, x2] - image_array[y2, x1] + image_array[y1, x1]

    if (image_array[i, j] * count) < (im_sum * (100. - t) / 100.):
        out_im[i, j] = 0
    else:
        out_im[i, j] = 1


cdef void _weight_egm_by_distance(double[:, ::1] egm_block_,
                                  double[:, ::1] dist_block,
                                  unsigned int window_size,
                                  unsigned int half_window) nogil:

    cdef:
        Py_ssize_t bi, bj
        double current_dist, current_dist_weight

    for bi in range(0, window_size):

        for bj in range(0, window_size):

            current_dist = dist_block[bi, bj]

            if current_dist > 0.0:

                current_dist_weight = 1.0 - (current_dist / 1.0)

                egm_block_[bi, bj] *= current_dist_weight


cdef void _weight_egm_by_angle(double[:, ::1] egm_block_,
                               double[:, ::1] angle_block,
                               unsigned int window_size,
                               unsigned int half_window) nogil:

    cdef:
        Py_ssize_t bi, bj
        double max_angle_diff = 0.0
        double center_angle = angle_block[half_window, half_window]
        double current_angle, current_angle_scaled

    for bi in range(0, window_size):

        for bj in range(0, window_size):

            current_angle = _abs(center_angle - angle_block[bi, bj])

            max_angle_diff = _get_max(max_angle_diff, current_angle)

    for bi in range(0, window_size):

        for bj in range(0, window_size):

            current_angle = 1.0 - (_abs(center_angle - angle_block[bi, bj]) / max_angle_diff)

            # Scale the data because we don't want zeros.
            current_angle_scaled = _scale_min_max(current_angle, 0.5, 1.0, 0.0, 1.0)

            egm_block_[bi, bj] *= current_angle_scaled


cdef np.ndarray[DTYPE_uint8_t, ndim=2] _threshold(double[:, ::1] image_array,
                                                  unsigned int window_size,
                                                  double ignore_thresh,
                                                  double rt,
                                                  int n_jobs,
                                                  str method,
                                                  int constant,
                                                  double bradley_t,
                                                  double k_param,
                                                  bint inverse_dist,
                                                  double dist_sigma,
                                                  double[:, ::1] edge_direction_array,
                                                  double[:, ::1] edge_distance_array):

    cdef:
        Py_ssize_t i, j
        unsigned int rows = image_array.shape[0]
        unsigned int cols = image_array.shape[1]
        unsigned int half_window = <int>(window_size / 2.0)
        unsigned int row_dims = rows - window_size + 1
        unsigned int col_dims = cols - window_size + 1
        double ath = 0.
        DTYPE_uint8_t[:, ::1] out_threshold = np.zeros((rows, cols), dtype='uint8')
        double rt_ = (1. + rt / 100.0)
        double[::1] otsu_hist = np.zeros(256, dtype='float64')
        unsigned int method_int = 1
        double hwf = float(half_window)
        double[:, ::1] dist_weights = np.empty((window_size, window_size), dtype='float64')
        double weights_sum, max_angle_diff
        double[:, ::1] egm_array

        unsigned int edge_direction_array_size = edge_direction_array.shape[0] * edge_direction_array.shape[1]
        unsigned int edge_distance_array_size = edge_distance_array.shape[0] * edge_distance_array.shape[1]

    if method == 'wmean':
        method_int = 1
    elif method == 'mean-c':
        method_int = 2
    elif method == 'otsu':
        method_int = 3
    elif method == 'bradley':
        method_int = 4
    elif method == 'sauvola':
        method_int = 5
    elif method == 'bernson':
        method_int = 6
    elif method == 'niblack':
        method_int = 7

    with nogil:#, parallel(num_threads=n_jobs):

        if method_int == 1:

            _create_weights(dist_weights, window_size, hwf, inverse_dist, dist_sigma)
            weights_sum = _get_block_sum(dist_weights, window_size)

        for i in range(0, row_dims):

            for j in range(0, col_dims):

                # im_b = image_array[i:i+window_size, j:j+window_size]
                # im_bc = im_b[half_window, half_window]

                if ignore_thresh == -999.:

                    if method_int == 1:

                        egm_array = image_array[i:i+window_size,
                                                j:j+window_size]

                        if edge_direction_array_size > 4:

                            # Weight the EGM by difference from edge direction
                            _weight_egm_by_angle(egm_array,
                                                 edge_direction_array[i:i+window_size,
                                                                      j:j+window_size],
                                                 window_size,
                                                 half_window)

                        if edge_distance_array_size > 4:

                            # Weight the EGM by distance from sure edge.
                            _weight_egm_by_distance(egm_array,
                                                    edge_distance_array[i:i+window_size,
                                                                        j:j+window_size],
                                                    window_size,
                                                    half_window)

                        ath = _get_weighted_mean(egm_array,
                                                 dist_weights,
                                                 window_size,
                                                 weights_sum)

                    elif method_int == 2:

                        ath = _get_mean_c(image_array[i:i+window_size,
                                                      j:j+window_size],
                                          window_size,
                                          constant)

                    elif method_int == 3:

                        ath = _get_otsu_threshold(image_array[i:i+window_size,
                                                              j:j+window_size],
                                                  window_size,
                                                  otsu_hist)

                    elif method_int == 4:

                        _bradley(image_array,
                                 out_threshold,
                                 i,
                                 j,
                                 window_size,
                                 rows,
                                 cols,
                                 bradley_t)

                        continue

                    elif method_int == 5:

                        ath = _sauvola(image_array[i:i+window_size,
                                                   j:j+window_size],
                                       window_size,
                                       k_param)

                    elif method_int == 6:

                        ath = _bernson(image_array[i:i+window_size,
                                                   j:j+window_size],
                                       window_size)

                    elif method_int == 7:

                        ath = _niblack(image_array[i:i+window_size,
                                                   j:j+window_size],
                                       window_size,
                                       k_param)

                else:

                    if image_array[i+half_window, j+half_window] > ignore_thresh:

                        if method_int == 1:

                            egm_array = image_array[i:i+window_size,
                                                    j:j+window_size]

                            if edge_direction_array_size > 4:

                                # Weight the EGM by difference from edge direction
                                _weight_egm_by_angle(egm_array,
                                                     edge_direction_array[i:i+window_size,
                                                                          j:j+window_size],
                                                     window_size,
                                                     half_window)

                            if edge_distance_array_size > 4:

                                # Weight the EGM by distance from sure edge.
                                _weight_egm_by_distance(egm_array,
                                                        edge_distance_array[i:i+window_size,
                                                                            j:j + window_size],
                                                        window_size,
                                                        half_window)

                            ath = _get_weighted_mean(egm_array,
                                                     dist_weights,
                                                     window_size,
                                                     weights_sum)

                        elif method_int == 2:

                            ath = _get_mean_c(image_array[i:i+window_size,
                                                          j:j+window_size],
                                              window_size,
                                              constant)

                        elif method_int == 3:

                            ath = _get_otsu_threshold(image_array[i:i+window_size,
                                                                  j:j+window_size],
                                                      window_size,
                                                      otsu_hist)

                        elif method_int == 4:

                            _bradley(image_array,
                                     out_threshold,
                                     i,
                                     j,
                                     window_size,
                                     rows,
                                     cols,
                                     bradley_t)

                            continue

                        elif method_int == 5:

                            ath = _sauvola(image_array[i:i+window_size,
                                                       j:j+window_size],
                                           window_size,
                                           k_param)

                        elif method_int == 6:

                            ath = _bernson(image_array[i:i+window_size,
                                                       j:j+window_size],
                                           window_size)

                        elif method_int == 7:

                            ath = _niblack(image_array[i:i+window_size,
                                                       j:j+window_size],
                                           window_size,
                                           k_param)

                    else:
                        ath = 0.

                if ath > 0:

                    if image_array[i+half_window, j+half_window] >= (ath * rt_):
                        out_threshold[i+half_window, j+half_window] = 1

    return np.uint8(out_threshold)


def threshold(image_array not None,
              window_size,
              ignore_thresh=-999.,
              rt=20.,
              n_jobs=1,
              method='wmean',
              c=5,
              t=15.,
              k=.5,
              inverse_dist=False,
              dist_sigma=10.0,
              edge_direction_array=None,
              edge_distance_array=None):

    """
    Args:
        image_array (2d array[rows(y) x cols(x)]): The array to threshold.
        window_size (int): The search window size, in pixels.
        ignore_thresh (Optional[float]): Any values equal to or below will be ignored.
            Default is -999, which considers all values.
        rt (Optional[float]): Relative percentage to the local threshold. Set to 1 for direct comparison. Default is 20.
        n_jobs (Optional[int]): The number of parallel jobs. Default is 1.
        method (Optional[str]): The method to use. Choices are ['mean-c', 'wmean', 'otsu',
            'bradley', 'sauvola', 'bernson', 'niblack']. Defaut is 'wmean'. *If `method` = 'bradley', then `image_array`
            should be the integral image.
        c (Optional[int]): The C (constant) parameter for `method` = 'mean-c'. Default is 5.
        t (Optional[float]): The Bradley T parameter. Default is 15.
        k (Optional[float]): The Sauvola k parameter. Default is 0.5.
        inverse_dist (Optional[bool]): Whether to invert the distance weights. Default is False.
        dist_sigma (Optional[float]): The logistic-weighted distance sigma. Default is 10.0.
        edge_direction_array (Optional[2d array]): An edge direction array. Default is None.
        edge_distance_array (Optional[2d array]): A euclidean distance array, normalized to 0-1. Default is None.
    """

    if method not in ['mean-c', 'wmean', 'otsu', 'bradley', 'sauvola', 'bernson', 'niblack']:
        raise ValueError('The method is not supported.')

    if not isinstance(edge_direction_array, np.ndarray):
        edge_direction_array = np.zeros((2, 2), dtype='float64')

    if not isinstance(edge_distance_array, np.ndarray):
        edge_distance_array = np.zeros((2, 2), dtype='float64')

    return _threshold(np.float64(np.ascontiguousarray(image_array)),
                      int(window_size),
                      float(ignore_thresh),
                      float(rt),
                      int(n_jobs),
                      str(method),
                      int(c),
                      float(t),
                      float(k),
                      inverse_dist,
                      dist_sigma,
                      edge_direction_array,
                      edge_distance_array)
