# distutils: language=c++
# cython: language_level=3
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

"""
The _draw_line() function was adapted from skimage (https://github.com/scikit-image)

See function docstring for full copyright disclaimer
"""

import cython
cimport cython

from copy import copy
import numpy as np
cimport numpy as np

# from libcpp.map cimport map

# from libc.stdlib cimport c_abs
# from libc.math cimport fabs

# from cython.parallel import prange
# from cython.parallel import parallel

try:
    import cv2
except ImportError:
    raise ImportError('OpenCV must be installed')

old_settings = np.seterr(all='ignore')

DTYPE_intp = np.intp
ctypedef np.intp_t DTYPE_intp_t

DTYPE_int32 = np.int32
ctypedef np.int32_t DTYPE_int32_t

DTYPE_int16 = np.int16
ctypedef np.int16_t DTYPE_int16_t

DTYPE_int64 = np.int64
ctypedef np.int64_t DTYPE_int64_t

DTYPE_uint8 = np.uint8
ctypedef np.uint8_t DTYPE_uint8_t

DTYPE_float32 = np.float32
ctypedef np.float32_t DTYPE_float32_t   

DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t


cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        void push_back(T&) nogil
        size_t size() nogil
        T& operator[](size_t) nogil
        void clear() nogil


cdef extern from 'math.h':
   double sqrt(double a) nogil


cdef extern from 'math.h':
   double cos(double a) nogil


cdef extern from 'math.h':
   double atan2(double a, double b) nogil


cdef extern from 'stdlib.h':
    double exp(double x)


cdef extern from 'numpy/npy_math.h':
    bint npy_isnan(double value) nogil


cdef extern from 'numpy/npy_math.h':
    bint npy_isinf(double value) nogil


# cdef extern from 'math.h':
#    double floor(double val) nogil


cdef inline double _multiply(double a, double b):
    return a * b


cdef inline double _subtract(double a, double b) nogil:
    return a - b


cdef inline double _nogil_get_min(double a, double b) nogil:
    return a if a <= b else b


cdef inline double _nogil_get_max(double a, double b) nogil:
    return a if a >= b else b


cdef inline unsigned int int_max(unsigned int a, unsigned int b) nogil:
    return a if a >= b else b


cdef inline double _abs(double m) nogil:
    return m*-1.0 if m < 0 else m


cdef inline Py_ssize_t _abs_pysize(Py_ssize_t m) nogil:
    return m*-1 if m < 0 else m


cdef inline int _abs_int(int m) nogil:
    return m*-1 if m < 0 else m


cdef inline double _perc_diff(double a, double b) nogil:
    return (b - a) / ((b + a) / 2.0)


cdef inline double _pow(double m, double n) nogil:
    return m**n


cdef inline double _spectral_distance(double x1, double x2) nogil:
    return _pow(x2 - x1, 2.0)


cdef inline double _euclidean_distance(double x1, double x2, double y1, double y2) nogil:
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5


cdef inline double normalize_eu_dist(double d, double max_d):
    return _abs(d - max_d) / max_d


cdef inline double euclidean_distance_color(double x1, double x2, double eu_dist):
    return ((x2 - x1)**2)**0.5 * eu_dist


cdef inline double simple_distance(double x1, double x2, double eu_dist):
    return ((x2 - x1)**2) * eu_dist


cdef inline double euclidean_distance_color_rgb(double r1, double g1, double b1, double r2, double g2, double b2):
    return (((r2 - r1)**2) + ((g2 - g1)**2) + ((b2 - b1)**2))**0.5


cdef inline double _translate_quadrant(double coord2translate, double wsh) nogil:
    return _abs(wsh - coord2translate)


# cdef inline int _get_rindex(int col_dims, Py_ssize_t index) nogil:
#     return <int>floor(<double>index / <double>col_dims)


# cdef inline int _get_cindex(int col_dims, Py_ssize_t index, int row_index) nogil:
#     return <int>(index - <double>col_dims * row_index)


cdef double _get_line_angle(double point1_y,
                            double point1_x,
                            double point2_y,
                            double point2_x) nogil:

    """
    Args:
        point1_y (int): Center point row coordinate.
        point1_x (int): Center point column coordinate.
        point2_y (int): End point row coordinate.
        point2_x (int): End point column coordinate.
        wsh (int): The line length.
    
    point1: [y1, x1]
    point2: [y2, x2]
    
    (0,0) (0,1) (0,2) (0,3) (0,4)
    (1,0) (1,1) (1,2) (1,3) (1,4)
    (2,0) (2,1) (2,2) (2,3) (2,4)
    (3,0) (3,1) (3,2) (3,3) (3,4)
    (4,0) (4,1) (4,2) (4,3) (4,4)    
    
    if (0,0):
        x_diff = -1 * (2 - 0)
            = -2
        y_diff = 2 - 0
            = 2
    """

    cdef:
        double x_diff
        double y_diff
        double theta
        double pi = 3.14159265

    # If necessary, translate from quadrant
    #   III or IV to quadrant I or II.
    # if point2_y > point1_y:
    #
    #     point2_y = _translate_quadrant(point2_y, float(wsh))
    #     point2_x = _translate_quadrant(point2_x, float(wsh))

    # x_diff = _subtract(point1_x, point2_x) * -1.
    # y_diff = _subtract(point1_y, point2_y)

    y_diff = point2_y - point1_y
    x_diff = point2_x - point1_x

    theta = atan2(y_diff, x_diff)

    if theta < 0:
        theta += 2.0 * pi

    # Invert the rotation
    theta = _abs((pi * 2.) - theta)

    # Convert to degrees
    theta *= 180.0 / pi

    return _abs(theta - 360.0)

    # Normalize to 0-180 (e.g., 270 becomes 45)
    # if theta > 180:
    #
    #     if theta >= 359:
    #         return 0.
    #     else:
    #         return theta - 180.
    #
    # else:
    #     return theta


# Define a function pointer to a metric.
ctypedef double (*metric_ptr)(double[:, ::1], DTYPE_intp_t, DTYPE_intp_t, double, double, double[:, ::1], unsigned int) nogil


cdef void _extract_values_byte(DTYPE_uint8_t[:, ::1] block,
                               double[::1] values,
                               double[:, ::1] rc_,
                               int fl) nogil:

    cdef:
        Py_ssize_t fi
        double fi_, fj_

    for fi in range(0, fl):

        fi_ = rc_[0, fi]
        fj_ = rc_[1, fi]

        values[fi] = float(block[<int>fi_, <int>fj_])


cdef void _extract_values_int(DTYPE_uint8_t[:, ::1] block,
                              double[::1] values,
                              double[:, ::1] rc_,
                              unsigned int fl) nogil:

    cdef:
        Py_ssize_t fi
        int fi_, fj_

    for fi in range(0, fl):

        fi_ = <int>rc_[0, fi]
        fj_ = <int>rc_[1, fi]

        values[fi] = float(block[fi_, fj_])


cdef void _extract_values_full(double[:, ::1] image_array,
                               Py_ssize_t i,
                               Py_ssize_t j,
                               double[::1] values,
                               double[:, ::1] rc_,
                               unsigned int fl) nogil:

    cdef:
        Py_ssize_t fi
        unsigned int fi_, fj_

    for fi in range(0, fl):

        fi_ = <int>rc_[0, fi]
        fj_ = <int>rc_[1, fi]

        values[fi] = image_array[i+fi_, j+fj_]


cdef void _extract_values(double[:, ::1] block,
                          double[::1] values,
                          double[:, ::1] rc_,
                          unsigned int fl) nogil:

    cdef:
        Py_ssize_t fi
        int fi_, fj_

    for fi in range(0, fl):

        fi_ = <int>rc_[0, fi]
        fj_ = <int>rc_[1, fi]

        values[fi] = block[fi_, fj_]


cdef void _draw_line(Py_ssize_t y0,
                     Py_ssize_t x0,
                     Py_ssize_t y1,
                     Py_ssize_t x1,
                     double[:, ::1] rc_) nogil:
    """Generates line pixel coordinates

    Adapted from original code at:
    https://github.com/scikit-image/scikit-image/blob/1af0071c8aadde6f7dbc06e53b7f38701b0f4b81/skimage/draw/_draw.pyx#L44

    Copyright (C) 2019, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    skimage/_shared/version_requirements.py:_check_version

        Copyright (c) 2013 The IPython Development Team
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this
          list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright notice,
          this list of conditions and the following disclaimer in the documentation
          and/or other materials provided with the distribution.

        * Neither the name of the copyright holder nor the names of its
          contributors may be used to endorse or promote products derived from
          this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    skimage/_shared/version_requirements.py:is_installed:

        Original Copyright (C) 2009-2011 Pierre Raybaut

        Permission is hereby granted, free of charge, to any person obtaining
        a copy of this software and associated documentation files (the
        "Software"), to deal in the Software without restriction, including
        without limitation the rights to use, copy, modify, merge, publish,
        distribute, sublicense, and/or sell copies of the Software, and to
        permit persons to whom the Software is furnished to do so, subject to
        the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
        EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
        MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
        NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
        LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
        OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
        WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Parameters
    ----------
    r0, c0 : int
        Starting position (row, column).
    r1, c1 : int
        End position (row, column).

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the line.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    See Also
    --------
    line_aa : Anti-aliased line generator
    """

    cdef:
        char steep = 0
        Py_ssize_t x = x0
        Py_ssize_t y = y0
        Py_ssize_t dx = _abs_pysize(x1 - x0)
        Py_ssize_t dy = _abs_pysize(y1 - y0)
        Py_ssize_t sx, sy, d, i

    if (x1 - x) > 0:
        sx = 1
    else:
        sx = -1

    if (y1 - y) > 0:
        sy = 1
    else:
        sy = -1

    if dy > dx:

        steep = 1
        x, y = y, x
        dx, dy = dy, dx
        sx, sy = sy, sx

    d = (2 * dy) - dx

    for i in range(0, dx):

        if steep:
            rc_[0, i] = x
            rc_[1, i] = y
        else:
            rc_[0, i] = y
            rc_[1, i] = x

        while d >= 0:

            y += sy
            d -= 2 * dx

        x += sx
        d += 2 * dy

    # Store the row index
    rc_[0, dx] = float(y1)

    # Store the column index
    rc_[1, dx] = float(x1)

    # Store the line length
    rc_[2, 0] = float(dx + 1)


cdef void draw_optimum_line(Py_ssize_t y0,
                            Py_ssize_t x0,
                            Py_ssize_t y1,
                            Py_ssize_t x1,
                            double[:, ::1] rc_,
                            double[:, ::1] resistance,
                            DTYPE_int16_t[::1] draw_indices,
                            double min_check) nogil:

    """
    Draws a line along the route of least resistance
    """

    cdef:
        Py_ssize_t ii__, jj__
        int ii_, jj_

        double start_value = resistance[y0, x0]
        double compare_value, min_diff, start_value_

        Py_ssize_t track_y = y0
        Py_ssize_t track_x = x0
        Py_ssize_t track_y_, track_x_

        Py_ssize_t gi = y0
        Py_ssize_t gj = x0

        unsigned int rrows = resistance.shape[0]
        unsigned int ccols = resistance.shape[1]

        unsigned int line_counter = 1

    # The starting y index
    rc_[0, 0] = y0

    # The starting x index
    rc_[1, 0] = x0

    # The line length
    rc_[2, 0] = 1

    # Continue until the end
    #   has been reached.
    while True:

        # Initiate a minimum distance.
        min_diff = 1000000.

        # Check neighbors
        for ii__ in range(0, 3):

            ii_ = draw_indices[ii__]

            # No change
            if y1 == track_y:

                if ii_ != 0:
                    continue

            # Increase
            elif y1 > track_y:

                if ii_ < 0:
                    continue

            # Decrease
            else:

                if ii_ > 0:
                    continue

            # Bounds checking
            if (track_y + ii_) > y1:
                track_y_ = y1
            else:
                track_y_ = track_y + ii_

            for jj__ in range(0, 3):

                jj_ = draw_indices[jj__]

                # No change
                if x1 == track_x:

                    if jj_ != 0:
                        continue

                # Increase
                elif x1 > track_x:

                    if jj_ < 0:
                        continue

                # Decrease
                else:

                    if jj_ > 0:
                        continue

                # Bounds checking
                if (track_x + jj_) > x1:
                    track_x_ = x1
                else:
                    track_x_ = track_x + jj_

                # Force movement
                if (ii__ == 1) and (jj__ == 1):
                    continue

                # Get the resistance value.
                compare_value = resistance[track_y, track_y]

                # Get the absolute difference
                #   between the two connecting
                #   values.
                if _abs(start_value - compare_value) < min_diff:

                    min_diff = _abs(start_value - compare_value)

                    gi = track_y_
                    gj = track_x_

                    start_value_ = start_value

        # Update the optimum
        #   row and column
        #   indices.
        rc_[0, line_counter] = gi
        rc_[1, line_counter] = gj
        rc_[2, 0] = line_counter + 1

        line_counter += 1

        # Update the tracking
        #   coordinates.
        track_y = gi
        track_x = gj

        start_value = start_value_

        # Check if the line is
        #   within 1 pixel of
        #   the end.
        if (_abs(gi - y1) == 1) and (_abs(gj - x1) == 1):
            break

        # Check if the line finished.
        if (gi == y1) and (gj == x1):
            break

        if line_counter >= rrows:
            break

        # Check if the minimum difference is
        #   greater than the minimum allowed.
        if min_check != -999.0:

            if min_diff > min_check:
                break

    # Ensure the last coordinate
    #   is the line end.
    if rc_[0, <int>rc_[2, 0]-1] != y1:

        rc_[0, line_counter] = y1
        rc_[1, line_counter] = x1
        rc_[2, 0] = line_counter + 1


cdef list _direction_list():

    d1_idx = [2, 2, 2, 2, 2], [0, 1, 2, 3, 4]
    d2_idx = [3, 3, 2, 1, 1], [0, 1, 2, 3, 4]
    d3_idx = [4, 3, 2, 1, 0], [0, 1, 2, 3, 4]
    d4_idx = [4, 3, 2, 1, 0], [1, 1, 2, 3, 3]
    d5_idx = [0, 1, 2, 3, 4], [2, 2, 2, 2, 2]
    d6_idx = [0, 1, 2, 3, 4], [1, 1, 2, 3, 3]
    d7_idx = [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]
    d8_idx = [1, 1, 2, 3, 3], [0, 1, 2, 3, 4]

    return [d1_idx, d2_idx, d3_idx, d4_idx, d5_idx, d6_idx, d7_idx, d8_idx]


cdef dict _direction_dict():

    return {0: np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]], dtype='uint8'),
            1: np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1],
                         [0, 0, 1, 0, 0],
                         [1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0]], dtype='uint8'),
            2: np.array([[0, 0, 0, 0, 1],
                         [0, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 0],
                         [1, 0, 0, 0, 0]], dtype='uint8'),
            3: np.array([[0, 0, 0, 1, 0],
                         [0, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 1, 0, 0, 0]], dtype='uint8'),
            4: np.array([[0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0]], dtype='uint8'),
            5: np.array([[0, 1, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 0],
                         [0, 0, 0, 1, 0]], dtype='uint8'),
            6: np.array([[1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 1]], dtype='uint8'),
            7: np.array([[0, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0]], dtype='uint8')}


cdef DTYPE_uint8_t _get_mean1d_int(DTYPE_uint8_t[::1] block_list,
                                   unsigned int length):

    cdef:
        Py_ssize_t ii
        DTYPE_uint8_t s = block_list[0]

    for ii in range(1, length):
        s += block_list[ii]

    return s / length


cdef double _get_mean1d(double[::1] block_list, unsigned int length) nogil:

    cdef:
        Py_ssize_t ii
        double s = block_list[0]

    for ii in range(1, length):
        s += block_list[ii]

    return s / length


cdef double _get_sum1df(double[::1] block_list, unsigned int length) nogil:

    cdef:
        Py_ssize_t fi
        double s = block_list[0]

    for fi in range(1, length):
        s += block_list[fi]

    return s


cdef DTYPE_uint8_t _get_sum1d(DTYPE_uint8_t[::1] block_list,
                              unsigned int length) nogil:

    cdef:
        Py_ssize_t fi
        DTYPE_uint8_t s = block_list[0]

    for fi in range(1, length):
        s += block_list[fi]

    return s


cdef void _get_sum1d_f(double[::1] block_list,
                       int length,
                       double[::1] sums_array__) nogil:

    cdef:
        Py_ssize_t fi
        double bv

    for fi in range(0, length):

        bv = block_list[fi]

        sums_array__[0] += bv    # EGM sum
        sums_array__[1] += 1     # pixel count


cdef double _get_std1d(double[::1] block_list, int length) nogil:

    cdef:
        Py_ssize_t ii
        double block_list_mean = _get_mean1d(block_list, length)
        double s = _pow(block_list[0] - block_list_mean, 2.0)

    for ii in range(1, length):
        s += _pow(block_list[ii] - block_list_mean, 2.0)

    return sqrt(s / length)


cdef DTYPE_uint8_t _get_argmin1d(double[::1] block_list, int length):

    cdef:
        Py_ssize_t ii
        double s = block_list[0]
        DTYPE_uint8_t argmin = 0
        double b_samp

    with nogil:

        for ii in range(1, length-1):

            b_samp = block_list[ii]

            s = _nogil_get_min(b_samp, s)

            if s == b_samp:
                argmin = ii

    return argmin


cdef DTYPE_uint8_t[:, ::1] _mu_std(double[:, ::1] imb2calc, list direction_list, dict se_dict):

    cdef:
        tuple d_directions
        double[::1] std_list = np.array([_get_std1d(imb2calc[d_directions], 5) for d_directions in direction_list], dtype='float64')
        int min_std_idx = _get_argmin1d(std_list, 5)

    return se_dict[min_std_idx]


cdef double _morph_pass(double[:, :] image_block, DTYPE_intp_t window_i, DTYPE_intp_t window_j,
                        DTYPE_intp_t target_value, DTYPE_intp_t ignore_value,
                        double[:, :] weights, int hw):

    """
    Reference:
        D. Chaudhuri, N. K. Kushwaha, and A. Samal (2012) 'Semi-Automated Road
            Detection From High Resolution Satellite Images by Directional
            Morphological Enhancement and Segmentation Techniques', IEEE JOURNAL
            OF SELECTED TOPICS IN APPLIED EARTH OBSERVATIONS AND REMOTE SENSING,
            5(5), OCTOBER.
    """

    cdef:
        Py_ssize_t half = int(window_i / 2)
        list ds = _direction_list()
        dict se_dict = _direction_dict()
        np.ndarray[DTYPE_uint8_t, ndim=2, mode='c'] se = _mu_std(np.array(image_block).astype(np.float64), ds, se_dict)
        np.ndarray[DTYPE_uint8_t, ndim=2, mode='c'] im_b = cv2.morphologyEx(np.array(image_block).astype(np.uint8), cv2.MORPH_OPEN, se, iterations=1)
        # np.ndarray[DTYPE_uint8_t, ndim=2, mode='c'] im_b = pymorph.asfrec(np.array(image_block).astype(np.uint8), seq='CO', B=se, Bc=se, N=1)

    # se = _mu_std(im_b.astype(np.float64), ds, se_dict)

    # im_b = pymorph.asfrec(im_b, seq='OC', B=se, Bc=se, N=1)
    # im_b = cv2.morphologyEx(im_b, cv2.MORPH_DILATE, se, iterations=1)

    # se = _mu_std(im_b.astype(np.float64), ds, se_dict)

    # im_b = cv2.morphologyEx(im_b.astype(np.uint8), cv2.MORPH_DILATE, se, iterations=1)

    # se = _mu_std(im_b.astype(np.float64), ds, se_dict)

    # return float(pymorph.asfrec(im_b, seq='OC', B=se, Bc=se, N=1)[half, half])
    # return float(cv2.morphologyEx(im_b, cv2.MORPH_CLOSE, se, iterations=1)[half, half])
    return float(im_b[half, half])


cdef double _get_median(double[:, :] block, DTYPE_intp_t window_i, DTYPE_intp_t window_j,
                        DTYPE_intp_t target_value, DTYPE_intp_t ignore_value,
                        double[:, :] weights, int hw):

    cdef:
        Py_ssize_t half = int((window_i * window_j) / 2)
        list sorted_list = sorted(list(np.asarray(block).ravel()))

    return sorted_list[half]


cdef double _get_min(double[:, ::1] block,
                     DTYPE_intp_t window_i,
                     DTYPE_intp_t window_j,
                     double target_value,
                     double ignore_value,
                     double[:, ::1] weights,
                     unsigned int hw) nogil:

    cdef:
        Py_ssize_t ii, jj
        double su = 999999.

    if ignore_value != -9999.:

        for ii in range(0, window_i):
            for jj in range(0, window_j):

                if block[ii, jj] != ignore_value:
                    su = _nogil_get_min(block[ii, jj], su)

    else:

        for ii in range(0, window_i):
            for jj in range(0, window_j):

                su = _nogil_get_min(block[ii, jj], su)

    return su


cdef double _get_max(double[:, ::1] block,
                     DTYPE_intp_t window_i,
                     DTYPE_intp_t window_j,
                     double target_value,
                     double ignore_value,
                     double[:, ::1] weights,
                     unsigned int hw) nogil:

    cdef:
        Py_ssize_t ii, jj
        double su = -999999.

    if ignore_value != -9999.:

        for ii in range(0, window_i):
            for jj in range(0, window_j):

                if block[ii, jj] != ignore_value:
                    su = _nogil_get_max(block[ii, jj], su)

    else:

        for ii in range(0, window_i):
            for jj in range(0, window_j):

                su = _nogil_get_max(block[ii, jj], su)

    return su


cdef DTYPE_uint8_t _fill_holes(DTYPE_uint8_t[:, :] block,
                               DTYPE_uint8_t[::1] rr,
                               DTYPE_uint8_t[::1] cc,
                               int window_size,
                               int n_neighbors) nogil:

    cdef:
        Py_ssize_t ii, jj
        int center = <int>(window_size / 2.)
        DTYPE_uint8_t s = 0
        DTYPE_uint8_t fill_value = block[center, center]

    if fill_value == 0:

        for ii in range(0, n_neighbors):
            s += block[rr[ii], cc[ii]]

        # Fill the pixel if it
        #   is surrounded.
        if s == n_neighbors:
            fill_value = 1

    return fill_value


cdef DTYPE_uint8_t _get_sum_int(DTYPE_uint8_t[:, ::1] block,
                                unsigned int window_i,
                                unsigned int window_j) nogil:

    cdef:
        Py_ssize_t ii, jj
        DTYPE_uint8_t su = 0

    # with nogil, parallel(num_threads=window_i):
    #
    #     for ii in prange(0, window_i, schedule='static'):
    for ii in range(0, window_i):
        for jj in range(0, window_j):
            su += block[ii, jj]

    return su


cdef DTYPE_uint8_t _get_sum_uint8(DTYPE_uint8_t[:, :] block, DTYPE_intp_t window_i, DTYPE_intp_t window_j):

    cdef:
        Py_ssize_t ii, jj
        DTYPE_uint8_t su = 0

    # with nogil, parallel(num_threads=window_i):
    #
    #     for ii in prange(0, window_i, schedule='static'):
    for ii in xrange(0, window_i):

        for jj in xrange(0, window_j):

            su += block[ii, jj]

    return su


cdef double _get_sum(double[:, ::1] block,
                     DTYPE_intp_t window_i,
                     DTYPE_intp_t window_j,
                     double target_value,
                     double ignore_value,
                     double[:, ::1] weights,
                     unsigned int hw) nogil:

    cdef:
        Py_ssize_t ii, jj
        double su = 0.

    if ignore_value != -9999.:

        # with nogil, parallel(num_threads=window_i):
        #
        #     for ii in prange(0, window_i, schedule='static'):
        for ii in range(0, window_i):

            for jj in range(0, window_j):

                if block[ii, jj] != ignore_value:
                    su += block[ii, jj]

    else:

        # with nogil, parallel(num_threads=window_i):
        #
        #     for ii in prange(0, window_i, schedule='static'):
        for ii in range(0, window_i):

            for jj in range(0, window_j):

                su += block[ii, jj]

    return su


cdef vector[double] _check_value_count(vector[double] counts_list, double block_value) nogil:

    cdef:
        Py_ssize_t cidx
        int ncounts = counts_list.size()

    if ncounts == 0:
        counts_list.push_back(block_value)
        return counts_list

    # Check if the value has been added
    for cidx in range(0, ncounts):
        if block_value == counts_list[cidx]:
            return counts_list

    counts_list.push_back(block_value)
    return counts_list


cdef double _get_unique_count(double[:, ::1] block,
                              DTYPE_intp_t window_i,
                              DTYPE_intp_t window_j,
                              double target_value,
                              double ignore_value,
                              double[:, ::1] weights,
                              unsigned int hw) nogil:

    cdef:
        Py_ssize_t ii, jj
        double value
        vector[double] counts

    if ignore_value != -9999.0:

        for ii in range(0, window_i):
            for jj in range(0, window_j):
                if block[ii, jj] != ignore_value:
                    value = block[ii, jj]
                    counts = _check_value_count(counts, value)

    else:

        for ii in range(0, window_i):
            for jj in range(0, window_j):
                value = block[ii, jj]
                counts = _check_value_count(counts, value)

    return <double>counts.size()


cdef double _get_mean(double[:, ::1] block,
                      DTYPE_intp_t window_i,
                      DTYPE_intp_t window_j,
                      double target_value,
                      double ignore_value,
                      double[:, ::1] weights,
                      unsigned int hw) nogil:

    cdef:
        Py_ssize_t ii, jj
        double su = 0.
        int good_values = 0

    if target_value != -9999.:
        if block[hw, hw] != target_value:
            return block[hw, hw]

    if ignore_value != -9999.:

        for ii in range(0, window_i):
            for jj in range(0, window_j):

                if block[ii, jj] != ignore_value:

                    su += block[ii, jj] * weights[ii, jj]
                    good_values += 1

    else:

        for ii in range(0, window_i):
            for jj in range(0, window_j):

                su += block[ii, jj] * weights[ii, jj]

        good_values = window_i * window_j

    if good_values == 0:
        return 0.
    else:
        return su / good_values


cdef double _get_std(double[:, ::1] block,
                     DTYPE_intp_t window_i,
                     DTYPE_intp_t window_j,
                     double target_value,
                     double ignore_value,
                     double[:, ::1] weights,
                     unsigned int hw) nogil:

    cdef:
        Py_ssize_t ii, jj
        double su = 0.0
        double stdv = 0.0
        int good_values = 0

    if target_value != -9999.0:
        if block[hw, hw] != target_value:
            return block[hw, hw]

    if ignore_value != -9999.0:

        for ii in range(0, window_i):
            for jj in range(0, window_j):

                if block[ii, jj] != ignore_value:

                    su += block[ii, jj] * weights[ii, jj]
                    good_values += 1

    else:

        for ii in range(0, window_i):
            for jj in range(0, window_j):

                su += block[ii, jj] * weights[ii, jj]

        good_values = window_i * window_j

    if good_values == 0:
        su = 0.0
    else:
        su /= good_values

    if ignore_value != -9999.0:

        for ii in range(0, window_i):
            for jj in range(0, window_j):
                if block[ii, jj] != ignore_value:
                    stdv += _pow(float(block[ii, jj]) - su, 2.0)

    else:

        for ii in range(0, window_i):
            for jj in range(0, window_j):
                stdv += _pow(float(block[ii, jj]) - su, 2.0)

    if good_values == 0:
        return 0.0
    else:
        return sqrt(stdv / good_values)


cdef double _get_distance(double[:, :, ::1] image_array_,
                          Py_ssize_t ii,
                          Py_ssize_t jj,
                          unsigned int nlayers,
                          unsigned  int window_size,
                          unsigned int hw,
                          DTYPE_int64_t[::1] x_offsets,
                          DTYPE_int64_t[::1] y_offsets,
                          double[::1] dist_weights,
                          double weights_sum) nogil:

    cdef:
        Py_ssize_t t, z
        double d_week, pow_week
        double pow_total = 0.0
        double week_total = 0.0
        double center_value

    # Iterate over each array layer
    for t in range(0, nlayers):

        center_value = image_array_[t, ii+hw, jj+hw]

        pow_week = _pow(center_value, 2.0)
        pow_total += pow_week

        d_week = 0.0

        # Iterate over the window
        for z in range(0, window_size*window_size):
            d_week += (_abs(center_value - image_array_[t, ii+y_offsets[z], jj+x_offsets[z]]) * dist_weights[z])

        week_total += (pow_week * (d_week / weights_sum))

    return week_total / pow_total


cdef double _get_distance_rgb(double[:, :, :] block,
                              DTYPE_intp_t window_i,
                              DTYPE_intp_t window_j,
                              int hw,
                              int dims):

    cdef:
        Py_ssize_t ii, jj
        # double d
        # double max_d = euclidean_distance(hw, 0., hw, 0.)
        double[::1] hw_values = np.zeros(3, dtype='float64')
        double color_d = 0.
        # double avg_d = 0.
        # double block_max = _get_max(block, window_i, window_j, ignore_value, weights, hw)
        # double block_min = _get_min(block, window_i, window_j, ignore_value, weights, hw)
        # double max_color_dist = euclidean_distance_color(block_min, block_max, 1.)
        # double max_color_dist = float(dims)**.5  # sqrt((1-0)^2 + (1-0)^2 + (1-0)^2)
        # double max_color_dist = 0.

    # Center values
    hw_values[0] = block[0, hw, hw]
    hw_values[1] = block[1, hw, hw]
    hw_values[2] = block[2, hw, hw]

    for ii in xrange(0, window_i):

        for jj in xrange(0, window_j):

            if (ii == hw) and (jj == hw):
                continue

            # Get the Euclidean distance between the center pixel
            #   and the surrounding pixels.
            # d = euclidean_distance(hw, jj, hw, ii)

            # d = normalize_eu_dist(d, max_d)
            #
            # if d == 0:
            #     d = .01

            # Get the distance between colors.
            color_d += euclidean_distance_color_rgb(hw_values[0], hw_values[1], hw_values[2],
                                                    block[0, ii, jj], block[1, ii, jj], block[2, ii, jj])

            # max_color_dist = max(color_d, max_color_dist)
            #
            # avg_d += color_d

    # Get the block average and
    #   normalize the block average.
    return color_d / ((hw * hw) - 1)


cdef int cy_argwhere(DTYPE_uint8_t[:, ::1] array1,
                     DTYPE_uint8_t[:, ::1] array2,
                     unsigned int dims,
                     DTYPE_int16_t[:, ::1] angles_dict) nogil:

    cdef:
        Py_ssize_t i_, j_, i_idx, j_idx
        unsigned int counter = 1

    for i_ in range(0, dims):

        for j_ in range(0, dims):

            if (array1[i_, j_] == 1) and (array2[i_, j_] == 0):

                if counter > 1:

                    i_idx = (i_idx + i_) / counter
                    j_idx = (j_idx + j_) / counter

                else:

                    i_idx = i_ + 0
                    j_idx = j_ + 0

                counter += 1

    if counter == 1:
        return 9999
    else:
        return angles_dict[i_idx, j_idx]


cdef tuple close_end(DTYPE_uint8_t[:, ::1] edge_block,
                     DTYPE_uint8_t[:, ::1] endpoints_block,
                     double[:, ::1] gradient_block,
                     int angle,
                     int center,
                     double[::1] dummy,
                     DTYPE_intp_t[::1] h_r,
                     DTYPE_intp_t[::1] h_c,
                     DTYPE_intp_t[::1] d_c,
                     DTYPE_int16_t[:, ::1] angles_dict,
                     int min_egm,
                     int max_gap,
                     double[:, ::1] rcc__):

    cdef:
        Py_ssize_t ip, rr_shape, ip_, jp_
        DTYPE_intp_t[::1] hr1, hr2
        int mtotal = 3      # The total number of orthogonal pixels required to connect a point with orthogonal lines.
        int connect_angle
        int min_line = 3    # The minimum line length to connect a point to an edge with sufficient EGM

        double[::1] rr_, cc_, line_values__

    if angle == 90:

        ip_ = -1
        jp_ = 0
        hr1 = h_r
        hr2 = h_c

    elif angle == -90:

        ip_ = 1
        jp_ = 0
        hr1 = h_r
        hr2 = h_c

    elif angle == 180:

        ip_ = 0
        jp_ = -1
        hr1 = h_c
        hr2 = h_r

    elif angle == -180:

        ip_ = 0
        jp_ = 1
        hr1 = h_c
        hr2 = h_r

    elif angle == 135:

        ip_ = -1
        jp_ = -1
        hr1 = d_c
        hr2 = h_c

    elif angle == -135:

        ip_ = 1
        jp_ = 1
        hr1 = d_c
        hr2 = h_c

    elif angle == 45:

        ip_ = -1
        jp_ = 1
        hr1 = h_c
        hr2 = h_c

    elif angle == -45:

        ip_ = 1
        jp_ = -1
        hr1 = h_c
        hr2 = h_c

    else:
        return dummy, dummy, 0, 9999

    for ip in range(1, max_gap-2):

        if edge_block[center+(ip*ip_), center+(ip*jp_)] == 1:
            _draw_line(center, center, center+(ip*ip_), center+(ip*jp_), rcc__)

            # Get the current line length.
            rr_shape = <int>rcc__[2, 0]

            # row of zeros, up to the line length
            line_values__ = rcc__[3, :rr_shape]

            rr_ = rcc__[0, :rr_shape]
            cc_ = rcc__[1, :rr_shape]

            # Extract the values along the line.
            _extract_values(gradient_block, line_values__, rcc__, rr_shape)

            # Connect the points if the line is
            #   small and has edge magnitude.
            if rr_shape <= min_line:

                if _get_mean1d(line_values__, rr_shape) >= min_egm:
                    return rr_, cc_, 1, 9999

            # Check if it is an endpoint or an edge.
            connect_angle = cy_argwhere(edge_block[center+(ip*ip_)-2:center+(ip*ip_)+3,
                                                   center+(ip*jp_)-2:center+(ip*jp_)+3],
                                        endpoints_block[center+(ip*ip_)-2:center+(ip*ip_)+3,
                                                        center+(ip*jp_)-2:center+(ip*jp_)+3], 3, angles_dict)

            # Connect lines of any length with
            #   inverse or orthogonal angles.
            if angle + connect_angle == 0 or \
                            _get_sum1d(extract_values(edge_block[center+(ip*ip_)-2:center+(ip*ip_)+3,
                                                                 center+(ip*jp_)-2:center+(ip*jp_)+3],
                                                      hr1, hr2, 5), 5) >= mtotal:

                if _get_mean1d(line_values__, rr_shape) >= min_egm:
                    return rr_, cc_, 1, 9999

            break

    return dummy, dummy, 0, 9999


cdef double[::1] extract_values_f(double[:, :] block, DTYPE_intp_t[::1] rr_, DTYPE_intp_t[::1] cc_, int fl):

    cdef:
        Py_ssize_t fi, fi_, fj_
        double[::1] values = np.zeros(fl, dtype='float64')

    with nogil:

        for fi in range(0, fl):

            fi_ = rr_[fi]
            fj_ = cc_[fi]

            values[fi] = block[fi_, fj_]

    return values


cdef DTYPE_uint8_t[::1] extract_values(DTYPE_uint8_t[:, ::1] block,
                                       DTYPE_intp_t[::1] rr_,
                                       DTYPE_intp_t[::1] cc_,
                                       unsigned int fl):

    cdef:
        Py_ssize_t fi, fi_, fj_
        DTYPE_uint8_t[::1] values = np.zeros(fl, dtype='uint8')

    for fi in range(0, fl):

        fi_ = rr_[fi]
        fj_ = cc_[fi]

        values[fi] = block[fi_, fj_]

    return values


cdef void _fill_block(DTYPE_uint8_t[:, ::1] block2fill_,
                      double[::1] rr_,
                      double[::1] cc_,
                      unsigned int fill_value,
                      unsigned int fl) nogil:

    cdef:
        Py_ssize_t fi
        unsigned int fi_, fj_

    for fi in range(0, fl):

        fi_ = <int>rr_[fi]
        fj_ = <int>cc_[fi]

        block2fill_[fi_, fj_] = fill_value


cdef void _link_endpoints(DTYPE_uint8_t[:, ::1] edge_block,
                          DTYPE_uint8_t[:, ::1] endpoints_block,
                          double[:, ::1] gradient_block,
                          unsigned int window_size_,
                          DTYPE_int16_t[:, ::1] angles_dict,
                          DTYPE_intp_t[::1] h_r,
                          DTYPE_intp_t[::1] h_c,
                          DTYPE_intp_t[::1] d_c,
                          int min_egm,
                          int smallest_allowed_gap,
                          int medium_allowed_gap,
                          double[:, ::1] rcc_,
                          DTYPE_int16_t[::1] draw_indices,
                          Py_ssize_t i_, Py_ssize_t j_):

    cdef:
        Py_ssize_t ii, jj, ii_, jj_

        unsigned int smallest_gap = window_size_ * window_size_   # The smallest gap found
        unsigned int center = <int>(window_size_ / 2.)
        int center_angle, connect_angle, ss, match

        double[::1] rr_, cc_
        # double[::1] dummy = np.zeros(window_size_, dtype='float64')

        unsigned int rc_shape, rc_length_

        double[::1] line_values_, line_values_g

        double line_mean
        double max_line_mean = 0.

        bint short_line

    if smallest_allowed_gap > window_size_:
        smallest_allowed_gap = window_size_

    if medium_allowed_gap > window_size_:
        medium_allowed_gap = window_size_

    # Get the origin angle of the center endpoint.
    center_angle = cy_argwhere(edge_block[center-1:center+2,
                                          center-1:center+2],
                               endpoints_block[center-1:center+2,
                                               center-1:center+2],
                               3,
                               angles_dict)

    if center_angle != 9999:    # and (center_angle != 0):

        with nogil:

            # There must be at least
            #   two endpoints in the block.
            if _get_sum_int(endpoints_block, window_size_, window_size_) > 1:

                for ii in range(1, window_size_-1):

                    for jj in range(1, window_size_-1):

                        # Cannot connect to direct neighbors or itself.
                        if (_abs(float(ii) - float(center)) <= 1) and (_abs(float(jj) - float(center)) <= 1):
                            continue

                        # Cannot connect with edges
                        #   because we cannot
                        #   get the angle at the
                        #   window edges.
                        # if (ii == 0) or (ii == window_size_-1) or (jj == 0) or (jj == window_size_-1):
                        #     continue

                        # Located another endpoint.
                        if endpoints_block[ii, jj] == 1:

                            # Draw a line between the two endpoints.
                            if (_abs(ii - center) <= smallest_allowed_gap) and \
                                    (_abs(jj - center) <= smallest_allowed_gap):

                                short_line = True

                                # Draw the straightest line.
                                _draw_line(center, center, ii, jj, rcc_)

                            else:

                                short_line = False

                                # Draw a line along the path
                                #   of least resistance.
                                draw_optimum_line(center, center, ii, jj, rcc_, gradient_block, draw_indices, -999.0)

                            # Get the current line length.
                            rc_length_ = <int>rcc_[2, 0]

                            # row of zeros, up to the line length
                            line_values_ = rcc_[3, :rc_length_]

                            # Extract the values along the line.
                            _extract_values_int(edge_block, line_values_, rcc_, rc_length_)

                            # CHECK IF THE CONNECTING
                            #   LINE CROSSES OTHER EDGES
                            if _get_sum1df(line_values_, rc_length_) > 2:

                                # Try a straighter line.
                                if not short_line:

                                    _draw_line(center, center, ii, jj, rcc_)

                                    # Get the current line length.
                                    rc_length_ = <int>rcc_[2, 0]

                                    # row of zeros, up to the line length
                                    line_values_ = rcc_[3, :rc_length_]

                                    # Extract the values along the line.
                                    _extract_values_int(edge_block, line_values_, rcc_, rc_length_)

                                if _get_sum1df(line_values_, rc_length_) > 2:
                                    continue

                            # CONNECT THE SMALLEST LINE
                            #   POSSIBLE WITH HIGH EGM
                            if rc_length_ >= smallest_gap:
                                continue

                            # Check the angles if the gap is large.

                            # CONNECT POINTS WITH SIMILAR ANGLES
                            connect_angle = cy_argwhere(edge_block[ii-1:ii+2, jj-1:jj+2],
                                                        endpoints_block[ii-1:ii+2, jj-1:jj+2],
                                                        3, angles_dict)

                            if connect_angle == 9999:     # or (connect_angle == 0):
                                continue

                            # Don't connect same angles.
                            # if center_angle == connect_angle:
                            #     continue

                            line_values_g = rcc_[3, :rc_length_]

                            # Extract edge gradient values
                            #   along the line.
                            _extract_values(gradient_block, line_values_g, rcc_, rc_length_)

                            if rc_length_ == 3:
                                line_mean = line_values_g[1]
                            else:

                                # Get the mean EGM along the line,
                                #   avoiding the endpoints.
                                line_mean = _get_mean1d(line_values_g[1:rc_length_-1], rc_length_)

                            # For small gaps allow any angle as long
                            #   as there is sufficient EGM.
                            if rc_length_ <= smallest_allowed_gap:

                                # There must be edge contrast
                                #   along the line.
                                if line_mean >= min_egm:

                                    # rr_, cc_ = rr.copy(), cc.copy()
                                    rr_ = rcc_[0, :rc_length_]
                                    cc_ = rcc_[1, :rc_length_]
                                    rc_shape = rc_length_

                                    ii_ = ii + 0
                                    jj_ = jj + 0

                                    smallest_gap = <int>_nogil_get_min(rc_length_, smallest_gap)

                                    max_line_mean = line_mean

                            # For medium-sized gaps allow similar angles, but no
                            #   asymmetric angles.
                            elif rc_length_ > smallest_allowed_gap:

                                match = 0

                                ####################
                                # The columns should
                                #   not overlap.
                                ####################
                                if (_abs(center_angle) == 180) and (_abs(connect_angle) == 180):

                                    # The endpoints must be within 5
                                    #   pixels along the perpendicular plane.
                                    # if _abs(center - ii) <= 5:

                                    if (center_angle == 180) and (connect_angle == -180):

                                        if jj <= center:
                                            match = 1

                                    elif (center_angle == -180) and (connect_angle == 180):

                                        if jj >= center:
                                            match = 1

                                elif (_abs(center_angle) == 135) and (_abs(connect_angle) == 180):

                                    if (center_angle == 135) and (connect_angle == -180):

                                        if jj <= center:
                                            match = 1

                                    elif (center_angle == -135) and (connect_angle == 180):

                                        if jj >= center:
                                            match = 1

                                elif (_abs(center_angle) == 180) and (_abs(connect_angle) == 135):

                                    if (center_angle == 180) and (connect_angle == -135):

                                        if jj <= center:
                                            match = 1

                                    elif (center_angle == -180) and (connect_angle == 135):

                                        if jj >= center:
                                            match = 1

                                elif (_abs(center_angle) == 180) and (_abs(connect_angle) == 45):

                                    if (center_angle == 180) and (connect_angle == 45):

                                        if jj <= center:
                                            match = 1

                                    elif (center_angle == -180) and (connect_angle == -45):

                                        if jj >= center:
                                            match = 1

                                #################
                                # The rows should
                                #   not overlap.
                                #################
                                elif (_abs(center_angle) == 90) and (_abs(connect_angle) == 90):

                                    # The endpoints must be within 5
                                    #   pixels along the perpendicular plane.
                                    # if _abs(center - jj) <= 5:

                                    if (center_angle == 90) and (connect_angle == -90):

                                        if ii <= center:
                                            match = 1

                                    elif (center_angle == -90) and (connect_angle == 90):

                                        if ii >= center:
                                            match = 1

                                elif (_abs(center_angle) == 90) and (_abs(connect_angle) == 135):

                                    if (center_angle == 90) and (connect_angle == -135):

                                        if ii <= center:
                                            match = 1

                                    elif (center_angle == -90) and (connect_angle == 135):

                                        if ii >= center:
                                            match = 1

                                elif (_abs(center_angle) == 90) and (_abs(connect_angle) == 45):

                                    if (center_angle == 90) and (connect_angle == -45):

                                        if ii <= center:
                                            match = 1

                                    elif (center_angle == -90) and (connect_angle == 45):

                                        if ii >= center:
                                            match = 1

                                #######################
                                # The rows and columns
                                #   should not overlap.
                                #######################
                                elif (center_angle == 135) and (connect_angle == -135):

                                    if (ii <= center) and (jj <= center):
                                        match = 1

                                elif (center_angle == -135) and (connect_angle == 135):

                                    if (ii >= center) and (jj >= center):
                                        match = 1

                                elif (center_angle == 45) and (connect_angle == -45):

                                    if (ii >= center) and (jj >= center):
                                        match = 1

                                elif (center_angle == -45) and (connect_angle == 45):

                                    if (ii <= center) and (jj <= center):
                                        match = 1

                                elif (center_angle == 0) and (connect_angle == 0):
                                    match = 1

                                else:
                                    line_mean *= 2.

                                if match == 1:

                                    # There must be edge contrast
                                    #   along the line.
                                    if line_mean >= min_egm:

                                        # rr_, cc_ = rr.copy(), cc.copy()
                                        rr_ = rcc_[0, :rc_length_]
                                        cc_ = rcc_[1, :rc_length_]
                                        rc_shape = rc_length_

                                        ii_ = ii + 0
                                        jj_ = jj + 0

                                        smallest_gap = <int>_nogil_get_min(rc_length_, smallest_gap)

                                        max_line_mean = line_mean

                            # All other gaps must be inverse angles and have
                            #   a mean edge gradient magnitude over the minimum
                            #   required.
                            # else:
                            #
                            #     # All other inverse angles.
                            #     if center_angle + connect_angle == 0:
                            #
                            #         # Extract the values along the line.
                            #         _extract_values(gradient_block, line_values_g, rcc_, rc_length_)
                            #
                            #         # There must be edge contrast along the line.
                            #         if _get_mean1d(line_values_g, rc_length_) >= min_egm:
                            #
                            #             # rr_, cc_ = rr.copy(), cc.copy()
                            #             rr_ = rcc_[0, :rc_length_]
                            #             cc_ = rcc_[1, :rc_length_]
                            #             rc_shape = rc_length_
                            #
                            #             ii_ = ii + 0
                            #             jj_ = jj + 0
                            #
                            #             smallest_gap = <int>_nogil_get_min(rc_length_, smallest_gap)

        # TRY TO CLOSE GAPS FROM ENDPOINTS

        # At this juncture, there doesn't have to
        #   be two endpoints.
        # if smallest_gap == window_size_ * window_size_:
        #
        #     rr_, cc_, ss, ii_ = close_end(edge_block,
        #                                   endpoints_block,
        #                                   gradient_block,
        #                                   center_angle,
        #                                   center,
        #                                   dummy,
        #                                   h_r,
        #                                   h_c,
        #                                   d_c,
        #                                   angles_dict,
        #                                   min_egm,
        #                                   center,
        #                                   rcc_)
        #
        #     if ss == 1:
        #         smallest_gap = 0

        if smallest_gap < (window_size_ * window_size_):

            # if rc_shape > 3:
            #     print np.uint8(edge_block)

            _fill_block(edge_block, rr_, cc_, 1, rc_shape)

            # if rc_shape > 3:
            #     print np.uint8(edge_block)
            #     import sys
            #     sys.exit()

            # Remove the endpoint
            endpoints_block[center, center] = 0

            if ii_ < 9999:
                endpoints_block[ii_, jj_] = 0


cdef double _duda_operator(double[:, ::1] block,
                                    DTYPE_intp_t window_i,
                                    DTYPE_intp_t window_j,
                                    double target_value,
                                    double ignore_value,
                                    double[:, ::1] weights,
                                    unsigned int hw) nogil:

    cdef:
        double duda = 0.
        double a, b, c

    a = block[hw, hw]

    b = block[0, hw]
    c = block[window_i-1, hw]

    if b < a > c:
        duda = _nogil_get_max(duda, 2.*a-b-c)

    b = block[0, 0]
    c = block[window_i-1, window_j-1]

    if b < a > c:
        duda = _nogil_get_max(duda, 2.*a-b-c)

    b = block[hw, 0]
    c = block[hw, window_j-1]

    if b < a > c:
        duda = _nogil_get_max(duda, 2.*a-b-c)

    b = block[window_i-1, 0]
    c = block[0, window_j-1]

    if b < a > c:
        duda = _nogil_get_max(duda, 2.*a-b-c)

    return duda


cdef double _get_percent(double[:, ::1] block,
                                  DTYPE_intp_t window_i,
                                  DTYPE_intp_t window_j,
                                  double target_value,
                                  double ignore_value,
                                  double[:, ::1] weights,
                                  unsigned int hw) nogil:

    cdef:
        Py_ssize_t ii, jj
        double su = 0.
        unsigned int good_values = 0

    if ignore_value != -9999.:

        for ii in range(0, window_i):
            for jj in range(0, window_j):

                if block[ii, jj] != ignore_value:
                    su += block[ii, jj]
                    good_values += 1

    else:

        for ii in range(0, window_i):

            for jj in range(0, window_j):

                su += block[ii, jj]

        good_values = window_i * window_j

    return (su / good_values) * 100.


cdef double[::1] _get_unique(double[:, ::1] block, DTYPE_intp_t window_i, DTYPE_intp_t window_j):

    cdef:
        Py_ssize_t ii, jj, cc
        double[::1] unique_values = np.zeros(window_i*window_j, dtype='float64') - 9999.0
        int counter = 0
        bint u_found

    for ii in range(0, window_i):

        for jj in range(0, window_j):

            if counter == 0:

                unique_values[counter] = block[ii, jj]
                counter += 1

            else:

                u_found = False

                for cc in range(0, counter):

                    if unique_values[cc] == block[ii, jj]:
                        u_found = True
                        break

                if not u_found:
                    unique_values[counter] = block[ii, jj]
                    counter += 1

    return unique_values[:counter]


cdef double _get_majority(double[:, ::1] block,
                                   DTYPE_intp_t window_i,
                                   DTYPE_intp_t window_j,
                                   double target_value,
                                   double ignore_value,
                                   double[:, ::1] weights,
                                   unsigned int hw):

    cdef:
        Py_ssize_t ii, jj, max_idx, kk
        double[::1] unique_values = _get_unique(block, window_i, window_j)
        int n_unique = unique_values.shape[0]
        DTYPE_uint8_t[::1] count_list = np.zeros(n_unique, dtype='uint8')
        Py_ssize_t samples = window_i * window_j
        Py_ssize_t half_samples = <int>(samples / 2.0)
        double block_value, max_count

    with nogil:

        if target_value != -9999:
            if block[hw, hw] != target_value:
                return block[hw, hw]

        if ignore_value != -9999:

            for ii in range(0, window_i):

                for jj in range(0, window_j):

                    block_value = block[ii, jj]

                    if block_value != ignore_value:

                        for kk in range(0, n_unique):

                            if unique_values[kk] == block_value:
                                count_list[kk] += 1

                                if count_list[kk] > half_samples:
                                    return block_value

                                break

        else:

            for ii in range(0, window_i):

                for jj in range(0, window_j):

                    block_value = block[ii, jj]

                    for kk in range(0, n_unique):

                        if unique_values[kk] == block_value:
                            count_list[kk] += 1

                            if count_list[kk] > half_samples:
                                return block_value

                            break

    # Get the largest count.
    max_count = count_list[0]
    max_idx = 0
    for kk in range(1, n_unique):

        if count_list[kk] > max_count:
            max_idx = copy(kk)

    return unique_values[max_idx]


cdef np.ndarray[DTYPE_uint8_t, ndim=2] link_window(DTYPE_uint8_t[:, ::1] edge_image,
                                                   unsigned int window_size,
                                                   DTYPE_uint8_t[:, ::1] endpoint_image,
                                                   double[:, ::1] gradient_image,
                                                   int min_egm,
                                                   int smallest_allowed_gap,
                                                   int medium_allowed_gap):

    """
    Links endpoints
    """

    cdef:
        Py_ssize_t cij, isub, jsub, iplus, jplus, sub_counter

        unsigned int rows = edge_image.shape[0]
        unsigned int cols = edge_image.shape[1]
        unsigned int half_window, window_size_adj

        DTYPE_int64_t[:, ::1] endpoint_idx
        DTYPE_int64_t[::1] endpoint_row

        unsigned int endpoint_idx_rows, block_size

        DTYPE_uint8_t[:, ::1] edge_block_, ep_block_
        double[:, ::1] gradient_block_

        DTYPE_int16_t[:, ::1] angles_array = np.array([[-135, -90, -45],
                                                       [-180, 0, 180],
                                                       [45, 90, 135]], dtype='int16')

        DTYPE_intp_t[::1] h_r = np.array([2, 2, 2, 2, 2], dtype='intp')
        DTYPE_intp_t[::1] h_c = np.array([0, 1, 2, 3, 4], dtype='intp')
        DTYPE_intp_t[::1] d_c = np.array([4, 3, 2, 1, 0], dtype='intp')

        DTYPE_int16_t[::1] draw_indices = np.array([-1, 0, 1], dtype='int16')

        double[:, ::1] rcc = np.zeros((4, window_size*2), dtype='float64')

        unsigned int max_iters

    endpoint_idx = np.ascontiguousarray(np.argwhere(np.uint8(endpoint_image) == 1))
    endpoint_idx_rows = endpoint_idx.shape[0]

    for cij in range(0, endpoint_idx_rows):

        endpoint_row = endpoint_idx[cij]

        half_window = <int>(window_size / 2.0)

        isub = endpoint_row[0] - half_window
        iplus = endpoint_row[0] + half_window
        jsub = endpoint_row[1] - half_window
        jplus = endpoint_row[1] + half_window

        # Bounds checking
        if (isub < 0) or (iplus >= rows) or (jsub < 0) or (jplus >= cols):

            sub_counter = 1

            max_iters = <int>(half_window / 2.0)
            window_size_adj = window_size

            # Try to find a smaller
            #   window that fits.
            while True:

                # Reduce the window size.
                window_size_adj -= 2

                # Get the new half-window size.
                half_window = <int>(window_size_adj / 2.0)

                isub = endpoint_row[0] - half_window
                iplus = endpoint_row[0] + half_window
                jsub = endpoint_row[1] - half_window
                jplus = endpoint_row[1] + half_window

                sub_counter += 1

                if (0 <= isub < rows) and (0 <= iplus < rows) and (0 <= jsub < cols) and (0 <= jplus < cols):
                    break

                if sub_counter >= max_iters:
                    break

                if sub_counter <= smallest_allowed_gap:
                    break

        if (isub < 0) or (iplus >= rows) or (jsub < 0) or (jplus >= cols):
            continue

        edge_block_ = edge_image[isub:iplus,
                                 jsub:jplus]

        ep_block_ = endpoint_image[isub:iplus,
                                   jsub:jplus]

        gradient_block_ = gradient_image[isub:iplus,
                                         jsub:jplus]

        block_size = edge_block_.shape[0]

        _link_endpoints(edge_block_,
                        ep_block_,
                        gradient_block_,
                        block_size,
                        angles_array,
                        h_r,
                        h_c,
                        d_c,
                        min_egm,
                        smallest_allowed_gap,
                        medium_allowed_gap,
                        rcc,
                        draw_indices,
                        isub+<int>(block_size / 2.),
                        jsub+<int>(block_size / 2.))

        edge_image[isub:iplus, jsub:jplus] = edge_block_
        endpoint_image[isub:iplus, jsub:jplus] = ep_block_

    return np.uint8(edge_image)


cdef unsigned int _orthogonal_opt(double y1,
                                  double x1,
                                  double y2,
                                  double x2,
                                  double center_value,
                                  unsigned int window_size,
                                  double[:, ::1] edge_image_block,
                                  double[:, ::1] rcc2_) nogil:

    """
    Finds the maximum number of pixels along an orthogonal
    line that are less than the center edge value

    Args:
        points of the line
        N: length of the line
        center_value
    """

    cdef:
        int y3_, x3_, y3, x3, y4, x4, li
        unsigned int rc_length_
        unsigned int max_consecutive1 = 0
        unsigned int max_consecutive2 = 0
        double[::1] line_values_

    # First, translate the coordinate
    #   to the Cartesian plane.
    y3_ = <int>y1 - <int>y2
    x3_ = -(<int>x1 - <int>x2)

    # Next, shift the coordinates 90 degrees.
    y3 = x3_
    x3 = y3_ * -1

    # Translate back to a Python grid.
    y3 = <int>(_abs(float(y3) - y1))
    x3 = x3 + <int>x1

    # Find the orthogonal line
    # Draw a line from the center pixel
    #   to the current end coordinate.
    _draw_line(<int>y1, <int>x1, y3, x3, rcc2_)

    # Get the current line length.
    rc_length_ = <int>rcc2_[2, 0]

    # row of zeros, up to the line length
    line_values_ = rcc2_[3, :rc_length_]

    # Extract the values along the line.
    _extract_values(edge_image_block, line_values_, rcc2_, rc_length_)

    # Get the the maximum number
    #   of consecutive pixels
    #   with values less than
    #   the center pixel.
    for li in range(1, rc_length_):

        if line_values_[li] >= center_value:
            break
        else:
            max_consecutive1 += 1

    # TODO: window size-1
    y4 = (window_size - 1) - <int>y3
    x4 = (window_size - 1) - <int>x3

    # Find the orthogonal line
    # Draw a line from the center pixel
    #   to the current end coordinate.
    _draw_line(<int>y1, <int>x1, y4, x4, rcc2_)

    # Get the current line length.
    rc_length_ = <int>rcc2_[2, 0]

    # row of zeros, up to the line length
    line_values_ = rcc2_[3, :rc_length_]

    # Extract the values along the line.
    _extract_values(edge_image_block, line_values_, rcc2_, rc_length_)

    # Get the the maximum number
    #   of consecutive pixels
    #   with values less than
    #   the center pixel.
    for li in range(1, rc_length_):

        if line_values_[li] >= center_value:
            break
        else:
            max_consecutive2 += 1

    return max_consecutive1 + max_consecutive2


cdef void _fill_zeros(double[::1] array2fill, unsigned int array_length) nogil:

    cdef:
        Py_ssize_t fill_idx

    for fill_idx in range(0, array_length):
        array2fill[fill_idx] = 0.


cdef void _get_angle_info(double[:, ::1] edge_image_block,
                          Py_ssize_t iy2,
                          Py_ssize_t jx2,
                          Py_ssize_t iy1,
                          Py_ssize_t jx1,
                          unsigned int window_size,
                          double[:, ::1] rcc1,
                          double[:, ::1] rcc2,
                          double[::1] sums_array,
                          double[::1] sums_array_,
                          double center_value,
                          double[::1] vars_array__) nogil:

    cdef:
        unsigned int rc_length
        unsigned int nc_opt
        double[::1] line_values
        double theta_opt

    # Draw a line from the center pixel
    #   to the current end coordinate.
    _draw_line(iy1, jx1, iy2, jx2, rcc1)

    # Get the current line length.
    rc_length = <int>rcc1[2, 0]

    # Extract a row of zeros, up to
    #   the line length.
    line_values = rcc1[3, :rc_length]

    # Extract the values along the line.
    _extract_values(edge_image_block, line_values, rcc1, rc_length)

    # Ensure the sums holder is empty.
    sums_array_[...] = sums_array

    # Get the sum of edge gradient
    #   magnitudes along the line.
    _get_sum1d_f(line_values, rc_length, sums_array_)

    # Get the angle if there is a new maximum
    #   edge gradient magnitude.
    if (sums_array_[0] / sums_array_[1]) > vars_array__[1]:

        # Get the angle of the line.
        theta_opt = _get_line_angle(rcc1[0, 0],
                                    rcc1[1, 0],
                                    rcc1[0, rc_length-1],
                                    rcc1[1, rc_length-1])

        # Get the maximum number of consecutive
        #   pixels with an edge intensity value
        #   less than the center pixel, found
        #   searching orthogonally to the
        #   optimal edge direction.
        nc_opt = _orthogonal_opt(rcc1[0, 0],
                                 rcc1[1, 0],
                                 rcc1[0, rc_length-1],
                                 rcc1[1, rc_length-1],
                                 center_value,
                                 window_size,
                                 edge_image_block,
                                 rcc2)

        # if nc_opt == 0:
        #     nc_opt = 1

        vars_array__[0] = theta_opt          # theta_opt
        vars_array__[1] = sums_array_[0] / sums_array_[1]    # si_opt
        vars_array__[2] = sums_array_[1]     # n_opt
        vars_array__[3] = nc_opt             # nc_opt
        vars_array__[4] = iy1                # y start
        vars_array__[5] = jx1                # x start
        vars_array__[6] = iy2                # y endpoint
        vars_array__[7] = jx2                # x endpoint


cdef void _optimal_edge_orientation(double[:, ::1] edge_image_block,
                                    unsigned int window_size,
                                    unsigned int half_window,
                                    unsigned int l_size,
                                    double[:, ::1] rcc1,
                                    double[:, ::1] rcc2,
                                    double[::1] vars_array_h,
                                    double[::1] sums_array,
                                    double[::1] sums_array_) nogil:

    """
    Returns:
        max_sum:  Maximum edge value sum along the optimal angle
        line_angle:  The optimal line angle
        n_opt:  The number of pixels along the optimal line angle
        nc_opt:  The maximum number pixels with an edge value less than the center
    """

    cdef:
        Py_ssize_t ii, jj, iy1_, jx1_

        Py_ssize_t end_idx, end_idx_r, pix_idx_r, j__, nc_opt
        unsigned int rc_length
        double si_sum
        double line_angle = 0.
        double si_max = -9999.
        double center_value = edge_image_block[half_window, half_window]
        double[::1] line_values
        double theta_opt

        # l = 4 :: 3-5
        unsigned int min_line_length = <int>((l_size / 2.) + 1)
        unsigned int max_line_length = <int>(l_size + 1)

    #########################
    # FIRST DO LINES WITH
    # ENDPOINTS AT THE CENTER
    #########################
    for ii in range(0, window_size):

        for jj in range(0, window_size):

            if (ii >= min_line_length) and (ii <= max_line_length) and \
                    (jj >= min_line_length) and (jj <= max_line_length):

                continue

            _get_angle_info(edge_image_block,
                            ii,
                            jj,
                            half_window,
                            half_window,
                            window_size,
                            rcc1,
                            rcc2,
                            sums_array,
                            sums_array_,
                            center_value,
                            vars_array_h)

    #####################
    # DO MAX LENGTH LINES
    # OVERLAPPING THE
    # CENTER BY 1
    #####################
    for ii in range(1, window_size-1):

        for jj in range(1, window_size-1):

            if (ii == half_window) and (jj == half_window):
                continue

            if ii == half_window:
                iy1_ = half_window
            else:

                if ii < half_window:
                    iy1_ = half_window + 1
                else:
                    iy1_ = half_window - 1

            if jj == half_window:
                jx1_ = half_window
            else:

                if jj < half_window:
                    jx1_ = half_window + 1
                else:
                    jx1_ = half_window - 1

            _get_angle_info(edge_image_block,
                            ii,
                            jj,
                            iy1_,
                            jx1_,
                            window_size,
                            rcc1,
                            rcc2,
                            sums_array,
                            sums_array_,
                            center_value,
                            vars_array_h)

    #####################
    # DO MAX LENGTH LINES
    # OVERLAPPING THE
    # CENTER BY 2
    #####################
    for ii in range(2, window_size-2):

        for jj in range(2, window_size-2):

            if (ii == half_window) and (jj == half_window):
                continue

            ix1_ = half_window + (half_window - ii)
            jx1_ = half_window + (half_window - jj)

            _get_angle_info(edge_image_block,
                            ii,
                            jj,
                            iy1_,
                            jx1_,
                            window_size,
                            rcc1,
                            rcc2,
                            sums_array,
                            sums_array_,
                            center_value,
                            vars_array_h)


cdef np.ndarray[double, ndim=2] remove_min(double[:, ::1] value_array,
                                           unsigned int window_size):

    """
    Removes values with a lower mean than the surrounding 
    """

    cdef:
        Py_ssize_t i, j, ii, jj

        unsigned int rows = value_array.shape[0]
        unsigned int cols = value_array.shape[1]

        unsigned int half_window = <int>(window_size / 2.)
        unsigned int row_dims = <int>(rows - (half_window * 2.))
        unsigned int col_dims = <int>(cols - (half_window * 2.))

        unsigned int block_counter
        double[:, ::1] block_array
        double center_value, block_mean
        double[:, ::1] out_array = value_array.copy()

        double rt_ = .1

    with nogil:

        for i in range(0, row_dims):

            for j in range(0, col_dims):

                block_array = value_array[i:i+window_size, j:j+window_size]
                center_value = block_array[half_window, half_window]

                block_mean = 0.
                block_counter = 0

                for ii in range(0, window_size):

                    for jj in range(0, window_size):

                        if (ii == half_window) and (jj == half_window):
                            continue

                        if block_array[ii, jj] > 0:

                            block_mean += block_array[ii, jj]
                            block_counter += 1

                if (block_mean / float(block_counter)) * rt_ > center_value:
                    out_array[i, j] = 0

    return np.float64(out_array)


cdef np.ndarray[DTYPE_uint8_t, ndim=2] seg_dist(double[:, ::1] value_array,
                                                unsigned int window_size):

    """
    Recodes a distance transform array by surrounding neighbors 
    """

    cdef:
        Py_ssize_t i, j, ii, jj

        unsigned int rows = value_array.shape[0]
        unsigned int cols = value_array.shape[1]

        unsigned int half_window = <int>(window_size / 2.)
        unsigned int row_dims = <int>(rows - (half_window * 2.))
        unsigned int col_dims = <int>(cols - (half_window * 2.))

        double[:, ::1] block_array
        double center_value, max_neighbor
        DTYPE_uint8_t[:, ::1] out_array = np.zeros((rows, cols), dtype='uint8')

    with nogil:

        for i in range(0, row_dims):

            for j in range(0, col_dims):

                block_array = value_array[i:i+window_size, j:j+window_size]
                center_value = block_array[half_window, half_window]

                if center_value < 1.4:

                    if center_value == 0:
                        out_array[i, j] = 1
                    else:

                        max_neighbor = 0.

                        # Check if any neighbor is >1
                        for ii in range(0, window_size):

                            for jj in range(0, window_size):

                                if (ii == half_window) and (jj == half_window):
                                    continue

                                max_neighbor = _nogil_get_max(max_neighbor, block_array[ii, jj])

                        if max_neighbor < 1.4:
                            out_array[i, j] = 1

    return np.uint8(out_array)


cdef double _get_edge_direction(double[:, ::1] gradient_block,
                                         unsigned int window_size,
                                         unsigned int half_window,
                                         double[:, ::1] rc_,
                                         DTYPE_uint8_t[:, ::1] disk) nogil:

    """
    Finds the direction that has the maximum value
    """

    cdef:
        Py_ssize_t i1, j1, i2, j2
        unsigned int rc_length
        double[::1] line_values
        double line_mean
        double angle = -1.
        double max_mean = 0.

    for i1 in range(0, window_size):

        for j1 in range(0, window_size):

            if disk[i1, j1] == 1:

                i2 = (window_size - i1) - 1
                j2 = (window_size - j1) - 1

                # Draw a line
                _draw_line(i1, j1, i2, j2, rc_)

                # Get the current line length.
                rc_length = <int>rc_[2, 0]

                # Row of zeros, up to the line length
                line_values = rc_[3, :rc_length]

                # Extract the values along the optimum angle.
                _extract_values(gradient_block, line_values, rc_, rc_length)

                # Get the mean along the line.
                line_mean = _get_mean1d(line_values, rc_length)

                if line_mean > max_mean:

                    angle = _get_line_angle(i1, j1, i2, j2)
                    max_mean = line_mean

    return angle


cdef double[:, ::1] get_edge_direction(double[:, ::1] gradient_array,
                                                unsigned int window_size,
                                                DTYPE_uint8_t[:, ::1] disk):

    """
    Finds the edge direction by maximum gradient
    """

    cdef:
        Py_ssize_t i, j

        unsigned int rows = gradient_array.shape[0]
        unsigned int cols = gradient_array.shape[1]
        unsigned int half_window = <int>(window_size / 2.)
        unsigned int row_dims = <int>(rows - (half_window * 2.))
        unsigned int col_dims = <int>(cols - (half_window * 2.))

        double[:, ::1] gradient_block
        double[:, ::1] rc = np.zeros((4, window_size*window_size), dtype='float64')

        double edge_gradient, edge_direction
        double[:, ::1] out_array = np.zeros((rows, cols), dtype='float64')

    with nogil:

        for i in range(0, row_dims):

            for j in range(0, col_dims):

                gradient_block = gradient_array[i:i+window_size, j:j+window_size]

                edge_direction = _get_edge_direction(gradient_block,
                                                     window_size,
                                                     half_window,
                                                     rc,
                                                     disk)

                if npy_isnan(edge_direction):
                    edge_direction = 0.

                out_array[i+half_window, j+half_window] = edge_direction

    return out_array


cdef DTYPE_uint8_t[::1] _extend_endpoints(DTYPE_uint8_t[:, ::1] edge_block_,
                                          unsigned int window_size,
                                          unsigned int hw,
                                          DTYPE_uint8_t[:, ::1] endpoint_block_,
                                          DTYPE_uint8_t[:, ::1] gradient_block_,
                                          DTYPE_uint8_t[:, ::1] direction_block_,
                                          DTYPE_uint8_t[::1] indices_) nogil:

    cdef:
        Py_ssize_t ii, jj
        double min_diff = 100000.
        double end_diff

    for ii in range(0, window_size):

        for jj in range(0, window_size):

            if (ii == hw) and (jj == hw):
                continue

            # Search for an endpoint.
            if endpoint_block_[ii, jj] == 1:

                # Compare the direction to
                #   the center direction.
                if direction_block_[ii, jj] == direction_block_[hw, hw]:

                    end_diff = _abs(float(gradient_block_[ii, jj]) - float(gradient_block_[hw, hw]))

                    # Compare the gradient value.
                    if (end_diff <= min_diff) and (end_diff < 25) and (gradient_block_[ii, jj] >= 25):

                        min_diff = end_diff

                        indices_[0] = ii
                        indices_[1] = jj
                        indices_[2] = 1

    return indices_


cdef np.ndarray[DTYPE_uint8_t, ndim=2] extend_endpoints(DTYPE_uint8_t[:, ::1] edge_array,
                                                        unsigned int window_size,
                                                        DTYPE_uint8_t[:, ::1] endpoint_array,
                                                        DTYPE_uint8_t[:, ::1] gradient_array,
                                                        DTYPE_uint8_t[:, ::1] direction_array):

    """
    Extends endpoints along path of same orientation
    """

    cdef:
        Py_ssize_t i, j

        unsigned int rows = gradient_array.shape[0]
        unsigned int cols = gradient_array.shape[1]
        unsigned int half_window = <int>(window_size / 2.)
        unsigned int row_dims = <int>(rows - (half_window * 2.))
        unsigned int col_dims = <int>(cols - (half_window * 2.))

        DTYPE_uint8_t[:, ::1] edge_block
        DTYPE_uint8_t[:, ::1] endpoint_block
        DTYPE_uint8_t[:, ::1] gradient_block
        DTYPE_uint8_t[:, ::1] direction_block

        DTYPE_uint8_t[::1] indices = np.zeros(3, dtype='uint8')
        DTYPE_uint8_t[::1] indices_ = indices.copy()

    with nogil:

        for i in range(0, row_dims):

            for j in range(0, col_dims):

                edge_block = edge_array[i:i+window_size, j:j+window_size]
                gradient_block = gradient_array[i:i+window_size, j:j+window_size]

                # Avoid endpoints and other edges.
                if (edge_array[half_window, half_window] == 0) and (gradient_block[half_window, half_window] >= 25):

                    indices_[...] = indices

                    endpoint_block = endpoint_array[i:i+window_size, j:j+window_size]
                    direction_block = direction_array[i:i+window_size, j:j+window_size]

                    indices_ = _extend_endpoints(edge_block,
                                                 window_size,
                                                 half_window,
                                                 endpoint_block,
                                                 gradient_block,
                                                 direction_block,
                                                 indices_)

                    if indices_[2] == 1:

                        edge_array[i+half_window, j+half_window] = 1
                        endpoint_array[i+indices_[0], j+indices_[1]] = 0

    return np.uint8(edge_array)


cdef double _get_disk_mean(double[:, ::1] gradient_block,
                                    DTYPE_uint8_t[:, ::1] disk_full,
                                    unsigned int window_size) nogil:

    cdef:
        Py_ssize_t ii, jj
        double disk_mu = 0.
        unsigned int disk_counter = 0

    for ii in range(0, window_size):

        for jj in range(0, window_size):

            if disk_full[ii, jj] == 1:

                disk_mu += gradient_block[ii, jj]
                disk_counter += 1

    return disk_mu / float(disk_counter)


cdef np.ndarray[double, ndim=2] suppression(double[:, ::1] gradient_array,
                                            unsigned int window_size,
                                            double diff_thresh,
                                            DTYPE_uint8_t[:, ::1] disk_full,
                                            DTYPE_uint8_t[:, ::1] disk_edge):

    cdef:
        Py_ssize_t i, j, ii

        unsigned int rows = gradient_array.shape[0]
        unsigned int cols = gradient_array.shape[1]
        unsigned int half_window = <int>(window_size / 2.)
        unsigned int row_dims = <int>(rows - (half_window * 2.))
        unsigned int col_dims = <int>(cols - (half_window * 2.))

        double[:, ::1] gradient_block, direction_block

        double edge_gradient, edge_direction, disk_mean
        double[:, ::1] out_array = np.zeros((rows, cols), dtype='float64')

        unsigned int window_iters

        double[:, ::1] direction_array = get_edge_direction(gradient_array,
                                                                     (window_size*2)+1,
                                                                     disk_edge)

    if half_window == 1:
        window_iters = 2
    else:
        window_iters = half_window

    with nogil:

        for i in range(0, row_dims):

            for j in range(0, col_dims):

                gradient_block = gradient_array[i:i+window_size, j:j+window_size]
                direction_block = direction_array[i:i+window_size, j:j+window_size]

                # Get the EGM mean over the disk.
                disk_mean = _get_disk_mean(gradient_block,
                                           disk_full,
                                           window_size)

                # Get the gradient value
                #   for the current pixel.
                edge_gradient = gradient_block[half_window, half_window]

                # Get the gradient direction
                #   for the current pixel.
                edge_direction = direction_block[half_window, half_window]

                if (edge_gradient >= .1) and (disk_mean >= .05):

                    out_array[i+half_window, j+half_window] = edge_gradient
                    continue

                # The EGM should be >= 125% of the disk mean.
                if edge_gradient >= (disk_mean + (disk_mean * diff_thresh)):

                    for ii in range(1, window_iters):

                        # Get the local maximum gradient
                        #   along the direction.

                        # 0 degrees
                        if (edge_direction >= 337.5) or (edge_direction < 22.5) or (157.5 <= edge_direction < 202.5):

                            if (edge_gradient >= gradient_block[half_window-ii, half_window]) and \
                                    (edge_gradient >= gradient_block[half_window+ii, half_window]):

                                out_array[i+half_window, j+half_window] = edge_gradient

                        # 45 degrees
                        elif (22.5 <= edge_direction < 67.5) or (202.5 <= edge_direction < 247.5):

                            if (edge_gradient >= gradient_block[half_window+ii, half_window+ii]) and \
                                    (edge_gradient >= gradient_block[half_window-ii, half_window-ii]):

                                out_array[i+half_window, j+half_window] = edge_gradient

                        # 90 degrees
                        elif (67.5 <= edge_direction < 112.5) or (247.5 <= edge_direction < 292.5):

                            if (edge_gradient >= gradient_block[half_window, half_window+ii]) and \
                                    (edge_gradient >= gradient_block[half_window, half_window-ii]):

                                out_array[i+half_window, j+half_window] = edge_gradient

                        # 135 degrees
                        elif (112.5 <= edge_direction < 157.5) or (292.5 <= edge_direction < 337.5):

                            if (edge_gradient >= gradient_block[half_window-ii, half_window+ii]) and \
                                    (edge_gradient >= gradient_block[half_window+ii, half_window-ii]):

                                out_array[i+half_window, j+half_window] = edge_gradient

    return np.float64(out_array)


cdef DTYPE_uint8_t[:, ::1] _fill_circles(DTYPE_uint8_t[:, ::1] image_block,
                                         DTYPE_uint8_t[:, ::1] circle_block,
                                         DTYPE_intp_t dims,
                                         double circle_match,
                                         double[::1] rr_,
                                         double[::1] cc_):

    cdef:
        Py_ssize_t i_, j_
        Py_ssize_t overlap_count = 0
        unsigned int fill_shape = rr_.shape[0]

    for i_ in range(0, dims):

        for j_ in range(0, dims):

            if (image_block[i_, j_] == 1) and (circle_block[i_, j_] == 1):
                overlap_count += 1

    if overlap_count >= circle_match:
        _fill_block(image_block, rr_, cc_, 1, fill_shape)

    return image_block


cdef tuple get_circle_locations(DTYPE_uint8_t[:, ::1] circle_block, int window_size):

    cdef:
        Py_ssize_t i_, j_
        Py_ssize_t counter = 0
        double[::1] rr = np.zeros(window_size*window_size, dtype='float64')
        double[::1] cc = rr.copy()

    for i_ in range(0, window_size):

        for j_ in range(0, window_size):

            if circle_block[i_, j_] == 1:

                rr[counter] = i_
                cc[counter] = j_

    return rr[:counter], cc[:counter]


cdef np.ndarray[DTYPE_uint8_t, ndim=2] fill_circles(DTYPE_uint8_t[:, ::1] image_array, list circle_list):

    """
    Fills circles
    """

    cdef:
        int rows = image_array.shape[0]
        int cols = image_array.shape[1]
        Py_ssize_t i, j, ci
        int half_window
        int n_circles = len(circle_list)
        DTYPE_intp_t window_size
        int row_dims
        int col_dims
        DTYPE_uint8_t[:, ::1] circle
        DTYPE_uint8_t circle_sum
        double required_percent = .3
        double circle_match
        double[::1] rr, cc

    for ci in range(0, n_circles):

        circle = circle_list[ci]

        window_size = circle.shape[0]

        half_window = int(window_size / 2)
        row_dims = rows - (half_window * 2)
        col_dims = cols - (half_window * 2)

        # Get the circle total
        circle_sum = _get_sum_uint8(circle, window_size, window_size)

        # Get the required percentage.
        circle_match = float(circle_sum) * required_percent

        rr, cc = get_circle_locations(circle, window_size)

        for i in range(0, row_dims):

            for j in range(0, col_dims):

                image_array[i:i+window_size, j:j+window_size] = _fill_circles(image_array[i:i+window_size,
                                                                                          j:j+window_size],
                                                                              circle,
                                                                              window_size,
                                                                              circle_match,
                                                                              rr,
                                                                              cc)

    return np.uint8(image_array)


cdef double _fill_peaks(double[:, ::1] image_block,
                                 unsigned int window_size,
                                 unsigned int half_window,
                                 unsigned int upper_thresh) nogil:

    cdef:
        Py_ssize_t ii, jj
        double bcv = image_block[half_window, half_window]
        double bnv
        double block_mean = 0.
        unsigned int block_counter = 0

    for ii in range(0, window_size):

        for jj in range(0, window_size):

            if (ii == half_window) and (jj == half_window):
                continue

            bnv = image_block[ii, jj]

            # Less than the center value?
            if bnv < bcv:

                block_mean += bnv
                block_counter += 1

    if block_counter >= upper_thresh:

        # Return the mean of the neighbors
        return block_mean / float(block_counter)

    else:

        # Return the original center value
        return bcv


cdef np.ndarray[double, ndim=2] fill_peaks(double[:, ::1] image2fill,
                                                    unsigned int window_size,
                                                    int iterations):

    """Fills peaks"""

    cdef:
        Py_ssize_t i, j, iteration
        int rows = image2fill.shape[0]
        int cols = image2fill.shape[1]
        unsigned int half_window = <int>(window_size / 2.)
        unsigned int row_dims = rows - (half_window * 2)
        unsigned int col_dims = cols - (half_window * 2)
        double[:, ::1] out_array
        unsigned int upper_thresh

    if window_size == 3:
        upper_thresh = (window_size * 2) + 1
    else:
        upper_thresh = (window_size * 2) + ((window_size - 2) * 2)

    for iteration in range(0, iterations):

        out_array = np.zeros((rows, cols), dtype='float64')

        with nogil:

            for i in range(0, row_dims):

                for j in range(0, col_dims):

                    out_array[i+half_window,
                              j+half_window] = _fill_peaks(image2fill[i:i+window_size,
                                                                      j:j+window_size],
                                                           window_size,
                                                           half_window,
                                                           upper_thresh)

            image2fill[...] = out_array

    return np.float64(image2fill)


cdef double _fill_basins(double[:, ::1] image_block,
                                  unsigned int window_size,
                                  unsigned int half_window,
                                  unsigned int upper_thresh) nogil:

    cdef:
        Py_ssize_t ii, jj
        double bcv = image_block[half_window, half_window]
        double bnv
        double block_mean = 0.
        unsigned int block_counter = 0

    for ii in range(0, window_size):

        for jj in range(0, window_size):

            if (ii == half_window) and (jj == half_window):
                continue

            bnv = image_block[ii, jj]

            # Greater than the center value?
            if bnv > bcv:

                block_mean += bnv
                block_counter += 1

    if block_counter >= upper_thresh:

        # Return the mean of the neighbors
        return block_mean / float(block_counter)

    else:

        # Return the original center value
        return bcv


cdef double _get_proba_mean(double[:, ::1] proba_block___,
                                     double[:, ::1] weight_sums___,
                                     unsigned int window_size) nogil:

    """Calculates the weighted mean of the relaxed probability sums"""

    cdef:
        Py_ssize_t ii, jj
        double proba_sum = 0.
        double weight_sum = 0.

    for ii in range(0, window_size):

        for jj in range(0, window_size):

            proba_sum += proba_block___[ii, jj]
            weight_sum += weight_sums___[ii, jj]

    return proba_sum / weight_sum


cdef void _block_weighted_sum(double[:, ::1] proba_block__,
                              double[:, ::1] probs_layer,
                              unsigned int window_size,
                              double pls,
                              double[:, ::1] dist_weights,
                              double[:, ::1] weight_sums__) nogil:

    """Calculates the weighted sum of posterior probabilities"""

    cdef:
        Py_ssize_t ii, jj, iii, jjj
        double weight

    for ii in range(0, window_size):

        for jj in range(0, window_size):

            weight = pls * dist_weights[ii, jj]

            # posterior class probability x class uncertainty x distance weight
            proba_block__[ii, jj] += (probs_layer[ii, jj] * weight)

            weight_sums__[ii, jj] += weight


cdef double _k_prob(double[:, :, ::1] probs,
                             double[:, ::1] proba_block_,
                             Py_ssize_t current_band,
                             unsigned int bands,
                             unsigned int window_size,
                             double[:, ::1] compatibility_matrix,
                             double[:, ::1] dist_weights,
                             double[:, ::1] weight_sums_) nogil:

    cdef:
        Py_ssize_t class_iter
        double[:, ::1] current_layer

    # Iterate over each class
    for class_iter in range(0, bands):

        # The current class probability window
        current_layer = probs[class_iter, :, :]

        # --------------------
        # Get the weighted sum
        # --------------------

        # Add the current class probabilities,
        #   weighted by the uncertainty.
        _block_weighted_sum(proba_block_,
                            current_layer,
                            window_size,
                            compatibility_matrix[current_band, class_iter],
                            dist_weights,
                            weight_sums_)

    # Get the weighted mean
    return _get_proba_mean(proba_block_,
                           weight_sums_,
                           window_size)


cdef double[:, ::1] _create_weights(unsigned int window_size,
                                             unsigned int half_window):

    cdef:
        Py_ssize_t ri, rj, ri2, rj2
        double[:, ::1] dist_weights = np.zeros((window_size, window_size), dtype='float64')
        double rcm = float(half_window)
        double max_distance

    for ri in range(0, window_size):
        for rj in range(0, window_size):
            dist_weights[ri, rj] = _euclidean_distance(float(rj), rcm, float(ri), rcm)

    max_distance = dist_weights[0, 0]

    for ri2 in range(0, window_size):
        for rj2 in range(0, window_size):

            if dist_weights[ri2, rj2] == 0:
                dist_weights[ri2, rj2] = 1.
            else:
                dist_weights[ri2, rj2] = 1.0 - (dist_weights[ri2, rj2] / max_distance)

    return dist_weights


cdef np.ndarray[double, ndim=3] plr(double[:, :, ::1] proba_array,
                                             unsigned int window_size,
                                             unsigned int iterations,
                                             double[:, ::1] uncertainties):

    """
    Posterior-probability Label Relaxation
    
    Args:
        proba_array (3d array): The class posterior probabilities.
        window_size (int): The moving window size, in pixels.
        iterations (int): The number of relaxation iterations.
        uncertainties (dict): A dictionary with class:class uncertainties. 
        
    Process:
        proba_q:
            The Q_i vector equals the sum of probabilities over all classes and neighborhoods.
    """

    cdef:
        Py_ssize_t i, j, iteration, band, bci, bcj
        unsigned int bands = proba_array.shape[0]
        unsigned int rows = proba_array.shape[1]
        unsigned int cols = proba_array.shape[2]
        unsigned int half_window = <int>(window_size / 2.)
        unsigned int row_dims = rows - (half_window * 2)
        unsigned int col_dims = cols - (half_window * 2)
        double[:, :, ::1] out_array = proba_array.copy()
        double[:, ::1] block_proba = np.zeros((window_size, window_size), dtype='float64')
        double[:, ::1] proba_block = block_proba.copy()
        double[:, ::1] weight_sums = block_proba.copy()
        double[:, :, ::1] proba_block_3d
        double[:, ::1] dist_weights = _create_weights(window_size, half_window)
        double[:, ::1] compatibility_matrix = np.zeros((bands, bands), dtype='float64')
        double proba_q, proba_p

    # TODO
    # -------------------------------
    # Create the compatibility matrix
    # -------------------------------

    if uncertainties.shape[0] == bands:
        compatibility_matrix[...] = uncertainties
    else:

        for bci in range(0, bands):

            for bcj in range(0, bands):

                if bci == bcj:
                    compatibility_matrix[bci, bcj] = 1.
                else:
                    compatibility_matrix[bci, bcj] = .5

    for iteration in range(0, iterations):

        with nogil:

            for i in range(0, row_dims):

                for j in range(0, col_dims):

                    # Current probabilities
                    proba_block_3d = proba_array[:,
                                                 i:i+window_size,
                                                 j:j+window_size]

                    # Iterate over each class.
                    for band in range(0, bands):

                        # Initiate the block to be filled with zeros.
                        proba_block[...] = block_proba
                        weight_sums[...] = block_proba

                        # Get the weighted mean
                        #   of probabilities.
                        proba_q = _k_prob(proba_block_3d,
                                          proba_block,
                                          band,
                                          bands,
                                          window_size,
                                          compatibility_matrix,
                                          dist_weights,
                                          weight_sums)

                        proba_p = proba_block_3d[band, half_window, half_window]

                        out_array[band, i+half_window, j+half_window] = proba_q * proba_p

            proba_array[...] = out_array

        # Normalize
        # proba_array = np.float64(out_array) / np.float64(out_array).max(axis=0)

    return np.float64(proba_array)


cdef np.ndarray[DTYPE_float64_t, ndim=2] fill_basins(double[:, ::1] image2fill,
                                                     unsigned int window_size,
                                                     int iterations):

    """Fills basins"""

    cdef:
        Py_ssize_t i, j, iteration
        int rows = image2fill.shape[0]
        int cols = image2fill.shape[1]
        unsigned int half_window = <int>(window_size / 2.)
        unsigned int row_dims = rows - (half_window * 2)
        unsigned int col_dims = cols - (half_window * 2)
        double[:, ::1] out_array
        unsigned int upper_thresh

    if window_size == 3:
        upper_thresh = (window_size * 2) + 1
    else:
        upper_thresh = (window_size * 2) + ((window_size - 2) * 2)

    for iteration in range(0, iterations):

        out_array = np.zeros((rows, cols), dtype='float64')

        with nogil:

            for i in range(0, row_dims):

                for j in range(0, col_dims):

                    out_array[i+half_window,
                              j+half_window] = _fill_basins(image2fill[i:i+window_size,
                                                                       j:j+window_size],
                                                            window_size,
                                                            half_window,
                                                            upper_thresh)

            image2fill[...] = out_array

    return np.float64(image2fill)


cdef double _get_window_var(double[:, ::1] e_block_,
                                     double e_mu,
                                     unsigned int window_size,
                                     unsigned int counter) nogil:

    cdef:
        Py_ssize_t ib, jb
        double i_var = 0.

    for ib in range(0, window_size):

        for jb in range(0, window_size):
            i_var += _pow(e_block_[ib, jb] - e_mu, 2.0)

    return i_var / float(counter)


cdef double _get_ones_var(double[:, ::1] w_block_,
                                   double[:, ::1] e_block_,
                                   double e_mu,
                                   unsigned int window_size,
                                   unsigned int ones_counter) nogil:

    cdef:
        Py_ssize_t ib, jb
        double i_var = 0.

    for ib in range(0, window_size):

        for jb in range(0, window_size):

            if w_block_[ib, jb] == 1:
                i_var += _pow(e_block_[ib, jb] - e_mu, 2.0)

    return i_var / float(ones_counter)


cdef double _egm_morph(double[:, ::1] image_block,
                                unsigned int window_size,
                                unsigned int hw,
                                double[:, :, ::1] window_stack,
                                double diff_thresh,
                                double var_thresh) nogil:

    cdef:
        Py_ssize_t ww, ii, jj
        double bcv = image_block[hw, hw]
        unsigned int n_windows = window_stack.shape[0]
        double[:, ::1] w_block
        unsigned int ones_counter, zeros_counter
        double ones_sum, zeros_sum
        double wv, bv, pdiff, edge_mean, edge_var, non_edge_mean, window_mean, window_var
        bint is_edge = False
        bint is_noisy = False

    for ww in range(0, n_windows):

        ones_counter = 0
        zeros_counter = 0

        ones_sum = 0.
        zeros_sum = 0.

        w_block = window_stack[ww, :, :]

        for ii in range(0, window_size):

            for jj in range(0, window_size):

                wv = w_block[ii, jj]
                bv = image_block[ii, jj]

                if wv == 1:

                    ones_sum += bv
                    ones_counter += 1

                else:

                    zeros_sum += bv
                    zeros_counter += 1

        # Get the mean of the window.
        window_mean = (ones_sum + zeros_sum) / float(ones_counter + zeros_counter)

        # Check for a high mean with low variance.
        # TODO: threshold parameters
        if not npy_isnan(window_mean) and (window_mean > .7):

            # Get the window variance.
            window_var = _get_window_var(image_block, window_mean, window_size, ones_counter+zeros_counter)

            if not npy_isnan(window_var) and (window_var < .1):

                is_noisy = True
                break

        if not is_noisy:

            if ones_sum > 0:

                # Get the mean along the edge.
                edge_mean = ones_sum / float(ones_counter)

                # Get the mean of non-edges.
                non_edge_mean = zeros_sum / float(zeros_counter)

                if not npy_isnan(edge_mean) and not npy_isnan(non_edge_mean):

                    # Get the percentage difference between the means.
                    pdiff = _perc_diff(non_edge_mean, edge_mean)

                    if not npy_isnan(pdiff):

                        # Get the variance along the edge.
                        edge_var = _get_ones_var(w_block, image_block, edge_mean, window_size, ones_counter)

                        if (pdiff >= diff_thresh) and (edge_var < var_thresh):

                            is_edge = True

                            break

    if is_edge:
        return edge_mean * 2.
    else:

        # non_edge_mean = _get_zeros_mean(image_block, window_size)
        # return sqrt(non_edge_mean)

        return (bcv / 2.) * -1.


cdef void _replace_with_mean(double[:, ::1] edge_block_,
                             double[:, ::1] bzc,
                             unsigned int window_size) nogil:

    cdef:
        Py_ssize_t ii, jj
        double line_mean = 1000000.
        # unsigned int line_counter = 0

    for ii in range(0, window_size):

        for jj in range(0, window_size):

            if bzc[ii, jj] == 1:

                # line_mean += edge_block_[ii, jj]
                # line_counter += 1
                line_mean = _nogil_get_min(line_mean, edge_block_[ii, jj])

    # line_mean /= float(line_counter)

    for ii in range(0, window_size):

        for jj in range(0, window_size):

            if bzc[ii, jj] == 1:
                edge_block_[ii, jj] = line_mean


cdef double[:, ::1] _edge_tracker(double[:, ::1] edge_block,
                                  unsigned int window_size,
                                  unsigned int hw,
                                  double[:, ::1] bzc,
                                  double[:, ::1] bzcz,
                                  double diff_thresh,
                                  bint force_line,
                                  DTYPE_uint8_t[::1] row_neighbors,
                                  DTYPE_uint8_t[::1] col_neighbors) nogil:

    cdef:

        Py_ssize_t iteration, n, i_, j_
        unsigned int iterations

        double center_value = edge_block[hw, hw]
        double compare_value, n_value, abs_diff, min_diff

        int rr, cc, ri, ci, bi, bj
        unsigned int chi, chj

        bint stop_iters, any_found, min_found

        unsigned int fill_window_size = 3
        unsigned int fill_half_window = <int>(fill_window_size / 2.)
        unsigned int window_dims = fill_window_size - (fill_half_window * 2)
        unsigned int upper_thresh = (fill_window_size * fill_window_size) - 2

    if force_line:
        iterations = 1
    else:
        iterations = 3

    bzc[hw, hw] = 1

    for iteration in range(0, iterations):

        stop_iters = False

        rr, cc = hw, hw

        compare_value = edge_block[hw, hw]

        # 1 = decreasing
        # 2 = increasing
        chi = 0
        chj = 0

        any_found = False

        while True:

            min_diff = diff_thresh
            min_found = False

            # Check neighboring values
            for n in range(0, 8):

                ri = rr + (row_neighbors[n] - 1)
                ci = cc + (col_neighbors[n] - 1)

                if (ri < 0) or (ci < 0):

                    stop_iters = True
                    break

                if (ri >= window_size) or (ci >= window_size):

                    stop_iters = True
                    break

                if force_line:

                    if chi > 0:

                        # Ensure decreasing
                        if chi == 1:
                            if ri > rr:
                                continue

                        # Ensure increasing
                        if chi == 2:
                            if ri < rr:
                                continue

                    if chj > 0:

                        # Ensure decreasing
                        if chj == 1:
                            if ci > cc:
                                continue

                        # Ensure increasing
                        if chj == 2:
                            if ci < cc:
                                continue

                if bzc[ri, ci] == 1:
                    continue

                n_value = edge_block[ri, ci]
                abs_diff = _abs(compare_value - n_value)
                # abs_pdiff = _abs(_perc_diff(compare_value, n_value))

                if abs_diff < min_diff:

                    min_diff = abs_diff

                    bi = ri
                    bj = ci

                    if chi == 0:

                        if bi < rr:
                            chi = 1
                        else:
                            chi = 2

                    if chj == 0:

                        if bj < cc:
                            chj = 1
                        else:
                            chj = 2

                    any_found = True
                    min_found = True

            if not min_found:
                break

            if stop_iters:
                break

            # Track the line
            bzc[bi, bj] = 1

            rr, cc = bi, bj

            # Continue along the path of least resistance.
            compare_value = edge_block[rr, cc]

        if not any_found:
            return edge_block

    for iteration in range(0, 2):

        bzcz[...] = bzc

        for i_ in range(0, window_dims):

            for j_ in range(0, window_dims):

                bzcz[i_+fill_half_window,
                     j_+fill_half_window] = _fill_basins(bzc[i_:i_+fill_window_size,
                                                             j_:j_+fill_window_size],
                                                         fill_window_size,
                                                         fill_half_window,
                                                         upper_thresh)

        bzc[...] = bzcz

    _replace_with_mean(edge_block, bzc, window_size)

    return edge_block


cdef double _get_line_straightness(double[:, ::1] rcc_, unsigned int n) nogil:

    cdef:
        Py_ssize_t a, b, c
        double x_avg = 0.0
        double y_avg = 0.0
        double var_x = 0.0
        double cov_xy = 0.0
        double temp, slope, intercept
        double alpha_sum = 0.0
        double yhat, yrmse
        #double[::1] x_ = rcc_[0, :n]
        #double[::1] y_ = rcc_[1, :n]

    for a in range(0, n):

        x_avg += rcc_[0, a]
        y_avg += rcc_[1, a]

    x_avg /= <double>n
    y_avg /= <double>n

    for b in range(0, n):

        temp = rcc_[0, b] - x_avg
        var_x += _pow(temp, 2.0)
        cov_xy += temp * (rcc_[1, b] - y_avg)

    slope = cov_xy / var_x

    intercept = y_avg - slope*x_avg

    # Get the RMSE between the original and fitted data.
    # slope*x + intercept
    for c in range(0, n):
        yhat = slope*rcc_[0, c] + intercept
        alpha_sum += _pow(rcc_[1, c] - yhat, 2.0)
        #alpha_sum += _abs((rcc_[1, c] - slope*rcc_[0, c] + intercept) / sqrt(1.0 + _pow(slope, 2.0)))

    alpha_sum /= <double>n
    yrmse = sqrt(alpha_sum)

    return yrmse


cdef np.ndarray[DTYPE_float64_t, ndim=2] pixel_continuity(double[:, ::1] image_array,
                                                          unsigned int window_size,
                                                          double min_thresh):

    """
    Calculates continuity from a center pixel
    """

    cdef:
        Py_ssize_t i, j, ii, jj

        unsigned int rows = image_array.shape[0]
        unsigned int cols = image_array.shape[1]

        unsigned int half_window = <int>(window_size / 2.0)
        unsigned int row_dims = rows - (half_window * 2)
        unsigned int col_dims = cols - (half_window * 2)

        double[:, ::1] gradient_block
        DTYPE_int16_t[::1] draw_indices = np.array([-1, 0, 1], dtype='int16')

        double[:, ::1] rcc = np.zeros((3, window_size*2), dtype='float64')
        unsigned int rc_length, max_line_length, max_line_length_
        double line_dev

        double[:, ::1] out_array = np.zeros((rows, cols), dtype='float64')

    with nogil:

        for i in range(0, row_dims):

            for j in range(0, col_dims):

                gradient_block = image_array[i:i+window_size, j:j+window_size]

                max_line_length = 0
                line_dev = 0.0

                # Get the longest line in the window.

                ii = 0
                for jj in range(0, window_size):

                    draw_optimum_line(half_window,
                                      half_window,
                                      ii,
                                      jj,
                                      rcc,
                                      gradient_block,
                                      draw_indices,
                                      min_thresh)

                    # Get the current line length.
                    rc_length = <int>rcc[2, 0]

                    max_line_length_ = int_max(max_line_length, rc_length)

                    if max_line_length_ > max_line_length:

                        max_line_length = max_line_length_

                        # Fit least squares and find the
                        #   deviation from the fit.
                        line_dev = _get_line_straightness(rcc, rc_length)

                jj = 0
                for ii in range(0, window_size):

                    draw_optimum_line(half_window,
                                      half_window,
                                      ii,
                                      jj,
                                      rcc,
                                      gradient_block,
                                      draw_indices,
                                      min_thresh)

                    # Get the current line length.
                    rc_length = <int>rcc[2, 0]

                    max_line_length_ = int_max(max_line_length, rc_length)

                    if max_line_length_ > max_line_length:

                        max_line_length = max_line_length_

                        # Fit least squares and find the
                        #   deviation from the fit.
                        line_dev = _get_line_straightness(rcc, rc_length)

                ii = window_size - 1
                for jj in range(0, window_size):

                    draw_optimum_line(half_window,
                                      half_window,
                                      ii,
                                      jj,
                                      rcc,
                                      gradient_block,
                                      draw_indices,
                                      min_thresh)

                    # Get the current line length.
                    rc_length = <int>rcc[2, 0]

                    max_line_length_ = int_max(max_line_length, rc_length)

                    if max_line_length_ > max_line_length:

                        max_line_length = max_line_length_

                        # Fit least squares and find the
                        #   deviation from the fit.
                        line_dev = _get_line_straightness(rcc, rc_length)

                jj = window_size - 1
                for ii in range(0, window_size):

                    draw_optimum_line(half_window,
                                      half_window,
                                      ii,
                                      jj,
                                      rcc,
                                      gradient_block,
                                      draw_indices,
                                      min_thresh)

                    # Get the current line length.
                    rc_length = <int>rcc[2, 0]

                    max_line_length_ = int_max(max_line_length, rc_length)

                    if max_line_length_ > max_line_length:

                        max_line_length = max_line_length_

                        # Fit least squares and find the
                        #   deviation from the fit.
                        line_dev = _get_line_straightness(rcc, rc_length)

                #out_array[i+half_window, j+half_window] = 1.0 - (max_line_length / <double>window_size)
                out_array[i+half_window, j+half_window] = line_dev

    return np.float64(out_array)


cdef np.ndarray[DTYPE_float64_t, ndim=2] egm_morph(double[:, ::1] image_array,
                                                   unsigned int window_size,
                                                   double diff_thresh,
                                                   bint force_line):

    cdef:
        Py_ssize_t i, j

        unsigned int rows = image_array.shape[0]
        unsigned int cols = image_array.shape[1]

        unsigned int half_window = <int>(window_size / 2.)
        unsigned int row_dims = rows - (half_window * 2)
        unsigned int col_dims = cols - (half_window * 2)

        double[:, ::1] bz = np.zeros((window_size, window_size), dtype='float64')
        double[:, ::1] bzc = bz.copy()
        double[:, ::1] bzcz = bz.copy()

        DTYPE_uint8_t[::1] row_neighbors = np.array([0, 0, 0, 1, 1, 2, 2, 2], dtype='uint8')
        DTYPE_uint8_t[::1] col_neighbors = np.array([0, 1, 2, 0, 2, 0, 1, 2], dtype='uint8')

    with nogil:

        for i in range(0, row_dims):

            for j in range(0, col_dims):

                bzc[...] = bz
                bzcz[...] = bz

                image_array[i:i+window_size,
                            j:j+window_size] = _edge_tracker(image_array[i:i+window_size,
                                                                         j:j+window_size],
                                                             window_size,
                                                             half_window,
                                                             bzc,
                                                             bzcz,
                                                             diff_thresh,
                                                             force_line,
                                                             row_neighbors,
                                                             col_neighbors)

    return np.float64(image_array)


cdef np.ndarray[DTYPE_uint8_t, ndim=2] fill_window(DTYPE_uint8_t[:, :] image_array,
                                                   int window_size,
                                                   int n_neighbors):

    """Fills binary holes"""

    cdef:
        int rows = image_array.shape[0]
        int cols = image_array.shape[1]
        Py_ssize_t i, j, ij
        int half_window = int(window_size / 2.)
        int row_dims = rows - (half_window*2)
        int col_dims = cols - (half_window*2)
        DTYPE_uint8_t[::1] rr
        DTYPE_uint8_t[::1] cc
        list idx_rr, idx_cc

    if n_neighbors == 4:

        rr = np.array([0, 1, 1, 2], dtype='uint8')
        cc = np.array([1, 0, 2, 1], dtype='uint8')

        with nogil:

            for i in range(0, row_dims):
                for j in range(0, col_dims):

                    image_array[i+half_window, j+half_window] = _fill_holes(image_array[i:i+window_size,
                                                                                        j:j+window_size],
                                                                            rr, cc, window_size, n_neighbors)

    elif n_neighbors == 2:

        idx_rr = [np.array([0, 2], dtype='uint8'),
                  np.array([1, 1], dtype='uint8')]

        idx_cc = [np.array([1, 1], dtype='uint8'),
                  np.array([0, 2], dtype='uint8')]

        for ij in range(0, 2):

            rr = idx_rr[ij]
            cc = idx_cc[ij]

            with nogil:

                for i in range(0, row_dims):
                    for j in range(0, col_dims):

                        image_array[i+half_window, j+half_window] = _fill_holes(image_array[i:i+window_size,
                                                                                            j:j+window_size],
                                                                                rr, cc, window_size, n_neighbors)

    return np.uint8(np.asarray(image_array))


cdef double _inhibition(double[:, ::1] block,
                                 DTYPE_uint8_t[:, ::1] corners_block,
                                 double[:, ::1] inhibition_w,
                                 double inhibition_scale,
                                 unsigned int window_size,
                                 unsigned int hw,
                                 unsigned int inhibition_operation) nogil:

    cdef:
        Py_ssize_t ii, jj
        double inhibition_term = 0.
        double center_term = 0.
        double ini_diff
        unsigned int term_counter = 0
        unsigned int center_counter = 0
        bint center_includes_corner = False

    # Check if there is a corner
    #   pixel in the center.
    for ii in range(0, window_size):

        for jj in range(0, window_size):

            if inhibition_w[ii, jj] == 2:

                if corners_block[ii, jj] == 1:

                    center_includes_corner = True
                    break

    if center_includes_corner:
        return 0.

    # Get the weighted average within the local window.
    #   This is the inhibition term, T.
    for ii in range(0, window_size):

        for jj in range(0, window_size):

            if inhibition_w[ii, jj] == 1:

                inhibition_term += block[ii, jj] * inhibition_scale
                term_counter += 1

            elif inhibition_w[ii, jj] == 2:

                center_term += block[ii, jj] * inhibition_scale
                center_counter += 1

    inhibition_term /= float(term_counter) * inhibition_scale
    center_term /= float(center_counter) * inhibition_scale

    ini_diff = inhibition_term - center_term

    if inhibition_operation == 1:

        # The outside term
        #   should be larger.
        if ini_diff < 0:
            ini_diff = 0.
        else:
            ini_diff *= -1.

    elif inhibition_operation == 2:

        # The outside term
        #   should be smaller.
        if ini_diff > 0:
            ini_diff = 0.
        else:
            ini_diff *= -1.

    return ini_diff


cdef np.ndarray[DTYPE_float64_t, ndim=2] inhibition(double[:, ::1] image_array,
                                                    unsigned int window_size,
                                                    double inhibition_scale,
                                                    double[:, ::1] inhibition_w,
                                                    unsigned int iterations,
                                                    DTYPE_uint8_t[:, ::1] corners_array,
                                                    unsigned int inhibition_operation):

    """
    Local EGM inhibition

    References:
        Giuseppi Papari and Patrizio Campisi (2007). Multilevel surround inhibition:
            A biologically inspired contour detector.
    """

    cdef:
        Py_ssize_t i, j

        unsigned int rows = image_array.shape[0]
        unsigned int cols = image_array.shape[1]
        unsigned int half_window = int(window_size / 2.)
        unsigned int row_dims = rows - (half_window*2)
        unsigned int col_dims = cols - (half_window*2)

        double[:, ::1] edge_block
        DTYPE_uint8_t[:, ::1] corners_block

        double[:, ::1] out_array = image_array.copy()

    for iteration in range(0, iterations):

        with nogil:

            for i in range(0, row_dims):

                for j in range(0, col_dims):

                    edge_block = image_array[i:i+window_size, j:j+window_size]
                    corners_block = corners_array[i:i+window_size, j:j+window_size]

                    out_array[i+half_window, j+half_window] -= _inhibition(edge_block,
                                                                           corners_block,
                                                                           inhibition_w,
                                                                           inhibition_scale,
                                                                           window_size,
                                                                           half_window,
                                                                           inhibition_operation)

            if iterations > 1:
                image_array[...] = out_array

    return np.float64(out_array)


cdef double _line_enhance(double[:, ::1] block,
                                   double[:, ::1] inhibition_w,
                                   double inhibition_scale,
                                   unsigned int window_size,
                                   unsigned int hw,
                                   double target_value) nogil:

    cdef:
        Py_ssize_t ii, jj
        double center_value = block[hw, hw]
        double inhibition_term = 0.
        double center_term = 0.
        double ini_diff
        unsigned int term_counter = 0
        unsigned int center_counter = 0

    # Get the weighted average within the local window.
    #   This is the inhibition term, T.
    for ii in range(0, window_size):

        for jj in range(0, window_size):

            if inhibition_w[ii, jj] == 1:

                inhibition_term += block[ii, jj] * inhibition_scale
                term_counter += 1

            elif inhibition_w[ii, jj] == 2:

                center_term += block[ii, jj] * inhibition_scale
                center_counter += 1

    inhibition_term /= float(term_counter) * inhibition_scale
    center_term /= float(center_counter) * inhibition_scale

    ini_diff = center_term - inhibition_term

    # The outside term
    #   should be smaller.
    if ini_diff < 0:
        ini_diff = 0.

    if target_value == -9999.:

        if (center_value + ini_diff) >= 1:
            return 1. - center_value
        else:
            return center_value + ini_diff

    else:

        if center_value >= target_value:
            return center_value
        elif (center_value + ini_diff) >= target_value:
            return target_value - center_value
        else:
            return center_value + ini_diff


cdef np.ndarray[DTYPE_float64_t, ndim=2] line_enhance(double[:, ::1] image_array,
                                                      unsigned int window_size,
                                                      double inhibition_scale,
                                                      double[:, :, ::1] inhibition_w,
                                                      unsigned int iterations,
                                                      double target_value,
                                                      double ignore_value):

    """
    Local line enhancement
    """

    cdef:
        Py_ssize_t i, j, n_disk

        unsigned int rows = image_array.shape[0]
        unsigned int cols = image_array.shape[1]
        unsigned int half_window = int(window_size / 2.)
        unsigned int row_dims = rows - (half_window*2)
        unsigned int col_dims = cols - (half_window*2)

        unsigned int n_disks = inhibition_w.shape[0]

        double[:, ::1] edge_block
        double[:, ::1] disk_block

        double[:, ::1] out_array = image_array.copy()

        double max_intersection, line_intersection

    for iteration in range(0, iterations):

        with nogil:

            for i in range(0, row_dims):

                for j in range(0, col_dims):

                    edge_block = image_array[i:i+window_size, j:j+window_size]

                    if ignore_value != -9999.:

                        if edge_block[half_window, half_window] < ignore_value:
                            continue

                    max_intersection = 0.

                    for n_disk in range(0, n_disks):

                        disk_block = inhibition_w[n_disk, :, :]

                        line_intersection = _line_enhance(edge_block,
                                                          disk_block,
                                                          inhibition_scale,
                                                          window_size,
                                                          half_window,
                                                          target_value)

                        max_intersection = _nogil_get_max(max_intersection, line_intersection)

                    out_array[i+half_window, j+half_window] = max_intersection

            if iterations > 1:
                image_array[...] = out_array

    return np.float64(out_array)


cdef np.ndarray[DTYPE_float64_t, ndim=2] dist_window(double[:, :, ::1] image_array):

    """
    Computes the spectral distance

    Args:
        image_array (3d array): A 3d ndarray, where (L, M, N), L = layers, M = rows, N = columns.
        window_size (int): The window size. Default is 3.
        ignore_value (int or float): A value to ignore in calculations. Default is None.
    """

    cdef:
        Py_ssize_t i, j, w

        unsigned int nlayers = image_array.shape[0]
        unsigned int nrows = image_array.shape[1]
        unsigned int ncols = image_array.shape[2]

        unsigned int window_size = 3

        unsigned int half_window = <int>(window_size / 2.0)
        unsigned int row_dims = <int>(nrows - (half_window*2.0))
        unsigned int col_dims = <int>(ncols - (half_window*2.0))

        DTYPE_int64_t[::1] x_offsets = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype='int64')
        DTYPE_int64_t[::1] y_offsets = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype='int64')
        double[::1] dist_weights = np.array([0.7071, 1.0, 0.7071, 1.0, 1.0, 1.0, 0.7071, 1.0, 0.7071], dtype='float64')
        double weights_sum = 0.0

        double[:, ::1] out_array = np.zeros((nrows, ncols), dtype='float64')

    with nogil:

        for w in range(0, 9):
            weights_sum += dist_weights[w]

        for i in range(0, row_dims):

            for j in range(0, col_dims):

                out_array[i+half_window, j+half_window] += _get_distance(image_array,
                                                                         i, j,
                                                                         nlayers,
                                                                         window_size,
                                                                         half_window,
                                                                         x_offsets,
                                                                         y_offsets,
                                                                         dist_weights,
                                                                         weights_sum)

    return np.float64(out_array)


cdef void _get_direction(double[:, ::1] imarray_,
                         Py_ssize_t ii,
                         Py_ssize_t jj,
                         unsigned int window_size,
                         unsigned int half_window,
                         unsigned int skip_factor,
                         bint is_row,
                         double thresh_hom,
                         double learning_rate,
                         unsigned int t_value,
                         double center_value,
                         bint with_std,
                         double[:, ::1] rc,
                         double[::1] rs) nogil:

    cdef:
        Py_ssize_t ija, lni, lni_f, rc_shape
        double ph_i, d_i, line_sd
        double[::1] line_values
        double max_line_length_ = -1e9
        double min_line_length_ = 1e9

    # Iterate over every other angle
    for ija from 0 <= ija < window_size by skip_factor:

        # Draw a line between the two endpoints.
        if is_row:
            _draw_line(half_window, half_window, ija, t_value, rc)
        else:
            _draw_line(half_window, half_window, t_value, ija, rc)

        # rc_shape = rc.shape[1]
        rc_shape = <int>rc[2, 0]  # the real line length

        # line_values = rc[0].copy()
        line_values = rc[3, :rc_shape]  # row of zeros, up to the line length

        # Extract the values along the line.
        _extract_values_full(imarray_, ii, jj, line_values, rc, rc_shape)

        # The cumulative line length
        lni_f = 0

        # The cumulative score along the line
        ph_i = 0.0

        # learning_rate_ = 1.0

        # Iterate over line values
        for lni in range(0, rc_shape):

            # Break if the line exceeds the threshold
            if ph_i >= thresh_hom:
                break
            else:

                if with_std:

                    # Calculate the line standard deviation
                    if lni_f > 0:

                        line_sd = _get_std1d(line_values[:lni_f+1], lni_f+1)

                        if (line_sd == 0) or npy_isnan(line_sd) or npy_isinf(line_sd):
                            line_sd = 1.0

                    else:
                        line_sd = 1.0

                    ph_i += ((<double>line_values[lni] / learning_rate) / line_sd)

                else:
                    ph_i += (<double>line_values[lni] / learning_rate)

                # Increase the score along the line
                # ph_i += (((1.0 - _abs(<double>line_values[lni] - center_value)) / learning_rate) / line_sd)
                # ph_i += (_abs(<double>line_values[lni] - center_value) / learning_rate))
                # ph_i += ((_abs(<double>line_values[lni] - center_value) / learning_rate) / line_sd)
                # ph_i += (<double>line_values[lni] / learning_rate)
                # ph_i += ((<double>line_values[lni] / learning_rate) / line_sd)

                # Increase the line total
                lni_f += 1

                # if learning_rate_ > 0.01:
                #     learning_rate_ -= learning_rate

                # learning_rate_ = 0.01 if learning_rate_ <= 0.01 else learning_rate_

        # Get the line euclidean distance (from center to end)
        d_i = _euclidean_distance(<double>half_window, rc[1, lni_f], <double>half_window, rc[0, lni_f])

        max_line_length_ = _nogil_get_max(max_line_length_, d_i)
        min_line_length_ = _nogil_get_min(min_line_length_, d_i)

    if (max_line_length_ == 0) or npy_isnan(max_line_length_) or npy_isinf(max_line_length_):
        max_line_length_ = 1.0

    if (min_line_length_ == 0) or npy_isnan(min_line_length_) or npy_isinf(min_line_length_):
        min_line_length_ = 1.0

    rs[0] = _nogil_get_min(min_line_length_, rs[0])
    rs[1] = _nogil_get_max(max_line_length_, rs[1])


cdef void _get_optimal_angle(double[:, ::1] imarray_,
                             Py_ssize_t ii,
                             Py_ssize_t jj,
                             unsigned int window_size,
                             unsigned int half_window,
                             unsigned int skip_factor,
                             unsigned int x0,
                             bint is_row,
                             double[:, ::1] rc,
                             double[::1] rs) nogil:

    cdef:
        Py_ssize_t y0, rc_shape, lni
        double si
        double[::1] line_values

    # Iterate over every other angle
    for y0 from 0 <= y0 < window_size by skip_factor:

        # Draw a line between the two endpoints.
        if is_row:
            _draw_line(half_window, half_window, y0, x0, rc)
        else:
            _draw_line(half_window, half_window, x0, y0, rc)

        rc_shape = <int>rc[2, 0]  # the real line length
        line_values = rc[3, :rc_shape]  # row of zeros, up to the line length

        # Extract the values along the line.
        _extract_values_full(imarray_, ii, jj, line_values, rc, rc_shape)

        # Iterate over line values
        si = 0.0
        for lni in range(0, rc_shape):
            si += line_values[lni]

        # [maximum value, y0, x0, is_row]
        if si > rs[0]:

            rs[0] = si
            rs[1] = <double>y0
            rs[2] = <double>x0

            if is_row:
                rs[3] = 1.0
            else:
                rs[3] = 0.0


cdef void _line_collinearity(double[:, ::1] imarray_,
                             Py_ssize_t ii,
                             Py_ssize_t jj,
                             unsigned int window_size,
                             unsigned int half_window,
                             unsigned int skip_factor,
                             unsigned int x0,
                             bint is_row,
                             double[:, ::1] rc,
                             double[::1] rs) nogil:

    cdef:
        Py_ssize_t y0, rc_shape, lni
        double[::1] line_values
        double center_value = imarray_[ii+half_window, jj+half_window]
        double cos_sum

    # Iterate over every other angle
    for y0 from 0 <= y0 < window_size by skip_factor:

        # Draw a line between the two endpoints.
        if is_row == 1:
            _draw_line(half_window, half_window, y0, x0, rc)
        else:
            _draw_line(half_window, half_window, x0, y0, rc)

        rc_shape = <int>rc[2, 0]  # the real line length

        line_values = rc[3, :rc_shape]  # row of zeros, up to the line length

        # Extract the optimal angle values along the line.
        _extract_values_full(imarray_, ii, jj, line_values, rc, rc_shape)

        cos_sum = 0.0
        for lni in range(0, rc_shape):
            cos_sum += cos(_abs(line_values[lni] - center_value))

        rs[0] = _nogil_get_max(cos_sum, rs[0])
        rs[1] = <double>rc_shape


cdef double _collinearity(double[:, ::1] optimal_angles_,
                          Py_ssize_t i,
                          Py_ssize_t j,
                          unsigned int window_size,
                          unsigned int half_window,
                          unsigned int skip_factor,
                          double[:, ::1] rcc_,
                          double[::1] results_) nogil:

    results_[0] = -1e9

    _line_collinearity(optimal_angles_, i, j, window_size, half_window, skip_factor, 0, True, rcc_, results_)
    _line_collinearity(optimal_angles_, i, j, window_size, half_window, skip_factor, window_size-1, True, rcc_, results_)
    _line_collinearity(optimal_angles_, i, j, window_size, half_window, skip_factor, 0, False, rcc_, results_)
    _line_collinearity(optimal_angles_, i, j, window_size, half_window, skip_factor, window_size-1, False, rcc_, results_)

    return results_[0] / results_[1]


cdef void _find_optimal_angles(double[:, ::1] image_array_,
                               double[:, ::1] optimal_angles_,
                               Py_ssize_t i,
                               Py_ssize_t j,
                               unsigned int window_size,
                               unsigned int half_window,
                               unsigned int skip_factor,
                               double[:, ::1] rcc_,
                               double[::1] results_) nogil:

    cdef:
        unsigned int y0, x0

    results_[0] = -1e9

    # if image_array_[i+half_window, j+half_window] < 0.6:
    #     optimal_angles_[i+half_window, j+half_window] = 0.0
    # else:

    # Find the optimum angle
    _get_optimal_angle(image_array_, i, j, window_size, half_window, skip_factor, 0, True, rcc_, results_)
    _get_optimal_angle(image_array_, i, j, window_size, half_window, skip_factor, window_size-1, True, rcc_, results_)
    _get_optimal_angle(image_array_, i, j, window_size, half_window, skip_factor, 0, False, rcc_, results_)
    _get_optimal_angle(image_array_, i, j, window_size, half_window, skip_factor, window_size-1, False, rcc_, results_)

    y0 = <int>results_[1]
    x0 = <int>results_[2]

    optimal_angles_[i+half_window, j+half_window] = _get_line_angle(0.0, 0.0, <double>(half_window-y0), <double>(x0-half_window))


cdef double _saliency(double[:, ::1] optimal_angles_,
                    Py_ssize_t i,
                    Py_ssize_t j,
                    unsigned int window_size,
                    unsigned int half_window,
                    unsigned int skip_factor,
                    double[:, ::1] rcc_,
                    double[::1] results_) nogil:

    return _collinearity(optimal_angles_,
                  i,
                  j,
                  window_size,
                  half_window,
                  skip_factor,
                  rcc_,
                  results_)


cdef np.ndarray saliency_window(double[:, ::1] image_array,
                                unsigned int window_size,
                                unsigned int skip_factor=2):

    """
    Edge saliency and linearity
    
    Args:
        image_array (ndarray): A 2d ndarray of double (float64) precision.
        window_size (Optional[int]): The window size. Default is 3.
        skip_factor (Optional[int])    
    """

    cdef:
        int rows = image_array.shape[0]
        int cols = image_array.shape[1]
        Py_ssize_t i, j, f
        unsigned int half_window = <int>(window_size / 2.0)
        unsigned int row_dims = <int>(rows - (half_window*2.0))
        unsigned int col_dims = <int>(cols - (half_window*2.0))

        double[:, ::1] out_array = np.zeros((rows, cols), dtype='float64')
        double[:, ::1] optimal_angles = np.zeros((rows, cols), dtype='float64')

        double[:, ::1] rcc = np.zeros((4, window_size), dtype='float64')

        # [maximum value, ija, t_value, is_row]
        double[::1] results = np.zeros(4, dtype='float64')

    with nogil:

        for i in range(0, row_dims):
            for j in range(0, col_dims):

                _find_optimal_angles(image_array,
                                     optimal_angles,
                                     i, j,
                                     window_size,
                                     half_window,
                                     skip_factor,
                                     rcc,
                                     results)

        for i in range(0, row_dims):
            for j in range(0, col_dims):

                out_array[i+half_window, j+half_window] = _saliency(optimal_angles,
                                                                    i, j,
                                                                    window_size,
                                                                    half_window,
                                                                    skip_factor,
                                                                    rcc,
                                                                    results)

    return np.float64(out_array)


cdef double _sfs(double[:, ::1] image_array_,
                 Py_ssize_t ii,
                 Py_ssize_t jj,
                 unsigned int window_size,
                 unsigned int half_window,
                 unsigned int skip_factor,
                 double thresh_hom,
                 double learning_rate,
                 bint with_std,
                 double[:, ::1] rcc_,
                 double[::1] results_,
                 unsigned int sfs_min_max) nogil:

    cdef:
        double center_value = image_array_[ii+half_window, jj+half_window]

    results_[0] = 1e9
    results_[1] = -1e9

    # Rows, 1st column
    _get_direction(image_array_, ii, jj, window_size, half_window, skip_factor, True, thresh_hom, learning_rate, 0, center_value, with_std, rcc_, results_)

    # Rows, last column
    _get_direction(image_array_, ii, jj, window_size, half_window, skip_factor, True, thresh_hom, learning_rate, window_size-1, center_value, with_std, rcc_, results_)

    # Columns, 1st row
    _get_direction(image_array_, ii, jj, window_size, half_window, skip_factor, False, thresh_hom, learning_rate, 0, center_value, with_std, rcc_, results_)

    # Columns, last row
    _get_direction(image_array_, ii, jj, window_size, half_window, skip_factor, False, thresh_hom, learning_rate, window_size-1, center_value, with_std, rcc_, results_)

    if sfs_min_max == 1:
        return results_[0] / _euclidean_distance(<double>half_window, 0.0, <double>half_window, 0.0)
    else:
        return results_[1] / _euclidean_distance(<double>half_window, 0.0, <double>half_window, 0.0)


cdef np.ndarray sfs_window(double[:, ::1] image_array,
                           unsigned int window_size,
                           unsigned int skip_factor=2,
                           double thresh_hom=10.0,
                           double learning_rate=0.1,
                           bint with_std=False,
                           unsigned int sfs_min_max=1,
                           int n_jobs=1):

    """
    Structural feature sets
    
    Args:
        image_array (ndarray): A 2d ndarray of double (float64) precision.
        window_size (Optional[int]): The window size. Default is 3.    
    """

    cdef:
        int rows = image_array.shape[0]
        int cols = image_array.shape[1]
        Py_ssize_t i, j, f
        unsigned int half_window = <int>(window_size / 2.0)
        unsigned int row_dims = <int>(rows - (half_window*2.0))
        unsigned int col_dims = <int>(cols - (half_window*2.0))
        # int nsamples = row_dims * col_dims
        double[:, ::1] out_array = np.zeros((rows, cols), dtype='float64')
        double[:, ::1] rcc = np.zeros((4, window_size), dtype='float64')
        double[::1] results = np.zeros(2, dtype='float64')

    with nogil:

        for i in range(0, row_dims):
            for j in range(0, col_dims):

                out_array[i+half_window, j+half_window] = _sfs(image_array,
                                                               i,
                                                               j,
                                                               window_size,
                                                               half_window,
                                                               skip_factor,
                                                               thresh_hom,
                                                               learning_rate,
                                                               with_std,
                                                               rcc,
                                                               results,
                                                               sfs_min_max)

    return np.float64(out_array)


cdef np.ndarray rgb_window(double[:, :, ::1] image_array, unsigned int window_size):

    """
    Computes focal (moving window) statistics.

    Args:
        image_array (ndarray): A 2d ndarray of double (float64) precision.
        statistic (Optional[str]): The statistic to compute. Default is 'mean'.
            Choices are ['mean', 'min', 'max', 'median', 'majority', 'morph', 'percent', 'sum', 'distance'].
        window_size (Optional[int]): The window size. Default is 3.
        ignore_value (Optional[int or float]): A value to ignore in calculations. Default is None.
        resample (Optional[bool]): Whether to resample to the kernel size. Default is False.
        weights (Optional[ndarray]): Must match ``window_size`` x ``window_size``.
    """

    cdef:
        int dims = image_array.shape[0]
        int rows = image_array.shape[1]
        int cols = image_array.shape[2]
        Py_ssize_t i, j
        int half_window = int(window_size / 2)
        int row_dims = rows - (half_window*2)
        int col_dims = cols - (half_window*2)
        double[:, ::1] out_array

    out_array = np.zeros((rows, cols), dtype='float64')

    for i in xrange(0, row_dims):

        for j in xrange(0, col_dims):

            out_array[i+half_window, j+half_window] = _get_distance_rgb(image_array[:, i:i+window_size, j:j+window_size],
                                                                        window_size, window_size, half_window, dims)

    return np.float64(np.asarray(out_array))


cdef np.ndarray window(double[:, ::1] image_array,
                       str statistic,
                       unsigned int window_size,
                       double target_value,
                       double ignore_value,
                       unsigned int iterations,
                       double[:, ::1] weights,
                       unsigned int skip_block):

    """
    Computes focal (moving window) statistics.
    """

    cdef:
        Py_ssize_t i, j, iters, ic, jc
        unsigned int rows = image_array.shape[0]
        unsigned int cols = image_array.shape[1]
        unsigned int half_window = <int>(window_size / 2.0)
        unsigned int row_dims = rows - (half_window*2)
        unsigned int col_dims = cols - (half_window*2)
        double[:, ::1] out_array
        metric_ptr window_function

    if statistic == 'mean':
        window_function = &_get_mean
    elif statistic == 'min':
        window_function = &_get_min
    elif statistic == 'max':
        window_function = &_get_max
    elif statistic == 'unique-count':
        window_function = &_get_unique_count
    # elif statistic == 'median':
    #     window_function = &_get_median
    # elif statistic == 'majority':
    #     window_function = &_get_majority
    elif statistic == 'percent':
        window_function = &_get_percent
    elif statistic == 'sum':
        window_function = &_get_sum
    # elif statistic == 'morph':
    #     window_function = &_morph_pass
    elif statistic == 'duda':
        window_function = &_duda_operator
    elif statistic == 'std':
        window_function = &_get_std
    else:
        raise ValueError('\n{} is not a supported statistic.\n'.format(statistic))

    if statistic == 'majority':
        out_array = image_array.copy()
    else:

        if skip_block > 0:

            out_array = np.zeros((int(np.ceil(rows / float(skip_block))),
                                  int(np.ceil(cols / float(skip_block)))), dtype='float64')

        else:
            out_array = np.zeros((rows, cols), dtype='float64')

    with nogil:

        for iters in range(0, iterations):

            if skip_block > 0:

                ic = 0
                for i from 0 <= i < rows by skip_block:

                    jc = 0
                    for j from 0 <= j < cols by skip_block:

                        out_array[ic, jc] = window_function(image_array[i:i+skip_block, j:j+skip_block],
                                                            skip_block,
                                                            skip_block,
                                                            target_value,
                                                            ignore_value,
                                                            weights,
                                                            half_window)

                        jc += 1

                    ic += 1

            else:

                for i in range(0, row_dims):

                    for j in range(0, col_dims):

                        out_array[i+half_window, j+half_window] = window_function(image_array[i:i+window_size,
                                                                                              j:j+window_size],
                                                                                  window_size,
                                                                                  window_size,
                                                                                  target_value,
                                                                                  ignore_value,
                                                                                  weights,
                                                                                  half_window)

    return np.float64(out_array)


def moving_window(np.ndarray image_array not None,
                  str statistic='mean',
                  DTYPE_intp_t window_size=3,
                  int skip_block=0,
                  float target_value=-9999.,
                  float ignore_value=-9999.,
                  int iterations=1,
                  weights=None,
                  endpoint_array=None,
                  gradient_array=None,
                  int n_neighbors=4,
                  list circle_list=[],
                  int min_egm=25,
                  int smallest_allowed_gap=3,
                  int medium_allowed_gap=7,
                  int inhibition_ws=3,
                  float inhibition_scale=0.5,
                  corners_array=None,
                  int inhibition_operation=1,
                  int l_size=4,
                  float diff_thresh=0.5,
                  float var_thresh=0.02,
                  bint force_line=False,
                  unsigned int skip_factor=2,
                  double thresh_hom=10.0,
                  double learning_rate=0.1,
                  bint with_std=False,
                  unsigned int sfs_min_max=1):

    """
    Args:
        image_array (2d array): The image to process.
        statistic (Optional[str]): The statistic to apply. Default is 'mean'.

            Choices are ['mean', 'min', 'max', 'median', 'majority', 'percent', 'sum',
                         'link', 'fill-basins', 'fill', 'circles', 'distance'. 'rgb-distance',
                         'inhibition', 'line-enhance', 'duda', 'suppression', 'edge-direction',
                         'extend-endpoints', 'remove-min', 'seg-dist', 'plr', 'unique-count'].

            circles: TODO
            distance: Computes the spectral distance between the center value and neighbors
            duda: TODO
            egm-morph:
            fill-basins: Fills pixels surrounded by higher values
            fill-peaks: Fills pixels surrounded by lower values
            fill: Fill 0s surrounded by 1s
            inhibition: TODO
            line-enhance:
            link: Links binary endpoints
            majority: Majority value within window
            max: Maximum of window
            mean: Mean of window
            median: Median of window
            min: Minimum of window
            percent: Percent of window filled by binary 1s
            plr: Posterior-Probability Label Relaxation
            rgb-distance: TODO
            sum: Sum of window
            unique-count: A count of unique values

        window_size (Optional[int]): The window size (of 1 side). Default is 3.
        skip_block (Optional[int]): A block size skip factor. Default is 0.
        target_value (Optional[int]): A value to target (i.e., only operate on this value).
            Default is -9999, or target all values.
        ignore_value (Optional[int]): A value to ignore in calculations. Default is -9999, or don't ignore any values.
        iterations (Optional[int]): The number of iterations to apply the filter. Default is 1.
        weights (Optional[2d array]): A ``window_size`` x ``window_size`` array of weights. Default is None.
        endpoint_array (Optional[2d array]): The endpoint image with ``statistic``='link'. Default is None.
        gradient_array (Optional[2d array]): The edge gradient image with ``statistic``='link'. Default is None.
        n_neighbors (Optional[int]): The number of neighbors with ``statistic``='fill'. Default is 4.
            Choices are [2, 4].
        circle_list (Optional[list]: A list of circles. Default is [].
        min_egm (Optional[int]): The minimum edge gradient magnitude with statistic='link'.
        smallest_allowed_gap (Optional[int]): The smallest allowed gap size (in pixels) with statistic='link'.
        medium_allowed_gap (Optional[int]): The medium allowed gap size (in pixels) with statistic='link'.
        inhibition_ws (Optional[int]): The window size with statistic='inhibition'.
        inhibition_scale (Optional[float]): The scale with statistic='inhibition'.
        corners_array (Optional[2d array]): The corners array with statistic='inhibition'.
        inhibition_operation (Optional[int]): The inhibition operation with statistic='inhibition'.
        l_size (Optional[int])
        diff_thresh (Optional[float])
        var_thresh (Optional[float])
        force_line (Optional[bool])

    Examples:
        >>> from mpglue import moving_window
        >>>
        >>> # Focal mean
        >>> output = moving_window(in_array,
        >>>                        statistic='mean',
        >>>                        window_size=3)
        >>>
        >>> # Focal max
        >>> output = moving_window(in_array,
        >>>                        statistic='max',
        >>>                        window_size=3)
        >>>
        >>> # Focal min
        >>> output = moving_window(in_array,
        >>>                        statistic='min',
        >>>                        window_size=3)
        >>>
        >>> # Focal majority
        >>> output = moving_window(in_array,
        >>>                        statistic='majority',
        >>>                        window_size=3)
        >>>
        >>> # Focal percentage of binary pixels
        >>> output = moving_window(in_array,
        >>>                        statistic='percent',
        >>>                        window_size=25)
        >>>
        >>> # Focal sum
        >>> output = moving_window(in_array,
        >>>                        statistic='sum',
        >>>                        window_size=15)
        >>>
        >>> # Fill local basins
        >>> output = moving_window(in_array,
        >>>                        statistic='fill-basins',
        >>>                        window_size=5)
        >>>
        >>> # Fill local peaks
        >>> output = moving_window(in_array,
        >>>                        statistic='fill-peaks',
        >>>                        window_size=5)
        >>>
        >>> # Non-maximum suppression
        >>> output = moving_window(in_array,
        >>>                        statistic='suppression',
        >>>                        window_size=5,
        >>>                        diff_thresh=.1)
        >>>
        >>> # Edge gradient direction
        >>> output = moving_window(in_array,
        >>>                        statistic='edge-direction',
        >>>                        window_size=15)

    Returns
    """

    if statistic not in ['circles',
                         'distance',
                         'edge-direction',
                         'egm-morph',
                         'extend-endpoints',
                         'continuity',
                         'duda',
                         'fill-basins',
                         'fill-peaks',
                         'fill',
                         'inhibition',
                         'line-enhance',
                         'link',
                         'max',
                         'majority',
                         'mean',
                         'median',
                         'min',
                         'percent',
                         'plr',
                         'remove-min',
                         'rgb-distance',
                         'seg-dist',
                         'std',
                         'sum',
                         'suppression',
                         'unique-count',
                         'sfs',
                         'saliency']:

        raise ValueError('The statistic {} is not an option.'.format(statistic))

    if not isinstance(weights, np.ndarray):

        if statistic == 'plr':
            weights = np.ones((1, 1), dtype='float64')
        else:
            weights = np.ones((window_size, window_size), dtype='float64')

    if not isinstance(endpoint_array, np.ndarray):
        endpoint_array = np.empty((2, 2), dtype='uint8')

    if not isinstance(gradient_array, np.ndarray):
        gradient_array = np.empty((2, 2), dtype='uint8')

    if statistic == 'link':

        return link_window(np.uint8(np.ascontiguousarray(image_array)),
                           window_size,
                           np.uint8(np.ascontiguousarray(endpoint_array)),
                           np.float64(np.ascontiguousarray(gradient_array)),
                           min_egm,
                           smallest_allowed_gap,
                           medium_allowed_gap)

    elif statistic == 'egm-morph':

        return egm_morph(np.float64(np.ascontiguousarray(image_array)),
                         window_size,
                         diff_thresh,
                         force_line)

    elif statistic == 'continuity':

        return pixel_continuity(np.float64(np.ascontiguousarray(image_array)),
                                window_size,
                                diff_thresh)

    elif statistic == 'remove-min':

        return remove_min(np.float64(np.ascontiguousarray(image_array)),
                           window_size)

    elif statistic == 'seg-dist':

        return seg_dist(np.float64(np.ascontiguousarray(image_array)),
                        window_size)

    elif statistic == 'suppression':

        try:
            import pymorph
        except ImportError:
            raise ImportError('Pymorph must be installed to use inhibition')

        disk1 = np.uint8(pymorph.sedisk(r=int(window_size / 2.),
                                        dim=2,
                                        metric='euclidean',
                                        flat=True,
                                        h=0) * 1)

        disk2 = np.uint8(pymorph.sedisk(r=int((window_size-2) / 2.),
                                        dim=2,
                                        metric='euclidean',
                                        flat=True,
                                        h=0) * 1)

        disk2 = np.pad(disk2,
                       (1, 1),
                       mode='constant',
                       constant_values=(0, 0))

        disk3 = np.where((disk1 == 1) & (disk2 == 1), 0, disk1)

        return suppression(np.float64(np.ascontiguousarray(image_array)),
                           window_size,
                           diff_thresh,
                           np.uint8(np.ascontiguousarray(disk1)),
                           np.uint8(np.ascontiguousarray(disk3)))

    elif statistic == 'edge-direction':

        try:
            import pymorph
        except ImportError:
            raise ImportError('Pymorph must be installed to use inhibition')

        disk1 = np.uint8(pymorph.sedisk(r=int(window_size / 2.),
                                        dim=2,
                                        metric='euclidean',
                                        flat=True,
                                        h=0) * 1)

        disk2 = np.uint8(pymorph.sedisk(r=int((window_size-2) / 2.),
                                        dim=2,
                                        metric='euclidean',
                                        flat=True,
                                        h=0) * 1)

        disk2 = np.pad(disk2,
                       (1, 1),
                       mode='constant',
                       constant_values=(0, 0))

        disk3 = np.where((disk1 == 1) & (disk2 == 1), 0, disk1)

        return np.float64(get_edge_direction(np.float64(np.ascontiguousarray(image_array)),
                                             window_size,
                                             np.uint8(np.ascontiguousarray(disk3))))

    elif statistic == 'extend-endpoints':

        return extend_endpoints(np.uint8(np.ascontiguousarray(image_array)),
                                window_size,
                                np.uint8(np.ascontiguousarray(endpoint_array)),
                                np.uint8(np.ascontiguousarray(gradient_array)),
                                np.uint8(np.ascontiguousarray(weights)))

    elif statistic == 'inhibition':

        try:
            import pymorph
        except ImportError:
            raise ImportError('Pymorph must be installed to use inhibition')

        disk1 = np.uint8(pymorph.sedisk(r=int(inhibition_ws / 2.),
                                        dim=2,
                                        metric='euclidean',
                                        flat=True,
                                        h=0) * 1)

        disk2 = np.uint8(pymorph.sedisk(r=int(window_size / 2.),
                                        dim=2,
                                        metric='euclidean',
                                        flat=True,
                                        h=0) * 1)

        disk1 = np.pad(disk1,
                       (int((window_size - inhibition_ws) / 2), int((window_size - inhibition_ws) / 2)),
                       mode='constant',
                       constant_values=(0, 0))

        disk2[disk1 == 1] = 2

        if not isinstance(corners_array, np.ndarray):
            corners_array = np.zeros((image_array.shape[0], image_array.shape[1]), dtype='uint8')
        else:
            corners_array = np.uint8(np.ascontiguousarray(corners_array))

        return inhibition(np.float64(np.ascontiguousarray(image_array)),
                          window_size,
                          inhibition_scale,
                          np.float64(np.ascontiguousarray(disk2)),
                          iterations,
                          corners_array,
                          inhibition_operation)

    elif statistic == 'line-enhance':

        try:
            import pymorph
        except ImportError:
            raise ImportError('Pymorph must be installed to use inhibition')

        from scipy.ndimage.interpolation import rotate

        disk_array = np.uint8(pymorph.sedisk(r=int(window_size / 2.),
                                             dim=2,
                                             metric='euclidean',
                                             flat=True,
                                             h=0) * 1)

        window_half = int(window_size / 2.)

        if window_size - inhibition_ws == 0:
            inhibition_ws_start = 0
            inhibition_ws_end = window_size
        else:
            inhibition_ws_start = (window_size - inhibition_ws) / 2
            inhibition_ws_end = window_size - inhibition_ws_start

        # The disk holder.
        disk_holder = np.zeros((len(range(0, 180, 15)) + len(range(0, 360, 15)),
                                window_size,
                                window_size),
                               dtype='uint8')

        disk_counter = 0

        # Flat line
        disk_arrayc = disk_array.copy()
        disk_arrayc[window_half, inhibition_ws_start:inhibition_ws_end] = 2

        for degree in range(0, 180, 15):

            disk_holder[disk_counter] = rotate(disk_arrayc, degree, order=1, reshape=False)
            disk_counter += 1

        # Half line
        disk_arrayc = disk_array.copy()
        disk_arrayc[window_half, window_half:window_half+inhibition_ws] = 2

        for degree in range(0, 360, 15):

            disk_holder[disk_counter] = rotate(disk_arrayc, degree, order=1, reshape=False)
            disk_counter += 1

        # Right angle
        # disk_arrayc = disk_array.copy()
        # disk_arrayc[window_half, window_half:window_half+inhibition_ws] = 2
        # disk_arrayc[window_half:window_half+inhibition_ws, window_half] = 2
        #
        # for degree in range(0, 360, 15):
        #     disk_holder[disk_counter] = rotate(disk_arrayc, degree, order=1, reshape=False)
        #     disk_counter += 1

        if not isinstance(corners_array, np.ndarray):
            corners_array = np.zeros((image_array.shape[0], image_array.shape[1]), dtype='uint8')
        else:
            corners_array = np.uint8(np.ascontiguousarray(corners_array))

        return line_enhance(np.float64(np.ascontiguousarray(image_array)),
                            window_size,
                            inhibition_scale,
                            np.float64(np.ascontiguousarray(disk_holder)),
                            iterations,
                            target_value,
                            ignore_value)

    elif statistic == 'fill-basins':

        return fill_basins(np.float64(np.ascontiguousarray(image_array)),
                           window_size,
                           iterations)

    elif statistic == 'fill-peaks':

        return fill_peaks(np.float64(np.ascontiguousarray(image_array)),
                          window_size,
                          iterations)

    elif statistic == 'fill':

        return fill_window(np.uint8(np.ascontiguousarray(image_array)),
                           window_size,
                           n_neighbors)

    elif statistic == 'circles':

        return fill_circles(np.uint8(np.ascontiguousarray(image_array)),
                            circle_list)

    elif statistic == 'plr':

        return plr(np.float64(np.ascontiguousarray(image_array)),
                   window_size,
                   iterations,
                   weights)

    elif statistic == 'distance':
        return dist_window(np.float64(np.ascontiguousarray(image_array)))

    elif statistic == 'rgb-distance':

        return rgb_window(np.float64(np.ascontiguousarray(image_array)),
                          window_size)

    elif statistic == 'saliency':

        return saliency_window(np.float64(np.ascontiguousarray(image_array)),
                               window_size,
                               skip_factor=skip_factor)

    elif statistic == 'sfs':

        return sfs_window(np.float64(np.ascontiguousarray(image_array)),
                          window_size,
                          skip_factor=skip_factor,
                          thresh_hom=thresh_hom,
                          learning_rate=learning_rate,
                          with_std=with_std,
                          sfs_min_max=sfs_min_max)

    else:

        return window(np.float64(np.ascontiguousarray(image_array)),
                      statistic,
                      window_size,
                      target_value,
                      ignore_value,
                      iterations,
                      np.float64(np.ascontiguousarray(weights)),
                      skip_block)
