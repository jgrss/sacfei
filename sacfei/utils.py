from __future__ import division

from future.utils import viewitems
from builtins import int, zip

import concurrent.futures

import os
import itertools

from ._adaptive_threshold import threshold as athreshold
from .pool import pooler
from ._moving_window import moving_window

# from mpglue.raster_tools import create_raster
# from mpglue import moving_window

import numpy as np
import cv2

# SciPy
from scipy.ndimage.measurements import label as nd_label
from scipy.ndimage.measurements import mean as nd_mean
import scipy.stats as sci_stats
from scipy.stats import mode as sci_mode

from sklearn.preprocessing import StandardScaler

# Scikit-image
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_local
from skimage.morphology import remove_small_objects, skeletonize
from skimage.morphology import thin as sk_thin
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.measure import label as sk_label

import pymorph

from mahotas import thin as mthin
from mahotas.morph import hitmiss as mhitmiss
# from tqdm import tqdm
# from joblib import Parallel, delayed


def local_straightness(arr, kernel_filter, w, sigma_color, sigma_space):

    """
    https://ieeexplore-ieee-org.ezproxy.library.uq.edu.au/document/1334256
    https://docs.opencv.org/master/d4/d70/tutorial_anisotropic_image_segmentation_by_a_gst.html

    Example:
        >>> conv_kernels = set_kernel_pairs(methods=['compass'])
        >>> kernel_filter = conv_kernels['compass']['kernels']
        >>> local_straightness(array, kernel_filter, 3, 1, 1)
    """

    diff_x = cv2.filter2D(np.float32(arr),
                          cv2.CV_32F,
                          kernel_filter[1],
                          borderType=cv2.BORDER_CONSTANT)

    diff_y = cv2.filter2D(np.float32(arr),
                          cv2.CV_32F,
                          kernel_filter[0],
                          borderType=cv2.BORDER_CONSTANT)

    diff_xy = diff_x * diff_y
    diff_xx = diff_x * diff_x
    diff_yy = diff_y * diff_y

    c11 = cv2.boxFilter(np.float32(diff_xx), cv2.CV_32F, (w, w))
    c22 = cv2.boxFilter(np.float32(diff_yy), cv2.CV_32F, (w, w))
    c12 = cv2.boxFilter(np.float32(diff_xy), cv2.CV_32F, (w, w))

    # c11 = cv2.bilateralFilter(np.float32(diff_xx), w, sigma_color, sigma_space)
    # c22 = cv2.bilateralFilter(np.float32(diff_yy), w, sigma_color, sigma_space)
    # c12 = cv2.bilateralFilter(np.float32(diff_xy), w, sigma_color, sigma_space)

    gamma_max = (c11 + c22 + np.sqrt((c11 - c22)**2 + 4*c12**2)) / 2.0
    gamma_min = (c11 + c22 - np.sqrt((c11 - c22)**2 + 4*c12**2)) / 2.0

    s = 1.0 - (gamma_min / gamma_max)

    return s


def logistic(x, **params):
    return sci_stats.logistic.cdf(x, **params)


def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-b * (x - a)))


def log_transform(egm, scale=1e-6, logistic_alpha=1.6, logistic_beta=0.5):

    """
    Transforms an EGM to probabilities

    Args:
        egm (2d array)
        scale (Optional[float]): The scaling factor
        logistic_alpha (Optional[float])
        logistic_beta (Optional[float])

    Returns:
        Probabilities (2d array)
    """

    # Mask
    egm[egm == 0] = np.nan
    log_min = np.nanpercentile(np.log(egm * scale), 2)
    egm[np.isnan(egm)] = 0

    # Log transform
    egm_proba = np.where(egm > 0, np.log(egm * scale), log_min)

    # Scale and clip
    r, c = egm_proba.shape

    zegm = np.where(egm_proba.ravel() > log_min)[0]

    scaler = StandardScaler().fit(egm_proba.ravel()[zegm][:, np.newaxis])
    egm_proba = scaler.transform(egm_proba.ravel()[:, np.newaxis]).reshape(r, c)
    egm_proba = rescale_intensity(egm_proba, in_range=(-3, 3), out_range=(-3, 3))

    # CDF
    return logistic(egm_proba,
                    loc=logistic_alpha,
                    scale=logistic_beta)


def bayes(prior_a, prior_b, likelihood):

    """
    Bayes rule

    Args:
        prior_a (float): The class prior probability.
        prior_b (float): The class prior probability.
        likelihood (float)
    """

    posterior = (likelihood * prior_a) / (likelihood * prior_a + prior_b * (1.0 - prior_a))

    posterior[np.isnan(posterior)] = 0

    return posterior


class Params(object):

    def __init__(self, **kwargs):

        for k, v in viewitems(kwargs):
            setattr(self, k, v)


def mopen(array2morph, se, iters=1):

    return cv2.morphologyEx(np.uint8(array2morph),
                            cv2.MORPH_OPEN,
                            se,
                            iterations=iters)


def mclose(array2morph, se, iters=1):

    return cv2.morphologyEx(np.uint8(array2morph),
                            cv2.MORPH_CLOSE,
                            se,
                            iterations=iters)


def merode(array2morph, se, iters=1):

    return cv2.morphologyEx(np.uint8(array2morph),
                            cv2.MORPH_ERODE,
                            se,
                            iterations=iters)


def mdilate(array2morph, se, iters=1):

    return cv2.morphologyEx(np.uint8(array2morph),
                            cv2.MORPH_DILATE,
                            se,
                            iterations=iters)


def closerec(array2morph, se, r=3, iters=5):

    """
    Close by reconstruction

    Args:
        array2morph (2d array)
        se (str)
        r (Optional[int])
        iters (Optional[int])
    """

    if se == 'disk':
        se = np.uint8(pymorph.sedisk(r=r))
    elif se == 'cross':
        se = np.uint8(pymorph.secross(r=r))

    evi2_dist = np.float32(cv2.distanceTransform(np.uint8(np.where(array2morph >= 20, 1, 0)), cv2.DIST_L2, 3))

    seed = np.uint8(np.where(evi2_dist >= 2,
                             cv2.morphologyEx(np.uint8(array2morph),
                                              cv2.MORPH_OPEN,
                                              se,
                                              iterations=1),
                             0))

    im_result = seed.copy()

    for iter in range(0, iters):

        im_dilated = cv2.morphologyEx(np.uint8(im_result),
                                      cv2.MORPH_DILATE,
                                      se,
                                      iterations=1)

        im_rec = np.minimum(im_dilated, array2morph)

        im_result = im_rec.copy()

        if np.allclose(seed, im_rec):
            break

    return im_result


def openrec(array2morph, se, iters=5):

    """
    Open by reconstruction

    Args:
        array2morph (2d array)
        se (2d array)
        iters (Optional[int])
    """

    evi2_dist = np.float32(cv2.distanceTransform(np.uint8(np.where(array2morph >= 20, 1, 0)), cv2.DIST_L2, 3))

    seed = np.uint8(np.where(evi2_dist >= 2,
                             cv2.morphologyEx(np.uint8(array2morph),
                                              cv2.MORPH_OPEN,
                                              se,
                                              iterations=1),
                             0))

    im_result = seed.copy()

    for iter in range(0, iters):

        im_dilated = merode(im_result, se, iters=1)

        im_rec = np.minimum(im_dilated, array2morph)

        im_result = im_rec.copy()

        if np.allclose(seed, im_rec):
            break

    return im_result


def set_kernel_pairs(methods=None):

    """
    Creates 2d convolution kernels

    Args:
        methods (Optional[str list]): Choices are ['compass', 'kirsch', 'prewitt', 'roberts', 'scharr', 'sobel'].

    Returns:
        List of kernel filters
    """

    returned_filters = dict()

    if methods:

        returned_filters['custom'] = dict(kernels=methods,
                                          compass=True)

    methods = ['compass', 'kirsch', 'prewitt', 'roberts', 'sobel']

    # Prewitt compass
    compass_filters = np.array([[[-1, -1, -1],
                                 [1, -2, 1],
                                 [1, 1, 1]],
                                [[-1, -1, 1],
                                 [-1, -2, 1],
                                 [1, 1, 1]],
                                [[-1, 1, 1],
                                 [-1, -2, 1],
                                 [-1, 1, 1]],
                                [[1, 1, 1],
                                 [-1, -2, 1],
                                 [-1, -1, 1]],
                                [[1, 1, 1],
                                 [1, -2, 1],
                                 [-1, -1, -1]],
                                [[1, 1, 1],
                                 [1, -2, -1],
                                 [1, -1, -1]],
                                [[1, 1, -1],
                                 [1, -2, -1],
                                 [1, 1, -1]]], dtype='float32')

    # Sobel
    sobel_filters = np.array([[[1, 2, 0],
                               [2, 0, -2],
                               [0, -2, -1]],
                              [[-1, -2, 0],
                               [-2, 0, 2],
                               [0, 2, 1]],
                              [[0, 2, 1],
                               [-2, 0, 2],
                               [-1, -2, 0]],
                              [[0, -2, -1],
                               [2, 0, -2],
                               [1, 2, 0]],
                              [[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]],
                              [[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]],
                              [[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]],
                              [[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]]], dtype='float32')

    # Scharr
    scharr_filters = np.array([[[10, 3, 0],
                                [3, 0, -3],
                                [0, -3, -10]],
                               [[-10, -3, 0],
                                [-3, 0, 3],
                                [0, 3, 10]],
                               [[0, 3, 10],
                                [-3, 0, 3],
                                [-10, -3, 0]],
                               [[0, -3, -10],
                                [3, 0, -3],
                                [10, 3, 0]],
                               [[-10, 0, 10],
                                [-3, 0, 3],
                                [-10, 0, 10]],
                               [[10, 0, -10],
                                [3, 0, -3],
                                [10, 0, -10]],
                               [[-10, -3, -10],
                                [0, 0, 0],
                                [10, 3, 10]],
                               [[10, 3, 10],
                                [0, 0, 0],
                                [-10, -3, -10]]], dtype='float32')

    # Roberts cross
    roberts_filters = np.array([[[0, -1],
                                 [1, 0]],
                                [[0, 1],
                                 [-1, 0]],
                                [[-1, 0],
                                 [0, 1]],
                                [[1, 0],
                                 [0, -1]]], dtype='float32')

    # Prewitt
    prewitt_filters = np.array([[[1, 1, 1],
                                 [0, 0, 0],
                                 [-1, -1, -1]],
                                [[-1, -1, -1],
                                 [0, 0, 0],
                                 [1, 1, 1]],
                                [[1, 1, 0],
                                 [1, 0, -1],
                                 [0, -1, -1]],
                                [[-1, -1, 0],
                                 [-1, 0, 1],
                                 [0, 1, 1]],
                                [[1, 0, -1],
                                 [1, 0, -1],
                                 [1, 0, -1]],
                                [[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]],
                                [[0, 1, 1],
                                 [-1, 0, 1],
                                 [-1, -1, 0]],
                                [[0, -1, -1],
                                 [1, 0, -1],
                                 [1, 1, 0]]], dtype='float32')

    # Kirsch compass
    kirsch_filters = np.array([[[5, 5, 5],
                                [-3, 0, -3],
                                [-3, -3, -3]],
                               [[5, 5, -3],
                                [5, 0, -3],
                                [-3, -3, -3]],
                               [[5, -3, -3],
                                [5, 0, -3],
                                [5, -3, -3]],
                               [[-3, -3, -3],
                                [5, 0, -3],
                                [5, 5, -3]],
                               [[-3, -3, -3],
                                [-3, 0, -3],
                                [5, 5, 5]],
                               [[-3, -3, -3],
                                [-3, 0, 5],
                                [-3, 5, 5]],
                               [[-3, -3, 5],
                                [-3, 0, 5],
                                [-3, -3, 5]]], dtype='float32')

    if 'compass' in methods:

        returned_filters['compass'] = dict(kernels=compass_filters,
                                           compass=True)

    if 'kirsch' in methods:

        returned_filters['kirsch'] = dict(kernels=kirsch_filters,
                                          compass=True)

    if 'prewitt' in methods:

        returned_filters['prewitt'] = dict(kernels=prewitt_filters,
                                           compass=False)

    if 'roberts' in methods:

        returned_filters['roberts'] = dict(kernels=roberts_filters,
                                           compass=False)

    if 'scharr' in methods:

        returned_filters['scharr'] = dict(kernels=scharr_filters,
                                          compass=False)

    if 'sobel' in methods:

        returned_filters['sobel'] = dict(kernels=sobel_filters,
                                         compass=False)

    return returned_filters


def find_circles(intensity_array, kernel_size):

    """
    Finds circles

    Args:
        intensity_array (2d array)
        kernel_size (int)
    """

    kernel_radius = int(kernel_size / 2.0)

    kernel_circle = np.uint8(pymorph.sedisk(r=kernel_radius,
                                            dim=2,
                                            metric='euclidean',
                                            flat=True,
                                            h=0) * 1)

    kernel_square = np.uint8(pymorph.sebox(r=kernel_radius) * 1)

    circles = cv2.filter2D(np.float32(intensity_array),
                           cv2.CV_32F,
                           kernel_circle,
                           borderType=cv2.BORDER_CONSTANT)

    squares = cv2.filter2D(np.float32(intensity_array),
                           cv2.CV_32F,
                           kernel_square,
                           borderType=cv2.BORDER_CONSTANT)

    diff = circles - squares

    local_max_coords = peak_local_max(diff,
                                      min_distance=kernel_size,
                                      indices=True)

    local_max = np.zeros(intensity_array.shape, dtype='uint8')

    for local_coord in local_max_coords:

        local_coord[0] -= kernel_radius
        local_coord[1] -= kernel_radius

        local_max[local_coord[0]:local_coord[0]+kernel_size,
                  local_coord[1]:local_coord[1]+kernel_size] = kernel_circle

    se = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype='uint8')

    return cv2.morphologyEx(local_max,
                            cv2.MORPH_GRADIENT,
                            se)


def _get_magnitude(image2convolve, kernel_filter):

    """
    Calculates the Edge Gradient Magnitude from x and y derivatives

    Args:
        image2convolve (2d array)
        kernel_filter (tuple)

    Returns:
        EGM as 2d array
    """

    return cv2.magnitude(cv2.filter2D(np.float32(image2convolve),
                                      cv2.CV_32F,
                                      kernel_filter[1],
                                      borderType=cv2.BORDER_CONSTANT),
                         cv2.filter2D(np.float32(image2convolve),
                                      cv2.CV_32F,
                                      kernel_filter[0],
                                      borderType=cv2.BORDER_CONSTANT))


def get_magnitude(im, kernels=None, pad=15):

    """
    Gets the Edge Gradient Magnitude (EGM) over multiple edge kernels

    Args:
        im (2d array)
        kernels (Optional[list]
        pad (Optional[int])

    Returns:
        Gradient edge magnitude as 2d array.
        [Mean EGM] * [Max EGM]
    """

    n_rows, n_cols = im.shape

    # Pad image edges.
    if pad > 0:
        im = np.float32(cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_REFLECT))

    # The convolution kernel pairs
    conv_kernels = set_kernel_pairs(methods=kernels)

    # Mean EGM
    # mag_p = np.zeros((len(conv_kernels), im.shape[0], im.shape[1]), dtype='float32')
    mag_p = np.zeros(im.shape, dtype='float32')

    for kernel_name, kernel_dict in viewitems(conv_kernels):

        kernel_filters = kernel_dict['kernels']

        mag_c = np.zeros(im.shape, dtype='float32')

        if kernel_dict['compass']:

            if isinstance(kernel_filters, list):

                kiter = len(kernel_filters)

                # Get the maximum EGM over all kernel pairs.
                for ki in range(0, kiter):

                    for kw in range(0, 2):

                        # Image convolution
                        temp_egm = cv2.filter2D(np.float32(im),
                                                cv2.CV_32F,
                                                np.array(kernel_filters[ki], dtype='float32')[kw],
                                                borderType=cv2.BORDER_CONSTANT)

                        mag_c = np.maximum(mag_c, temp_egm)

            else:

                # Get the maximum EGM over all kernels.
                for ki in range(0, kernel_filters.shape[0]):

                    # Image convolution
                    temp_egm = cv2.filter2D(np.float32(im),
                                            cv2.CV_32F,
                                            kernel_filters[ki],
                                            borderType=cv2.BORDER_CONSTANT)

                    mag_c = np.maximum(mag_c, temp_egm)

        else:

            if isinstance(kernel_filters, list):

                kiter = len(kernel_filters)

                # Get the maximum EGM over all kernel pairs.
                for ki in range(0, kiter):

                    # EGM
                    temp_egm = _get_magnitude(im, np.array(kernel_filters[ki], dtype='float32'))

                    mag_c = np.maximum(mag_c, temp_egm)

            else:

                kiter = kernel_filters.shape[0]

                # Get the maximum EGM over all kernel pairs.
                for ki in range(0, kiter, 2):

                    # EGM
                    temp_egm = _get_magnitude(im, kernel_filters[ki:ki+2])

                    mag_c = np.maximum(mag_c, temp_egm)

        mag_p += mag_c

    if pad > 0:
        # mag_p = mag_p.mean(axis=0)[pad:n_rows+pad, pad:n_cols+pad] * mag_p.max(axis=0)[pad:n_rows+pad, pad:n_cols+pad]
        mag_p = mag_p[pad:n_rows+pad, pad:n_cols+pad] / len(conv_kernels)
    else:
        # mag_p = mag_p.mean(axis=0) * mag_p.max(axis=0)
        mag_p = mag_p / len(conv_kernels)

    mag_p[np.isnan(mag_p) | np.isinf(mag_p)] = 0.0

    return mag_p


def get_mag_egm(ts_array, ts_r, ts_c, kernels):

    # EGM holder
    mag_egm = np.zeros((ts_array.shape[0], ts_r, ts_c), dtype='float32')

    se = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype='uint8')

    # count = np.zeros((ts_r, ts_c), dtype='uint8')

    # Get the EGM from each day.
    for ti in range(0, ts_array.shape[0]):

        mask = mdilate(np.where(ts_array[ti] == 0, 1, 0), se, iters=10)

        # count[mask == 0] += 1

        # Get the EGM over all 'kernels'.
        magg_ = get_magnitude(ts_array[ti],
                              kernels=kernels,
                              pad=0)

        # magg_[mask == 1] = 0
        magg_[mask == 1] = np.nan

        mag_egm[ti] = magg_

    # Get the mean EGM over all layers
    # mag_egm_mean = mag_egm.sum(axis=0) / np.float32(count)
    mag_egm_mean = np.nanmean(mag_egm, axis=0)
    mag_egm_med = np.nanmedian(mag_egm, axis=0)
    mag_egm_cv = np.nanstd(mag_egm, axis=0) / mag_egm_med
    mag_egm_cv = ((mag_egm_cv + mag_egm_med) / 2.0) * 10000.0

    return mag_egm_mean, mag_egm_cv


def get_mag_dist(ts_array, ts_r, ts_c, cvm):

    # EGM holder
    mag_dist = np.zeros((ts_r, ts_c), dtype='float32')

    # Get the edge distance from each day.
    for ti in range(0, ts_array.shape[0]-3):

        mag_dist_ = moving_window(ts_array[ti:ti+3],
                                  statistic='distance',
                                  window_size=3,
                                  weights=cvm)

        mag_dist += mag_dist_

    return mag_dist / float(ts_array.shape[0]-3)


def _do_clahe(image2adjust, clip_perc, grid_tile):

    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE)

    Args:
        image2adjust (2d array)
        clip_perc (float)
        grid_tile (int)

    Returns:
        CLAHE adjusted 2d array
    """

    clahe = cv2.createCLAHE(clipLimit=clip_perc, tileGridSize=grid_tile)

    return clahe.apply(image2adjust)


def local_hist_eq(image2adjust, clip_percentages=None, grid_tiles=None, method='mean'):

    """
    Computes multi-scale Contrast Limited Adaptive Histogram Equalization (CLAHE)

    Args:
        image2adjust (ndarray): The edge gradient magnitude array to adjust. Should be uint8 data type.
        clip_percentages (Optional[float list]): A list of clip percentages for CLAHE. Default is [1.].
        grid_tiles (Optional[tuple list]): A list of grid tuples for CLAHE. Default is [(16, 16)].
        method (Optional[str]): The aggregation method.

    Returns:
        Adjusted image as 2d array.
    """

    if not clip_percentages:
        clip_percentages = [1.]

    if grid_tiles:
        grid_tiles = [(gk, gk) for gk in grid_tiles]
    else:
        grid_tiles = [(16, 16)]

    rws, cls = image2adjust.shape

    if method == 'mean':
        temp_arr_eq = np.zeros((rws, cls), dtype='uint64')
    elif method == 'median' or method == 'min':
        temp_arr_eq = np.zeros((len(clip_percentages) * len(grid_tiles), rws, cls), dtype='uint64')
        counter = 0

    # Iterate over each clip percentage.
    for clip_perc in clip_percentages:

        # Iterate over each grid tile.
        for grid_tile in grid_tiles:

            # Compute CLAHE and add it to the output array.
            if method == 'mean':

                temp_arr_eq += _do_clahe(image2adjust, clip_perc, grid_tile)

                # temp_arr_eq += rescale_intensity(exposure.equalize_adapthist(image2adjust,
                #                                                                       kernel_size=grid_tile[0],
                #                                                                       clip_limit=clip_perc),
                #                                           in_range=(0., 1.), out_range=(0, 255))

            elif method == 'median' or method == 'min':

                temp_arr_eq[counter] = _do_clahe(image2adjust, clip_perc, grid_tile)
                counter += 1

    # Return the mean CLAHE-adjusted edge gradient magnitude
    if method == 'mean':

        return np.float32(temp_arr_eq / float(len(clip_percentages) * len(grid_tiles))) / 255.0

        # return np.uint8(np.divide(temp_arr_eq, float(len(clip_percentages) * len(grid_tiles))) / 255.)

    elif method == 'median':
        return np.float32(np.median(temp_arr_eq, axis=0) / 255.0)
    elif method == 'min':
        return np.float32(temp_arr_eq.min(axis=0) / 255.0)


def locate_endpoints(edge_image, locations='all'):

    """
    Locates edge endpoints

    Args:
        edge_image (2d array)
        locations (Optional[str]): Choices are ['all', 'small', 'broken'].

    Returns:
        Image endpoints, where endpoints = 1.
    """

    # Setup the endpoint structuring elements for
    #    hit or miss morphology.
    if locations == 'all':

        endpoints = [np.array([[0, 0, 0], [0, 1, 0], [2, 1, 2]], dtype='uint8'),
                     np.array([[0, 0, 0], [0, 1, 2], [0, 2, 1]], dtype='uint8'),
                     np.array([[0, 0, 2], [0, 1, 1], [0, 0, 2]], dtype='uint8'),
                     np.array([[0, 2, 1], [0, 1, 2], [0, 0, 0]], dtype='uint8'),
                     np.array([[2, 1, 2], [0, 1, 0], [0, 0, 0]], dtype='uint8'),
                     np.array([[1, 2, 0], [2, 1, 0], [0, 0, 0]], dtype='uint8'),
                     np.array([[2, 0, 0], [1, 1, 0], [2, 0, 0]], dtype='uint8'),
                     np.array([[0, 0, 0], [2, 1, 0], [1, 2, 0]], dtype='uint8'),
                     np.array([[0, 0, 0], [0, 1, 0], [1, 2, 1]], dtype='uint8'),
                     np.array([[0, 0, 1], [0, 1, 2], [0, 0, 1]], dtype='uint8'),
                     np.array([[1, 2, 1], [0, 1, 0], [0, 0, 0]], dtype='uint8')]

    elif locations == 'small':

        endpoints = [np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]], dtype='uint8'),
                     np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype='uint8'),
                     np.array([[0, 0, 1], [0, 1, 1], [0, 0, 1]], dtype='uint8'),
                     np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype='uint8'),
                     np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype='uint8'),
                     np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype='uint8'),
                     np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype='uint8'),
                     np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype='uint8'),
                     np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]], dtype='uint8'),
                     np.array([[0, 0, 1], [0, 1, 1], [0, 0, 1]], dtype='uint8'),
                     np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype='uint8'),
                     np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype='uint8')]

    elif locations == 'broken':

        endpoints = [np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1]], dtype='uint8'),
                     np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]], dtype='uint8'),
                     np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]], dtype='uint8'),
                     np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype='uint8')]

    end_points = np.zeros(edge_image.shape, dtype='uint8')

    # Find the endpoints.
    for endpoint in endpoints:
        end_points += mhitmiss(np.uint8(edge_image), endpoint)

    end_points[end_points > 1] = 1

    return end_points


def _locate_islands(edge_image):

    """
    Locates single pixel islands

    Args:
        edge_image (2d array)

    Returns:
        Image endpoint islands, where islands = 1.
    """

    # Setup the endpoint structuring elements for
    #    hit or miss morphology.
    endpoint = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype='uint8')

    end_points = np.zeros(edge_image.shape, dtype='uint8')

    end_points += mhitmiss(edge_image, endpoint)

    end_points[end_points > 1] = 1

    return end_points


def _trim_endpoints(edge_image,
                    iterations,
                    locations='all',
                    filter=False,
                    filter_ws=15,
                    filter_pct=.1,
                    skeleton=False):

    """
    Trims unconnected lines, starting from endpoints

    Args:
        edge_image (2d array)
        iterations (int)
        locations (str)
        filter (bool)
        filter_ws (int)
        filter_pct (float)
        skeleton (bool)
    """

    if filter:
        edge_image_sum = moving_window(edge_image, statistic='sum', window_size=filter_ws)

    for iter in range(0, iterations):

        # Locate the endpoints
        ep = locate_endpoints(edge_image, locations=locations)

        # Filter high density areas.
        if filter:
            ep[edge_image_sum >= int((filter_ws * filter_ws) * filter_pct)] = 0

        # Remove the endpoints from the edge image.
        edge_image[ep == 1] = 0

        # Fill small gaps after the first iteration.
        if iter == 0:
            edge_image = moving_window(edge_image, statistic='fill', window_size=3, n_neighbors=2)

    # Remove remaining single pixels.
    ep = _locate_islands(edge_image)

    edge_image[ep == 1] = 0

    if skeleton:
        return _do_skeleton(edge_image)
    else:
        return edge_image


def _link_edge_endpoints(cr, max_gap, mag_image, **kwargs):

    """
    Links edge endpoints

    Args:
         cr (2d array)
         max_gap (int)
         mag_image (2d array)
    """

    # Link endpoints
    cr = moving_window(np.uint8(cr*1),
                       statistic='link',
                       window_size=max_gap,
                       endpoint_array=locate_endpoints(np.uint8(cr*1)),
                       gradient_array=mag_image,
                       **kwargs)

    # Fill broken links
    #   __--__ to
    #   ______
    cr = _trim_endpoints(cr, 1, locations='broken')
    cr = moving_window(cr, statistic='fill', window_size=3)

    # A little cleanup before linking.
    cr = _trim_endpoints(cr, 1, locations='all', filter=True, filter_ws=15, filter_pct=.1)

    # Link endpoints.
    return moving_window(cr * 1,
                         statistic='link',
                         window_size=max_gap,
                         endpoint_array=locate_endpoints(cr * 1),
                         gradient_array=mag_image,
                         **kwargs)


def canny_morphology(value_array, egm_array, l1, l2, k_size, l_egm, link_window):

    """
    Args:
        value_array (2d array): Float32 0-1
        egm_array (2d array): Float32 0-1
        l1 (int): Canny lower threshold.
        l2 (int): Canny upper threshold.
        k_size (int): Canny aperture size.
        l_egm (float): The EGM lower threshold.
        link_window (int): The link window size.
    """

    canny_edge = cv2.Canny(np.uint8(value_array * 255.),
                           l1,
                           l2,
                           apertureSize=k_size,
                           L2gradient=True)

    # canny_edge = moving_window(egm_array,
    #                            window_size=3,
    #                            weights=egd,
    #                            statistic='suppression')

    canny_edge[canny_edge > 0] = 1

    canny_edge = _trim_endpoints(canny_edge, 1, locations='broken')

    # Remove small edge objects.
    # canny_edge = nd_label(canny_edge)[0]
    canny_edge = sk_label(np.uint8(canny_edge), connectivity=2)
    # canny_edge = np.uint64(remove_small_objects(canny_edge, min_size=5, connectivity=1))

    # Remove objects with low EGM.
    props = regionprops(canny_edge, intensity_image=egm_array)
    canny_edge = np.float32(canny_edge)

    for prop in props:
        canny_edge[canny_edge == prop.label] = prop.mean_intensity

    canny_edge[canny_edge <= l_egm] = 0
    canny_edge[canny_edge > 0] = 1

    # Link endpoints
    canny_edge = _trim_endpoints(np.uint8(canny_edge), 1, locations='broken')

    canny_edge = moving_window(np.uint8(canny_edge),
                               statistic='link',
                               window_size=link_window,
                               endpoint_array=locate_endpoints(np.uint8(canny_edge)),
                               gradient_array=egm_array,
                               smallest_allowed_gap=5)

    # Remove small objects.
    # canny_edge = nd_label(np.uint8(canny_edge))[0]
    canny_edge = sk_label(np.uint8(canny_edge), connectivity=2)
    canny_edge = np.uint64(remove_small_objects(canny_edge, min_size=10, connectivity=1))

    # props = regionprops(canny_edge, intensity_image=egm_array)

    # canny_edge = np.float32(canny_edge)

    canny_edge[canny_edge > 0] = 1

    return _trim_endpoints(canny_edge, 1)

    # for prop in props:
    #
    #     if (prop.eccentricity < .4) and (prop.area < 100):
    #         canny_edge[canny_edge == prop.label] = 0
    #
    #     # if ((prop.major_axis_length + .00001) / (prop.minor_axis_length + .00001) < 2) and (prop.area < 100):
    #     #     canny_edge[canny_edge == prop.label] = 0
    #
    # canny_edge[canny_edge > 0] = 1

    # cannycv_r = cv2.threshold(np.uint8(canny_edge), 0, 1, cv2.THRESH_BINARY_INV)[1]
    #
    # dist = cv2.distanceTransform(np.uint8(cannycv_r), cv2.DIST_L2, 3)
    #
    # canny_edge = moving_window(dist, statistic='seg-dist', window_size=3)
    #
    # canny_edge = moving_window(np.uint8(canny_edge),
    #                            statistic='link',
    #                            window_size=link_window,
    #                            endpoint_array=locate_endpoints(np.uint8(canny_edge)),
    #                            gradient_array=egm_array,
    #                            smallest_allowed_gap=5)

    return canny_edge


def _do_skeleton(cr):

    """
    Computes the morphological skeleton

    Args:
        cr (2d array)

    Returns:
        Image skeleton as 2d array
    """

    # Fill holes to keep straighter skeleton lines.
    return np.uint8(skeletonize(moving_window(np.uint8(cr), statistic='fill', window_size=3)))


def morphological_cleanup(cr,
                          min_line_size,
                          theta_45_iters=0,
                          theta_90_iters=0,
                          theta_180_iters=0,
                          pre_thin=False,
                          endpoint_iterations=0,
                          skeleton=False,
                          link_ends=False,
                          egm_array=None,
                          extend_endpoints=False,
                          max_gap=25,
                          min_egm=25,
                          smallest_allowed_gap=3,
                          medium_allowed_gap=7,
                          link_iters=1,
                          link_window_size=7,
                          extend_iters=1,
                          value_array=None):

    """
    A function to morphologically clean binary edges

    Args:
        cr (2d array)
        min_line_size (int)
        theta_45_iters (Optional[int])
        theta_90_iters (Optional[int])
        theta_180_iters (Optional[int])
        pre_thin (Optional[bool])
        endpoint_iterations (Optional[int])
        skeleton (Optional[bool])
        link_ends (Optional[bool])
        egm_array (Optional[2d array]): Edge gradient magnitude
        extend_endpoints (Optional[bool])
        max_gap (Optional[int])
        min_egm (Optional[int])
        smallest_allowed_gap (Optional[int])
        medium_allowed_gap (Optional[int])
        link_iters (Optional[int])
        link_window_size (Optional[int])
        extend_iters (Optional[int])
        value_array (Optional[2d array])

    Returns:
        Morphologically cleaned edges as 2d array
    """

    if isinstance(value_array, np.ndarray):
        low_value_edge_idx = np.where((cr == 1) & (value_array < 0.2))

    if pre_thin:

        # Thin edges with 1 iteration
        cr = pymorph.thin(pymorph.binary(cr), n=1, Iab=pymorph.endpoints())

    # Remove small edge objects.
    # cr = nd_label(cr)[0]
    cr = sk_label(np.uint8(cr), connectivity=2)
    cr = np.uint64(remove_small_objects(cr, min_size=min_line_size, connectivity=1))
    cr[cr > 0] = 1

    # Extend endpoints along
    #   the same gradient
    #   orientation.
    if extend_endpoints:

        # The edge gradient direction
        egd_array = moving_window(egm_array,
                                  window_size=link_window_size,
                                  statistic='edge-direction')

        for iter in range(0, extend_iters):

            cr = moving_window(cr,
                               statistic='extend-endpoints',
                               window_size=3,
                               endpoint_array=locate_endpoints(cr),
                               gradient_array=egm_array*255.,
                               weights=egd_array)

    # Thin edges
    if (theta_180_iters > 0) and (theta_90_iters > 0) and (theta_45_iters > 0):

        # cr = np.uint8(pymorph.thin(pymorph.binary(np.uint8(cr)), theta=180, n=theta_180_iters))
        # cr2 = np.uint8(pymorph.thin(pymorph.binary(np.uint8(cr)), theta=90, n=theta_90_iters))
        # cr3 = np.uint8(pymorph.thin(pymorph.binary(np.uint8(cr)), n=theta_45_iters))
        #
        # cr[(cr2 == 1) | (cr3 == 1)] = 1

        cr = sk_thin(np.uint8(cr), max_iter=1)

    else:

        if theta_180_iters > 0:
            cr = np.uint8(pymorph.thin(pymorph.binary(np.uint8(cr)), theta=180, n=theta_180_iters))

        if theta_90_iters > 0:
            cr = np.uint8(pymorph.thin(pymorph.binary(np.uint8(cr)), theta=90, n=theta_90_iters))

        if theta_45_iters > 0:
            cr = np.uint8(mthin(np.uint8(cr), max_iter=theta_45_iters))

    # Remove small objects again after
    #   thinning and trimming.
    if min_line_size > 0:

        # cr, __ = nd_label(cr)
        cr = sk_label(np.uint8(cr), connectivity=2)
        cr = np.uint64(remove_small_objects(cr, min_size=min_line_size, connectivity=1))
        cr[cr > 0] = 1

    # if skeleton:
    #     crc = _do_skeleton(cr.copy())

    # Link endpoints with small gaps.
    if link_ends:

        for link_iter in range(0, link_iters):

            cr = _link_edge_endpoints(cr,
                                      max_gap,
                                      egm_array,
                                      min_egm=min_egm,
                                      smallest_allowed_gap=smallest_allowed_gap,
                                      medium_allowed_gap=medium_allowed_gap)

            cr = _trim_endpoints(cr, 1)

        # import matplotlib.pyplot as plt
        # cr = _do_skeleton(cr)
        # plt.subplot(121)
        # plt.imshow(crc)
        # plt.subplot(122)
        # plt.imshow(cr)
        # plt.show()
        # import sys
        # sys.exit()

    # Compute the morphological skeleton.
    #   The skeleton is morphological thinning with
    #   infinite iterations.
    if skeleton:
        cr = _do_skeleton(cr)

    # Trim endpoints with ``endpoint_iterations`` iterations.
    if endpoint_iterations > 0:
        cr = _trim_endpoints(cr, endpoint_iterations, skeleton=True)

    # Fill small holes
    if isinstance(value_array, np.ndarray):

        cr[low_value_edge_idx] = 1
        cr = moving_window(cr, statistic='fill', window_size=3, n_neighbors=2)

    # Fill broken links
    #   __--__ to
    #   ______
    cr = _trim_endpoints(cr, 1, locations='broken')
    return moving_window(cr, statistic='fill', window_size=3)


def init_distance(egm_array, threshold):

    """
    Initializes a euclidean distance transform array

    Args:
        egm_array (2d array)
        threshold (float or int)
    """

    # Threshold the EGM into a binary edge/no edge array.
    binary_array = np.uint8(np.where(egm_array < threshold, 1, 0))

    # Get the euclidean distance from edge pixels.
    dist = np.float32(cv2.distanceTransform(binary_array, cv2.DIST_L2, 3))

    dist[dist < 0] = 0
    dist /= dist.max()

    return dist


def init_level_set(egm_array, threshold):

    """
    Initializes a level set array

    Args:
        egm_array (2d array)
        threshold (float or int)
    """

    # Threshold the EGM into a binary edge/no edge array.
    binary_array = np.uint8(np.where(egm_array < threshold, 1, 0))

    # Get the euclidean distance from edge pixels.
    dist = np.float32(cv2.distanceTransform(binary_array, cv2.DIST_L2, 3))

    dist = np.where((binary_array == 1) & (dist > 1), dist, 0)

    binary_array_r = np.uint8(cv2.threshold(binary_array, 0, 1, cv2.THRESH_BINARY_INV)[1])

    dist_r = cv2.distanceTransform(binary_array_r, cv2.DIST_L2, 3)

    return np.where(dist == 0, dist_r * -1., dist)


def multiscale_threshold(egm_array,
                         min_object_size,
                         windows=None,
                         link_ends=False,
                         theta_180_iters=1,
                         theta_90_iters=1,
                         theta_45_iters=1,
                         skeleton=False,
                         endpoint_iterations=1,
                         method='wmean',
                         ignore_thresh=15.0,
                         inverse_dist=True,
                         n_jobs=-1):

    """
    Computes multi-scale adaptive threshold and morphological "cleaning"

    Args:
        egm_array (ndarray):
        min_object_size (int):
        windows (Optional[int list]):
        link_ends (Optional[bool]):
        theta_180_iters (Optional[int]):
        theta_90_iters (Optional[int]):
        theta_45_iters (Optional[int]):
        skeleton (Optional[bool]):
        endpoint_iterations (Optional[int]):
        method (Optional[str]): Choices area ['gaussian', 'mean', 'median', 'weighted'].
        ignore_thresh (Optional[float])
        inverse_dist (Optional[bool])
        n_jobs (Optional[int])

    Returns:
        Binary edges as 2d array
    """

    if not isinstance(windows, list):
        windows = [11, 21, 31, 41, 51, 61, 71]

    # Get the image shape.
    im_rows, im_cols = egm_array.shape

    # Setup the output binary edge array holder.
    thresholded_edges = np.zeros((im_rows, im_cols), dtype='uint8')

    wp = 64
    egm_array = cv2.copyMakeBorder(egm_array, wp, wp, wp, wp, cv2.BORDER_REFLECT)

    for w in windows:

        # Threshold the array with the current window size.

        if method == 'gaussian':

            # The gaussian threshold is a weighted sum of the window,
            #   where the weights are a gaussian window.
            binary_adaptive_m = cv2.adaptiveThreshold(egm_array, 1,
                                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                      cv2.THRESH_BINARY, w, 15.)

        elif method == 'mean-c':

            binary_adaptive_m = cv2.adaptiveThreshold(egm_array, 1,
                                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, w, 15.)

        elif method == 'median':
            binary_adaptive_m = threshold_local(egm_array, w, method=method)

        elif method == 'wmean':

            dist_transform = np.float64(init_distance(egm_array, 30))
            dist_transform = np.float64(closerec(np.uint8(dist_transform*255.0), 'disk', r=3, iters=5))
            dist_transform /= dist_transform.max()

            binary_adaptive_m = athreshold(np.ascontiguousarray(egm_array, dtype='float64'),
                                           w,
                                           ignore_thresh=ignore_thresh,
                                           rt=-25.0,
                                           n_jobs=n_jobs,
                                           method=method,
                                           inverse_dist=inverse_dist,
                                           edge_direction_array=None,
                                           edge_distance_array=dist_transform)

        elif method == 'bernson':

            binary_adaptive_m = athreshold(np.ascontiguousarray(egm_array, dtype='float64'),
                                           w,
                                           ignore_thresh=15.,
                                           rt=-10.,
                                           n_jobs=n_jobs,
                                           method=method)

        elif method == 'niblack':

            binary_adaptive_m = athreshold(np.ascontiguousarray(egm_array, dtype='float64'),
                                           w,
                                           ignore_thresh=15.,
                                           rt=-10.,
                                           k=-.01,
                                           n_jobs=n_jobs,
                                           method=method)

        elif method == 'sauvola':

            binary_adaptive_m = athreshold(np.ascontiguousarray(egm_array, dtype='float64'),
                                           w,
                                           ignore_thresh=15.,
                                           rt=-10.,
                                           k=-.01,
                                           n_jobs=n_jobs,
                                           method=method)

        elif method == 'bradley':

            binary_adaptive_m = athreshold(np.ascontiguousarray(egm_array, dtype='float64'),
                                           w,
                                           ignore_thresh=15.,
                                           rt=1.,
                                           n_jobs=n_jobs,
                                           method=method)

        elif method == 'otsu':

            binary_adaptive_m = athreshold(np.ascontiguousarray(egm_array, dtype='float64'),
                                           w,
                                           ignore_thresh=15.,
                                           rt=1.,
                                           n_jobs=n_jobs,
                                           method=method)

        elif method == '60':

            func = lambda arr: np.percentile(arr, 60)
            binary_adaptive_m = threshold_local(egm_array, w, 'generic', param=func)

        else:
            raise ValueError('The method was not recognized.')

        # Cleanup the binary edges with image morphology.
        thresholded_edges += morphological_cleanup(binary_adaptive_m[wp:-wp, wp:-wp],
                                                   min_object_size,
                                                   theta_180_iters=theta_180_iters,
                                                   theta_90_iters=theta_90_iters,
                                                   theta_45_iters=theta_45_iters,
                                                   skeleton=skeleton,
                                                   endpoint_iterations=endpoint_iterations,
                                                   link_ends=link_ends,
                                                   egm_array=egm_array)

    thresholded_edges[thresholded_edges > 1] = 1

    return thresholded_edges


# def _remove_interior_islands(prop,
#                              min_area_int_,
#                              mean_threshold_,
#                              boundary_mean_,
#                              prop_area_weight_,
#                              bbox_pad,
#                              arows,
#                              acols,
#                              segments_g,
#                              original_binary_edge_g,
#                              se_cross):

def _remove_interior_islands(*args):

    """
    Gets indices to remove interior island objects
    """

    prop, min_area_int_, mean_threshold_, boundary_mean_, prop_area_weight_, bbox_pad, arows, acols, segments_g, original_binary_edge_g, se_cross = list(itertools.chain(*args))

    # mean_threshold_ = 0.2   # The minimum EVI2 threshold allowed for objects
    # boundary_mean_ = 0.25   # The maximum EVI2 threshold allowed for boundaries
    # min_area_int_ = 222     # The minimum pixel count allowed for interior objects

    # Get the bounding box of the current segment.
    min_row, min_col, max_row, max_col = prop.bbox

    # Expand the box.
    min_row = min_row - bbox_pad if (min_row - bbox_pad) > 0 else 0
    max_row = max_row + bbox_pad if (max_row + bbox_pad) < (arows - 1) else arows - 1
    min_col = min_col - bbox_pad if (min_col - bbox_pad) > 0 else 0
    max_col = max_col + bbox_pad if (max_col + bbox_pad) < (acols - 1) else acols - 1

    # Get a subset of the current object.
    labels_sub = segments_g[min_row:max_row, min_col:max_col]

    # Get a subset of the pre-cleaned edges
    if isinstance(original_binary_edge_g, np.ndarray):

        binary_sub = original_binary_edge_g[min_row:max_row, min_col:max_col]

        # Get the count of pre-cleaned
        #   edges in the object.
        binary_edge_count = ((binary_sub == 1) & (labels_sub == prop.label)).sum()

        # Don't include objects half covered by pre-cleaned edges.
        if binary_edge_count >= int(prop.area * prop_area_weight_):

            idx = list(np.where(labels_sub == prop.label))

            idx[0] = idx[0] + min_row
            idx[1] = idx[1] + min_col

            return list(idx[0]), list(idx[1])

    # Don't include objects with low EVI2.
    if hasattr(prop, 'mean_intensity'):

        if prop.mean_intensity < mean_threshold_:

            idx = list(np.where(labels_sub == prop.label))

            idx[0] = idx[0] + min_row
            idx[1] = idx[1] + min_col

            return list(idx[0]), list(idx[1])

    # Get the current object.
    labels_sub_center = np.uint8(np.where(labels_sub == prop.label, 1, 0))

    # Get the boundary labels.
    label_boundary = cv2.morphologyEx(labels_sub_center,
                                      cv2.MORPH_DILATE,
                                      se_cross,
                                      iterations=2) - labels_sub_center

    boundary_idx = np.where(label_boundary == 1)

    # Check if the current object is completely
    #   surrounded by 1-2 other objects.
    if np.any(boundary_idx):

        boundary_values = labels_sub[boundary_idx]

        # The parcel should be surrounded
        #   by other vegetation.
        if boundary_values.mean() >= boundary_mean_:

            unique_boundary_values = list(np.unique(boundary_values))

            if (0 in unique_boundary_values) and (0 < len(unique_boundary_values) <= 2) and (prop.area < min_area_int_):

                idx = list(np.where(labels_sub_center == 1))

                idx[0] = idx[0] + min_row
                idx[1] = idx[1] + min_col

                return list(idx[0]), list(idx[1])

            else:
                return list(), list()

        else:
            return list(), list()

    else:
        return list(), list()


# def _clean_objects(prop,
#                    min_area_,
#                    min_area_int_,
#                    mean_threshold_,
#                    boundary_mean_,
#                    bbox_pad,
#                    arows,
#                    acols,
#                    segments_g,
#                    morphed_sep,
#                    morphed,
#                    se_cross,
#                    se_square):

def _clean_objects(*args):

    """
    Area:
        15m:
            0.1 ha / [(15m x 15m) x 0.0001] = 5 pixels
            5 ha / [(15m x 15m) x 0.0001] = 222 pixels
            10 ha / [(15m x 15m) x 0.0001] = 444 pixels
            20 ha / [(15m x 15m) x 0.0001] = 888 pixels
            5,000 ha / [(15m x 15m) x 0.0001] = 222,222 pixels
            10,000 ha / [(15m x 15m) x 0.0001] = 444,444 pixels
            20,000 ha / [(15m x 15m) x 0.0001] = 888,888 pixels
    """

    prop, min_area_, min_area_int_, mean_threshold_, boundary_mean_, bbox_pad, arows, acols, segments_g, morphed_sep, morphed, se_cross, se_square = list(itertools.chain(*args))

    el_ = []

    # mean_threshold_ = 0.2   # The minimum EVI2 threshold allowed for objects
    # boundary_mean_ = 0.25   # The maximum EVI2 threshold allowed for boundaries
    # min_area_ = 5           # The minimum pixel count allowed for any object
    # max_area_ = 250000      # The maximum pixel count allowed for any object
    # min_area_int_ = 222     # The minimum pixel count allowed for interior objects

    # if prop.area > 10000:
    #     return el_, el_, el_, el_, el_

    if hasattr(prop, 'mean_intensity'):

        if prop.mean_intensity < mean_threshold_:
            return el_, el_, el_, el_, el_

    # Get the bounding box of the current segment.
    min_row, min_col, max_row, max_col = prop.bbox

    # Expand the box.
    min_row = min_row - bbox_pad if (min_row - bbox_pad) > 0 else 0
    max_row = max_row + bbox_pad if (max_row + bbox_pad) < (arows - 1) else arows - 1
    min_col = min_col - bbox_pad if (min_col - bbox_pad) > 0 else 0
    max_col = max_col + bbox_pad if (max_col + bbox_pad) < (acols - 1) else acols - 1

    # Get a subset of the current object.
    labels_sub = segments_g[min_row:max_row, min_col:max_col]
    morphed_sep_sub = morphed_sep[min_row:max_row, min_col:max_col]
    morphed_sub = morphed[min_row:max_row, min_col:max_col]

    # Get the current object.
    labels_sub_center = np.uint8(np.where(labels_sub == prop.label, 1, 0))

    # Get the boundary labels.
    label_boundary = cv2.morphologyEx(labels_sub_center,
                                      cv2.MORPH_DILATE,
                                      se_cross,
                                      iterations=2) - labels_sub_center

    boundary_idx = np.where(label_boundary == 1)

    # Check if the current object is completely
    #   surrounded by 1-3 other objects.
    if np.any(boundary_idx):

        boundary_values = labels_sub[boundary_idx]

        # The parcel should be surrounded
        #   by other vegetation.
        if boundary_values.mean() >= boundary_mean_:

            unique_boundary_values = list(np.unique(boundary_values))

            if (0 in unique_boundary_values) and (0 < len(unique_boundary_values) <= 2) and (prop.area < min_area_int_):
                return el_, el_, el_, el_, el_

    # Morphological closing by reconstruction
    closerec_sub = pymorph.closerec(pymorph.binary(labels_sub_center))

    closerec_sub = merode(closerec_sub, se_cross)
    closerec_sub = mopen(closerec_sub, se_square)

    if (closerec_sub == 1).sum() < min_area_:
        return el_, el_, el_, el_, el_
    else:

        idxs = list(np.where((morphed_sep_sub == 0) & (closerec_sub == 1)))

        idxs[0] = idxs[0] + min_row
        idxs[1] = idxs[1] + min_col

        # Decrease the gaps between land cover
        closerec_sub = cv2.morphologyEx(np.uint8(closerec_sub),
                                        cv2.MORPH_DILATE,
                                        se_cross,
                                        iterations=2)

        idx = list(np.where((morphed_sub == 0) & (closerec_sub == 1)))

        idx[0] = idx[0] + min_row
        idx[1] = idx[1] + min_col

        return list(idxs[0]), list(idxs[1]), \
               list(np.zeros(len(idx[0]), dtype='uint64') + prop.label), \
               list(idx[0]), list(idx[1])


def clean_objects(segments,
                  intensity_array=None,
                  original_binary_edge=None,
                  binary=True,
                  min_object_area=5,
                  min_interior_count=222,
                  mean_threshold=0.2,
                  boundary_mean=0.25,
                  prop_area_weight=0.9,
                  bbox_pad=10,
                  chunk_size=100000,
                  n_jobs=1):

    """
    Cleans objects with morphological operations

    Args:
        segments (2d array): The segmented objects array to be cleaned.
        intensity_array (2d array): The intensity values.
        original_binary_edge (2d array): The original edges as binary.
        binary (Optional[bool]): Whether the input segments are binary (True) or labelled (False). Default is True.
        min_object_area (Optional[int]): The minimum object area.
        min_interior_count (Optional[int]): The minimum pixel count of interior pixels.
        mean_threshold (float): The vegetation index mean threshold.
        boundary_mean (float): The vegetation index boundary threshold.
        prop_area_weight (float): The object property area weighting.
        bbox_pad (Optional[int]): The `regionprops bbox` padding. Default is 10.
        chunk_size (Optional[int]): The chunk size for multiprocessing. Default is 100,000.
        n_jobs (Optional[int]):

    Returns:
        Segments dilated, Segments with separate boundaries
    """

    # global morphed, segments_g, morphed_sep, se_cross, se_square, original_binary_edge_g

    # segments_g = segments
    # original_binary_edge_g = original_binary_edge

    arows, acols = segments.shape

    if binary:
        segments = nd_label(segments)[0]
        # segments = sk_label(np.uint8(segments), connectivity=1)

    morphed_sep = np.zeros((arows, acols), dtype='uint8')
    morphed = np.zeros((arows, acols), dtype='uint64')

    se_cross = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype='uint8')

    se_square = np.array([[1, 1],
                          [1, 1]], dtype='uint8')

    props = regionprops(segments,
                        intensity_image=intensity_array)

    # Clean parcels
    for chi in range(0, len(props), chunk_size):

        data_gen = ((prop_,
                     min_object_area,
                     min_interior_count,
                     mean_threshold,
                     boundary_mean,
                     bbox_pad,
                     arows,
                     acols,
                     segments,
                     morphed_sep,
                     morphed,
                     se_cross,
                     se_square) for prop_ in props[chi:chi+chunk_size])

        cleaned_parcels = []

        with concurrent.futures.ThreadPoolExecutor(n_jobs) as executor:

            for res in executor.map(_clean_objects, data_gen):
                cleaned_parcels.append(res)

        # cleaned_parcels = Parallel(backend='multiprocessing',
        #                            n_jobs=n_jobs)(delayed(_clean_objects)(prop_,
        #                                                                   min_object_area,
        #                                                                   min_interior_count,
        #                                                                   mean_threshold,
        #                                                                   boundary_mean,
        #                                                                   bbox_pad,
        #                                                                   arows,
        #                                                                   acols,
        #                                                                   segments,
        #                                                                   morphed_sep,
        #                                                                   morphed,
        #                                                                   se_cross,
        #                                                                   se_square) for prop_ in props[chi:chi+chunk_size])

        rowidx_sep_list, colidx_sep_list, labels_list, rowidx_list, colidx_list = zip(*cleaned_parcels)

        labels_list = np.array(list(itertools.chain.from_iterable(labels_list)), dtype='uint64')

        # piece together the parcels
        if np.any(labels_list):

            # Objects with separate boundaries
            rowidx_sep_list = np.array(list(itertools.chain.from_iterable(rowidx_sep_list)), dtype='uint64')
            colidx_sep_list = np.array(list(itertools.chain.from_iterable(colidx_sep_list)), dtype='uint64')

            morphed_sep[(rowidx_sep_list, colidx_sep_list)] = 1

            # Objects with dilated boundaries
            rowidx_list = np.array(list(itertools.chain.from_iterable(rowidx_list)), dtype='uint64')
            colidx_list = np.array(list(itertools.chain.from_iterable(colidx_list)), dtype='uint64')

            morphed[(rowidx_list, colidx_list)] = labels_list

    # One last check for interior islands
    props = regionprops(morphed,
                        intensity_image=intensity_array)

    # segments_g = morphed

    for chi in range(0, len(props), chunk_size):

        data_gen = ((prop_,
                     min_interior_count,
                     mean_threshold,
                     boundary_mean,
                     prop_area_weight,
                     bbox_pad,
                     arows,
                     acols,
                     morphed,
                     original_binary_edge,
                     se_cross) for prop_ in props[chi:chi+chunk_size])

        cleaned_parcels = []

        with concurrent.futures.ThreadPoolExecutor(n_jobs) as executor:

            for res in executor.map(_remove_interior_islands, data_gen):
                cleaned_parcels.append(res)

        # cleaned_parcels = Parallel(backend='multiprocessing',
        #                            n_jobs=n_jobs)(delayed(_remove_interior_islands)(prop_,
        #                                                                             min_interior_count,
        #                                                                             mean_threshold,
        #                                                                             boundary_mean,
        #                                                                             prop_area_weight,
        #                                                                             bbox_pad,
        #                                                                             arows,
        #                                                                             acols,
        #                                                                             morphed,
        #                                                                             original_binary_edge,
        #                                                                             se_cross) for prop_ in props[chi:chi+chunk_size])

        rowidx_list, colidx_list = zip(*cleaned_parcels)

        rowidx_list = np.array(list(itertools.chain.from_iterable(rowidx_list)), dtype='uint64')
        colidx_list = np.array(list(itertools.chain.from_iterable(colidx_list)), dtype='uint64')

        # piece together the parcels
        if np.any(rowidx_list):
            morphed[(rowidx_list, colidx_list)] = 0

        morphed_sep[morphed == 0] = 0

    return morphed, morphed_sep


def invert_size_check(im_edges, min_size_object, binary=True):

    """
    Inverts binary edges and checks sizes

    Args:
        im_edges (2d array): Edge array, where edges = 1.
        min_size_object (int): The minimum size of line to be retained.
        binary (Optional[bool]): Whether to recode the output labelled edges to binary. Default is True.

    Returns:
         Image objects as 2d array
    """

    # Invert edges to objects
    im_objects = cv2.threshold(np.uint8(im_edges), 0, 1, cv2.THRESH_BINARY_INV)[1]

    # Remove potential field objects smaller
    #   than size threshold.
    im_objects = nd_label(im_objects)[0]
    # im_objects = sk_label(np.uint8(im_objects), connectivity=1)

    im_objects = np.uint64(remove_small_objects(im_objects,
                                                min_size=min_size_object,
                                                connectivity=1))

    if binary:
        im_objects[im_objects > 0] = 1

    return im_objects


def _intersect_objects(prop):

    lc_values = list()
    area_values = list()
    orient_values = list()
    solidity_values = list()
    eccentricity_values = list()

    if prop.label == 0:
        return lc_values, area_values, orient_values, solidity_values, eccentricity_values, list(), list()

    # Get the bounding box of the current segment.
    min_row, min_col, max_row, max_col = prop.bbox

    # Get the label sub-array
    # labels_sub = lc_objects_g[min_row:max_row, min_col:max_col]

    # Get the indices of the current object.
    # labels_sub_idx = list(np.where(labels_sub == prop.label))
    labels_sub_idx = (prop.coords[:, 0], prop.coords[:, 1])
    labels_sub_idx_object = (prop.coords[:, 0] - min_row, prop.coords[:, 1] - min_col)

    n_samples = len(labels_sub_idx[0])

    # labels_sub_idx[0] = labels_sub_idx[0] + min_row
    # labels_sub_idx[1] = labels_sub_idx[1] + min_col

    ##################################
    # LAND COVER CLASS ID INTERSECTION
    ##################################

    if get_class_id_g:

        # Get the land cover
        #   class for the object.
        lc_array_sub = lc_array_g[min_row:max_row, min_col:max_col]

        # Get the majority land cover class
        lc_mode_object = sci_mode(lc_array_sub[tuple(labels_sub_idx_object)])

        lc_mode = int(lc_mode_object.mode)
        lc_count = int(lc_mode_object.count)

        # Check that the land cover count
        #   is above the required threshold.

        # The pixel count needed
        #   to meet the threshold
        pix_count = int(prop.area * object_fraction_g)

        # There must be at least
        #   `object_fraction_g` of the
        #   target class in the object.
        if lc_count >= pix_count:

            # Return the majority class
            lc_values = list(np.zeros(n_samples, dtype='uint8') + lc_mode)

        else:

            # Return empty
            lc_values = list(np.zeros(n_samples, dtype='uint8'))

    # Get the current object.
    # labels_sub_center = np.uint8(np.where(labels_sub == idx, 1, 0))

    ##########################
    # OBJECT AREA INTERSECTION
    ##########################

    if get_object_area_g:

        object_area = round(prop.area * pixel_ha, 2)

        area_values = list(np.zeros(n_samples, dtype='float32') + object_area)

    ########################
    # OBJECT ID INTERSECTION
    ########################

    if get_orientation_g:
        orient_values = list(np.zeros(n_samples, dtype='float32') + prop.orientation)

    if get_solidity_g:
        solidity_values = list(np.zeros(n_samples, dtype='float32') + prop.solidity)

    if get_eccentricity_g:
        eccentricity_values = list(np.zeros(n_samples, dtype='float32') + prop.eccentricity)

    # Return the object value
    return lc_values, \
           area_values, \
           orient_values, \
           solidity_values, \
           eccentricity_values, \
           list(labels_sub_idx[0]), \
           list(labels_sub_idx[1])


def intersect_objects(lc_objects,
                      lc_objects_sep=None,
                      lc_array=None,
                      var_array=None,
                      objects_are_unique=False,
                      object_fraction=0.5,
                      get_object_area=True,
                      get_object_id=False,
                      get_class_id=False,
                      get_orientation=False,
                      get_solidity=False,
                      get_eccentricity=False,
                      cell_size=30.0,
                      n_jobs=1,
                      chunk_size=100000):

    """"
    Intersects land cover objects with a thematic land cover map

    Args:
        lc_objects (2d array): The segmented objects.
        lc_objects_sep (2d array): The eroded segmented objects.
        lc_array (Optional[2d array]): The land cover array, needed if `get_object_area = False`. Default is None.
        var_array (Optional[2d array]): The image variables array. Default is None.
        objects_are_unique (Optional[bool]): Whether the land cover objects of `lc_objects` are unique.
            Default is False.
        object_fraction (Optional[float]): The fraction of an object in `lc_objects` to be considered for intersection.
            Default is 0.5.
        get_object_area (Optional[bool]): Whether to return the object area. Default is True.
        get_object_id (Optional[bool]): Whether to return the object id from `lc_objects`. Default is False.
        get_class_id (Optional[bool]): Whether to return the land cover class id from `lc_array`. Default is False.
        get_orientation (Optional[bool]): Whether to return the object orientation. Default is False.
        get_solidity (Optional[bool]): Whether to return the object solidity. Default is False.
        get_eccentricity (Optional[bool]): Whether to return the object eccentricity. Default is False.
        cell_size (Optional[float]): The cell size, used when `get_object_area = True`. Default is 30.
        n_jobs (Optional[int]): The number of parallel jobs. Default is 1.
        chunk_size (Optional[int]): The chunk size for Pool. Default is 100,000.
    """

    global object_fraction_g, get_object_area_g, get_object_id_g, \
        get_class_id_g, get_orientation_g, get_solidity_g, get_eccentricity_g, \
        pixel_ha, lc_objects_g, lc_array_g

    object_fraction_g = object_fraction
    get_object_area_g = get_object_area
    get_object_id_g = get_object_id
    get_class_id_g = get_class_id
    get_orientation_g = get_orientation
    get_solidity_g = get_solidity
    get_eccentricity_g = get_eccentricity

    lc_objects_g = lc_objects
    lc_array_g = lc_array

    # Square meters to hectares
    pixel_ha = (cell_size * cell_size) * 0.0001

    out_array = np.zeros((5, lc_objects.shape[0], lc_objects.shape[1]), dtype='float32')

    # Get unique object ids.
    if not objects_are_unique:

        lc_objects[lc_objects > 0] = 1
        lc_objects, n_objects = nd_label(lc_objects)

    # Get object properties.
    #   zo = prop.area
    props_int = regionprops(lc_objects)

    for chi in range(0, len(props_int), chunk_size):

        with pooler(processes=n_jobs) as pool:

            # Get object statistics
            intersected_objects = pool.map(_intersect_objects,
                                           props_int[chi:chi+chunk_size],
                                           chunk_size)

        lc_values_, area_values_, ori_values_, sol_values_, ecc_values_, rowidx_list, colidx_list = zip(*intersected_objects)

        # Join the lists
        lc_values_ = np.array(list(itertools.chain.from_iterable(lc_values_)), dtype='uint8')
        area_values_ = np.array(list(itertools.chain.from_iterable(area_values_)), dtype='float32')
        ori_values_ = np.array(list(itertools.chain.from_iterable(ori_values_)), dtype='float32')
        sol_values_ = np.array(list(itertools.chain.from_iterable(sol_values_)), dtype='float32')
        ecc_values_ = np.array(list(itertools.chain.from_iterable(ecc_values_)), dtype='float32')

        rowidx_list = np.array(list(itertools.chain.from_iterable(rowidx_list)), dtype='uint64')
        colidx_list = np.array(list(itertools.chain.from_iterable(colidx_list)), dtype='uint64')

        # Piece together the parcels

        # land cover
        if lc_values_.shape[0] > 0:
            out_array[0, rowidx_list, colidx_list] = lc_values_

        # area
        if area_values_.shape[0] > 0:
            out_array[1, rowidx_list, colidx_list] = area_values_

        # orientation
        if ori_values_.shape[0] > 0:
            out_array[2, rowidx_list, colidx_list] = ori_values_

        # solidarity
        if sol_values_.shape[0] > 0:
            out_array[3, rowidx_list, colidx_list] = sol_values_

        # eccentricity
        if ecc_values_.shape[0] > 0:
            out_array[4, rowidx_list, colidx_list] = ecc_values_

    if isinstance(lc_objects_sep, np.ndarray):

        # Swap the land cover with the eroded objects
        out_array[0] = np.where(lc_objects_sep > 0, out_array[0], 0)

    if isinstance(var_array, np.ndarray):

        # Give the objects unique labels
        lc_objects_labs_ = sk_label(np.uint8(lc_objects_sep), connectivity=2)
        lc_objects_labs_index = np.unique(lc_objects_labs_)

        # Get the mean for each variable layer
        for var_idx, var_layer in enumerate(var_array):

            layer_means = nd_mean(var_layer, labels=lc_objects_labs_, index=lc_objects_labs_index)

            new_layer = np.zeros(var_layer.shape, dtype='float32')

            # Update the layer values
            for object_index in lc_objects_labs_index:
                new_layer[lc_objects_labs_ == object_index] = layer_means[object_index]

            # Add the layer to the output array
            out_array = np.vstack((out_array, new_layer[np.newaxis, :, :]))

    return out_array


def smooth(ts_array,
           iterations=1,
           window_size=5,
           sigma_color=1.0,
           sigma_space=1.0):

    """
    Spatially smooths a series of arrays

    Args:
        ts_array (3d array): The time series array to smooth [time dimensions x rows x columns].
        iterations (Optional[int]): The number of smoothing iterations. Default is 1.
        window_size (Optional[int]): The bi-lateral filter window size (in pixels). Default is 5.
        sigma_color (Optional[float]): The bi-lateral filter color sigma. Default is 1.0.
        sigma_space (Optional[float]): The bi-lateral filter space (distance) sigma. Default is 1.0.

    Returns:
        The smoothed 3d `ts_array`.
    """

    for tsp in range(0, ts_array.shape[0]):

        for iteration in range(0, iterations):

            # Subtract the inhibition term.
            # ts_array[tsp] = moving_window(ts_array[tsp],
            #                               statistic='inhibition',
            #                               window_size=5,
            #                               inhibition_ws=3,
            #                               iterations=1,
            #                               inhibition_scale=.5,
            #                               inhibition_operation=1)

            # Add the inhibition term.
            # ts_array[tsp] = moving_window(ts_array[tsp],
            #                               statistic='inhibition',
            #                               window_size=5,
            #                               inhibition_ws=3,
            #                               iterations=1,
            #                               inhibition_scale=.5,
            #                               inhibition_operation=2)

            # Fill basins
            ts_array[tsp] = moving_window(ts_array[tsp],
                                          statistic='fill-basins',
                                          window_size=3,
                                          iterations=2)

            # Fill peaks
            ts_array[tsp] = moving_window(ts_array[tsp],
                                          statistic='fill-peaks',
                                          window_size=3,
                                          iterations=2)

            # Bilateral filter
            ts_array[tsp] = cv2.bilateralFilter(ts_array[tsp],
                                                window_size,
                                                sigma_color,
                                                sigma_space)

    return ts_array


class PixelStats(object):

    @staticmethod
    def calc(array2proc, statistic, lower_percentile, upper_percentile):

        if statistic == 'cv':

            ts_stat = array2proc.std(axis=0) / np.median(array2proc, axis=0)
            ts_stat = np.where(ts_stat > 1, 1, ts_stat)

        elif statistic == 'max':
            ts_stat = array2proc.max(axis=0)

        elif statistic == 'mean':
            ts_stat = array2proc.mean(axis=0)

        elif statistic in ['25', '50', '75']:
            ts_stat = np.percentile(array2proc, int(statistic), axis=0)

        ts_stat = moving_window(ts_stat, statistic='fill-basins', window_size=3, iterations=2)
        ts_stat = moving_window(ts_stat, statistic='fill-peaks', window_size=3, iterations=2)

        ts_stat[np.isnan(ts_stat) | np.isinf(ts_stat)] = 0

        return rescale_intensity(ts_stat,
                                 in_range=(np.percentile(ts_stat, lower_percentile),
                                           np.percentile(ts_stat, upper_percentile)),
                                 out_range=(0, 1))


class ArrayWriter(object):

    def write2file(self,
                   output_image,
                   write_mag=False,
                   write_egm=False,
                   write_edges=False,
                   write_objects=False,
                   write_cv=False):

        out_bands = 0

        if write_mag:
            out_bands += 1

        if write_egm:
            out_bands += 1

        if write_edges:
            out_bands += 1

        if write_objects:
            out_bands += 1

        if write_cv:
            out_bands += 1

        band_pos = 1

        if os.path.isfile(output_image):
            os.remove(output_image)

        with create_raster(output_image,
                           None,
                           compress='deflate',
                           rows=self.parameter_info.n_rows,
                           cols=self.parameter_info.n_cols,
                           bands=out_bands,
                           projection=self.parameter_info.projection,
                           cellY=self.parameter_info.extent_dict['cell_y'],
                           cellX=self.parameter_info.extent_dict['cell_x'],
                           left=self.parameter_info.extent_dict['left'],
                           top=self.parameter_info.extent_dict['top'],
                           storage='float32') as out_rst:

            if write_mag:

                out_rst.write_array(self.mag[self.parameter_info.ipad:self.parameter_info.ipad+self.parameter_info.n_rows,
                                             self.parameter_info.jpad:self.parameter_info.jpad+self.parameter_info.n_cols],
                                    band=band_pos)

                out_rst.close_band()

                band_pos += 1

            if write_egm:

                out_rst.write_array(self.egm[self.parameter_info.ipad:self.parameter_info.ipad+self.parameter_info.n_rows,
                                             self.parameter_info.jpad:self.parameter_info.jpad+self.parameter_info.n_cols],
                                    band=band_pos)

                out_rst.close_band()

                band_pos += 1

            if write_edges:

                out_rst.write_array(self.im_edges, band=band_pos)
                out_rst.close_band()

                band_pos += 1

            if write_objects:

                out_rst.write_array(self.objects[self.parameter_info.ipad:self.parameter_info.ipad+self.parameter_info.n_rows,
                                                 self.parameter_info.jpad:self.parameter_info.jpad+self.parameter_info.n_cols],
                                    band=band_pos)

                out_rst.close_band()

                band_pos += 1

            if write_cv:

                out_rst.write_array(self.ts_array.var(axis=0) / self.ts_array.mean(axis=0), band=band_pos)
                out_rst.close_band()

        out_rst.close_file()
        del out_rst


def adjust_edges(original_edges,
                 edges2adjust,
                 lower_threshold=.1,
                 scale_factor=2.,
                 scale_range=(0, 1),
                 min_object_size=100,
                 iterations=1):

    for iteration in range(0, iterations):

        edges2adjust[edges2adjust > lower_threshold] = 1
        edges2adjust = sk_label(np.uint8(edges2adjust), connectivity=2)
        edges2adjust = np.uint64(remove_small_objects(edges2adjust, min_size=min_object_size, connectivity=1))
        edges2adjust[edges2adjust > 0] = 1

        edges2adjust = np.float32(np.where(edges2adjust == 1,
                                           original_edges * scale_factor,
                                           original_edges / scale_factor))

        edges2adjust = rescale_intensity(edges2adjust, in_range=scale_range, out_range=(0, 1))

        if iterations > 1:
            original_edges = edges2adjust.copy()

        scale_factor += .1
        scale_range = scale_range[0], scale_range[1]+.1

    return rescale_intensity((edges2adjust + (original_edges * .5)) / 1.5,
                             in_range=(0, 1),
                             out_range=(0, 1))

    # import matplotlib.pyplot as plt
    # plt.imshow(edges2adjust)
    # plt.show()
    # import sys
    # sys.exit()


def hist_match(source, template):

    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Args:
        source (2d array): Image to transform; the histogram is computed over the flattened array.
        template (2d array): Template image; can have different dimensions to source.

    Reference:
        https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x

    Returns:
        matched (2d array): The transformed output image.
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # Get the set of unique pixel values and
    #   their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(source,
                                            return_inverse=True,
                                            return_counts=True)

    t_values, t_counts = np.unique(template,
                                   return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.float32(s_counts.cumsum())
    s_quantiles /= s_quantiles[-1]

    t_quantiles = np.float32(t_counts.cumsum())
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)
