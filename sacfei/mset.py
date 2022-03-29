from . import utils
from .errors import logger
from .relax import relax

import numpy as np
from skimage.exposure import rescale_intensity
import cv2


SCALING = dict(Landsat_pan=(0.001, 0.05),
               Sentinel2_10m=(2e-5, 0.003),
               RGB=(0, 0.25),
               cropnet=(0, 1),
               log=(-14, -8))

COMPAT = np.array([[1.0, 0.7],
                   [0.7, 1.0]], dtype='float32')


class MSET(object):

    """
    [M]ulti-[S]cale [E]dge [T]hreshold
    """

    def __init__(self,
                 threshold_windows=None,
                 adapth_thinning=0,
                 adapth_endpoint=0,
                 min_size_line=15,
                 endpoint_iterations=10,
                 extend_iters=3,
                 link_ends=False,
                 link_iters=1,
                 link_window_size=15,
                 smallest_allowed_gap=7,
                 max_gap=15,
                 min_egm=20,
                 hist_equal=True,
                 clip_percentages=None,
                 grid_tiles=None,
                 min_size_object=5,
                 logistic_alpha=1.5,
                 logistic_beta=0.5,
                 log=True,
                 scale=1e-8,
                 template=None,
                 satellite='Sentinel2_10m',
                 relax_probas=False,
                 apply_clean=False,
                 clean_min_object_area=5,
                 clean_min_area_int=222,
                 ignore_thresh=10.0,
                 n_jobs=-1):

        self.threshold_windows = threshold_windows
        self.adapth_thinning = adapth_thinning
        self.adapth_endpoint = adapth_endpoint
        self.min_size_line = min_size_line
        self.endpoint_iterations = endpoint_iterations
        self.extend_iters = extend_iters
        self.link_ends = link_ends
        self.link_iters = link_iters
        self.link_window_size = link_window_size
        self.smallest_allowed_gap = smallest_allowed_gap
        self.max_gap = max_gap
        self.min_egm = min_egm
        self.hist_equal = hist_equal
        self.clip_percentages = clip_percentages
        self.grid_tiles = grid_tiles
        self.min_size_object = min_size_object
        self.logistic_alpha = logistic_alpha
        self.logistic_beta = logistic_beta
        self.log = log
        self.scale = scale
        self.template = template
        self.satellite = satellite
        self.relax_probas = relax_probas
        self.apply_clean = apply_clean
        self.clean_min_object_area = clean_min_object_area
        self.clean_min_area_int = clean_min_area_int
        self.ignore_thresh = ignore_thresh
        self.n_jobs = n_jobs

        if not isinstance(self.clip_percentages, list):
            self.clip_percentages = [2.0]

        if not isinstance(self.grid_tiles, list):
            self.grid_tiles = [16]

        if not isinstance(self.threshold_windows, list):
            self.threshold_windows = [31, 51]

    def egm_to_proba(self):

        if not hasattr(self, 'egm'):

            logger.error('  The EGM array does not exist.')
            raise AttributeError

        if not hasattr(self, 'egm_cv_array'):
            logger.warning('  The EGM CV array does not exist.')

        if not hasattr(self, 'vi_cv_array'):
            logger.warning('  The EVI2 CV array does not exist.')

        if isinstance(self.template, np.ndarray):

            if self.verbose > 0:
                logger.info('  Matching template ...')

            self.egm = utils.hist_match(self.egm, self.template)

        if self.verbose > 0:
            logger.info('  Data scaling ...')

        # Scale the EGM array.
        if self.log:

            # self.egm_proba = np.float32(rescale_intensity(np.log(self.egm * self.scale),
            #                                               in_range=self.in_range,
            #                                               out_range=(-1, 1)))

            self.egm_proba = utils.log_transform(self.egm,
                                                 self.scale,
                                                 self.logistic_alpha,
                                                 self.logistic_beta)

            # epr = relax(ep, COMPAT, iterations=3, shape=3)
            # fig,ax=plt.subplots(1, 2);ax[0].imshow(self.egm);ax[1].imshow(ep);plt.show()
            # self.egm_proba = np.float32(utils.logistic(self.egm_proba,
            #                                           self.logistic_alpha,
            #                                           self.logistic_beta))

        else:
            self.egm_proba = self.egm.copy()

        if self.relax_probas:

            # Relax the EGM likelihoods
            self.egm_proba = relax(self.egm_proba, COMPAT, iterations=3, shape=3)

        egm_cv_array = None
        vi_cv_array = None

        if hasattr(self, 'egm_cv_array'):

            if isinstance(self.egm_cv_array, np.ndarray):

                # Scale the EGM CV
                self.egm_cv_array = np.uint8(rescale_intensity(self.egm_cv_array,
                                                               in_range=(0, 2000),
                                                               out_range=(0, 255)))

        if hasattr(self, 'vi_cv_array'):

            if isinstance(self.vi_cv_array, np.ndarray):

                # Scale the EVI2 CV
                self.vi_cv_array = np.float32(rescale_intensity(self.vi_cv_array,
                                                                in_range=(0, 10000),
                                                                out_range=(0, 1)))

        if self.hist_equal:

            if self.verbose > 0:
                logger.info('  Histogram equalization ...')

            self.egm_proba = utils.local_hist_eq(np.uint8(self.egm_proba * 255.0),
                                                 clip_percentages=self.clip_percentages,
                                                 grid_tiles=self.grid_tiles)

        if self.verbose > 0:
            logger.info('  EVI2 morphology ...')

        egm_dist = np.float32(cv2.distanceTransform(np.uint8(np.where(self.egm_proba < 0.2, 1, 0)), cv2.DIST_L2, 3))

        if isinstance(self.egm_cv_array, np.ndarray):

            self.egm_cv_array = utils.closerec(self.egm_cv_array, 'disk', r=3, iters=5)
            egm_cv_dist = np.float32(cv2.distanceTransform(np.uint8(np.where(self.egm_cv_array >= 10, 1, 0)), cv2.DIST_L2, 3))

            self.egm_proba[(egm_cv_dist >= 5) & (egm_dist >= 5)] = 0

        else:
            self.egm_proba[(egm_dist >= 5)] = 0

    def _segment(self):

        if self.verbose > 0:
            logger.info('  Thresholding edges ...')

        im_objects = utils.multiscale_threshold(np.uint8(self.egm_proba * 255.0),
                                                self.min_size_object,
                                                windows=self.threshold_windows,
                                                theta_180_iters=0,
                                                theta_90_iters=0,
                                                theta_45_iters=0,
                                                endpoint_iterations=0,
                                                method='wmean',
                                                inverse_dist=False,
                                                ignore_thresh=self.ignore_thresh)

        im_edges_copy = im_objects.copy()

        if self.verbose > 0:
            logger.info('  Object morphology ...')

        im_objects = utils.morphological_cleanup(im_objects,
                                                 10,
                                                 endpoint_iterations=self.endpoint_iterations,
                                                 theta_45_iters=1,
                                                 theta_90_iters=1,
                                                 theta_180_iters=1,
                                                 extend_endpoints=True,
                                                 extend_iters=self.extend_iters,
                                                 link_ends=self.link_ends,
                                                 link_iters=self.link_iters,
                                                 link_window_size=self.link_window_size,
                                                 smallest_allowed_gap=self.smallest_allowed_gap,
                                                 max_gap=self.max_gap,
                                                 min_egm=self.min_egm,
                                                 skeleton=False,
                                                 egm_array=np.uint8(self.egm_proba * 255.0),
                                                 pre_thin=False,
                                                 value_array=self.vi_cv_array)

        if self.verbose > 0:
            logger.info('  Object size check ...')

        # [15m x 15m] = 225 m^2
        #   = 0.0225 ha
        #   0.0225 x 5 = 0.1125 ha
        im_objects = utils.invert_size_check(im_objects,
                                             self.min_size_object,
                                             binary=False)

        if self.verbose > 0:
            logger.info('  Cleaning objects ...')

        if self.apply_clean:

            self.objects, self.objects_sep = utils.clean_objects(im_objects,
                                                                 intensity_array=self.vi_cv_array,
                                                                 original_binary_edge=im_edges_copy,
                                                                 binary=False,
                                                                 min_object_area=self.clean_min_object_area,
                                                                 min_interior_count=self.clean_min_area_int,
                                                                 n_jobs=self.n_jobs)

        else:
            self.objects, self.objects_sep = im_objects.copy(), im_objects.copy()

    def _intersect(self, **kwargs):
        self.objects_intersected = utils.intersect_objects(self.objects, **kwargs)
