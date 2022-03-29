from .errors import logger
from . import utils
from .mset import MSET

import numpy as np


class MTMSEGM(utils.PixelStats, utils.ArrayWriter, MSET):

    """
    [M]ulti-[T]emporal and [M]ulti-[S]cale [E]dge [G]radient [M]agnitude (MTMSEGM)
    """

    def __init__(self,
                 ts_array,
                 dims=None,
                 nrows=None,
                 ncols=None,
                 smooth_iterations=1,
                 gamma=0.1,
                 pre_smooth=True,
                 smooth_window=7,
                 sigma_color=1.0,
                 sigma_space=1.0,
                 kernels=None,
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
                 parameter_info=None,
                 output_image=None,
                 cdl_array=None,
                 scale_factor=0.0001,
                 logistic_alpha=1.5,
                 logistic_beta=0.5,
                 log=True,
                 scale=1e-8,
                 template=None,
                 satellite='Sentinel2_10m',
                 relax_probas=True,
                 apply_clean=False,
                 clean_min_object_area=5,
                 clean_min_area_int=222,
                 ignore_thresh=10.0,
                 n_jobs=-1,
                 verbose=1):

        MSET.__init__(self,
                      threshold_windows=threshold_windows,
                      adapth_thinning=adapth_thinning,
                      adapth_endpoint=adapth_endpoint,
                      min_size_line=min_size_line,
                      endpoint_iterations=endpoint_iterations,
                      extend_iters=extend_iters,
                      link_ends=link_ends,
                      link_iters=link_iters,
                      link_window_size=link_window_size,
                      smallest_allowed_gap=smallest_allowed_gap,
                      max_gap=max_gap,
                      min_egm=min_egm,
                      hist_equal=hist_equal,
                      clip_percentages=clip_percentages,
                      grid_tiles=grid_tiles,
                      min_size_object=min_size_object,
                      logistic_alpha=logistic_alpha,
                      logistic_beta=logistic_beta,
                      log=log,
                      scale=scale,
                      template=template,
                      satellite=satellite,
                      relax_probas=relax_probas,
                      apply_clean=apply_clean,
                      clean_min_object_area=clean_min_object_area,
                      clean_min_area_int=clean_min_area_int,
                      ignore_thresh=ignore_thresh,
                      n_jobs=n_jobs)

        self.dims = dims
        self.nrows = nrows
        self.ncols = ncols

        if isinstance(ts_array, np.ndarray):

            self.ts_array = np.float32(ts_array)

            if not isinstance(self.dims, int):
                self.dims = self.ts_array.shape[0]

            if not isinstance(self.nrows, int):
                self.nrows = self.ts_array.shape[1]

            if not isinstance(self.ncols, int):
                self.ncols = self.ts_array.shape[2]

        self.smooth_iterations = smooth_iterations

        self.gamma = gamma
        self.pre_smooth = pre_smooth
        self.smooth_window = smooth_window
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

        self.kernels = kernels

        self.parameter_info = parameter_info
        self.output_image = output_image
        self.cdl_array = cdl_array
        self.scale_factor = scale_factor
        self.verbose = verbose

        # if len(self.ts_array.shape) == 3:
        #     self.melted = False
        # else:
        #     self.melted = True

    def _extract_egm(self):

        """
        Extracts the Edge Gradient Magnitude
        """

        # if self.melted:
        #
        #     self.ts_array = np.float32(self.ts_array.T.reshape(self.dims,
        #                                                        self.nrows,
        #                                                        self.ncols))

        self.ts_array *= self.scale_factor

        ts_d, ts_r, ts_c = self.ts_array.shape

        if self.pre_smooth:

            if self.verbose > 0:
                logger.info('  Smoothing input image ...')

            self.ts_array = utils.smooth(self.ts_array,
                                         iterations=self.smooth_iterations,
                                         window_size=self.smooth_window,
                                         sigma_color=self.sigma_color,
                                         sigma_space=self.sigma_space)

        if self.verbose > 0:
            logger.info('  Extracting edge info ...')

        self.egm = utils.get_mag_egm(self.ts_array,
                                     ts_r,
                                     ts_c,
                                     self.kernels)[0]
