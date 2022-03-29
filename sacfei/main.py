from .mtmsegm import MTMSEGM


class SACFEI(MTMSEGM):

    """
    Args:
        ts_array (2d or 3d array)
        dims (Optional[int]): Default is None.
        nrows (Optional[int]): Default is None.
        ncols (Optional[int]): Default is None.
        smooth_iterations (Optional[int]): Default is 1.
        gamma (Optional[float]): Default is 0.1.
        pre_smooth (Optional[bool]): Default is True.
        kernels (Optional[tuple list]): Default is None.
        threshold_windows (Optional[int list]): Default is None.
        adapth_thinning (Optional[int]): Default is 0.
        adapth_endpoint (Optional[int]): Default is 0.
        min_size_line (Optional[int]): Default is 15.
        endpoint_iterations (Optional[int]): Default is 10.
        min_size_object (Optional[int]): Default is 5.
        parameter_info (Optional[object]): Default is None.
        output_image (Optional[str]): Default is None.
        cdl_array (Optional[2d array]): Default is None.
        verbose (Optional[int]): Default is 1.

    Examples:
        >>> # Compute EGM and segment
        >>> sco = SACFEI(ts_array)
        >>>
        >>> sco.extract_egm()
        >>> sco.segment_egm()
        >>>
        >>> # Segment only
        >>> sco = SACFEI(None)
        >>>
        >>> sco.egm = <EGM array>
        >>> sco.egm_cv_array = <EGM CV array>
        >>> sco.vi_cv_array = <VI CV array>
        >>>
        >>> sco.segment_egm()
    """

    def __init__(self,
                 ts_array,
                 **kwargs):

        MTMSEGM.__init__(self,
                         ts_array,
                         **kwargs)

    def extract_egm(self):

        self._extract_egm()
        self.egm_to_proba()

    def segment_egm(self):
        self._segment()

    def intersect_objects(self, **kwargs):
        self._intersect(**kwargs)
