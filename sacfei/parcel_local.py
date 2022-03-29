#!/usr/bin/env python

import os
import argparse
from contextlib import contextmanager
import multiprocessing as multi

from sacfei.errors import logger

import numpy as np
import geopandas as gpd
import shapely

from joblib import Parallel, delayed
# from tqdm import tqdm

shapely.speedups.enable()


@contextmanager
def pooler(*args, **kwargs):

    pool = multi.Pool(*args, **kwargs)
    yield pool
    pool.close()
    pool.join()
    pool.terminate()


def worker(pool_iter):

    parcel_row = parcel_df.iloc[pool_iter].geometry.buffer(radius)

    int_points = sorted(list(parcel_index.intersection(parcel_row.bounds)))

    if int_points:

        near_parcels = parcel_df.iloc[int_points]
        near_parcels = near_parcels.loc[near_parcels.intersects(parcel_row)]

        parcel_count = near_parcels.shape[0]
        mean_area = near_parcels.Area.mean()
        crop_count = near_parcels.DN.unique().shape[0]

    else:

        parcel_count = -999.0
        mean_area = -999.0
        crop_count = -999

    return mean_area, crop_count, parcel_count


def parcel_heterogeneity(parcels, buffer_dist=10000.0, n=None, verbose=0, n_jobs=1):

    """
    Calculates local statistics around individual parcels

    Args:
        parcels (str or GeoDataFrame): The parcels.
        buffer_dist (Optional[float]): The buffer distance around each parcel.
        n (Optional[int]): The subsample count.
        verbose (Optional[int]): The verbosity level.
        n_jobs (Optional[int]): The number of parallel workers.
    """

    global parcel_df, parcel_index, radius

    parcel_df = None
    parcel_index = None
    radius = None

    if isinstance(parcels, str):
        is_file = True
    else:
        is_file = False

    # with pooler(processes=n_jobs) as pool:

    radius = buffer_dist

    if verbose > 0:
        logger.info('  Reading data ...')

    if isinstance(parcels, str):
        parcel_df = gpd.read_file(parcels)
    else:
        parcel_df = parcels

    if verbose > 0:
        logger.info('  Querying DataFrame ...')

    parcel_df = parcel_df.query("DN != [78, 111, 124, 131, 138, 144, 152, 195]")

    if isinstance(n, int):
        parcel_df = parcel_df.sample(n=n)

    if verbose > 0:
        logger.info('  Setting up spatial index ...')

    parcel_index = parcel_df.sindex

    n_parcels = parcel_df.shape[0]

    if verbose > 0:
        logger.info('  Calculating on {:,d} parcels ...'.format(n_parcels))

    # results = list(map(worker, range(0, n_parcels)))
    # results = pool.map(worker, range(0, n_parcels))

    results = Parallel(n_jobs=n_jobs,
                       max_nbytes=None)(delayed(worker)(pidx) for pidx in range(0, n_parcels))

    results = np.array(results, dtype='float32')

    parcel_df['mean_local'] = results[:, 0]
    parcel_df['crop_count_local'] = np.uint16(results[:, 1])
    parcel_df['parcel_count_local'] = np.uint16(results[:, 2])

    if is_file:

        os.remove(parcels)
        parcel_df.to_file(parcels)

    else:
        return parcel_df
