#!/usr/bin/env python

import os
import argparse

from sacfei.errors import logger

import pandas as pd
import geopandas as gpd
import shapely

from joblib import Parallel, delayed
# from tqdm import tqdm

shapely.speedups.enable()


def point_checker(pool_iter):

    parcel_row = parcel_df.iloc[pool_iter]

    if parcel_row.geometry.centroid.within(adm1_geom):
        return True
    else:
        return False


def clip_parcels(parcels,
                 adm1_shp=None,
                 adm1_df=None,
                 adm0='Argentina',
                 adm1=None,
                 n_jobs=1,
                 verbose=0):

    global adm1_geom, parcel_df

    adm1_geom = None
    parcel_df = None

    if verbose > 0:
        logger.info('  Loading ADM1 ...')

    if not isinstance(adm1_df, pd.DataFrame):

        if not isinstance(adm1_shp, str):
            logger.exception('  A DataFrame or vector file must be given.')

        adm1_df = gpd.read_file(adm1_shp)
        adm1_df = adm1_df.query("(NAME_0 == '{}') & (ID_1 == {:d})".format(adm0, adm1))

    adm1_geom = adm1_df.geometry.values[0]

    if verbose > 0:
        logger.info('  Loading parcels ...')

    parcel_df = gpd.read_file(parcels)

    if verbose > 0:
        logger.info('  Clipping parcels ...')

    # point_list = [pidx for pidx, drow in tqdm(parcel_df.iterrows()) if drow.geometry.centroid.within(adm1_geom)]

    n_parcels = parcel_df.shape[0]

    point_list = Parallel(n_jobs=n_jobs,
                          max_nbytes=None)(delayed(point_checker)(pidx) for pidx in range(0, n_parcels))

    if verbose > 0:
        logger.info('  Subsetting and writing to file ...')

    parcel_df = parcel_df.loc[point_list]

    os.remove(parcels)

    if not parcel_df.empty:
        parcel_df.to_file(parcels)


def main():

    parser = argparse.ArgumentParser(description='Subsets parcels',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('--parcels', dest='parcels', help='The parcel file', default=None)
    parser.add_argument('--adm1-shp', dest='adm1_shp', help='The ADM1 file', default=None)
    parser.add_argument('--adm0', dest='adm0', help='The ADM0 name', default='Argentina')
    parser.add_argument('--adm1', dest='adm1', help='The ADM1 id', default=None, type=int)
    parser.add_argument('--n-jobs', dest='n_jobs', help='The number of parallel jobs for clipping', default=1, type=int)

    args = parser.parse_args()

    clip_parcels(args.parcels,
                 adm1_shp=args.adm1_shp,
                 adm0=args.adm0,
                 adm1=args.adm1,
                 n_jobs=args.n_jobs)


if __name__ == '__main__':
    main()
