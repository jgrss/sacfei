#!/usr/bin/env python

import os
import argparse
import time

from sacfei.errors import logger

import geopandas as gpd
import shapely

from joblib import Parallel, delayed

shapely.speedups.enable()


def inter_worker(geom, b_geoms):

    for j in range(0, b_geoms.shape[0]):

        if geom.intersects(b_geoms[j]):
            return True

    return False


def intersects(a, b, output=None, query_a=None, query_b=None, verbose=0, n_jobs=1):

    """
    Intersects to vector files

    Args:
        a (str or GeoDataFrame): The first vector file.
        b (str or GeoDataFrame): The second vector file.
        output (Optional[str]): A vector output to write to. Default is None.
        query_a (Optional[str]): A DataFrame query for `a`. Default is None.
        query_b (Optional[str]): A DataFrame query for `b`. Default is None.
        verbose (Optional[int]): The verbosity level. Default is 0.
        n_jobs (Optional[int]): The number of parallel processes. Default is 1.
    """

    if isinstance(a, str):

        if verbose > 0:
            logger.info('  Reading A ...')

        a = gpd.read_file(a)

    if isinstance(b, str):

        if verbose > 0:
            logger.info('  Querying B ...')

        b = gpd.read_file(b)

    if isinstance(query_a, str):

        if verbose > 0:
            logger.info('  Querying A ...')

        a = a.query(query_a)

    if isinstance(query_b, str):

        if verbose > 0:
            logger.info('  Querying B ...')

        b = b.query(query_b)

    a_geoms = a.geometry.values
    b_geoms = b.geometry.values

    n_features = a_geoms.shape[0]

    if verbose > 0:
        logger.info('  Checking intersection of A and B ...')

    mask = Parallel(n_jobs=n_jobs)(delayed(inter_worker)(a_geoms[i], b_geoms) for i in range(0, n_features))

    if isinstance(output, str):

        if verbose > 0:
            logger.info('  Writing to file ...')

        dirname = os.path.dirname(output)

        ext = os.path.splitext(os.path.split(output)[1])[1]

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        a_inter = a[mask]
        a_null = a[~mask]

        a_inter.to_file(output.replace(ext, '_int' + ext))
        a_null.to_file(output.replace(ext, '_int_null' + ext))

    return mask


def main():

    parser = argparse.ArgumentParser(description='Intersect parcels',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('--dfa', dest='dfa', help='The first file', default=None)
    parser.add_argument('--dfb', dest='dfb', help='The second file', default=None)
    parser.add_argument('--output', dest='output', help='The output file', default=None)
    parser.add_argument('--query-a', dest='query_a', help='A DataFrame query for a', default=None)
    parser.add_argument('--query-b', dest='query_b', help='A DataFrame query for b', default=None)
    parser.add_argument('--verbose', dest='verbose', help='The verbosity level', default=1, type=int)
    parser.add_argument('--n-jobs', dest='n_jobs', help='The number of parallel jobs', default=1, type=int)

    args = parser.parse_args()

    logger.info('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    mask = intersects(args.dfa,
                      args.dfb,
                      output=args.output,
                      query_a=args.query_a,
                      query_b=args.query_b,
                      verbose=args.verbose,
                      n_jobs=args.n_jobs)

    logger.info('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
                    (time.asctime(time.localtime(time.time())), (time.time() - start_time)))


if __name__ == '__main__':
    main()
