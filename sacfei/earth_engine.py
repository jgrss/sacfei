#!/usr/bin/env python

import os

import ee
import numpy as np


ee.Initialize()

FMASK_CLEAR_GROUND = 0
FMASK_WATER = 1
FMASK_CLOUD_SHADOW = 2
FMASK_SNOW = 3
FMASK_CLOUD = 4

L8_PAN_BAND = 'B8'
S2_GREEN_BAND = 'B3'
S2_RED_BAND = 'B4'
S2_NIR_BAND = 'B8'
S2_SWIR2_BAND = 'B11'
S2_SWIR2_BAND = 'B12'

WRS_PATH1 = 227
WRS_PATH2 = 227
WRS_ROW1 = 82
WRS_ROW2 = 82

MGRS_TILE = '32TPQ'

START_DATE = '2016-05-01'
END_DATE = '2017-08-01'

# Define a Gaussian kernel.
GAUSSIAN_KERNEL1 = ee.Kernel.gaussian(15, 0.5)
GAUSSIAN_KERNEL2 = ee.Kernel.gaussian(9, 0.75)


def mask_landsat(image):

    """
    Masks clouds in Landsat imagery

    Args:
        image (object)
    """

    fmask = image.select('fmask')

    # Setup the cloud mask var
    cloud_mask = fmask.neq(FMASK_CLOUD).And(fmask.neq(FMASK_SNOW)).And(fmask.neq(FMASK_CLOUD_SHADOW))

    # Mask the image
    return image.updateMask(cloud_mask)


def get_magnitude(image, kx, ky):

    """
    Extracts the edge gradient magnitude (equivalent to `sacfei.utils._get_magnitude`)

    Args:
        image (object)
        kx (int)
        ky (int)
    """
  
    # Apply the edge-detection kernel.
    egm_x = image.convolve(kx)
    egm_y = image.convolve(ky)

    # Compute the magnitude of the gradient.
    mag = egm_x.pow(2).add(egm_y.pow(2)).sqrt()

    return ee.Image(mag)


def compute_egm(image):

    """
    Computes the multi-kernel EGM

    Args:
        image (object)
    """

    # Define the kernels

    # Sobel
    kernel_x_s = ee.Kernel.sobel(1, True)
    kernel_y_s = ee.Kernel.sobel(1, True).rotate(1)

    # Roberts
    kernel_x_r = ee.Kernel.roberts(1, True)
    kernel_y_r = ee.Kernel.roberts(1, True).rotate(1)

    # Prewitt
    kernel_x_p = ee.Kernel.prewitt(1, True)
    kernel_y_p = ee.Kernel.prewitt(1, True).rotate(1)

    # Kirsch
    kernel_k = ee.Kernel.kirsch(1, True)

    # Compass
    kernel_c = ee.Kernel.compass(1, True)

    # Smooth the image by convolving with a Gaussian kernel.
    image_smooth = image.convolve(GAUSSIAN_KERNEL1).convolve(GAUSSIAN_KERNEL2)

    # Compute the magnitude of the gradient.
    egm_s = get_magnitude(image_smooth, kernel_x_s, kernel_y_s)
    egm_r = get_magnitude(image_smooth, kernel_x_r, kernel_y_r)
    egm_p = get_magnitude(image_smooth, kernel_x_p, kernel_y_p)
    egm_k = image_smooth.convolve(kernel_k)
    egm_c = image_smooth.convolve(kernel_c)

    egm = egm_s.add(egm_r).add(egm_p).add(egm_k).add(egm_c)

    return ee.Image(egm).divide(5.0)


def egm_reduce(image, image_list):

    """
    Function helper for EE collection iteration

    Args:
        image (object)
        image_list (list)
    """

    # Get the Pan image
    pan_image = image.select(L8_PAN_BAND)

    # Compute the magnitude of the gradient and add metdata
    egm = compute_egm(pan_image).set('system:time_start', image.get('system:time_start'))

    # Return the list with the new EGM added
    return ee.List(image_list).add(egm)


def main():

    # The Landsat 8 collection
    sr_l8_collection = ee.ImageCollection('LANDSAT/LC8_L1T_TOA_FMASK')

    # Filter the collection with a 2x2 path/row
    wrs_filter = ee.Filter.Or(ee.Filter.And(ee.Filter.eq('WRS_PATH', WRS_PATH1),
                                            ee.Filter.eq('WRS_ROW', WRS_ROW1)),
                              ee.Filter.And(ee.Filter.eq('WRS_PATH', WRS_PATH1),
                                            ee.Filter.eq('WRS_ROW', WRS_ROW2)),
                              ee.Filter.And(ee.Filter.eq('WRS_PATH', WRS_PATH2),
                                            ee.Filter.eq('WRS_ROW', WRS_ROW1)),
                              ee.Filter.And(ee.Filter.eq('WRS_PATH', WRS_PATH2),
                                            ee.Filter.eq('WRS_ROW', WRS_ROW2)))

    sr_l8_collection_filtered = sr_l8_collection.filterDate(START_DATE,
                                                            END_DATE).filter(wrs_filter).sort('system:time_start',
                                                                                              False)

    sr_l8_c_masked = sr_l8_collection_filtered.map(mask_landsat)

    # Get the first pan layer
    pan_first = ee.Image(sr_l8_c_masked.first()).select(L8_PAN_BAND)

    # Extract the EGM from the Pan
    pan_egm = compute_egm(pan_first)

    # Get the timestamp from the most recent image in the reference collection.
    time0 = sr_l8_c_masked.first().get('system:time_start')

    first = ee.List([ee.Image(pan_egm).set('system:time_start', time0)])

    egm_c = ee.ImageCollection(ee.List(sr_l8_c_masked.iterate(egm_reduce, first)))

    egm_med = egm_c.median()

    # print egm_med.varName()

    # print help(egm_med.toArray)

    # print np.array((ee.Array(egm_med.get('B8')).getInfo()))

    # url = egm_med.getDownloadURL({'scale': 14,
    #                               'CRS': 'EPSG:4326'})


if __name__ == '__main__':
    main()
