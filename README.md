This repository is no longer maintained.  


SACFEI
---

Semi-Automatic Crop Field Extraction from Imagery

##### Google Earth Engine Python API

* [Earth Engine Python API](https://developers.google.com/earth-engine/python_install_manual)
    * SACFEI can be used as a stand-alone Python library, but also has scripts for extracting edges from
    Sentinel-2 10m and Landsat 7/8 15m pan from Google Cloud storage 
    
On Linux, the installation steps for the GEE Python package consist of:    
    
```commandline
> pip3.6 install --user earthengine-api
> python3.6 -c "import ee; ee.Initialize()"
```

...and then following the on-screen and browser steps.

SACFEI installation
---

```commandline
> git clone https://github.com/jgrss/sacfei.git
> cd sacfei/
> python3.6 setup.py build && pip3.6 install --user --upgrade . && rm -rf build/
```

Python usage 
---

#### Load the library:
    
```python
>>> import sacfei
```

#### Basic processing with default parameters:

```python
>>> # Here, `ts_array` is a multi-dimensional 
>>> #   time-series array 
>>> saf = sacfei.SACFEI(ts_array)
>>>
>>> # Extract edges
>>> saf.extract_egm()
>>>
>>> # Edge gradient magnitude
>>> print(saf.egm)
>>>
>>> # Segment edges
>>> saf.segment_egm()
>>>
>>> print(saf.objects)
```

Google Earth Engine
---

- The Google Earth Engine API (see above) is needed to run the GEE Python interface.

```commandline
cd /sacfei/scripts/
```

#### Print help 

```commandline
python3.6 gee.py -h
```

#### Processing examples

- When running `gee.py`, the output GeoTiffs will be saved to your Google Drive folder that is specified with `--drive-folder`.
  - When finished, use `gdcp` (below) to batch download files.

Process one year of data for MGRS grid 11SPS over Europe

```commandline
python3.6 gee.py --start-date 2016-1-1 --end-date 2017-1-1 --mgrs-grids 11SPS --export-name my_image --export-cell 10 --export-crs euaeac --drive-folder GEE_dir
```

Process MGRS grids over South America, where the grids to process are given by a shapefile

```commandline
python3.6 gee.py --start-date 2016-1-1 --end-date 2017-1-1 --mgrs-region /mgrs_grids.shp --export-name my_image --export-cell 15 --export-crs saaeac --drive-folder GEE_dir
```

Post-Google Earth Engine
---

#### Batch download Google Drive directories

* [Google Drive cp](https://github.com/ctberthiaume/gdcp)

E.g.,

```commandline
gdcp download -i <Google Drive folder id> <output location>
```

#### Edit projection and data type

```commandline
python3.6 post_gee.py -i /GEE -o /EGM --export-crs euaeac
```

#### Run the SACFEI batch script over an image, block by block

Print command-line options

```command
python3 batch.py -h
```

Basic usage intersecting a land cover map

```commandline
python3 batch.py -i /EGM/my_image_11SPS_edited.tif -o /objects/objects.tif --lc-image /land_cover.tif --method all --row-block 1000 --col-block 1000 --n-jobs 8
```

The command below specifies:

1. the image normalization grid tile sizes
2. the edge thresholding window sizes
3. relaxes edge probabilities
4. applies post-segmentation object cleaning
5. specifies the logistic scaling paramaters
6. appends additional image variables to each object
7. uses WGS84 lat/lon CRS

```commandline
python3 batch.py -i /EGM/edited_mosaic.vrt -o /objects/objects.tif --grid-tiles 16 32 --thresh-windows 25 51 --relax-probas --apply-clean --logistic-alpha 1.5 --logistic-beta 0.7 --projection wgs84 --var-image /extra_variables.vrt --method all --row-block 1000 --col-block 1000 --n-jobs 8
```
