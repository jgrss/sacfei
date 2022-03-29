import setuptools
from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

try:
    from Cython.Distutils import build_ext
except:
    from distutils.command import build_ext

import numpy as np


__version__ = '0.1.2'

pkg_name = 'sacfei'
maintainer = 'Jordan Graesser'
maintainer_email = ''
description = 'Semi-Automatic Crop Field Extraction from Imagery'
git_url = 'https://github.com/jgrss/sacfei.git'

with open('README.md') as f:
    long_description = f.read()

with open('LICENSE.txt') as f:
    license_file = f.read()

with open('AUTHORS.txt') as f:
    author_file = f.read()

required_packages = ['matplotlib>=2.0.0',
                     'joblib>=0.11.0',
                     'future',
                     'six>=1.11.0',
                     'numpy>=1.14',
                     'scipy>=0.19.0',
                     'scikit-image>=0.13',
                     'cython>=0.28.0',
                     'opencv-python>=3.4.0',
                     'scikit-learn>=0.19.0',
                     'pandas>=0.22.0',
                     'geopandas>=0.3.0',
                     'mahotas']


def get_extensions():

    extensions = [Extension('*',
                            sources=['sacfei/_moving_window.pyx']),
                  Extension('*',
                            sources=['sacfei/_adaptive_threshold.pyx'])]

    return extensions


def get_packages():
    return setuptools.find_packages()


def get_package_data():

    return {'': ['*.md', '*.txt'],
            'data': ['*.tfw',
                     '*.tif'],
            'files': ['*.dbf',
                      '*.prj',
                      '*.qpj',
                      '*.shp',
                      '*.shx'],
            'sacfei': ['*.pyx']}


def setup_package():

    metadata = dict(name=pkg_name,
                    maintainer=maintainer,
                    maintainer_email=maintainer_email,
                    description=description,
                    license=license_file,
                    version=__version__,
                    long_description=long_description,
                    author=author_file,
                    packages=get_packages(),
                    package_data=get_package_data(),
                    ext_modules=cythonize(get_extensions()),
                    cmdclass=dict(build_ext=build_ext),
                    install_requires=required_packages,
                    zip_safe=False,
                    download_url=git_url,
                    include_dirs=[np.get_include()])

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
