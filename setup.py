#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from os import path

import setuptools
from setuptools import find_packages
from setuptools import setup

version = int(setuptools.__version__.split('.')[0])
assert version > 30, 'requires setuptools > 30'

this_directory = path.abspath(path.dirname(__file__))

try:
    import pypandoc
    long_description = pypandoc.convert_file(path.join(this_directory, 'README.md'), 'rst')
except ModuleNotFoundError:
    long_description = ''


__version__ = '0.5.0'


setup(
    name='ml-dataloader',
    url='https://github.com/ericxsun/ml-dataloader.git',
    keywords='machine learning, deep learning, neural network, data processing',
    version=__version__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['examples', 'tests']),
    zip_safe=False,
    install_requires=[
        'numpy>=1.14',
        'six',
        'msgpack>=0.5.2',
        'msgpack-numpy>=0.4.4.2',
        'pyzmq>=16',
        'psutil>=5',
        'multiprocess',
        'prefetch_generator'
    ],
    extras_require={
        'all: "linux" in sys_platform': ['python-prctl'],
    },
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#universal-wheels
    options={'bdist_wheel': {'universal': '1'}}
)
