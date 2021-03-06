import sys
from setuptools import setup, find_packages
from setuptools import Extension
import os

SETUP_METADATA = \
               {
    "name": "vamb",
    "description": "Variational autoencoder for metagenomic binning",
    "url": "https://github.com/RasmussenLab/vamb",
    "author": "Jakob Nybo Nissen and Simon Rasmussen",
    "author_email": "jakobnybonissen@gmail.com",
    "version": "1.0.0",
    "license": "MIT",
    "packages": find_packages(),
    "package_data": {"vamb": ["kernel.npz"]},
    "entry_points": {'console_scripts': [
        'vamb = vamb.__main__:main'
        ]
    },
    "scripts": ['src/concatenate.py'],
    "ext_modules": [Extension("vamb._vambtools",
                               sources=["src/_vambtools.pyx"],
                               language="c")],
    # "install_requires": ["numpy>=1.15", "torch>=1.9.0", "pysam>=0.14", "torchvision>=0.10"],
    "install_requires": ["numpy>=1.15", "pysam>=0.14"],
    "setup_requires": ['Cython>=0.25.2', "setuptools>=38.6.0"],
    "python_requires": ">=3.7",
    "classifiers":[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    }

setup(**SETUP_METADATA)
