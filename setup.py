#! /ur/bin/env python
from setuptools

DESCRIPTION = "Finetune Mask R-CNN"
URL = "https://github.com/cwood1967/xlearn"
LICENSE = 'MIT'
VERSION = '0.0.1'
PYTHON_REQUIRES = ">=3.6"

INSTALL_REQUIRES = [
    'numpy>=1.15',
    'scipy>=1.0',
    'pandas>=0.23',
    'matplotlib>=2.2',
    'nd2reader>=3.2.3',
    'tifffile>=2020.7.24',
]

if __name__ == "__main__":
    from setuptools import setup

    import sys
    if sys.version_info[:2] < (3, 6):
        raise RuntimeError("xlearn python >= 3.6.")

    setup(
        name='xlearn',
        author="Chris Wood",
        author_email="cjw@stowers.org",
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        packages=setuptools.find_packages(),
    )