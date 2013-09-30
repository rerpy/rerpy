#!/usr/bin/env python

import os
import sys

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
try:
    import numpy as np
except ImportError:
    sys.exit("numpy must be installed before this package can be built")

DESC = """XX"""

# This should be valid ReST.
LONG_DESC = (DESC + "\n"
             "XX")

setup(
    name="rerpy",
    version="0.0.0+dev",
    description=DESC,
    long_description=LONG_DESC,
    author="Nathaniel J. Smith",
    author_email="njs@pobox.com",
    license="XX None yet",
    packages=["rerpy"],
    url="https://github.com/rerpy",
    requires=["numpy", "scipy", "pandas", "patsy"],
    classifiers =
      [ "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 2",
        "Topic :: Scientific/Engineering",
        ],
    ext_modules = [
        Extension("rerpy.io._erpss",
                  ["rerpy/io/_erpss.pyx"],
                  include_dirs=[np.get_include()],
                  ),
        Extension("rerpy._artifact",
                  ["rerpy/_artifact.pyx"],
                  include_dirs=[np.get_include()],
                  libraries=["m"],
                  ),
        ],
    cmdclass={"build_ext": build_ext},
    )
