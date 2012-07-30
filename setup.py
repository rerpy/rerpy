#!/usr/bin/env python

import os
import sys
# Add our fake Pyrex at the end of the Python search path
# in order to fool setuptools into allowing compilation of
# pyx files to C files. Importing Cython.Distutils then
# makes Cython the tool of choice for this rather than
# (the possibly nonexisting) Pyrex.
project_path = os.path.split(__file__)[0]
sys.path.append(os.path.join(project_path, 'fake_pyrex'))

from setuptools import setup, Extension
from Cython.Distutils import build_ext
try:
    import numpy as np
except ImportError:
    sys.exit("numpy must be installed before this package")

DESC = """XX"""

# This should be valid ReST.
LONG_DESC = (DESC + "\n"
             "XX")

setup(
    name="pyrerp",
    version="0.0.0+dev",
    description=DESC,
    long_description=LONG_DESC,
    author="Nathaniel J. Smith",
    author_email="njs@pobox.com",
    license="XX None yet",
    packages=["pyrerp"],
    url="https://github.com/njsmith/pyrerp",
    install_requires=["numpy", "scipy", "patsy"],
    classifiers =
      [ "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 2",
        "Topic :: Scientific/Engineering",
        ],
    ext_modules = [
        Extension("pyrerp._kutaslab",
                  ["pyrerp/_kutaslab.pyx"],
                  include_dirs=[np.get_include()],
                  ),
        ],
    cmdclass={"build_ext": build_ext},
    )
