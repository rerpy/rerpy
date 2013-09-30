# This file is part of rERPy
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

__version__ = "0.0.0+dev"

# Do this first, to make it easy to check for warnings while testing:
import os
if "RERPY_WARNINGS_MODE" in os.environ:
    import warnings
    warnings.filterwarnings(os.environ["RERPY_WARNINGS_MODE"],
                            module="^rerpy")
    del warnings
del os

from rerpy.data import DataSet, DataFormat
from rerpy.io.erpss import load_erpss
