# This file is part of pyrerp
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

__version__ = "0.0.0+dev"

# Do this first, to make it easy to check for warnings while testing:
import os
if os.environ.get("PYRERP_FORCE_NO_WARNINGS"):
    import warnings
    warnings.filterwarnings("error", module="^pyrerp")
    del warnings
del os
