# This file is part of pyrerp
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

__version__ = "0.0.0+dev"

# Do this first, to make it easy to check for warnings while testing:
import os
if "PYRERP_WARNINGS_MODE" in os.environ:
    import warnings
    warnings.filterwarnings(os.environ["PYRERP_WARNINGS_MODE"],
                            module="^pyrerp")
    del warnings
del os
