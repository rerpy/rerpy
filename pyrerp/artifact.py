# This file is part of pyrerp
# Copyright (C) 2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# Can we efficiently calculate these for sliding windows?

# mxflat is done

# for ppa, this can be done with scipy.ndimage.maximum_filter1d,
# minimum_filter1d (with mode="nearest")

# for polinv/pinv, let x = the difference between the channels. then we need
# the maximum (or absolute maximum?) and mean of each window, and we compute
# the difference between these.

import numpy as np
from pyrerp._artifact import flat_spans

def test_flat_spans():
    assert np.all(flat_spans(10.0, np.asarray([
                    10, 15, 20, 23, -10, -5, 100, 100, 105, 95], dtype=float))
                  == [3, 3, 2, 1, 2, 1, 4, 3, 2, 1])
    assert np.all(flat_spans(10.0, np.asarray([
                    10, 15, 5, 0, 110, 110, 105, 100], dtype=float))
                  == [3, 2, 2, 1, 4, 3, 2, 1])

# yields (start, stop) tuples that each have length at least min_length, and
# for which all the values fall into some range value_low <= value <=
# value_low + 2*flatness_threshold.
def reject_flat(data, flatness_threshold, min_length):
    pass
