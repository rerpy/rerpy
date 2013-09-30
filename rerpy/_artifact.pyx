# This file is part of rERPy
# Copyright (C) 2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

cimport numpy as np
import numpy as np
from libc.math cimport fmin, fmax

# cdef extern from "math.h":
#     double fmax(double x, double y)
#     double fmin(double x, double y)

# Checks for blocking by finding the length of spans in which the data values
# do not vary by more than 'limit'. Does this by simply calculating the
# longest such span which begins at each point in the array; to find all
# fixed-width windows that contain a span with length > some constant, just
# run a maximum filter over this array. Or just find all entries whose values
# are > that constant. Equivalent to kutaslab artifact rejection function
# "mxflat".

# Technically this is O(n^2), but in practice we'll usually terminate in
# just a few iterations for almost all points, so it's more like n*5 or
# something.

# XX FIXME: should probably also accept float32 here (use fused types?)
def flat_spans(np.float64_t limit, np.float64_t [:] data not None):
    cdef np.npy_intp [:] spans
    cdef int i, j
    cdef np.float64_t low, high
    py_spans = np.empty(data.shape[0], dtype=int)
    spans = py_spans
    for i in range(data.shape[0]):
        low = high = data[i]
        j = i
        while j < data.shape[0]:
            low = fmin(low, data[j])
            high = fmax(high, data[j])
            if high - low > limit:
                break
            j += 1
        spans[i] = j - i
    return py_spans

def find_flat_spans(np.float64_t [:] data not None,
                    np.float64_t max_variation,
                    int min_length,
                    int pad=0):
    cdef np.npy_intp [:] spans
    cdef int i, j
    cdef np.float64_t low, high
    for i in range(data.shape[0]):
        low = high = data[i]
        j = i
        while j < data.shape[0]:
            low = fmin(low, data[j])
            high = fmax(high, data[j])
            if high - low > max_variation:
                break
            j += 1
        # now j points at the first item for which [i, j] is not a valid flat
        # span (either b/c data[j] is undefined, or because adding
        # data[j] to the span exceeded the max_variation threshold). Therefore
        # [i, j) is the longest flat span starting at i.
        if j - i > min_length:
            yield (i - pad, j + pad)
            
