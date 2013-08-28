# This file is part of pyrerp
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# Code for efficiently computing overlap-corrected rERPs in bounded memory, by
# *incrementally* building and calculating with a *sparse* design matrix.

import numpy as np
from scipy import sparse
import ctypes
import multiprocessing

from pyrerp.incremental_ls import XtXIncrementalLS

# How many data points/rows of the expanded X matrix should we process in each
# batch? Larger numbers produce a smaller number of larger jobs, so there's
# some trade-off: we want something small enough to effectively parallelize
# and avoid overwhelming memory usage, but there's some overhead associated
# with setting up each job so we don't want ridiculously small jobs. Currently
# this number is just a guess; TODO: try some different values to tune it
# better.
X_STRIP_SIZE = 3000

def incremental_rerp(eeg_data, artifact_starts, artifact_stops,
                     event_idx, design_matrix,
                     pre_event_samples, post_event_samples):
    XX

def _subdivide_slice(start, stop, piece_size):
    for sub_start in xrange(start, stop, piece_size):
        yield (sub_start,
               min(stop, sub_start + piece_size))

def test__subdivide_slice():
    assert (list(_subdivide_slice(4, 10, 2))
            == [(4, 6), (6, 8), (8, 10)])
    assert (list(_subdivide_slice(4, 10, 3))
            == [(4, 7), (7, 10)])
    assert (list(_subdivide_slice(4, 10, 4))
            == [(4, 8), (8, 10)])

def pick_slices(data_len, artifact_starts, artifact_stops,
                piece_size=X_STRIP_SIZE):
    good_start = 0
    for i in xrange(len(artifact_starts) + 1):
        if i < len(artifact_starts):
            good_stop = artifact_starts[i]
        else:
            good_stop = data_len
        for sub_slice in _subdivide_slice(good_start, good_stop, piece_size):
            yield sub_slice
        if i < len(artifact_starts):
            good_start = artifact_stops[i]

def test_pick_slices():
    assert (list(pick_slices(100, [10, 40], [12, 80], piece_size=8))
            == [(0, 8), (8, 10), (12, 20), (20, 28), (28, 36), (36, 40),
                (80, 88), (88, 96), (96, 100)])

def sparse_design_slice(event_idx, design_matrix,
                        pre_event_samples, post_event_samples,
                        slice_start, slice_stop):
    """Compute a slice of the 'expanded' design matrix.

    Given the locations of a set of events ('event_idx'), and their properties
    ('design_matrix'), whose effect is allowed to be non-zero on on
    'pre_event_samples' before and 'post_event_samples' after the event,
    efficiently computes a 

    Returns: a scipy.sparse.csc_matrix.
    """
    assert event_idx.ndim == 1
    # event_idx (and the parallel design matrix) must be sorted from low to
    # high. (If we want to support multiple simultaneous events then this will
    # need some adjustment; need to preprocess the design matrix to combine
    # any such rows by adding them together.)
    assert np.all(np.diff(event_idx) > 0)
    assert design_matrix.ndim == 2
    assert event_idx.shape[0] == design_matrix.shape[0]
    num_predictors = design_matrix.shape[1]
    samples_per_predictor = pre_event_samples + 1 + post_event_samples
    assert samples_per_predictor >= 1
    num_columns = num_predictors * samples_per_predictor

    # Each column in the expanded design matrix has non-zero entries which are
    # just a shifted slice from the original design matrix (plus a bunch of
    # zeros in between). So for the expanded column corresponding to latency
    # 0, we can put it directly into CSC format by using a slice of event_idx
    # (offset by slice_start) for the row indices, and a slice of
    # design_matrix for the non-zero values. The only tricky bit is shifting
    # the indices correctly for the other latencies, and handling the edge
    # conditions where some event might come into or out of view part way
    # through the edge of our slice, which means that different columns might
    # have different numbers of non-zeros.

    # latency + event_index - slice_start = location in output array
    # if this is in the range [0, slice_stop - slice_start), then there will
    # be an entry.
    # The smallest event index for which this is true is:
    #   latency + event_index - slice_start >= 0
    #   => event_index >= slice_start - latency
    # The strict upper bound on event_index is:
    #   latency + event_index - slice_start < slice_stop - slice_start
    #   => event_index < slice_stop - latency
    # Latency ranges over [-pre_event_samples, +post_events_samples]
    relevant_idx_start = np.searchsorted(event_idx,
                                         slice_start - post_event_samples)
    relevant_idx_stop = np.searchsorted(event_idx,
                                        slice_stop - -pre_event_samples)
    relevant_event_idx = event_idx[relevant_idx_start:relevant_idx_stop]
    print relevant_event_idx
    relevant_design_matrix = design_matrix[relevant_idx_start:relevant_idx_stop]
    # len(relevant_event_idx) is an upper bound on the number of nnz's you can
    # have in a single column. So we'll use this to get a conservative
    # estimate on how much space we'll need and use that to preallocate our
    # array:
    nnz_reserve = len(relevant_event_idx) * num_columns
    data = np.empty(nnz_reserve, dtype=design_matrix.dtype)
    # scipy.sparse matrices always use 32-bit indices
    indices = np.empty(nnz_reserve, dtype=np.int32)
    indptr = np.empty(num_columns + 1, dtype=np.int32)
    indptr[0] = 0
    data_idx = 0
    column_idx = 0
    for predictor_idx in xrange(num_predictors):
        for latency_idx in xrange(samples_per_predictor):
            latency = latency_idx - pre_event_samples
            this_start = np.searchsorted(relevant_event_idx,
                                         slice_start - latency)
            this_stop = np.searchsorted(relevant_event_idx,
                                        slice_stop - latency)
            this_nnz = this_stop - this_start
            this_data = relevant_design_matrix[this_start:this_stop, predictor_idx]
            data[data_idx:data_idx + this_nnz] = this_data
            this_event_idx = relevant_event_idx[this_start:this_stop]
            indices[data_idx:data_idx + this_nnz] = this_event_idx
            # Convert from original-data indices to indices within our matrix
            # slice
            indices[data_idx:data_idx + this_nnz] += (-slice_start + latency)
            data_idx += this_nnz
            indptr[column_idx + 1] = data_idx
            column_idx += 1
    return sparse.csc_matrix((data, indices, indptr),
                             shape=(slice_stop - slice_start, num_columns))

def test_sparse_design_slice():
    design_matrix = np.asarray([[1, 2],
                                [1, 3],
                                [1, 9]])
    event_idx = np.asarray([3, 4, 7])
    pre_event_samples = 2
    post_event_samples = 3

    expanded = np.asarray([[0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0], # 0
                           [1, 0, 0, 0, 0, 0,  2, 0, 0, 0, 0, 0], # 1
                           [1, 1, 0, 0, 0, 0,  3, 2, 0, 0, 0, 0], # 2
                           [0, 1, 1, 0, 0, 0,  0, 3, 2, 0, 0, 0], # 3
                           [0, 0, 1, 1, 0, 0,  0, 0, 3, 2, 0, 0], # 4
                           [1, 0, 0, 1, 1, 0,  9, 0, 0, 3, 2, 0], # 5
                           [0, 1, 0, 0, 1, 1,  0, 9, 0, 0, 3, 2], # 6
                           [0, 0, 1, 0, 0, 1,  0, 0, 9, 0, 0, 3], # 7
                           [0, 0, 0, 1, 0, 0,  0, 0, 0, 9, 0, 0], # 8
                           [0, 0, 0, 0, 1, 0,  0, 0, 0, 0, 9, 0], # 9
                           [0, 0, 0, 0, 0, 1,  0, 0, 0, 0, 0, 9], # 10
                           [0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0], # 11
                           ])
    
    for start in xrange(0, 11):
        for stop in xrange(start + 1, 12):
            # start = 0; stop = 2
            # import pdb; pdb.set_trace()
            expected = expanded[start:stop, :]
            got = sparse_design_slice(event_idx, design_matrix,
                                      pre_event_samples, post_event_samples,
                                      start, stop)
            print start, stop
            print expected
            print got.todense()
            assert np.all(got.todense() == expected)
                                 
