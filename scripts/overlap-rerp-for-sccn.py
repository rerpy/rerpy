#!/usr/bin/env python

# Copyright (C) 2012 Nathaniel Smith <njs@pobox.com>
# Licensed under the GNU GPL v2 as released by the Free Software Foundation
# or, at your option, any later version.

# To use:
#   python overlap-rerp-for-sccn.py INPUT.mat OUTPUT.mat
# where:
#   INPUT.mat is a MATLAB .mat file saved with -v7 (NOT -v7.3 or later!)
#   containing variables with the following names:
#     eeg_data: a NxM matrix where M is the number of distinct channels in the
#       data, and N is the number of distinct time points
#     event_idx: a vector containing the offset of each event in the eeg_data
#       matrix, using 1-based (MATLAB-style) indexing. Must be sorted in
#       ascending order.
#     design_matrix: a matrix where each row corresponds to one event, and
#       each column gives the attributes of each event as coded numerically
#       for linear regression.
#     pre_event_samples: the length of the baseline period, in samples
#     post_event_samples: the length of the post-event epoch, in samples
#     artifact_starts: a vector containing the offset of each chunk of the
#       eeg_data which should be discarded due to artifacts. This should name
#       the first data point which you wish to ignore, with 1-based
#       (MATLAB-style) indexing.
#     artifact_stops: likewise, but giving the offsets of the ends of each
#       chunk of artifactual data. This should name the last data point which
#       you wish to ignore in each chunk, again with 1-based (MATLAB style)
#       indexing.
# This will produce a file OUTPUT.mat, which is a MATLAB .mat file containing
# a single NxM matrix named "betas", where each column corresponds to one
# channel in the input data, and each row gives the regression (beta)
# coefficients for that channel. N = (epoch length * num predictors). The
# coefficients are ordered as, first all the coefficients for the first
# predictor, then all the coefficients for the second predictor, etc.

# Use --help to see all options.

# This file contains some self-tests (though not of the final algorithm). Use
# 'nosetests <this file>' to run them.

# How many data points/rows of the expanded X matrix should we process in each
# batch? Larger numbers produce a smaller number of larger jobs, so there's
# some trade-off: we want something small enough to effectively parallelize
# and avoid overwhelming memory usage, but there's some overhead associated
# with setting up each job so we don't want ridiculously small jobs. Currently
# this number is just a guess.
DEFAULT_X_STRIP_SIZE = 5000

import multiprocessing
import itertools
from optparse import OptionParser
import os
import numpy as np
from scipy import sparse
from scipy.io import loadmat, savemat

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
                piece_size):
    assert len(artifact_starts) == len(artifact_stops)
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
                                 
def compute_XtX_XtY_for_slice(start, stop):
    try:
        global _GLOBAL_SHARED_DATA_
        g = _GLOBAL_SHARED_DATA_
        X = sparse_design_slice(g["event_idx"], g["design_matrix"],
                                g["pre_event_samples"], g["post_event_samples"],
                                start, stop)
        XtX = X.T * X
        XtY = X.T * g["eeg_data"]
        return XtX, XtY
    except KeyboardInterrupt:
        # Avoid annoying console spew when someone hits Control-C
        return None

def main():
    parser = OptionParser(usage="usage: %prog [options] INPUT.mat OUTPUT.mat")
    if os.name == "posix":
        parser.add_option("-p", "--parallel", dest="parallel",
                          help="number of threads "
                          "(default: available CPUs; use 0 to disable threading code)",
                          default=None,
                          metavar="N")
    parser.add_option("-c", "--chunksize", dest="chunk_size",
                      default=DEFAULT_X_STRIP_SIZE,
                      metavar="N",
                      help="number of matrix rows to process at a time in each "
                           "parallel job (higher -> more memory usage, "
                           "less parallelism, reduced threading overhead)")
    options, args = parser.parse_args()
    if len(args) != 2:
        parser.error("wrong number of arguments")
    if os.name != "posix":
        options.parallel = 0

    (input_path, output_path) = args
    input_data = loadmat(input_path, matlab_compatible=True)
    # A hack: we stash the big data into a global variable before using
    # multiproessing to fork off threads, so the child processes will inherit
    # the data in shared memory. (This only works on Unix, which is why we
    # disable the multi-threading code on other systems.)
    global _GLOBAL_SHARED_DATA_
    g = {}
    g["eeg_data"] = input_data["eeg_data"]
    # -1 converts from MATLAB-style indexing to Python-style indexing
    g["event_idx"] = input_data["event_idx"].squeeze() - 1
    g["design_matrix"] = input_data["design_matrix"]
    g["pre_event_samples"] = input_data["pre_event_samples"].item()
    g["post_event_samples"] = input_data["post_event_samples"].item()
    g["artifact_starts"] = input_data["artifact_starts"].squeeze() - 1
    # Need a -1 to convert this to 0-based indexing, then a +1 to convert it
    # to Python-style half-open interval ranges instead of MATLAB-style closed
    # interval ranges... which cancel out, so in fact we can use the numbers
    # as is:
    g["artifact_stops"] = input_data["artifact_stops"].squeeze()
    _GLOBAL_SHARED_DATA_ = g

    num_channels = g["eeg_data"].shape[1]
    epoch_len = g["pre_event_samples"] + 1 + g["post_event_samples"]
    X_columns = epoch_len * g["design_matrix"].shape[1]
    XtX_accumulator = np.zeros((X_columns, X_columns))
    XtY_accumulator = np.zeros((X_columns, num_channels))

    if options.parallel == 0:
        # Serial code, in-process
        pool = None
        imap_fn = itertools.imap
    else:
        # With any other value for parallel, we spawn worker processes. (This
        # includes parallel=1. parallel=1 won't be any faster than parallel=0,
        # but it might be useful for testing the worker process code.)
        pool = multiprocessing.Pool(options.parallel)
        imap_fn = pool.imap_unordered

    try:
        for (XtX, XtY) in imap_fn(compute_XtX_XtY_for_slice,
                                  pick_slices(g["eeg_data"].shape[0],
                                              g["artifact_starts"],
                                              g["artifact_stops"],
                                              options.chunksize)):
            XtX_accumulator += XtX
            XtY_accumulator += XtY
    finally:
        if pool is not None:
            pool.terminate()
    betas = np.solve(XtX_accumulator, XtY_accumulator)
    savemat(output_path, {"betas": betas}, oned_as="column")

if __name__ == "__main__":
    main()
