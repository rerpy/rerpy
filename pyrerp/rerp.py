# This file is part of pyrerp
# Copyright (C) 2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

import itertools
from collections import namedtuple, OrderedDict

import numpy as np
import scipy.sparse as sp
from patsy import build_design_matrices

from pyrerp.incremental_ls import XtXIncrementalLS
from pyrerp.parimap import parimap_unordered

# Can't use a namedtuple for this because we want __eq__ and __hash__ to use
# identity semantics, not value semantics.
class _Epoch(object):
    def __init__(self, start_idx, epoch_len,
                 design_row, design_offset, expanded_design_offset):
        self.start_idx = start_idx
        self.epoch_len = epoch_len
        self.design_row = design_row
        self.design_offset = design_offset
        self.expanded_design_offset = expanded_design_offset

# If an artifact has a non-None "epoch" field, then it means that when overlap
# correction is disabled, this artifact only applies to the given epoch. (This
# is useful for "artifacts" like, that event missing some predictors. This
# needs to be treated as an artifact when overlap rejection is turned on, but
# not otherwise.)
_Artifact = namedtuple("_Artifact", ["type", "epoch"])

def multi_rerp_impl(data_set, rerp_specs, artifact_query, artifact_type_field,
                    overlap_correction, regression_strategy):
    # We need to be able to refer to individual recording spans in a
    # convenient, sortable way -- but Recording objects aren't sortable and
    # are otherwise inconvenient to work with. So we assign a sequential
    # integer id to each (Recording, span_id) pair.
    recording_span_lengths = data_set.span_lengths
    recording_span_intern_table = {}
    for (i, recording_and_span_id) in enumerate(recording_span_lengths):
        recording_span_intern_table[recording_and_span_id] = i

    # This list contains 3-tuples that record the position of interesting
    # events in the data, like epochs that are to be analyzed and
    # artifacts that are to be rejected. Each entry is of the form:
    #   (start, stop, tag)
    # where tag is an arbitrary hashable object, and start and stop are
    # pairs (interned recording span, offset). (Most code just cares that
    # they sort correctly, though.) Artifacts use _Artifact objects,
    # epochs use _Epoch objects.
    spans = []

    # Get artifacts.
    spans.extend(_artifact_spans(recording_span_intern_table, data_set,
                                 artifact_query, artifact_type_field))

    # And now get the events themselves, and calculate the associated epochs
    # (and any associated artifacts).
    epoch_span_info = _epoch_spans(recording_span_intern_table,
                                   data_set, rerp_specs)
    (rerp_infos, epoch_spans,
     design_width, expanded_design_width) = epoch_span_info
    spans.extend(epoch_spans)
    del epoch_spans, epoch_span_info

    # Now we have a list of all interesting events, and where they occur, but
    # it's organized so that for each event we can see where in the data it
    # happened. What we need is for each (interesting) position in the data,
    # to be see all events that are happening then. _epoch_subspans pulls out
    # subspans of the epoch spans, and annotates each with the epochs and
    # artifacts that apply to it.

    epochs_with_data = set()
    epochs_with_artifacts = set()
    total_wanted_ticks = 0
    total_good_ticks = 0
    # _epoch_subspans gives us chunks of data that we would *like* to analyze,
    # along with the epochs and artifacts that are relevant to it. This
    # representation already incorporates the effects of having overlap
    # correction enabled or disabled; if overlap correction is disabled, then
    # we will have multiple spans that refer to the same data but with
    # different associated epochs. So have_multi_epoch_data in practice will
    # be True iff overlap_correction=True AND there are actually-overlapping
    # epochs.
    have_multi_epoch_data = False
    analysis_spans = []
    for epoch_subspan in _epoch_subspans(spans, overlap_correction):
        start, stop, epochs, artifacts = epoch_subspan
        assert epochs
        total_wanted_ticks += stop[1] - start[1]
        if artifacts:
            epochs_with_artifacts.update(epochs)
            _count_artifacts(artifact_counts, stop[1] - start[1], artifacts)
        else:
            epochs_with_data.update(epochs)
            if len(epochs) > 1:
                have_multi_epoch_data = True
            total_good_ticks += stop[1] - start[1]
            analysis_spans.append((start, stop, epochs))

    regression_strategy = _choose_strategy(regression_strategy,
                                           epochs_with_data,
                                           epochs_with_artifacts,
                                           have_multi_epoch_data)

    if regression_strategy == "by-epoch":
        worker = _ByEpochWorker(design_width)
    elif regression_strategy == "continuous":
        worker = _ContinuousWorker(expanded_design_width)
    else:
        assert False
    jobs_iter = _analysis_jobs(data_set, recording_span_intern_table,
                               analysis_spans)
    model = XtXIncrementalLS()
    for job_result in parimap_unordered(worker, jobs_iter):
        model.append_bottom_half(job_result)
    result = model.fit()
    betas = result.coef()
    # For continuous fits, this has no effect. For by-epoch fits, it
    # rearranges the beta matrix so that each column contains results for one
    # channel, and the rows go:
    #   predictor 1, latency 1
    #   ...
    #   predictor 1, latency n
    #   predictor 2, latency 1
    #   ...
    #   predictor 2, latency n
    #   ...
    betas.resize((-1, len(data_set.data_format.channel_names)))

    return rERPAnalysis(data_set.data_format,
                        rerp_infos, overlap_correction, regression_strategy,
                        total_wanted_ticks, total_good_ticks,
                        artifact_counts, betas)

def _artifact_spans(recording_span_intern_table,
                    data_set, artifact_query, artifact_type_field):
    # Create "artifacts" to prevent us from trying to analyze any
    # data the falls outwith the bounds of our recordings.
    for recspan, length in data_set.span_lengths.iteritems():
        recording_span_intern = recording_span_intern_table[recspan]
        neg_inf = (recording_span_intern, -2**31)
        zero = (recording_span_intern, 0)
        end = (recording_span_intern, length)
        pos_inf = (recording_span_intern, 2**31)
        yield (neg_inf, zero, _Artifact("_NO_RECORDING", None))
        yield (end, pos_inf, _Artifact("_NO_RECORDING", None))

    # Now lookup the actual artifacts recorded in the events structure.
    for artifact_event in self.events.find(artifact_query):
        artifact_type = artifact_event.get(artifact_type_field, "_UNKNOWN")
        if not isinstance(artifact_type, basestring):
            raise TypeError("artifact type must be a string, not %r"
                            % (artifact_type,))
        recspan = (artifact_event.recording, artifact_event.span_id)
        recording_span_intern = recording_span_intern_table[recspan]
        yield ((recording_span_intern, artifact_event.start_idx)
               (recording_span_intern, artifact_event.stop_idx),
               _Artifact(artifact_type, None))

def _epoch_spans(recording_span_intern_table, data_set, rerp_specs):
    rerp_infos = []
    rerp_names = set()
    spans = []
    design_offset = 0
    expanded_design_offset = 0
    data_format = data_set.data_format
    for rerp_spec in rerp_specs:
        start_offset = data_format.ms_to_samples(rerp_spec.start_time)
        # Offsets are half open: [start, stop)
        # But, it's more intuitive for times to be closed: [start, stop]
        # So we interpret the user times as a closed interval, and add 1
        # sample when converting to offsets.
        stop_offset = 1 + data_format.ms_to_samples(rerp_spec.stop_time)
        if start_offset >= stop_offset:
            raise ValueError("Epochs must be >1 sample long!")
        event_set = self.events.find(rerp_spec.event_query)
        design = patsy.dmatrix(rerp_spec.formula, event_set,
                               return_type="dataframe")
        for i in xrange(len(event_set)):
            event = event_set[i]
            recording_span_intern = recspan_intern_from_event(event)
            epoch_start = start_offset + event.start_idx
            epoch_stop = stop_offset + event.start_idx
            epoch_span = ((recording_span_intern, epoch_start),
                          (recording_span_intern, epoch_stop))
            if i not in design.index:
                design_row = None
            else:
                design_row = np.asarray(design.loc[i, :])
            epoch = _Epoch(epoch_start, epoch_stop - epoch_start,
                           design_row, design_offset, expanded_design_offset)
            if design_row is None:
                # Event thrown out due to missing predictors; this
                # makes its whole epoch into an artifact -- but if overlap
                # correction is disabled, then this artifact only affects
                # this epoch, not anything else. (We still want to treat
                # it as an artifact though so we get proper accounting at
                # the end.)
                artifact = _Artifact("_MISSING_PREDICTOR", epoch)
                spans.append(epoch_span + (artifact,))
            spans.append(epoch_span + (epoch,))
        if rerp_spec.name in rerp_names:
            raise ValueError("name %r used for two different sub-analyses"
                             % (rerp_spec.name,))
        rerp_names.add(rerp_spec.name]
        rerp_infos.append({
            "spec": rerp_spec,
            "design_info": design.design_info,
            "start_offset": start_offset,
            "stop_offset": stop_offset,
            "design_offset": design_offset,
            "expanded_design_offset": expanded_design_offset,
            })
        design_offset += design.shape[1]
        epoch_samples = stop_offset - start_offset
        expanded_design_offset += epoch_samples * design.shape[1]

    return rerp_infos, spans, design_offset, expanded_design_offset

################################################################
# These span handling functions are rather confusing at first glance.
#
# General theory of operation:
#
# We have a list of when each artifact starts and stops, and when each
# analysis epoch starts and stops.
#
# We want to know:
# 1) Which spans of data are viable for analysis, and which events are live in
#    each? (This lets us construct the regression design matrix.)
# 2) Which spans of data do we want to analyze, but can't because of
#    artifacts? And which artifacts? (This lets us keep track of which
#    artifacts are making us throw out data.)
# 3) Are there any epochs in which some-but-not-all points are viable for
#    analysis? (This is needed to know whether the by-epoch regression
#    strategy is possible.)
# 4) If overlap_correction=True, are there any epochs that do in fact overlap?
#    (This is needed to know whether the by-epoch regression
#    strategy is possible.)
#
# Annoying wrinkles that apply when overlap_correction=False:
# In this mode, then each epoch effectively gets its own copy of the
# data. Also, in this mode, most artifacts are shared across this virtual
# copies, but there are some "artifacts" that are specific to a particular
# epoch (specifically, those that have something to do with the event itself,
# such as missing data).
#
# Our strategy:
# Convert our input span-based representation into an event-based
# representation, where an event is "such-and-such starts happening at
# position p" or "such-and-such stops happening at position p". Then, we can
# scan all such events from left to right, and incrementally generate a
# representation of *all* epochs/artifacts are happening at each position.

def _epoch_subspans(spans, overlap_correction):
    state_changes = []
    for span in spans:
        start, stop, tag = span
        state_changes.append((start, tag, +1))
        state_changes.append((stop, tag, -1))
    state_changes.sort()
    last_position = None
    current_epochs = {}
    current_artifacts = {}
    for position, tag, change in state_changes:
        if (last_position is not None
            and position != last_position
            and current_epochs):
            if overlap_correction:
                yield (last_position, position,
                       tuple(current_epochs), tuple(current_artifacts))
            else:
                for epoch in current_epochs:
                    yield (last_position, position,
                           (epoch,),
                           tuple(art for art in current_artifacts
                                 if art.epoch is None or art.epoch is epoch))
        if isinstance(tag, _Epoch):
            tag_dict = current_epochs
        else:
            assert isinstance(tag, _Artifact)
            tag_dict = current_artifacts
        if incr == 1 and tag not in tag_dict:
            tag_dict[tag] = 0
        tag_dict[tag] += incr
        if incr == -1 and tag_dict[tag] == 0:
            del tag_dict[tag]
    assert current_epochs == current_artifacts == {}

def _count_artifacts(counts, num_points, artifacts):
    for artifact in artifacts:
        if artifact.type not in counts:
            counts[artifact.type] = {
                "total": 0,
                "unique": 0,
                "proportional": 0,
                }
        subcounts = counts[artifact.type]
        subcounts["total"] += num_points
        if len(artifacts) == 1:
            subcounts["unique"] += num_points
        subcounts["proportional"] += num_points * 1.0 / len(artifacts)

def _choose_strategy(requested_strategy,
                     epochs_with_data, epochs_with_artifacts,
                     have_multi_epoch_data):
    if requested_strategy == "continuous":
        return requested_strategy
    have_partial_epochs = bool(epochs_with_data.intersection(epochs_with_artifacts))
    by_epoch_possible = not (have_partial_epochs or have_multi_epoch_data)
    if requested_strategy == "auto":
        if by_epoch_possible:
            return "by-epoch"
        else:
            return "continuous"
    elif requested_strategy == "by-epoch":
        if not by_epoch_possible:
            reasons = []
            if have_partial_epochs:
                reasons.append("at least one epoch is partially but not "
                               "fully eliminated by an artifact")
            if overlap_correction and have_overlap:
                reasons.append("there is overlap and overlap correction was "
                               "requested")
            raise ValueError("'by-epoch' regression strategy is not possible "
                             "because: " + "; also, ".join(reasons))
        else:
            return requested_strategy
    else:
        raise ValueError("Unknown regression strategy %r requested; must be "
                         "\"by-epoch\", \"continuous\", or \"auto\""
                         % (requested_strategy,))

def _analysis_jobs(data_set, recording_span_intern_table, analysis_spans):
    recording_spans = []
    for start, stop, epochs in analysis_spans:
        assert start[0] == stop[0]
        recording_spans.append(recording_span_intern_table[start[0]])
    data_iter = data_set.span_values(recording_spans)
    for data, analysis_span in itertools.izip(data_iter, analysis_spans):
        start, stop, epochs = analysis_span
        yield data[start[1]:stop[1], :], start[1], epochs

class _ByEpochWorker(object):
    def __init__(self, design_width):
        self._design_width = design_width

    def __call__(self, job):
        data, data_start_idx, epochs = job
        assert len(epochs) == 1
        epoch = epochs[0]
        assert epoch.start_idx == data_start_idx
        assert data.shape[0] == epoch.stop_idx - epoch.start_idx
        # XX FIXME: making this a sparse array could be more efficient
        x_strip = np.zeros((1, design_width))
        x_idx = slice(epoch.design_offset,
                      epoch.design_offset + len(epoch.design_row))
        x_strip[x_idx] = epoch.design_row
        y_strip = data.reshape((1, -1))
        return XtXIncrementalLS.append_top_half(x_strip, y_strip)

class _ContinuousWorker(object):
    def __init__(self, expanded_design_width):
        self._expanded_design_width = expanded_design_width

    def __call__(self, job):
        data, data_start_idx, epochs = job
        nnz = 0
        for epoch in epochs:
            nnz += epoch.design_row.shape[0] * data.shape[0]
        design_data = np.empty(nnz, dtype=float)
        design_i = np.empty(nnz, dtype=int)
        design_j = np.empty(nnz, dtype=int)
        write_ptr = 0
        # This code would be more complicated if it couldn't rely on the
        # following facts:
        # - Every epoch in 'epochs' is guaranteed to span the entire chunk of
        #   data, so we don't need to fiddle about finding start and end
        #   positions, and every epoch generates the same number of non-zero
        #   values.
        # - In a coo_matrix, if you have two different entries at the same (i,
        #   j) coordinate, then they get added together. This is the correct
        #   thing to do in our case (though it should be very rare -- in
        #   practice I guess it only happens if you have two events of the
        #   same type that occur at exactly the same time).
        for epoch in epochs:
            for i, x_value in enumerate(epoch.design_row):
                write_slice = slice(write_ptr, write_ptr + data.shape[0])
                design_data[write_slice] = x_value
                design_i[write_slice] = np.arange(data.shape[0])
                col_start = epoch.expanded_design_offset
                col_start += i * epoch.epoch_len
                col_start += data_start_idx - epoch.epoch_start
                design_j[write_slice] = np.arange(col_start,
                                                  col_start + data.shape[0])
            write_ptr += design_row.shape[0] * data.shape[0]
        x_strip_coo = sp.coo_matrix((design_data, (design_i, design_j)),
                                    shape=(data.shape[0],
                                           self._expanded_design_width))
        x_strip = x_strip_coo.tocsc()
        y_strip = data
        return XtXIncrementalLS.append_top_half(x_strip, y_strip)

class rERP(object):
    def __init__(self, rerp_info, data_format, betas):
        self.name = rerp_info["name"]
        self.spec = rerp_info["spec"]
        self.design_info = rerp_info["design_info"]
        self.start_offset = rerp_info["start_offset"]
        self.stop_offset = rerp_info["stop_offset"]
        self.data_format = data_format

        self.epoch_ticks = self.stop_offset - self.start_offset
        num_predictors = len(self.design_info.column_names)
        num_channels = len(self.data_format.channel_names)
        assert (num_predictors, self.epoch_ticks, num_channels) == betas.shape

        # This is always floating point (even if in fact all the values are
        # integers) which means that if you do regular [] indexing with
        # integers, then you will always get by-location indexing, and if you
        # use floating point values, you will always get by-label indexing.
        latencies = np.linspace(self.spec.start_time, self.spec.stop_time,
                                self.epoch_ticks)

        self.data = pandas.Panel(betas,
                                 items=self.design_info.column_names,
                                 major_axis=latencies,
                                 minor_axis=self.data_format.channel_names)
        self.data.data_format = self.data_format

    def predict(self, predictors, which_terms=None):
        if which_terms is not None:
            builder = self.design_info.builder.subset(which_terms)
            columns = []
            column_idx = np.arange(len(self.design_info.column_names))
            for term_name in builder.design_info.term_names:
                slice_ = self.design_info.term_name_slices[term_name]
                columns.append(column_idx[slice_])
            betas_idx = np.concatenate(columns)
        else:
            builder = self.design_info.builder
            betas_idx = slice(None)
        design = build_design_matrices([builder], predictors,
                                       return_type="dataframe")
        predicted = np.dot(np.asarray(design).T,
                           np.asarray(self.data)[betas_idx, :, :])
        as_pandas = pandas.Panel(predicted,
                                 items=design.index,
                                 major_axis=self.data.major_axis,
                                 minor_axis=self.data.minor_axis)
        as_pandas.data_format = self.data_format
        return as_pandas

class rERPAnalysis(object):
    def __init__(self, data_format,
                 rerp_infos, overlap_correction, regression_strategy,
                 total_wanted_ticks, total_good_ticks,
                 artifact_counts, betas):
        self.overlap_correction = overlap_correction
        self.regression_strategy = regression_strategy
        self.artifact_counts = artifact_counts
        self.total_wanted_ticks = total_wanted_ticks
        self.total_good_ticks = total_good_ticks
        self.total_bad_ticks = total_wanted_ticks - total_good_ticks

        self.rerps = OrderedDict()
        for rerp_info in self.rerp_infos:
            i = rerp_info["expanded_design_offset"]
            epoch_len = rerp_info["stop_offset"] - rerp_info["start_offset"]
            num_predictors = len(rerp_info["design_info"].column_names)
            this_betas = betas[i:i + epoch_len * num_predictors, :]
            this_betas.resize((num_predictors, epoch_len, this_betas.shape[1]))
            self.rerps[rerp_info["name"]] = rERP(rerp_info,
                                                 data_format,
                                                 betas)
