# This file is part of pyrerp
# Copyright (C) 2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

import itertools
from collections import namedtuple, OrderedDict

import numpy as np
import scipy.sparse as sp
import pandas
from patsy import dmatrices, ModelDesc, Term, build_design_matrices

from pyrerp.incremental_ls import XtXIncrementalLS
from pyrerp.parimap import parimap_unordered

# Can't use a namedtuple for this because we want __eq__ and __hash__ to use
# identity semantics, not value semantics.
class _Epoch(object):
    def __init__(self, start_idx, ticks,
                 design_row, design_offset, expanded_design_offset,
                 rerp_idx):
        self.start_idx = start_idx
        self.ticks = ticks
        self.design_row = design_row
        self.design_offset = design_offset
        self.expanded_design_offset = expanded_design_offset
        self.rerp_idx = rerp_idx

# If an artifact has a non-None "epoch" field, then it means that when overlap
# correction is disabled, this artifact only applies to the given epoch. (This
# is useful for "artifacts" like, that event missing some predictors. This
# needs to be treated as an artifact when overlap rejection is turned on, but
# not otherwise.)
_Artifact = namedtuple("_Artifact", ["type", "epoch"])

def multi_rerp_impl(data_set, rerp_specs, artifact_query, artifact_type_field,
                    overlap_correction, regression_strategy,
                    eval_env):
    # We need to be able to refer to individual recording spans in a
    # convenient, sortable way -- but Recording objects aren't sortable and
    # are otherwise inconvenient to work with. So we represent each
    # (Recording, span) pair by its index in the data_set.
    recspan_lengths = data_set.span_lengths
    recspan_intern_table = {}
    for (i, recording_and_span_id) in enumerate(recspan_lengths):
        recspan_intern_table[recording_and_span_id] = i

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
    spans.extend(_artifact_spans(recspan_intern_table, data_set,
                                 artifact_query, artifact_type_field))

    # And now get the events themselves, and calculate the associated epochs
    # (and any associated artifacts).
    epoch_span_info = _epoch_spans(recspan_intern_table,
                                   data_set, rerp_specs,
                                   eval_env)
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
    total_overlap_ticks = 0
    total_overlap_multiplicity = 0
    artifact_counts = {}
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
        ticks = stop[1] - start[1]
        total_wanted_ticks += ticks
        if artifacts:
            epochs_with_artifacts.update(epochs)
            _count_artifacts(artifact_counts, ticks, artifacts)
        else:
            epochs_with_data.update(epochs)
            if len(epochs) > 1:
                have_multi_epoch_data = True
                total_overlap_ticks += ticks
            total_good_ticks += ticks
            total_overlap_multiplicity += ticks * len(epochs)
            analysis_spans.append((start, stop, epochs))

    for epoch in epochs_with_data:
        rerp_infos[epoch.rerp_idx]["epochs_with_data"] += 1
    for epoch in epochs_with_artifacts:
        rerp_infos[epoch.rerp_idx]["epochs_with_artifacts"] += 1
    partial_epochs = epochs_with_data.intersection(epochs_with_artifacts)
    regression_strategy = _choose_strategy(regression_strategy,
                                           bool(partial_epochs),
                                           have_multi_epoch_data)

    if regression_strategy == "by-epoch":
        worker = _ByEpochWorker(design_width)
    elif regression_strategy == "continuous":
        worker = _ContinuousWorker(expanded_design_width)
    else:
        assert False
    jobs_iter = _analysis_jobs(data_set, analysis_spans)
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
    betas = betas.reshape((-1, len(data_set.data_format.channel_names)))

    return rERPAnalysis(data_set.data_format,
                        rerp_infos, overlap_correction, regression_strategy,
                        total_wanted_ticks, total_good_ticks,
                        total_overlap_ticks, total_overlap_multiplicity,
                        artifact_counts, betas)

def _artifact_spans(recspan_intern_table,
                    data_set, artifact_query, artifact_type_field):
    # Create "artifacts" to prevent us from trying to analyze any
    # data the falls outwith the bounds of our recordings.
    for recspan, length in data_set.span_lengths.iteritems():
        recspan_intern = recspan_intern_table[recspan]
        neg_inf = (recspan_intern, -2**31)
        zero = (recspan_intern, 0)
        end = (recspan_intern, length)
        pos_inf = (recspan_intern, 2**31)
        yield (neg_inf, zero, _Artifact("_NO_RECORDING", None))
        yield (end, pos_inf, _Artifact("_NO_RECORDING", None))

    # Now lookup the actual artifacts recorded in the events structure.
    for artifact_event in data_set.events.find(artifact_query):
        artifact_type = artifact_event.get(artifact_type_field, "_UNKNOWN")
        if not isinstance(artifact_type, basestring):
            raise TypeError("artifact type must be a string, not %r"
                            % (artifact_type,))
        recspan = (artifact_event.recording, artifact_event.span_id)
        recspan_intern = recspan_intern_table[recspan]
        yield ((recspan_intern, artifact_event.start_idx),
               (recspan_intern, artifact_event.stop_idx),
               _Artifact(artifact_type, None))

class _ArangeFactor(object):
    def __init__(self, n):
        self._n = n
        self.origin = None

    def name(self):
        return "arange(%s)" % (self._n,)

    def memorize_passes_needed(self, state):
        return 0

    def eval(self, state, data):
        return np.arange(self._n)

def _epoch_spans(recspan_intern_table, data_set, rerp_specs, eval_env):
    rerp_infos = []
    rerp_names = set()
    spans = []
    design_offset = 0
    expanded_design_offset = 0
    data_format = data_set.data_format
    for rerp_idx, rerp_spec in enumerate(rerp_specs):
        start_offset = data_format.ms_to_samples(rerp_spec.start_time)
        # Offsets are half open: [start, stop)
        # But, it's more intuitive for times to be closed: [start, stop]
        # So we interpret the user times as a closed interval, and add 1
        # sample when converting to offsets.
        stop_offset = 1 + data_format.ms_to_samples(rerp_spec.stop_time)
        if start_offset >= stop_offset:
            raise ValueError("Epochs must be >1 sample long!")
        event_set = data_set.events.find(rerp_spec.event_query)
        # Tricky bit: the specifies a RHS-only formula, but really we have an
        # implicit LHS (determined by the event_query). This makes things
        # complicated when it comes to e.g. keeping track of which items
        # survived NA removal, determining the number of rows in an
        # intercept-only formula, etc. Really we want patsy to just treat all
        # this stuff the same way as it normally handles a LHS~RHS
        # formula. So, we convert our RHS formula into a LHS~RHS formula,
        # using a special LHS that represents each event by a placeholder
        # integer!
        desc = ModelDesc.from_formula(rerp_spec.formula, eval_env)
        if desc.lhs_termlist:
            raise ValueError("Formula cannot have a left-hand side")
        desc.lhs_termlist = [Term([_ArangeFactor(len(event_set))])]
        fake_lhs, design = dmatrices(desc, event_set)
        surviving_event_idxes = np.asarray(fake_lhs, dtype=int).ravel()
        design_row_idxes = np.empty(len(event_set))
        design_row_idxes.fill(-1)
        design_row_idxes[surviving_event_idxes] = np.arange(design.shape[0])
        # Now design_row_idxes[i] is -1 if event i was thrown out, and
        # otherwise gives the row in 'design' which refers to event 'i'.
        for i in xrange(len(event_set)):
            event = event_set[i]
            # -1 for non-existent
            design_row_idx = design_row_idxes[i]
            recspan = (event.recording, event.span_id)
            recspan_intern = recspan_intern_table[recspan]
            epoch_start = start_offset + event.start_idx
            epoch_stop = stop_offset + event.start_idx
            epoch_span = ((recspan_intern, epoch_start),
                          (recspan_intern, epoch_stop))
            if design_row_idx == -1:
                design_row = None
            else:
                design_row = design[design_row_idx, :]
            epoch = _Epoch(epoch_start, epoch_stop - epoch_start,
                           design_row, design_offset, expanded_design_offset,
                           rerp_idx)
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
        rerp_names.add(rerp_spec.name)
        rerp_info = {
            "spec": rerp_spec,
            "design_info": design.design_info,
            "start_offset": start_offset,
            "stop_offset": stop_offset,
            "design_offset": design_offset,
            "expanded_design_offset": expanded_design_offset,
            "total_epochs": len(event_set),
            "epochs_with_data": 0,
            "epochs_with_artifacts": 0,
            }
        rerp_infos.append(rerp_info)
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
        if change == +1 and tag not in tag_dict:
            tag_dict[tag] = 0
        tag_dict[tag] += change
        if change == -1 and tag_dict[tag] == 0:
            del tag_dict[tag]
        last_position = position
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
                     have_partial_epochs, have_multi_epoch_data):
    by_epoch_possible = not (have_partial_epochs or have_multi_epoch_data)
    if requested_strategy == "continuous":
        return requested_strategy
    elif requested_strategy == "auto":
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

def _analysis_jobs(data_set, analysis_spans):
    recspans = list(data_set.span_lengths)
    wanted_recspans = []
    for start, stop, epochs in analysis_spans:
        assert start[0] == stop[0]
        # Un-intern
        wanted_recspans.append(recspans[start[0]])
    data_iter = data_set.span_values(wanted_recspans)
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
                write_ptr += data.shape[0]
                design_data[write_slice] = x_value
                design_i[write_slice] = np.arange(data.shape[0])
                col_start = epoch.expanded_design_offset
                col_start += i * epoch.ticks
                col_start += data_start_idx - epoch.start_idx
                design_j[write_slice] = np.arange(col_start,
                                                  col_start + data.shape[0])
        x_strip_coo = sp.coo_matrix((design_data, (design_i, design_j)),
                                    shape=(data.shape[0],
                                           self._expanded_design_width))
        x_strip = x_strip_coo.tocsc()
        y_strip = data
        return XtXIncrementalLS.append_top_half(x_strip, y_strip)

class rERP(object):
    def __init__(self, rerp_info, data_format, betas):
        self.spec = rerp_info["spec"]
        self.design_info = rerp_info["design_info"]
        self.start_offset = rerp_info["start_offset"]
        self.stop_offset = rerp_info["stop_offset"]
        self.total_epochs = rerp_info["total_epochs"]
        self.epochs_with_data = rerp_info["epochs_with_data"]
        self.epochs_with_artifacts = rerp_info["epochs_with_artifacts"]
        self.partial_epochs = ((self.epochs_with_data
                                + self.epochs_with_artifacts)
                               - self.total_epochs)
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
                 total_overlap_ticks, total_overlap_multiplicity,
                 artifact_counts, betas):
        self.overlap_correction = overlap_correction
        self.regression_strategy = regression_strategy
        self.artifact_counts = artifact_counts
        self.total_wanted_ticks = total_wanted_ticks
        self.total_good_ticks = total_good_ticks
        self.total_bad_ticks = total_wanted_ticks - total_good_ticks
        self.total_overlap_ticks = total_overlap_ticks
        self.mean_overlap = total_overlap_multiplicity * 1.0 / total_good_ticks

        self.rerps = OrderedDict()
        for rerp_info in rerp_infos:
            i = rerp_info["expanded_design_offset"]
            epoch_len = rerp_info["stop_offset"] - rerp_info["start_offset"]
            num_predictors = len(rerp_info["design_info"].column_names)
            this_betas = betas[i:i + epoch_len * num_predictors, :]
            this_betas.resize((num_predictors, epoch_len, this_betas.shape[1]))
            self.rerps[rerp_info["spec"].name] = rERP(rerp_info,
                                                      data_format,
                                                      this_betas)
