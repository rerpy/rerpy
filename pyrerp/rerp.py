# This file is part of pyrerp
# Copyright (C) 2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

import itertools
from collections import namedtuple, OrderedDict
import inspect

import numpy as np
import scipy.sparse as sp
import pandas
from patsy import (EvalEnvironment, dmatrices, ModelDesc, Term,
                   build_design_matrices)
from patsy.util import repr_pretty_delegate, repr_pretty_impl

from pyrerp.incremental_ls import XtXIncrementalLS
from pyrerp.parimap import parimap_unordered

################################################################
# Public interface
################################################################

# XX FIXME: add (and implement) bad_event_query and all_or_nothing arguments
class rERPRequest(object):
    def __init__(self, event_query, start_time, stop_time,
                 formula="~ 1", name=None, eval_env=0):
        if name is None:
            name = "%s: %s" % (event_query, formula)
        if stop_time < start_time:
            raise ValueError("start time %s comes after stop time %s"
                             % (start_time, stop_time))
        self.event_query = event_query
        self.start_time = start_time
        self.stop_time = stop_time
        self.formula = formula
        self.name = name
        self.eval_env = EvalEnvironment.capture(eval_env, reference=1)

    __repr__ = repr_pretty_delegate
    def _repr_pretty_(self, p, cycle):
        assert not cycle
        return repr_pretty_impl(p, self,
                                [self.event_query,
                                 self.start_time, self.stop_time],
                                {"formula": self.formula,
                                 "name": self.name,
                                 })

def test_rERPRequest():
    x = object()
    req = rERPRequest("useful query", -100, 1000, formula="x")
    assert req.name == "useful query: x"
    assert req.eval_env.namespace["x"] is x
    def one_deeper(x, level):
        return rERPRequest("foo", -100, 1000, eval_env=level)
    x2 = object()
    assert one_deeper(x2, 1).eval_env.namespace["x"] is x
    assert one_deeper(x2, 0).eval_env.namespace["x"] is x2
    from nose.tools import assert_raises
    assert_raises(ValueError, rERPRequest, "asdf", 100, 0)

class rERPInfo(object):
    def __init__(self, **attrs):
        self.__dict__.update(attrs)
        self._attr_names = sorted(attrs)

    __repr__ = repr_pretty_delegate
    def _repr_pretty_(self, p, cycle):
        assert not cycle
        return repr_pretty_impl(p, self, [],
                                [(attr, getattr(self, attr))
                                 for attr in self._attr_names])

# TODO:
# let's make the return value a list of rerp objects
# where each has a ref to an analysis-global-info object
# which itself has some weak record of all the results (rerp requests, but
# not a circular link)
# then rerp(...) can just return multi_rerp(...)[0]

# regression_strategy can be "continuous", "by-epoch", or "auto". If
# "continuous", we always build one giant regression model, treating the data
# as continuous. If "auto", we use the (much faster) approach of generating a
# single regression model and then applying it to each latency separately --
# but *only* if this will produce the same result as doing the full
# regression. If "epoch", then we either use the fast method, or else error
# out. Changing this argument never affects the actual output of this
# function. If it does, that's a bug! In general, we can do the fast thing if:
# -- any artifacts affect either all or none of each
#    epoch, and
# -- either, overlap_correction=False,
# -- or, overlap_correction=True and there are in fact no
#    overlaps.
def multi_rerp(dataset,
               rerp_requests,
               artifact_query="has _ARTIFACT_TYPE",
               artifact_type_field="_ARTIFACT_TYPE",
               overlap_correction=True,
               regression_strategy="auto"):
    return multi_rerp_impl(dataset, rerp_requests, artifact_query,
                           artifact_type_field, overlap_correction,
                           regression_strategy)

################################################################
# Implementation
################################################################

# Can't use a namedtuple for this because we want __eq__ and __hash__ to use
# identity semantics, not value semantics.
class _Epoch(object):
    def __init__(self, start_tick, ticks,
                 design_row, rerp_info, intrinsic_artifacts):
        self.start_tick = start_tick
        self.ticks = ticks
        self.design_row = design_row
        self.rerp_info = rerp_info
        # 'intrinsic artifacts' are artifact types that inhere in this epoch's
        # copy of the data. In "classic", no-overlap-correction rerps, this
        # means that they knock out all and only the data for this epoch, but
        # not overlapping epochs. In overlap-correcting rerps, this means that
        # they knock out all the epoch in this data, *and* the portions of any
        # other epochs that overlap this epoch, i.e., they become regular
        # artifacts.
        self.intrinsic_artifacts = intrinsic_artifacts

_DataSpan = namedtuple("_DataSpan", ["start", "stop", "epoch", "artifact"])
_DataSubSpan = namedtuple("_DataSubSpan",
                          ["start", "stop", "epochs", "artifacts"])

def multi_rerp_impl(dataset, rerp_requests,
                    artifact_query, artifact_type_field,
                    overlap_correction, regression_strategy):
    _check_unique_names(rerp_requests)

    # A bunch of _DataSpan objects representing all relevant artifacts and
    # epochs.
    spans = []
    # Get artifacts.
    spans.extend(_artifact_spans(dataset, artifact_query, artifact_type_field))
    # And get the events, and calculate the associated epochs (and any
    # associated artifacts).
    epoch_span_info = _epoch_spans(dataset, rerp_requests)
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
    jobs_iter = _analysis_jobs(dataset, analysis_spans)
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
    betas = betas.reshape((-1, len(dataset.data_format.channel_names)))

    return rERPAnalysis(dataset.data_format,
                        rerp_infos, overlap_correction, regression_strategy,
                        total_wanted_ticks, total_good_ticks,
                        total_overlap_ticks, total_overlap_multiplicity,
                        artifact_counts, betas)

def _check_unique_names(rerp_requests):
    rerp_names = set()
    for rerp_request in rerp_requests:
        if rerp_request.name in rerp_names:
            raise ValueError("name %r used for two different sub-analyses"
                             % (rerp_request.name,))
        rerp_names.add(rerp_request.name)

def test__check_unique_names():
    req1 = rERPRequest("asdf", -100, 1000, name="req1")
    req1_again = rERPRequest("asdf", -100, 1000, name="req1")
    req2 = rERPRequest("asdf", -100, 1000, name="req2")
    from nose.tools import assert_raises
    _check_unique_names([req1, req2])
    _check_unique_names([req1_again, req2])
    assert_raises(ValueError, _check_unique_names, [req1, req2, req1_again])

def _artifact_spans(dataset, artifact_query, artifact_type_field):
    # Create fake "artifacts" covering all regions where we don't actually
    # have any recordings:
    for recspan_info in dataset.recspan_infos:
        neg_inf = (recspan_info.id, -2**31)
        zero = (recspan_info.id, 0)
        end = (recspan_info.id, recspan_info.ticks)
        pos_inf = (recspan_info.id, 2**31)
        yield _DataSpan(neg_inf, zero, None, "_NO_RECORDING")
        yield _DataSpan(end, pos_inf, None, "_NO_RECORDING")

    # Now lookup the actual artifacts recorded in the events structure.
    for artifact_event in dataset.events_query(artifact_query):
        artifact_type = artifact_event.get(artifact_type_field, "_UNKNOWN")
        if not isinstance(artifact_type, basestring):
            raise TypeError("artifact type must be a string, not %r"
                            % (artifact_type,))
        yield _DataSpan((artifact_event.recspan_id, artifact_event.start_tick),
                        (artifact_event.recspan_id, artifact_event.stop_tick),
                        None,
                        artifact_type)

def test__artifact_spans():
    from pyrerp.test_data import mock_dataset
    ds = mock_dataset(num_recspans=2, ticks_per_recspan=30)
    ds.add_event(0, 5, 10, {"_ARTIFACT_TYPE": "just-bad"})
    ds.add_event(1, 15, 20, {"_ARTIFACT_TYPE": "even-worse"})
    ds.add_event(1, 10, 20, {"not-an-artifact": "nope"})

    spans = list(_artifact_spans(ds, "has _ARTIFACT_TYPE", "_ARTIFACT_TYPE"))
    assert sorted(spans) == sorted([
            _DataSpan((0, -2**31), (0, 0), None, "_NO_RECORDING"),
            _DataSpan((0, 30), (0, 2**31), None, "_NO_RECORDING"),
            _DataSpan((1, -2**31), (1, 0), None, "_NO_RECORDING"),
            _DataSpan((1, 30), (1, 2**31), None, "_NO_RECORDING"),
            _DataSpan((0, 5), (0, 10), None, "just-bad"),
            _DataSpan((1, 15), (1, 20), None, "even-worse"),
            ])

# A patsy "factor" that just returns arange(n); we use this as the LHS of our
# formula.
class _RangeFactor(object):
    def __init__(self, n):
        self._n = n
        self.origin = None

    def name(self):
        return "arange(%s)" % (self._n,)

    def memorize_passes_needed(self, state):
        return 0

    def eval(self, state, data):
        return np.arange(self._n)

def test__RangeFactor():
    f = _RangeFactor(5)
    assert np.array_equal(f.eval({}, {"x": 10, "y": 100}),
                          [0, 1, 2, 3, 4])

# Two little adapter classes to allow for rERP formulas like
#   ~ 1 + stimulus_type + _RECSPAN_INFO.subject
class _FormulaEnv(object):
    def __init__(self, events):
        self._events = events

    def __getitem__(self, key):
        if key == "_RECSPAN_INFO":
            return _FormulaRecspanInfo([ev.recspan_info for ev in self._events])
        else:
            # This will raise a KeyError for any events where the field is
            # just undefined, and will return None otherwise. This could be
            # done more efficiently by querying the database directly, but
            # let's not fret about that until it matters.
            #
            # We use pandas.Series here because it has much more sensible
            # NaN/None handling than raw numpy.
            #   np.asarray([None, 1, 2]) -> object (!) array
            #   np.asarray([np.nan, "a", "b"]) -> ["nan", "a", "b"] (!)
            # but
            #   pandas.Series([None, 1, 2]) -> [nan, 1, 2]
            #   pandas.Series([None, "a", "b"]) -> [None, "a", "b"]
            return pandas.Series([ev[key] for ev in self._events])

class _FormulaRecspanInfo(object):
    def __init__(self, recspan_infos):
        self._recspan_infos = recspan_infos

    def __getattr__(self, attr):
        return pandas.Series([ri[attr] for ri in self._recspan_infos])

def test__FormulaEnv():
    from pyrerp.test_data import mock_dataset
    ds = mock_dataset(num_recspans=3)
    ds.recspan_infos[0]["subject"] = "s1"
    ds.recspan_infos[1]["subject"] = "s1"
    ds.recspan_infos[2]["subject"] = "s2"
    ds.add_event(0, 10, 20, {"a": 1, "b": True, "c": None,
                             "d": "not all there"})
    ds.add_event(1, 1, 30, {"a": 2, "b": None, "c": "ev2"
                            # no "d" field
                            })
    ds.add_event(2, 10, 11, {"a": None, "b": False, "c": "ev3",
                             "d": "oops"})

    env = _FormulaEnv(ds.events())
    from pandas.util.testing import assert_series_equal
    np.testing.assert_array_equal(env["a"].values, [1, 2, np.nan])
    np.testing.assert_array_equal(env["b"].values, [True, None, False])
    np.testing.assert_array_equal(env["c"].values, [None, "ev2", "ev3"])
    from nose.tools import assert_raises
    assert_raises(KeyError, env.__getitem__, "d")

    np.testing.assert_array_equal(env["_RECSPAN_INFO"].subject.values,
                                  ["s1", "s1", "s2"])
    assert_raises(KeyError, env["_RECSPAN_INFO"].__getattr__, "subject_name")

def _rerp_design(formula, events, eval_env):
    # Tricky bit: the specifies a RHS-only formula, but really we have an
    # implicit LHS (determined by the event_query). This makes things
    # complicated when it comes to e.g. keeping track of which items survived
    # NA removal, determining the number of rows in an intercept-only formula,
    # etc. Really we want patsy to just treat all this stuff the same way as
    # it normally handles a LHS~RHS formula. So, we convert our RHS formula
    # into a LHS~RHS formula, using a special LHS that represents each event
    # by a placeholder integer!
    desc = ModelDesc.from_formula(formula, eval_env)
    if desc.lhs_termlist:
        raise ValueError("Formula cannot have a left-hand side")
    desc.lhs_termlist = [Term([_RangeFactor(len(events))])]
    fake_lhs, design = dmatrices(desc, _FormulaEnv(events))
    surviving_event_idxes = np.asarray(fake_lhs, dtype=int).ravel()
    design_row_idxes = np.empty(len(events))
    design_row_idxes.fill(-1)
    design_row_idxes[surviving_event_idxes] = np.arange(design.shape[0])
    # Now design_row_idxes[i] is -1 if event i was thrown out, and
    # otherwise gives the row in 'design' which refers to event 'i'.
    return design, design_row_idxes

def test__rerp_design():
    from pyrerp.test_data import mock_dataset
    ds = mock_dataset(num_recspans=3)
    ds.recspan_infos[0]["subject"] = "s1"
    ds.recspan_infos[1]["subject"] = "s1"
    ds.recspan_infos[2]["subject"] = "s2"
    ds.add_event(0, 10, 20, {"a": 1, "b": True, "c": None})
    ds.add_event(1, 1, 30, {"a": 2, "b": None, "c": "red"})
    ds.add_event(1, 30, 31, {"a": 20, "b": True, "c": "blue"})
    ds.add_event(2, 10, 11, {"a": None, "b": False, "c": "blue"})
    ds.add_event(2, 12, 14, {"a": 40, "b": False, "c": "red"})
    x = np.arange(5)
    eval_env = EvalEnvironment.capture()
    design, design_row_idxes = _rerp_design("a + b + c + x",
                                            ds.events(), eval_env)
    from numpy.testing import assert_array_equal
    assert_array_equal(design,
                       #Int b  c   a  x (b/c of rule: categorical b/f numeric)
                       [[1, 1, 0, 20, 2],
                        [1, 0, 1, 40, 4]])
    assert_array_equal(design_row_idxes, [-1, -1, 0, -1, 1])

def _epoch_info_and_spans(dataset, rerp_request):
    spans = []
    data_format = dataset.data_format
    # We interpret the time interval as a closed interval [start, stop],
    # just like pandas.
    start_tick = data_format.ms_to_ticks(rerp_request.start_time,
                                         round="up")
    stop_tick = data_format.ms_to_ticks(rerp_request.stop_time,
                                        round="down")
    # Convert closed tick interval to half-open tick interval [start, stop)
    stop_tick += 1
    # This is actually still needed even though rERPRequest also checks for
    # something similar, because e.g. if we have a sampling rate of 250 Hz and
    # they request an epoch of [1 ms, 3 ms], then stop_time is > start_time,
    # but nonetheless there is no data there.
    if stop_tick <= start_tick:
        raise ValueError("requested epoch span of [%s, %s] contains no "
                         "data points"
                         % (rerp_request.start_time,
                            rerp_request.stop_time))
    events = dataset.events(rerp_request.event_query)
    design, design_row_idxes = _rerp_design(rerp_request.formula,
                                            events,
                                            rerp_request.eval_env)
    rerp_info = {
        "request": rerp_request,
        "design_info": design.design_info,
        "start_tick": start_tick,
        "stop_tick": stop_tick,
        "total_epochs": len(events),
        # Filled in later
        "design_offset": None,
        "expanded_design_offset": None,
        # XX something about counting epochs with no/partial/all rejections,
        # and also by data point
        }
    for i, event in enumerate(events):
        epoch_start = start_tick + event.start_tick
        epoch_stop = stop_tick + event.start_tick
        # -1 for non-existent
        design_row_idx = design_row_idxes[i]
        if design_row_idx == -1:
            design_row = None
        else:
            design_row = design[design_row_idx, :]
        intrinsic_artifacts = []
        if design_row is None:
            # Event thrown out due to missing predictors; this
            # makes its whole epoch into an artifact -- but if overlap
            # correction is disabled, then this artifact only affects
            # this epoch, not anything else. (We still want to treat
            # it as an artifact though so we get proper accounting at
            # the end.)
            intrinsic_artifacts.append("_MISSING_PREDICTOR")
        epoch = _Epoch(epoch_start, epoch_stop - epoch_start,
                       design_row, rerp_info, intrinsic_artifacts)
        spans.append(_DataSpan((event.recspan_id, epoch_start),
                               (event.recspan_id, epoch_stop),
                               epoch,
                               None))
    return rerp_info, spans

def test__epoch_info_and_spans():
    from pyrerp.test_data import mock_dataset
    ds = mock_dataset(num_recspans=2, hz=250)
    ds.add_event(0, 0, 1, {"include": True, "a": 1})
    ds.add_event(0, 10, 11, {"include": True, "a": 2})
    ds.add_event(0, 20, 21, {"include": False, "a": 3})
    ds.add_event(0, 30, 31, {"a": 4}) # include unspecified
    ds.add_event(1, 40, 41, {"include": True, "a": None}) # missing predictor
    ds.add_event(1, 50, 51, {"include": True, "a": 6})

    req = rERPRequest("include", -10, 10, formula="a")
    rerp_info, spans = _epoch_info_and_spans(ds, req)
    # [-10 ms, 10 ms] -> [-8 ms, 8 ms] -> [-2 tick, 2 tick] -> [-2 tick, 3 tick)
    assert rerp_info["start_tick"] == -2
    assert rerp_info["stop_tick"] == 3
    assert rerp_info["total_epochs"] == 4
    assert [s.start for s in spans] == [
        (0, 0 - 2), (0, 10 - 2), (1, 40 - 2), (1, 50 - 2),
        ]
    assert [s.stop for s in spans] == [
        (0, 0 + 3), (0, 10 + 3), (1, 40 + 3), (1, 50 + 3),
        ]
    assert [s.artifact for s in spans] == [None] * 4
    assert [s.epoch.start_tick for s in spans] == [
        0 - 2, 10 - 2, 40 - 2, 50 - 2,
        ]
    assert [s.epoch.ticks for s in spans] == [5] * 4
    for span in spans:
        assert span.epoch.rerp_info is rerp_info
    from numpy.testing import assert_array_equal
    assert_array_equal(spans[0].epoch.design_row, [1, 1])
    assert_array_equal(spans[1].epoch.design_row, [1, 2])
    assert spans[2].epoch.design_row is None
    assert_array_equal(spans[3].epoch.design_row, [1, 6])
    assert [s.epoch.intrinsic_artifacts for s in spans] == [
        [], [], ["_MISSING_PREDICTOR"], [],
        ]

    # Check that rounding works right. Recall that our test dataset is 250
    # Hz.
    for times, exp_ticks in [[(-10, 10), (-2, 3)],
                             [(-11.99, 11.99), (-2, 3)],
                             [(-12, 12), (-3, 4)],
                             [(-20, 20), (-5, 6)],
                             [(18, 22), (5, 6)],
                             [(20, 20), (5, 6)],
                             ]:
        req = rERPRequest("include", times[0], times[1], formula="a")
        rerp_info, _ = _epoch_info_and_spans(ds, req)
        assert rerp_info["start_tick"] == exp_ticks[0]
        assert rerp_info["stop_tick"] == exp_ticks[1]

    # No samples -> error
    from nose.tools import assert_raises
    req_tiny = rERPRequest("include", 1, 3, formula="a")
    assert_raises(ValueError, _epoch_info_and_spans, ds, req_tiny)
    # But the same req it's fine with a higher-res data set
    ds_high_hz = mock_dataset(hz=1000)
    ds_high_hz.add_event(0, 0, 1, {"include": True, "a": 1})
    rerp_info, _ = _epoch_info_and_spans(ds_high_hz, req_tiny)
    assert rerp_info["start_tick"] == 1
    assert rerp_info["stop_tick"] == 4

def _epoch_spans(dataset, rerp_requests, eval_env):
    rerp_infos = []
    spans = []
    design_offset = 0
    expanded_design_offset = 0
    data_format = dataset.data_format
    for rerp_idx, rerp_request in enumerate(rerp_requests):
        start_tick = data_format.ms_to_samples(rerp_request.start_time)
        # Offsets are half open: [start, stop)
        # But, it's more intuitive for times to be closed: [start, stop]
        # So we interpret the user times as a closed interval, and add 1
        # sample when converting to offsets.
        stop_tick = 1 + data_format.ms_to_samples(rerp_request.stop_time)
        if start_tick >= stop_tick:
            raise ValueError("Epochs must be >1 sample long!")
        events = dataset.events(rerp_request.event_query)
        design, design_row_idxes = _rerp_design(rerp_request.formula,
                                                events, eval_env)
        # design_row_idxes[i] is -1 if event i was thrown out, and otherwise
        # gives the row in 'design' which refers to event 'i'.
        for i in xrange(len(events)):
            event = events[i]
            # -1 for non-existent
            design_row_idx = design_row_idxes[i]
            recspan = (event.recording, event.span_id)
            recspan_intern = recspan_intern_table[recspan]
            epoch_start = start_tick + event.start_tick
            epoch_stop = stop_tick + event.start_tick
            if design_row_idx == -1:
                design_row = None
            else:
                design_row = design[design_row_idx, :]
            epoch = _Epoch(epoch_start, epoch_stop - epoch_start,
                           design_row, rerp_info, [])
            if design_row is None:
                # Event thrown out due to missing predictors; this
                # makes its whole epoch into an artifact -- but if overlap
                # correction is disabled, then this artifact only affects
                # this epoch, not anything else. (We still want to treat
                # it as an artifact though so we get proper accounting at
                # the end.)
                epoch.intrinsic_artifacts.append("_MISSING_PREDICTOR")
            spans.append(_DataSpan((recspan_intern, epoch_start),
                                   (recspan_intern, epoch_stop),
                                   epoch,
                                   None))
        rerp_info = {
            "spec": rerp_request,
            "design_info": design.design_info,
            "start_tick": start_tick,
            "stop_tick": stop_tick,
            "design_offset": design_offset,
            "expanded_design_offset": expanded_design_offset,
            "total_epochs": len(events),
            "epochs_with_data": 0,
            "epochs_with_artifacts": 0,
            }
        rerp_infos.append(rerp_info)
        design_offset += design.shape[1]
        epoch_samples = stop_tick - start_tick
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
        start, stop, epoch, artifact = span
        state_changes.append((start, epoch, artifact, +1))
        state_changes.append((stop, epoch, artifact, -1))
    state_changes.sort()
    last_position = None
    current_epochs = {}
    current_artifacts = {}
    for position, epoch, artifact, change in state_changes:
        if (last_position is not None
            and position != last_position
            and current_epochs):
            if overlap_correction:
                effective_artifacts = set(current_artifacts)
                for epoch in current_epochs:
                    effective_artifacts.update(epoch.intrinsic_artifacts)
                yield _DataSubSpan(last_position, position,
                                   tuple(current_epochs),
                                   tuple(effective_artifacts))
            else:
                for epoch in current_epochs:
                    effective_artifacts = set(current_artifacts)
                    effective_artifacts.update(epoch.intrinsic_artifacts)
                    yield _DataSubSpan(last_position, position,
                                       (epoch,),
                                       tuple(effective_artifacts))
        for (state, state_dict) in [(epoch, current_epochs),
                                    (artifact, current_artifacts)]:
            if change == +1 and state not in state_dict:
                state_dict[state] = 0
            state_dict[state] += change
            if change == -1 and state_dict[state] == 0:
                del state_dict[state]
        last_position = position
    assert current_epochs == current_artifacts == {}

def _count_artifacts(counts, num_points, artifacts):
    for artifact in artifacts:
        if artifact not in counts:
            counts[artifact] = {
                "total": 0,
                "unique": 0,
                "proportional": 0,
                }
        subcounts = counts[artifact]
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

def _analysis_jobs(dataset, analysis_spans):
    recspans = list(dataset.span_lengths)
    wanted_recspans = []
    for start, stop, epochs in analysis_spans:
        assert start[0] == stop[0]
        # Un-intern
        wanted_recspans.append(recspans[start[0]])
    data_iter = dataset.span_values(wanted_recspans)
    for data, analysis_span in itertools.izip(data_iter, analysis_spans):
        start, stop, epochs = analysis_span
        yield data[start[1]:stop[1], :], start[1], epochs

class _ByEpochWorker(object):
    def __init__(self, design_width):
        self._design_width = design_width

    def __call__(self, job):
        data, data_start_tick, epochs = job
        assert len(epochs) == 1
        epoch = epochs[0]
        assert epoch.start_tick == data_start_tick
        assert data.shape[0] == epoch.stop_tick - epoch.start_tick
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
        data, data_start_tick, epochs = job
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
                col_start += data_start_tick - epoch.start_tick
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
        self.start_tick = rerp_info["start_tick"]
        self.stop_tick = rerp_info["stop_tick"]
        self.total_epochs = rerp_info["total_epochs"]
        self.epochs_with_data = rerp_info["epochs_with_data"]
        self.epochs_with_artifacts = rerp_info["epochs_with_artifacts"]
        self.partial_epochs = ((self.epochs_with_data
                                + self.epochs_with_artifacts)
                               - self.total_epochs)
        self.data_format = data_format

        self.epoch_ticks = self.stop_tick - self.start_tick
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
            epoch_len = rerp_info["stop_tick"] - rerp_info["start_tick"]
            num_predictors = len(rerp_info["design_info"].column_names)
            this_betas = betas[i:i + epoch_len * num_predictors, :]
            this_betas.resize((num_predictors, epoch_len, this_betas.shape[1]))
            self.rerps[rerp_info["spec"].name] = rERP(rerp_info,
                                                      data_format,
                                                      this_betas)
