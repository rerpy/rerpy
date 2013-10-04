# This file is part of rerpy
# Copyright (C) 2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

import itertools
from collections import namedtuple
import inspect
import sys

import numpy as np
import scipy.sparse as sp
import pandas
from patsy import (EvalEnvironment, dmatrices, ModelDesc, Term,
                   build_design_matrices)
from patsy.util import repr_pretty_delegate, repr_pretty_impl

from rerpy.util import indent, ProgressBar

################################################################
# Public interface
################################################################

class rERPRequest(object):
    # WARNING: if you modify this function's arguments in any way, you must
    # also update DataSet.rerp() to match!
    def __init__(self, event_query, start_time, stop_time, formula,
                 name=None, eval_env=0,
                 bad_event_query=None, all_or_nothing=False):
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
        self.bad_event_query = bad_event_query
        self.all_or_nothing = all_or_nothing

    __repr__ = repr_pretty_delegate
    def _repr_pretty_(self, p, cycle):
        assert not cycle
        kwargs = [("name", self.name)]
        if self.bad_event_query:
            kwargs.append(("bad_event_query", self.bad_event_query))
        if self.all_or_nothing:
            kwargs.append(("all_or_nothing", self.all_or_nothing))
        return repr_pretty_impl(p, self,
                                [self.event_query,
                                 self.start_time, self.stop_time,
                                 self.formula],
                                kwargs)

def test_rERPRequest():
    x = object()
    req = rERPRequest("useful query", -100, 1000, "x")
    assert req.name == "useful query: x"
    assert req.eval_env.namespace["x"] is x
    def one_deeper(x, level):
        return rERPRequest("foo", -100, 1000, "1", eval_env=level)
    x2 = object()
    assert one_deeper(x2, 1).eval_env.namespace["x"] is x
    assert one_deeper(x2, 0).eval_env.namespace["x"] is x2
    from nose.tools import assert_raises
    assert_raises(ValueError, rERPRequest, "asdf", 100, 0, "1")
    # smoke test
    repr(rERPRequest("useful query", -100, 1000, "x"))
    repr(rERPRequest("useful query", -100, 1000, "x",
                     all_or_nothing=True, bad_event_query="asdf"))

def multi_rerp_impl(dataset, rerp_requests,
                    artifact_query, artifact_type_field,
                    overlap_correction,
                    regression_strategy):
    if not rerp_requests:
        return []
    _check_unique_names(rerp_requests)

    ## Find all the requested epochs and artifacts
    sys.stdout.write("Locating epochs and artifacts\n")
    spans = []
    # And allocate the rERP objects that we will eventually return.
    rerps = []
    for i in xrange(len(rerp_requests)):
        rerp, epoch_spans = _epoch_info_and_spans(dataset, rerp_requests, i)
        rerps.append(rerp)
        spans.extend(epoch_spans)
    spans.extend(_artifact_spans(dataset, artifact_query, artifact_type_field))
    # Small optimization: only check for all_or_nothing artifacts
    if any(rerp_request.all_or_nothing for rerp_request in rerp_requests):
        _propagate_all_or_nothing(spans, overlap_correction)

    ## Find the good data, gather artifact/overlap/good data statistics
    accountant = _Accountant(rerps)
    analysis_subspans = []
    for subspan in _epoch_subspans(spans, overlap_correction):
        accountant.count(subspan.stop[1] - subspan.start[1],
                         subspan.epochs, subspan.artifacts)
        if not subspan.artifacts:
            analysis_subspans.append(subspan)
    accountant.save()

    ## Do the regression
    regression_strategy = _choose_strategy(regression_strategy,
                                           rerps[0].global_stats)
    for rerp in rerps:
        rerp._set_fit_info(regression_strategy, overlap_correction)
    sys.stdout.write("Fitting model to %s ticks with strategy %r\n"
                     % (rerps[0].global_stats.ticks.accepted,
                        regression_strategy))
    # _fit_* functions fill in .betas field on rerps.
    if regression_strategy == "by-epoch":
        _fit_by_epoch(dataset, analysis_subspans, rerps)
    elif regression_strategy == "continuous":
        all_betas = _fit_continuous(dataset, analysis_subspans, rerps)
    else: # pragma: no cover
        assert False

    for rerp in rerps:
        assert rerp._is_complete()
    sys.stdout.write("Done.\n")
    return rerps

################################################################
# Implementation
################################################################

# Types

# Can't use a namedtuple for this because we want __eq__ and __hash__ to use
# identity semantics, not value semantics.
class _Epoch(object):
    def __init__(self, recspan_id, start_tick, stop_tick,
                 design_row, rerp, intrinsic_artifacts):
        self.recspan_id = recspan_id
        self.start_tick = start_tick
        self.stop_tick = stop_tick
        self.design_row = design_row
        self.rerp = rerp
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

################################################################

def _check_unique_names(rerp_requests):
    rerp_names = set()
    for rerp_request in rerp_requests:
        if rerp_request.name in rerp_names:
            raise ValueError("name %r used for two different sub-analyses"
                             % (rerp_request.name,))
        rerp_names.add(rerp_request.name)

def test__check_unique_names():
    req1 = rERPRequest("asdf", -100, 1000, "1", name="req1")
    req1_again = rERPRequest("asdf", -100, 1000, "1", name="req1")
    req2 = rERPRequest("asdf", -100, 1000, "1", name="req2")
    from nose.tools import assert_raises
    _check_unique_names([req1, req2])
    _check_unique_names([req1_again, req2])
    assert_raises(ValueError, _check_unique_names, [req1, req2, req1_again])

################################################################

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
    from rerpy.test_data import mock_dataset
    ds = mock_dataset(num_recspans=2, ticks_per_recspan=30)
    ds.add_event(0, 5, 10, {"_ARTIFACT_TYPE": "just-bad",
                            "number": 1, "string": "a"})
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

    spans = list(_artifact_spans(ds, "has _ARTIFACT_TYPE", "string"))
    assert sorted(spans) == sorted([
            _DataSpan((0, -2**31), (0, 0), None, "_NO_RECORDING"),
            _DataSpan((0, 30), (0, 2**31), None, "_NO_RECORDING"),
            _DataSpan((1, -2**31), (1, 0), None, "_NO_RECORDING"),
            _DataSpan((1, 30), (1, 2**31), None, "_NO_RECORDING"),
            _DataSpan((0, 5), (0, 10), None, "a"),
            _DataSpan((1, 15), (1, 20), None, "_UNKNOWN"),
            ])

    from nose.tools import assert_raises
    assert_raises(TypeError,
                  list, _artifact_spans(ds, "has _ARTIFACT_TYPE", "number"))

################################################################

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
    from rerpy.test_data import mock_dataset
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
    from rerpy.test_data import mock_dataset
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

    # LHS not allowed
    from nose.tools import assert_raises
    assert_raises(ValueError, _rerp_design, "a ~ b", ds.events(), eval_env)

def _epoch_info_and_spans(dataset, rerp_requests, i):
    rerp_request = rerp_requests[i]
    spans = []
    data_format = dataset.data_format
    # We interpret the time interval as a closed interval [start, stop],
    # just like pandas.
    start_tick, stop_tick = data_format.ms_span_to_ticks(
        rerp_request.start_time, rerp_request.stop_time)
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
    if not events:
        raise ValueError("No events found for rERP %r" % (rerp_request.name,))
    design, design_row_idxes = _rerp_design(rerp_request.formula,
                                            events,
                                            rerp_request.eval_env)
    rerp = rERP(rerp_request, dataset.data_format, design.design_info,
                start_tick, stop_tick, i, len(rerp_requests))
    if rerp_request.bad_event_query is None:
        bad_event_query = None
    else:
        bad_event_query = dataset.events_query(rerp_request.bad_event_query)
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
        if bad_event_query is not None and event.matches(bad_event_query):
            intrinsic_artifacts.append("_BAD_EVENT_QUERY")
        epoch = _Epoch(event.recspan_id, epoch_start, epoch_stop,
                       design_row, rerp, intrinsic_artifacts)
        spans.append(_DataSpan((event.recspan_id, epoch_start),
                               (event.recspan_id, epoch_stop),
                               epoch,
                               None))
    return rerp, spans

def test__epoch_info_and_spans():
    from rerpy.test_data import mock_dataset
    ds = mock_dataset(num_recspans=2, hz=250)
    ds.add_event(0, 0, 1, {"include": True, "a": 1})
    ds.add_event(0, 10, 11, {"include": True, "a": 2})
    ds.add_event(0, 20, 21, {"include": False, "a": 3})
    ds.add_event(0, 30, 31, {"a": 4}) # include unspecified
    ds.add_event(1, 40, 41, {"include": True, "a": None}) # missing predictor
    ds.add_event(1, 50, 51, {"include": True, "a": 6})

    req = rERPRequest("include", -10, 10, "a")
    rerp, spans = _epoch_info_and_spans(ds, [req], 0)
    # [-10 ms, 10 ms] -> [-8 ms, 8 ms] -> [-2 tick, 2 tick] -> [-2 tick, 3 tick)
    assert rerp.start_tick == -2
    assert rerp.stop_tick == 3
    assert rerp.this_rerp_index == 0
    assert rerp.total_rerps == 1
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
    assert [s.epoch.stop_tick for s in spans] == [
        0 + 3, 10 + 3, 40 + 3, 50 + 3,
        ]
    for span in spans:
        assert span.epoch.rerp is rerp
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
        req = rERPRequest("include", times[0], times[1], "a")
        rerp, _ = _epoch_info_and_spans(ds, [req], 0)
        assert rerp.start_tick == exp_ticks[0]
        assert rerp.stop_tick == exp_ticks[1]

    # No samples -> error
    from nose.tools import assert_raises
    req_tiny = rERPRequest("include", 1, 3, "a")
    assert_raises(ValueError, _epoch_info_and_spans, ds, [req_tiny], 0)
    # But the same req it's fine with a higher-res data set
    ds_high_hz = mock_dataset(hz=1000)
    ds_high_hz.add_event(0, 0, 1, {"include": True, "a": 1})
    rerp, _ = _epoch_info_and_spans(ds_high_hz, [req_tiny], 0)
    assert rerp.start_tick == 1
    assert rerp.stop_tick == 4

    # no epochs -> error
    assert_raises(ValueError, _epoch_info_and_spans, ds,
                  [rERPRequest("False", -10, 10, "1")], 0)

    # bad_event_query
    req = rERPRequest("include", -10, 10, "a",
                      bad_event_query="a == None or a == 6")
    rerp, spans = _epoch_info_and_spans(ds, [req], 0)
    assert [s.epoch.intrinsic_artifacts for s in spans] == [
        [],
        [],
        ["_MISSING_PREDICTOR", "_BAD_EVENT_QUERY"],
        ["_BAD_EVENT_QUERY"]
        ]

################################################################
#
# _epoch_subspans theory of operation:
#
# Given a list of spans, like
#
#  span 1: [----------)
#  span 2:     [------------)
#  span 3:                             [--------------)
#  span 4:         [------------)
#
# we generate a list of subspans + annotations that looks like this (in
# order):
#
#          [--)                                         spans: 1
#              [---)                                    spans: 1, 2
#                  [--)                                 spans: 1, 2, 4
#                     [-----)                           spans: 2, 4
#                           [---)                       spans: 4
#                                      [--------------) spans: 3
#
# The idea is that the subspans don't overlap, and each is is either
# completely inside or completely outside *all* the different spans, so we can
# figure out everything we need about how to analyze the data in each subspan
# by just looking at a list of the spans that it falls within. (In the
# implementation, we track epochs and artifacts spans seperately, but this is
# the core idea.)
#
# In the classic algorithms literature these are called "canonical subspans",
# and are usually generated as the leaves of a "segment tree". (But here
# we just iterate over them instead of arranging them into a lookup tree.)
#
# The algorithm for generating these is, first we convert our span-based
# representation ("at positions n1-n2, this epoch/artifact applies") into an
# event-based representation ("at position n1, this epoch/artifact starts
# applying", "at position n2, this epoch/artifact stops applying"). Then we
# can easily sort these events (a cheap O(n log n) operation), and iterate
# through them in O(n) time. Each event then marks the end of one subspan and
# the beginning of another.
#
# Subtlety that applies when overlap_correction=False: In this mode, then we
# want to act as if each epoch gets its own copy of the data. This function
# takes care of that -- if two epochs overlap, then it will generate two
# subspan objects that refer to the same portion of the data, and each will
# say that only one of the epochs applies. So effectively each subspan
# represents a private copy of that bit of data.
#
# Second subtlety around overlap_correction=False: the difference between a
# regular artifact (as generated by _artifact_spans) and an "intrinsic"
# artifact (as noted in an epoch objects's .intrinsic_artifacts field) is that
# a regular artifact is considered to be a property of the *data*, and so in
# overlap_correction=False mode, when we make a "virtual copy" of the data for
# each epoch, these artifacts get carried along to all of the copies and
# effect all of the artifacts. An "intrinsic" artifact, by contrast, is
# considered to be a property of the epoch, and so it applies to that epoch's
# copy of the data, but not to any other possibly overlapping epochs. Of
# course, in overlap_correction=True mode, there is only one copy of the data,
# so both types of artifacts must be treated the same. This function also
# takes care of setting the artifact field of returned subspans appropriately
# in each case.
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
    for position, state_epoch, state_artifact, change in state_changes:
        if (last_position is not None
            and position != last_position
            and current_epochs):
            if overlap_correction:
                effective_artifacts = set(current_artifacts)
                for epoch in current_epochs:
                    effective_artifacts.update(epoch.intrinsic_artifacts)
                yield _DataSubSpan(last_position, position,
                                   set(current_epochs),
                                   effective_artifacts)
            else:
                # Sort epochs to ensure determinism for testing
                for epoch in sorted(current_epochs,
                                    key=lambda e: e.start_tick):
                    effective_artifacts = set(current_artifacts)
                    effective_artifacts.update(epoch.intrinsic_artifacts)
                    yield _DataSubSpan(last_position, position,
                                       set([epoch]),
                                       effective_artifacts)
        for (state, state_dict) in [(state_epoch, current_epochs),
                                    (state_artifact, current_artifacts)]:
            if state is not None:
                if change == +1 and state not in state_dict:
                    state_dict[state] = 0
                state_dict[state] += change
                if change == -1 and state_dict[state] == 0:
                    del state_dict[state]
        last_position = position
    assert current_epochs == current_artifacts == {}

def test__epoch_subspans():
    def e(start_tick, intrinsic_artifacts=[]):
        return _Epoch(0, start_tick, None, None, {}, intrinsic_artifacts)
    e0, e1, e2 = [e(i) for i in xrange(3)]
    e_ia0 = e(10, intrinsic_artifacts=["ia0"])
    e_ia0_ia1 = e(11, intrinsic_artifacts=["ia0", "ia1"])
    s = _DataSpan
    def t(spans, overlap_correction, expected):
        got = list(_epoch_subspans(spans, overlap_correction))
        # This is a verbose way of writing 'assert got == expected' (but
        # if there's a problem it locates it for debugging, instead of just
        # saying 'probably somewhere')
        assert len(got) == len(expected)
        for i in xrange(len(got)):
            got_subspan = got[i]
            expected_subspan = expected[i]
            assert got_subspan == expected_subspan
    t([s(-5, 10, e0, None),
       s( 0, 15, e1, None),
       s( 5,  8, None, "a1"),
       s( 7, 10, None, "a1"),
       s(12, 20, e_ia0, None),
       s(30, 40, e_ia0_ia1, None),
       ],
      True,
      [(-5,  0, {e0}, set()),
       ( 0,  5, {e0, e1}, set()),
       # XX maybe we should coalesce identical subspans like this? not sure if
       # it's worth bothering.
       ( 5,  7, {e0, e1}, {"a1"}),
       ( 7,  8, {e0, e1}, {"a1"}),
       ( 8, 10, {e0, e1}, {"a1"}),
       (10, 12, {e1}, set()),
       (12, 15, {e1, e_ia0}, {"ia0"}),
       (15, 20, {e_ia0}, {"ia0"}),
       (30, 40, {e_ia0_ia1}, {"ia0", "ia1"}),
       ])
    t([s(-5, 10, e0, None),
       s( 0, 15, e1, None),
       s( 5,  8, None, "a1"),
       s( 7, 10, None, "a1"),
       s(12, 20, e_ia0, None),
       s(30, 40, e_ia0_ia1, None),
       ],
      False,
      [(-5,  0, {e0}, set()),
       ( 0,  5, {e0}, set()),
       ( 0,  5, {e1}, set()),
       ( 5,  7, {e0}, {"a1"}),
       ( 5,  7, {e1}, {"a1"}),
       ( 7,  8, {e0}, {"a1"}),
       ( 7,  8, {e1}, {"a1"}),
       ( 8, 10, {e0}, {"a1"}),
       ( 8, 10, {e1}, {"a1"}),
       (10, 12, {e1}, set()),
       (12, 15, {e1}, set()),
       (12, 15, {e_ia0}, {"ia0"}),
       (15, 20, {e_ia0}, {"ia0"}),
       (30, 40, {e_ia0_ia1}, {"ia0", "ia1"}),
       ])

################################################################

def _propagate_all_or_nothing(spans, overlap_correction):
    # We need to find all epochs that have some artifact on them and
    # all_or_nothing requested, and mark the entire epoch as having an
    # artifact. and then we  need to find any overlapping epochs that
    # themselves have all_or_nothing requested, and repeat...
    # Both of these data structures will contain only epochs that have
    # all_or_nothing requested.
    overlap_graph = {}
    epochs_needing_artifact = set()
    for subspan in _epoch_subspans(spans, overlap_correction):
        relevant_epochs = [epoch for epoch in subspan.epochs
                           if epoch.rerp.all_or_nothing]
        for epoch_a in relevant_epochs:
            for epoch_b in relevant_epochs:
                if epoch_a is not epoch_b:
                    overlap_graph.setdefault(epoch_a, set()).add(epoch_b)
        if subspan.artifacts:
            for epoch in relevant_epochs:
                if not epoch.intrinsic_artifacts:
                    epochs_needing_artifact.add(epoch)
    while epochs_needing_artifact:
        epoch = epochs_needing_artifact.pop()
        assert not epoch.intrinsic_artifacts
        epoch.intrinsic_artifacts.append("_ALL_OR_NOTHING")
        for overlapped_epoch in overlap_graph.get(epoch, []):
            if not overlapped_epoch.intrinsic_artifacts:
                epochs_needing_artifact.add(overlapped_epoch)

def test__propagate_all_or_nothing():
    # convenience function for making mock epoch spans
    def e(start, stop, all_or_nothing, expected_survival,
          starts_with_intrinsic=False):
        req = rERPRequest("asdf", -100, 1000, "1",
                          all_or_nothing=all_or_nothing)
        rerp = rERP(req, None, None, -25, 250, 0, 1)
        i_a = []
        if starts_with_intrinsic:
            i_a.append("born_bad")
        epoch = _Epoch(None, None, None, None, rerp, i_a)
        epoch.expected_survival = expected_survival
        return _DataSpan((0, start), (0, stop), epoch, None)
    # convenience function for making mock artifact spans
    def a(start, stop):
        return _DataSpan((0, start), (0, stop), None, "_MOCK_ARTIFACT")
    def t(overlap_correction, spans):
        _propagate_all_or_nothing(spans, overlap_correction)
        for _, _, epoch, _ in spans:
            if epoch is not None:
                survived = "_ALL_OR_NOTHING" not in epoch.intrinsic_artifacts
                assert epoch.expected_survival == survived
    # One artifact at the beginning can knock out a whole string of
    # all_or_nothing epochs
    t(True, [e(0, 100, True, False),
             e(50, 200, True, False),
             e(150, 300, True, False),
             e(250, 400, True, False),
             a(10, 11)])
    # Also true if the artifact appears at the end
    t(True, [e(0, 100, True, False),
             e(50, 200, True, False),
             e(150, 300, True, False),
             e(250, 400, True, False),
             a(350, 360)])
    # But if overlap correction turned off, then epochs directly overlapping
    # artifacts get hit, but it doesn't spread
    t(False, [e(0, 100, True, False),
              e(50, 200, True, True),
              e(150, 300, True, True),
              e(250, 400, True, False),
              a(10, 11),
              a(300, 301)])
    # Epochs with all_or_nothing=False break the chain -- even if they have
    # artifacts on them directly
    t(True, [e(0, 100, True, False),
             e(50, 200, True, False),
             e(150, 300, False, True),
             e(250, 400, True, True),
             a(10, 11),
             a(200, 210)])
    # Epochs with intrinsic artifacts don't get marked with another
    # _ALL_OR_NOTHING artifact (this doesn't really matter semantically, only
    # for speed, but is how it works), so we put True for their
    # "survival". But they can trigger _ALL_OR_NOTHING artifacts in other
    # epochs.
    t(True, [e(0, 100, True, False),
             e(50, 200, True, True, starts_with_intrinsic=True),
             e(150, 300, True, False),
             e(250, 400, True, False),
             a(10, 11),
             ])
    # Intrinsic artifacts don't trigger spreading when overlap correction is
    # turned off.
    t(False, [e(0, 100, True, False),
             e(50, 200, True, True, starts_with_intrinsic=True),
             e(150, 300, True, True),
             e(250, 400, True, True),
             a(10, 11),
             ])

################################################################

# Epochs:
#   Requested: xx
#     Accepted: xx (x%)
#     Partially accepted: xx (x%)
#     Rejected: xx (x%)
# Ticks:
#   Requested: xx
#     Accepted: xx (x%)
#     Rejected: xx (x%)
#     Rejection causes:
#       Artifact 1: xx (xx uniquely)
#       Artifact 2: xx (xx uniquely)
# Event x ticks:
#   Requested: xx
#     Accepted: xx (x%)
#     Rejected: xx (x%)
#     Rejection causes:
#       ...
# Average events per accepted tick (>1 indicates overlap): 1.xx

def _break_down_rejections(superclass, subclasses):
    total = np.sum(count for (_, count) in subclasses)
    result = ("%s:\n" % (superclass,)
           + "  Requested: %s" % (total,))
    if total > 0:
        result += "\n"
        result += "\n".join(["    %s: %s (%.1f%%)"
                             % (name, value, value * 100. / total)
                             for (name, value) in subclasses])
    return result

class EpochRejectionStats(object):
    def __init__(self):
        self.fully_accepted = 0
        self.partially_accepted = 0
        self.fully_rejected = 0

    @property
    def requested(self):
        return (self.fully_accepted
                + self.partially_accepted
                + self.fully_rejected)

    def __repr__(self):
        return _break_down_rejections(
            "Epochs", [("Fully accepted", self.fully_accepted),
                       ("Partially accepted", self.partially_accepted),
                       ("Fully rejected", self.fully_rejected)])
    def _repr_pretty_(self, p, cycle): # pragma: no cover
        assert not cycle
        p.text(indent(repr(self), p.indentation, indent_first=False))

class PointRejectionStats(object):
    def __init__(self, name):
        self.name = name
        self.accepted = 0
        self.rejected = 0
        self.artifacts = {}

    @property
    def requested(self):
        return self.accepted + self.rejected

    def __repr__(self):
        result = _break_down_rejections(
            self.name, [("Accepted", self.accepted),
                        ("Rejected", self.rejected)])
        if self.artifacts:
            causes = []
            for artifact, counts in self.artifacts.items():
                causes.append("%s: %s (%s uniquely)"
                              % (artifact,
                                 counts["affected"],
                                 counts["unique"]))
            result += "\n"
            result += indent("\n".join(causes), 6)
        return result
    def _repr_pretty_(self, p, cycle): # pragma: no cover
        assert not cycle
        p.text(indent(repr(self), p.indentation, indent_first=False))

class RejectionOverlapStats(object):
    def __init__(self):
        self.epochs = EpochRejectionStats()
        self.ticks = PointRejectionStats("Ticks")
        self.event_ticks = PointRejectionStats("Event-ticks")
        self.no_overlap_ticks = PointRejectionStats("Ticks without overlap")

    def __repr__(self):
        chunks = [repr(self.epochs),
                  repr(self.ticks),
                  repr(self.event_ticks),
                  repr(self.no_overlap_ticks)]
        if self.ticks.accepted > 0:
            chunks.append(
                "Average events per accepted tick: %.2f\n"
                " (>1 indicates overlap between the epochs included in these statistics)"
                % (self.event_ticks.accepted * 1.0 / self.ticks.accepted))
        return "\n".join(chunks)
    def _repr_pretty_(self, p, cycle): # pragma: no cover
        assert not cycle
        p.text(indent(repr(self), p.indentation, indent_first=False))

class _Accountant(object):
    def __init__(self, rerps):
        self._rerps = rerps
        self._global_bucket = RejectionOverlapStats()
        self._rerp_buckets = [RejectionOverlapStats() for _ in rerps]

        self._epochs_with_artifacts = set()
        self._epochs_with_data = set()

    def count(self, ticks, epochs, artifacts):
        if not epochs:
            return
        if len(epochs) == 1:
            no_overlap_ticks = ticks
        else:
            no_overlap_ticks = 0
        bucket_events = {
            self._global_bucket: len(epochs),
            }
        for epoch in epochs:
            bucket = self._rerp_buckets[epoch.rerp.this_rerp_index]
            bucket_events.setdefault(bucket, 0)
            bucket_events[bucket] += 1

        if artifacts:
            self._epochs_with_artifacts.update(epochs)
            # Special case: _ALL_OR_NOTHING artifacts should logically be seen
            # as covering over all the parts of an epoch that aren't
            # *otherwise* covered by a real artifact. So for accounting
            # purposes, when an _ALL_OR_NOTHING artifact overlaps with a real
            # artifact, we give the real artifact full credit.
            if len(artifacts) >= 2 and "_ALL_OR_NOTHING" in artifacts:
                artifacts = set(artifacts)
                artifacts.remove("_ALL_OR_NOTHING")
            is_unique = (len(artifacts) == 1)
            def add_artifact(artifact, point_rej_info, points):
                point_rej_info.artifacts.setdefault(artifact,
                                                    {"affected": 0,
                                                     "unique": 0})
                point_rej_info.artifacts[artifact]["affected"] += points
                if is_unique:
                    point_rej_info.artifacts[artifact]["unique"] += points
            for bucket, events in bucket_events.items():
                bucket.ticks.rejected += ticks
                bucket.event_ticks.rejected += ticks * events
                bucket.no_overlap_ticks.rejected += no_overlap_ticks
            for artifact in artifacts:
                for bucket, events in bucket_events.items():
                    add_artifact(artifact, bucket.ticks, ticks)
                    add_artifact(artifact, bucket.event_ticks, ticks * events)
                    if len(epochs) == 1:
                        add_artifact(artifact, bucket.no_overlap_ticks, ticks)
        else:
            self._epochs_with_data.update(epochs)
            for bucket, events in bucket_events.items():
                bucket.ticks.accepted += ticks
                bucket.event_ticks.accepted += ticks * events
                bucket.no_overlap_ticks.accepted += no_overlap_ticks

    def _set_epoch_stats(self, bucket, all_epochs, rerp):
        for epoch in all_epochs:
            if rerp is None or epoch.rerp is rerp:
                if epoch not in self._epochs_with_artifacts:
                    bucket.epochs.fully_accepted += 1
                elif epoch not in self._epochs_with_data:
                    bucket.epochs.fully_rejected += 1
                else:
                    bucket.epochs.partially_accepted += 1

    def save(self):
        all_epochs = self._epochs_with_data.union(self._epochs_with_artifacts)
        self._set_epoch_stats(self._global_bucket, all_epochs, None)
        for rerp, rerp_bucket in zip(self._rerps, self._rerp_buckets):
            self._set_epoch_stats(rerp_bucket, all_epochs, rerp)
            rerp._set_accounting(self._global_bucket, rerp_bucket)

def test__Accountant():
    def make_rerps(N):
        rerps = [rERP(rERPRequest("asdf", -100, 1000, "1"),
                      None, None, 0, 0, i, N) for i in xrange(N)]
        return rerps
    def e(rerps, i):
        return _Epoch(None, None, None, None, rerps[i], None)
    def make_infos(ticks_epochs_artifacts, rerps):
        accountant = _Accountant(rerps)
        for ticks, epoch_or_nums, artifacts in ticks_epochs_artifacts:
            epochs = []
            for epoch_or_num in epoch_or_nums:
                if not isinstance(epoch_or_num, _Epoch):
                    epoch_or_num = e(rerps, epoch_or_num)
                epochs.append(epoch_or_num)
            accountant.count(ticks, epochs, artifacts)
        accountant.save()
        for rerp in rerps:
            assert rerp.global_stats is rerps[0].global_stats
        return (rerps[0].global_stats,
                [rerp.this_rerp_stats for rerp in rerps])

    g, (r0, r1, r2) = make_infos(
        [(1, [], []),
         (2, [], ["blah"]),
         (3, [0], []),
         (4, [0, 1], []),
         (5, [0, 2], ["a0"]),
         (6, [0, 0], ["a0", "a1"]),
         (7, [1], ["a2", "_ALL_OR_NOTHING"]),
         (8, [1, 2], ["_ALL_OR_NOTHING"]),
         ],
        make_rerps(3))
    assert g.epochs.requested == 10
    assert g.epochs.fully_accepted == 3
    assert g.epochs.partially_accepted == 0
    assert g.epochs.fully_rejected == 7
    assert g.ticks.requested == 3 + 4 + 5 + 6 + 7 + 8
    assert g.ticks.accepted == 3 + 4
    assert g.ticks.artifacts == {
        "a0": {"affected": 5 + 6, "unique": 5},
        "a1": {"affected": 6, "unique": 0},
        # _ALL_OR_NOTHING ignored except when it occurs on its own
        "a2": {"affected": 7, "unique": 7},
        "_ALL_OR_NOTHING": {"affected": 8, "unique": 8}
        }
    assert g.event_ticks.requested == 3*1 + 4*2 + 5*2 + 6*2 + 7*1 + 8*2
    assert g.event_ticks.accepted == 3*1 + 4*2
    assert g.event_ticks.artifacts == {
        "a0": {"affected": 5*2 + 6*2, "unique": 5*2},
        "a1": {"affected": 6*2, "unique": 0},
        "a2": {"affected": 7*1, "unique": 7*1},
        "_ALL_OR_NOTHING": {"affected": 8*2, "unique": 8*2}
        }
    assert g.no_overlap_ticks.requested == 3 + 7
    assert g.no_overlap_ticks.accepted == 3
    assert g.no_overlap_ticks.artifacts == {
        "a2": {"affected": 7*1, "unique": 7*1},
        }

    assert r0.epochs.requested == 5
    assert r0.epochs.fully_accepted == 2
    assert r0.epochs.partially_accepted == 0
    assert r0.epochs.fully_rejected == 3
    assert r0.ticks.requested == 3 + 4 + 5 + 6
    assert r0.ticks.accepted == 3 + 4
    assert r0.ticks.artifacts == {
        "a0": {"affected": 5 + 6, "unique": 5},
        "a1": {"affected": 6, "unique": 0},
        }
    assert r0.event_ticks.requested == 3*1 + 4*1 + 5*1 + 6*2
    assert r0.event_ticks.accepted == 3*1 + 4*1
    assert r0.event_ticks.artifacts == {
        "a0": {"affected": 5*1 + 6*2, "unique": 5*1},
        "a1": {"affected": 6*2, "unique": 0},
        }
    assert r0.no_overlap_ticks.requested == 3
    assert r0.no_overlap_ticks.accepted == 3
    assert r0.no_overlap_ticks.artifacts == {}

    assert r1.epochs.requested == 3
    assert r1.epochs.fully_accepted == 1
    assert r1.epochs.partially_accepted == 0
    assert r1.epochs.fully_rejected == 2
    assert r1.ticks.requested == 4 + 7 + 8
    assert r1.ticks.accepted == 4
    assert r1.ticks.artifacts == {
        "a2": {"affected": 7, "unique": 7},
        "_ALL_OR_NOTHING": {"affected": 8, "unique": 8}
        }
    assert r1.event_ticks.requested == 4*1 + 7*1 + 8*1
    assert r1.event_ticks.accepted == 4*1
    assert r1.event_ticks.artifacts == {
        "a2": {"affected": 7*1, "unique": 7*1},
        "_ALL_OR_NOTHING": {"affected": 8*1, "unique": 8*1}
        }
    assert r1.no_overlap_ticks.requested == 7
    assert r1.no_overlap_ticks.accepted == 0
    assert r1.no_overlap_ticks.artifacts == {
        "a2": {"affected": 7, "unique": 7},
        }

    assert r2.epochs.requested == 2
    assert r2.epochs.fully_accepted == 0
    assert r2.epochs.partially_accepted == 0
    assert r2.epochs.fully_rejected == 2
    assert r2.ticks.requested == 5 + 8
    assert r2.ticks.accepted == 0
    assert r2.ticks.artifacts == {
        "a0": {"affected": 5, "unique": 5},
        "_ALL_OR_NOTHING": {"affected": 8, "unique": 8}
        }
    assert r2.event_ticks.requested == 5*1 + 8*1
    assert r2.event_ticks.accepted == 0
    assert r2.event_ticks.artifacts == {
        "a0": {"affected": 5*1, "unique": 5*1},
        "_ALL_OR_NOTHING": {"affected": 8*1, "unique": 8*1}
        }
    assert r2.no_overlap_ticks.requested == 0
    assert r2.no_overlap_ticks.accepted == 0
    assert r2.no_overlap_ticks.artifacts == {}

    # partial rejects
    rerps = make_rerps(2)
    e00 = e(rerps, 0)
    e01 = e(rerps, 0)
    e10 = e(rerps, 1)
    g, (r0, r1) = make_infos(
        [(1, [e00, e01], []),
         (2, [e01], ["a0"]),
         (3, [e01, e10], ["a1"])
         ],
        rerps)
    assert g.epochs.requested == 3
    assert g.epochs.fully_accepted == 1
    assert g.epochs.partially_accepted == 1
    assert g.epochs.fully_rejected == 1
    assert g.ticks.requested == 1 + 2 + 3
    assert g.ticks.accepted == 1
    assert g.ticks.artifacts == {
        "a0": {"affected": 2, "unique": 2},
        "a1": {"affected": 3, "unique": 3},
        }
    assert g.event_ticks.requested == 1*2 + 2*1 + 3*2
    assert g.event_ticks.accepted == 1*2
    assert g.event_ticks.artifacts == {
        "a0": {"affected": 2*1, "unique": 2*1},
        "a1": {"affected": 3*2, "unique": 3*2},
        }
    assert g.no_overlap_ticks.requested == 2
    assert g.no_overlap_ticks.accepted == 0
    assert g.no_overlap_ticks.artifacts == {
        "a0": {"affected": 2, "unique": 2},
        }

    assert r0.epochs.requested == 2
    assert r0.epochs.fully_accepted == 1
    assert r0.epochs.partially_accepted == 1
    assert r0.epochs.fully_rejected == 0
    assert r0.ticks.requested == 1 + 2 + 3
    assert r0.ticks.accepted == 1
    assert r0.ticks.artifacts == {
        "a0": {"affected": 2, "unique": 2},
        "a1": {"affected": 3, "unique": 3},
        }
    assert r0.event_ticks.requested == 1*2 + 2*1 + 3*1
    assert r0.event_ticks.accepted == 1*2
    assert r0.event_ticks.artifacts == {
        "a0": {"affected": 2*1, "unique": 2*1},
        "a1": {"affected": 3*1, "unique": 3*1},
        }
    assert r0.no_overlap_ticks.requested == 2
    assert r0.no_overlap_ticks.accepted == 0
    assert r0.no_overlap_ticks.artifacts == {
        "a0": {"affected": 2, "unique": 2},
        }

    assert r1.epochs.requested == 1
    assert r1.epochs.fully_accepted == 0
    assert r1.epochs.partially_accepted == 0
    assert r1.epochs.fully_rejected == 1
    assert r1.ticks.requested == 3
    assert r1.ticks.accepted == 0
    assert r1.ticks.artifacts == {
        "a1": {"affected": 3, "unique": 3},
        }
    assert r1.event_ticks.requested == 3*1
    assert r1.event_ticks.accepted == 0
    assert r1.event_ticks.artifacts == {
        "a1": {"affected": 3*1, "unique": 3*1},
        }
    assert r1.no_overlap_ticks.requested == 0
    assert r1.no_overlap_ticks.accepted == 0
    assert r1.no_overlap_ticks.artifacts == {}

    # smoke test
    repr(g)
    repr(r0)
    repr(r1)

################################################################

def _choose_strategy(requested_strategy, global_stats):
    gs = global_stats
    # If there is any overlap, then by_epoch is impossible. (Recall that at
    # this phase in the code, if overlap_correction=False then all overlapping
    # epochs have been split out into independent non-overlapping epochs that
    # happen to refer to the same data, so any overlap really is the sort of
    # overlap that prevents by-epoch regression from working.)
    have_overlap = (gs.event_ticks.accepted > gs.ticks.accepted)
    # If there are any partially accepted, partially not-accepted epochs, then
    # by_epoch is impossible.
    have_partial_epochs = (gs.epochs.partially_accepted > 0)
    by_epoch_possible = not (have_overlap or have_partial_epochs)
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
                               "fully eliminated by an artifact (maybe you "
                               "want all_or_nothing=True?)")
            if have_overlap:
                reasons.append("there is overlap and overlap correction was "
                               "requested")
            raise ValueError("\"by-epoch\" regression strategy is not "
                             "possible because: %s. "
                             "Use \"continuous\" strategy instead."
                             % ("; also, ".join(reasons),))
        else:
            return requested_strategy
    else:
        raise ValueError("Unknown regression strategy %r requested; must be "
                         "\"by-epoch\", \"continuous\", or \"auto\""
                         % (requested_strategy,))

def test__choose_strategy():
    from nose.tools import assert_raises
    overlapped = RejectionOverlapStats()
    overlapped.ticks.accepted = 10
    overlapped.event_ticks.accepted = 11
    partial_rej = RejectionOverlapStats()
    partial_rej.epochs.partially_accepted = 1
    both = RejectionOverlapStats()
    both.ticks.accepted = 10
    both.event_ticks.accepted = 11
    both.epochs.partially_accepted = 1
    clean = RejectionOverlapStats()
    clean.ticks.accepted = 10
    clean.event_ticks.accepted = 10
    clean.epochs.fully_rejected = 5

    for stats in [overlapped, partial_rej, both, clean]:
        assert _choose_strategy("continuous", stats) == "continuous"
        assert_raises(ValueError, _choose_strategy, "asdf", stats)
    assert  _choose_strategy("auto", overlapped) == "continuous"
    assert  _choose_strategy("auto", partial_rej) == "continuous"
    assert  _choose_strategy("auto", both) == "continuous"
    assert  _choose_strategy("auto", clean) == "by-epoch"
    assert_raises(ValueError, _choose_strategy, "by-epoch", overlapped)
    assert_raises(ValueError, _choose_strategy, "by-epoch", partial_rej)
    assert_raises(ValueError, _choose_strategy, "by-epoch", both)
    assert  _choose_strategy("by-epoch", clean) == "by-epoch"

################################################################

# We used to do an incremental fit in here, but it was ridiculously
# slower. Like for a simple 80 epochs/32 channels/2 predictors problem, the
# naive incremental fit took 120 s, batching everything up took 6 s, just
# calling np.linalg.lstsq took 200 ms, and doing the lstsq by hand takes
# something under 10 ms (!!). The code could always be resurrected from git if
# needed for scalability though.
def _fit_by_epoch(dataset, analysis_subspans, rerps):
    # Throw out all that nice subspan information and just get a list of
    # epochs. We're guaranteed that every epoch which appears in
    # analysis_spans is fully included in the regression.
    epochs = set()
    for subspan in analysis_subspans:
        assert len(subspan.epochs) == 1
        epochs.update(subspan.epochs)
    # Process recspans in order, to improve data locality
    epochs = sorted(epochs, key=lambda e: (e.recspan_id, e.start_tick))
    channels = dataset.data_format.num_channels
    Xs_Ys_by_rerp = {rerp: ([], []) for rerp in rerps}
    for epoch in epochs:
        this_X = epoch.design_row
        recspan = dataset[epoch.recspan_id]
        y_data = recspan.iloc[epoch.start_tick:epoch.stop_tick, :]
        ticks = epoch.stop_tick - epoch.start_tick
        this_Y = np.asarray(y_data).reshape((1, ticks * channels))
        Xs, Ys = Xs_Ys_by_rerp[epoch.rerp]
        Xs.append(this_X)
        Ys.append(this_Y)
    for rerp, (Xs, Ys) in Xs_Ys_by_rerp.items():
        X = np.row_stack(Xs)
        Y = np.row_stack(Ys)
        if X.shape[0] < X.shape[1]:
            raise ValueError("rerp %r has more predictors than data points. "
                             "I'm afraid this isn't going to work out."
                             % (rerp.name,))
        # implementing lstsq ourselves is dramatically faster than using lstsq
        # (like, factor of 20?). This is because lstsq spends the majority of
        # its time calculating residuals, which we don't need. Some
        # discussion:
        #   http://mail.scipy.org/pipermail/scipy-user/2013-October/035016.html
        # How this works:
        #   svd produces the factorization: X = USV'
        # where U and V are unitary, S is diagonal. Therefore, starting from
        # the normal equations:
        #  B = (X'X)^-1 X'Y
        #    = (V S' U' U S V')^-1 V S' U' Y
        #    = V S^-1 S'^-1 V^-1 V S' U' Y
        #    = V S^-1 U' Y
        # And V S^-1 U' is in fact pinv(X), modulo some fiddling with
        # near-zero singular values; in fact this is exactly how
        # np.linalg.pinv is implemented. So basically this is equivalent to
        # doing
        #   betas = np.dot(np.linalg.pinv(X), Y)
        # except that we get a chance to peek at the singular values and check
        # for collinearity.
        #
        # Remember, np.linalg.svd gives V' instead of V, and gives S as a
        # vector rather than a matrix.
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        # R's lm() has a default tolerance of 1e-7, so I'll arbitrarily steal
        # that. s[0] / s[-1] is the condition number.
        if s[0] / s[-1] > 1e7:
            raise ValueError("Your predictors appear to be perfectly "
                             "collinear. I could make up an answer, but I'd "
                             "rather not.")
        # If this were a real generic least-norm solver we'd want to go in by
        # hand and zero out any singular values that were "too small", but the
        # above code guarantees that this will never happen, so we can just
        # use the naive formula directly:
        betas = np.dot(Vt.T * 1/s, np.dot(U.T, Y))
        betas = betas.reshape((-1, rerp.ticks, channels), order="C")
        rerp._set_betas(betas)

################################################################

def _fit_continuous(dataset, analysis_subspans, rerps):
    # Work out the size of the full design matrices, and the offset of each
    # rerp's individual design matrix within this overall design matrix.
    full_design_width = 0
    design_offsets = {}
    for rerp in rerps:
        # However many columns have been seen so far, that's the offset of
        # this rerp's design columns within the larger design matrix
        design_offsets[rerp] = full_design_width
        # Now figure out how many columns it takes up
        this_design_width = len(rerp.design_info.column_names)
        full_design_width += this_design_width * rerp.ticks
    # This was originally parallelized, but the parallel version always went
    # slower than the serial version, and both seem to be pretty fast,
    # so... eh. Low-hanging fruit for speeding it up would be to use CHOLMOD.
    XtX = np.zeros((full_design_width, full_design_width))
    XtY = np.zeros((full_design_width, dataset.data_format.num_channels))
    rows = 0
    with ProgressBar(len(analysis_subspans)) as progress_bar:
        for subspan in analysis_subspans:
            recspan = dataset[subspan.start[0]]
            data = np.asarray(recspan.iloc[subspan.start[1]:subspan.stop[1], :])
            rows += data.shape[0]
            nnz = 0
            for epoch in subspan.epochs:
                nnz += epoch.design_row.shape[0] * data.shape[0]
            design_data = np.empty(nnz, dtype=float)
            design_i = np.empty(nnz, dtype=int)
            design_j = np.empty(nnz, dtype=int)
            write_ptr = 0
            # This code would be more complicated if it couldn't rely on the
            # following facts:
            # - Every epoch in 'epochs' is guaranteed to span the entire chunk
            #   of data, so we don't need to fiddle about finding start and
            #   end positions, and every epoch generates the same number of
            #   non-zero values.
            # - In a coo_matrix, if you have two different entries at the same
            #   (i, j) coordinate, then they get added together. This is the
            #   correct thing to do in our case (though it should be very rare
            #   -- in practice I guess it only happens if you have two events
            #   of the same type that occur at exactly the same time).
            for epoch in subspan.epochs:
                for i, x_value in enumerate(epoch.design_row):
                    write_slice = slice(write_ptr, write_ptr + data.shape[0])
                    write_ptr += data.shape[0]
                    design_data[write_slice] = x_value
                    design_i[write_slice] = np.arange(data.shape[0])
                    col_start = design_offsets[epoch.rerp]
                    col_start += i * (epoch.stop_tick - epoch.start_tick)
                    col_start += subspan.start[1] - epoch.start_tick
                    design_j[write_slice] = np.arange(col_start,
                                                      col_start + data.shape[0])
            x_strip = sp.coo_matrix((design_data, (design_i, design_j)),
                                    shape=(data.shape[0], full_design_width))
            x_strip = x_strip.tocsc()
            # This actually transmutes XtX and XtY into np.matrix
            # objects. Weird and annoying, but not harmful.
            XtX += x_strip.T * x_strip
            XtY += x_strip.T * data
            progress_bar.increment()
    # Turn them back into ndarrays, to avoid any surprises later.
    XtX = np.asarray(XtX)
    XtY = np.asarray(XtY)
    if rows < full_design_width:
        raise ValueError("This analysis has more predictors than data "
                         "points. I'm afraid this isn't going to work out.")
    if np.linalg.cond(XtX) > 1e7:
        raise ValueError("Your predictors appear to be perfectly "
                         "collinear. I could make up an answer, but I'd "
                         "rather not.")
    all_betas = np.linalg.solve(XtX, XtY)
    # Extract each rerp's betas from the big beta matrix.
    for rerp in rerps:
        i = design_offsets[rerp]
        num_predictors = len(rerp.design_info.column_names)
        num_columns = rerp.ticks * num_predictors
        betas = all_betas[i:i + num_columns, :]
        rerp._set_betas(betas.reshape((num_predictors, rerp.ticks, -1)))

################################################################

class rERP(object):
    # This object is built up over time, which is a terrible design on my
    # part, because it creates the possibility of complex interrelationships
    # between the different parts of the code. But unfortunately I'm not
    # clever enough to come up with a better solution. At least the different
    # pieces don't interact too much. To somewhat mitigate the problem, let's
    # at least explicitly keep track of how built up it is at each point, so
    # we can have assertions about that.
    _ALL_PARTS = frozenset(["accounting", "fit-info", "betas"])
    def _has(self, *parts):
        return self._parts.issuperset(parts)
    def _is_complete(self):
        return self._has(*self._ALL_PARTS)
    def _add_part(self, part):
        assert not self._has(part)
        self._parts.add(part)

    def __init__(self, request, data_format, design_info,
                 start_tick, stop_tick, this_rerp_index, total_rerps):
        self._parts = set()
        self.name = str(request.name)
        self.event_query = str(request.event_query)
        self.start_time = float(request.start_time)
        self.stop_time = float(request.stop_time)
        self.formula = str(request.formula)
        self.bad_event_query = request.bad_event_query
        if self.bad_event_query is not None:
            self.bad_event_query = str(self.bad_event_query)
        self.all_or_nothing = request.all_or_nothing

        self.data_format = data_format
        self.design_info = design_info
        self.start_tick = start_tick
        self.stop_tick = stop_tick
        self.ticks = stop_tick - start_tick

        assert 0 <= this_rerp_index < total_rerps
        self.this_rerp_index = this_rerp_index
        self.total_rerps = total_rerps

    def _set_accounting(self, global_stats, this_rerp_stats):
        self.global_stats = global_stats
        self.this_rerp_stats = this_rerp_stats
        self._add_part("accounting")

    def _set_fit_info(self, regression_strategy, overlap_correction):
        self.regression_strategy = regression_strategy
        self.overlap_correction = overlap_correction
        self._add_part("fit-info")

    def _set_betas(self, betas):
        num_predictors = len(self.design_info.column_names)
        num_channels = len(self.data_format.channel_names)
        assert (num_predictors, self.ticks, num_channels) == betas.shape
        tick_array = np.arange(self.start_tick, self.stop_tick)
        time_array = self.data_format.ticks_to_ms(tick_array)
        self.betas = pandas.Panel(betas,
                                  items=self.design_info.column_names,
                                  major_axis=time_array,
                                  minor_axis=self.data_format.channel_names)
        self._add_part("betas")

    ################################################################
    # Public API
    ################################################################

    def predict_many(self, predictors, which_terms=None, NA_action="raise"):
        if not isinstance(predictors, pandas.DataFrame):
            # This pandas-based preprocessing accomplishes two purposes:
            # 1) If predictors has a mix of scalars and arrays, like
            #      {"x": [1, 2, 3], "type": "target"}
            #    then pandas will broadcast them against each other, which is
            #    nice.
            # 2) If predictors is a dict-of-scalars, like
            #      {"x": 1, "type": "target"}
            #    then DataFrame raises a ValueError unless index=[0] is
            #    specified. We want to support dict-of-scalars, so we attempt
            #    that.
            # See also:
            #   https://github.com/pydata/patsy/issues/24
            try:
                predictors = pandas.DataFrame(predictors)
            except ValueError:
                predictors = pandas.DataFrame(predictors, index=[0])
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
        (design,) = build_design_matrices([builder], predictors,
                                          return_type="dataframe",
                                          NA_action=NA_action)
        betas = np.asarray(self.betas)[betas_idx, :, :]
        # design has shape (i, n)
        # betas has shape (n, j, k)
        # we want to matrix-multiply these to get a single array with shape
        # (i, j, k)
        # to do that we have to collapse betas down to (n, j*k), multiply to
        # get (i, j*k), and then expand back to (i, j, k)
        i, n = design.shape
        assert n == betas.shape[0]
        n, j, k = betas.shape
        predicted = np.dot(np.asarray(design), betas.reshape(n, j * k))
        predicted = predicted.reshape((i, j, k))
        as_pandas = pandas.Panel(predicted,
                                 items=design.index,
                                 major_axis=self.betas.major_axis,
                                 minor_axis=self.betas.minor_axis)
        as_pandas.data_format = self.data_format
        return as_pandas

    # This gives a 2-d DataFrame instead of a 3-d Panel, and is otherwise
    # identical to predict_many
    def predict(self, predictors, which_terms=None, NA_action="raise"):
        prediction = self.predict_many(predictors,
                                       which_terms=which_terms,
                                       NA_action=NA_action)
        if prediction.shape[0] != 1:
            raise ValueError("multiple values given for predictors; to make "
                             "several predictions at once, use predict_many")
        return prediction.iloc[0, :, :]

    # Not sure what more API to provide, some ideas:

    # def events_predictor(self, events):
    #     return _FormulaEnv(events)

    # def design_matrix(self, events, which_terms=None, NA_action="raise"):
    #     if which_terms is not None:
    #         builder = self.design_info.builder.subset(which_terms)
    #     else:
    #         builder = self.design_info.builder
    #     data = _FormulaEnv(events)
    #     (design,) = build_design_matrices([builder], data,
    #                                       return_type="dataframe",
    #                                       NA_action=NA_action)
    #     return design

    # def predict_events(self, events, which_terms=None, NA_action="raise"):
    #     predictors = _FormulaEnv(events)
    #     return self.predict_many(predictors,
    #                              which_terms=which_terms,
    #                              NA_action=NA_action)
