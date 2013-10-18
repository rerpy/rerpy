# This file is part of rERPy
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

import cPickle
from collections import OrderedDict, namedtuple
from itertools import groupby, izip
import abc
import csv

import numpy as np
import pandas
from patsy import DesignInfo, EvalEnvironment

import rerpy.events
from rerpy.rerp import rERPRequest, multi_rerp_impl

# TODO: add sensor metadata, esp. locations, referencing. make units be
# by-sensor. (There's some code for locations that may be resurrectable from
# my old stuff.) Notes on formats:
# http://robertoostenveld.nl/?p=5
# https://sccn.ucsd.edu/svn/software/eeglab/functions/sigprocfunc/readlocs.m
# http://sccn.ucsd.edu/eeglab/channellocation.html
# kutaslab: topo.1, topofiles.5 (this latter has the actual data embedded in it)
# spherical griddata (use s=0): http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectSphereBivariateSpline.html
# some MEG sensors:
#   https://wiki.umd.edu/meglab/images/6/61/KIT_sensor_pos.txt
# at that point we'll probably also want to extend this to have a notion of
# "compatible" distinct from "==", b/c if two DataFormats are identical except
# that one has no sensor location information in it, then we can totally
# combine that data (and should be able to union the sensor metadata).
class DataFormat(object):
    def __init__(self, exact_sample_rate_hz, units, channel_names):
        self.exact_sample_rate_hz = exact_sample_rate_hz
        sample_period_ms = 1. / exact_sample_rate_hz * 1000
        # If sample period is exactly an integer, use an integer type to store
        # it. The >0 check is to avoid a divide-by-zero for high sampling
        # rates.
        if (int(sample_period_ms) > 0
            and 1000. / int(sample_period_ms) == exact_sample_rate_hz):
            sample_period_ms = int(sample_period_ms)
        self.approx_sample_period_ms = sample_period_ms
        self.units = units
        self.channel_names = np.asarray(channel_names)
        self.num_channels = self.channel_names.shape[0]
        if not len(self.channel_names) == len(set(self.channel_names)):
            raise ValueError("sensor names must be distinct")

    def __eq__(self, other):
        return (self.exact_sample_rate_hz == other.exact_sample_rate_hz
                and self.units == other.units
                and np.all(self.channel_names == other.channel_names))

    def __ne__(self, other):
        return not (self == other)

    def ms_to_ticks(self, ms, round="nearest"):
        float_tick = ms * self.exact_sample_rate_hz / 1000.0
        if round == "nearest":
            tick = np.round(float_tick)
        elif round == "down":
            tick = np.floor(float_tick)
        elif round == "up":
            tick = np.ceil(float_tick)
        else:
            raise ValueError("round= must be \"nearest\", \"up\", or \"down\"")
        return int(tick)

    def ticks_to_ms(self, ticks):
        return np.asarray(ticks) * 1000.0 / self.exact_sample_rate_hz

    def ms_span_to_ticks(self, start_time, stop_time):
        # Converts a closed [start, stop] time interval to a half-open [start,
        # stop) tick interval
        start_tick = self.ms_to_ticks(start_time, round="up")
        stop_tick = self.ms_to_ticks(stop_time, round="down")
        return (start_tick, stop_tick + 1)

    def compute_symbolic_transform(self, expression, exclude=[]):
        # This converts symbolic expressions like "-A1/2" into
        # matrices which perform that transformation. (Actually it is a bit of
        # a hack. The parser/interpreter from patsy that we re-use actually
        # converts arbitrary *combinations* of linear *constraints* into
        # matrices, and is designed to interpret strings like:
        #    "A1=2, rhz*2=lhz"
        # We re-use this code, but interpret the output differently:
        # only one expression is allowed, and it specifies some value that
        # is computed from the data, and then added to each channel
        # not mentioned in 'exclude'.
        transform = np.eye(self.num_channels)
        lc = DesignInfo(self.channel_names).linear_constraint(expression)
        # Check for the weird things that make sense for linear
        # constraints, but not for our hack here:
        if lc.coefs.shape[0] != 1:
            raise ValueError("only one expression allowed!")
        if np.any(lc.constants != 0):
            raise ValueError("transformations must be linear, not affine!")
        for i, channel_name in enumerate(self.channel_names):
            if channel_name not in exclude:
                transform[i, :] += lc.coefs[0, :]
        return transform

def test_DataFormat():
    from nose.tools import assert_raises
    df = DataFormat(1024, "uV", ["MiCe", "A2", "rle"])
    assert df.exact_sample_rate_hz == 1024
    assert np.allclose(df.approx_sample_period_ms, 1000. / 1024)
    assert isinstance(DataFormat(1000, "uV", []).approx_sample_period_ms,
                      int)
    assert df.units == "uV"
    assert isinstance(df.channel_names, np.ndarray)
    assert np.all(df.channel_names == ["MiCe", "A2", "rle"])
    assert df.num_channels == 3
    # no duplicate channel names
    assert_raises(ValueError, DataFormat, 1024, "uV", ["MiCe", "MiCe"])

    assert df.ms_to_ticks(1000) == 1024
    assert df.ticks_to_ms(1024) == 1000

    assert np.array_equal(df.ticks_to_ms([512, 1024]), [500.0, 1000.0])

    assert df.ms_to_ticks(1000.1) == 1024
    assert df.ms_to_ticks(1000.9) == 1025
    assert df.ms_to_ticks(1000, round="nearest") == 1024
    assert df.ms_to_ticks(1000.1, round="nearest") == 1024
    assert df.ms_to_ticks(1000.9, round="nearest") == 1025
    assert df.ms_to_ticks(1000, round="down") == 1024
    assert df.ms_to_ticks(1000.1, round="down") == 1024
    assert df.ms_to_ticks(1000.9, round="down") == 1024
    assert df.ms_to_ticks(1000, round="up") == 1024
    assert df.ms_to_ticks(1000.1, round="up") == 1025
    assert df.ms_to_ticks(1000.9, round="up") == 1025

    assert_raises(ValueError, df.ms_to_ticks, 1000, round="sideways")

    assert df.ms_span_to_ticks(-1000, 1000) == (-1024, 1025)
    assert df.ms_span_to_ticks(-999.99, 999.99) == (-1023, 1024)
    assert df.ms_span_to_ticks(-1000.01, 1000.01) == (-1024, 1025)

    assert df == df
    assert not (df != df)
    assert df != DataFormat(1000, "uV", ["MiCe", "A2", "rle"])
    assert df != DataFormat(1024, "raw", ["MiCe", "A2", "rle"])
    assert df != DataFormat(1024, "uV", ["MiCe", "rle", "A2"])

    tr = df.compute_symbolic_transform("-A2/2", exclude=["rle"])
    assert np.allclose(tr, [[1, -0.5, 0],
                            [0,  0.5, 0],
                            [0,    0, 1]])

    assert_raises(ValueError, df.compute_symbolic_transform, "A2/2, A2/3")
    assert_raises(ValueError, df.compute_symbolic_transform, "A2/2 + 1")

class Dataset(object):
    def __init__(self, data_format):
        self.data_format = data_format
        self._events = rerpy.events.Events()
        self._recspans = []
        self._lazy_recspans = []
        self._lazy_transforms = []
        self.recspan_infos = []

    def transform(self, matrix, exclude=[]):
        if isinstance(matrix, basestring):
            matrix = self.data_format.compute_symbolic_transform(matrix,
                                                                 exclude)
        else:
            if exclude:
                raise ValueError("exclude= can only be specified if matrix= "
                                 "is a symbolic expression")
        matrix = np.asarray(matrix)
        for i in xrange(len(self._recspans)):
            if self._recspans[i] is not None:
                recspan = self._recspans[i]
                new_data = np.dot(recspan, matrix.T)
                self._recspans[i] = pandas.DataFrame(new_data,
                                                     columns=recspan.columns,
                                                     index=recspan.index)
            else:
                old_transform = self._lazy_transforms[i]
                if old_transform is None:
                    old_transform = np.eye(self.data_format.num_channels)
                self._lazy_transforms[i] = np.dot(matrix, old_transform)

    def _add_recspan_info(self, ticks, metadata):
        recspan_id = len(self.recspan_infos)
        recspan_info = self._events.add_recspan_info(recspan_id,
                                                     ticks,
                                                     metadata)
        self.recspan_infos.append(recspan_info)

    def _decorate_recspan(self, data):
        ticks = data.shape[0]
        index = np.arange(ticks, dtype=float)
        index *= self.data_format.approx_sample_period_ms
        df = pandas.DataFrame(data,
                              columns=self.data_format.channel_names,
                              index=index)
        return df

    def add_recspan(self, data, metadata):
        data = np.asarray(data, dtype=np.float64)
        if data.shape[1] != self.data_format.num_channels:
            raise ValueError("wrong number of channels, array should have "
                             "shape (ticks, %s)"
                             % (self.data_format.num_channels,))
        ticks = data.shape[0]
        self._add_recspan_info(ticks, metadata)
        self._recspans.append(self._decorate_recspan(data))
        self._lazy_recspans.append(None)
        self._lazy_transforms.append(None)

    def add_lazy_recspan(self, loader, ticks, metadata):
        self._add_recspan_info(ticks, metadata)
        self._recspans.append(None)
        self._lazy_recspans.append(loader)
        self._lazy_transforms.append(None)

    def add_dataset(self, dataset):
        # Metadata
        if self.data_format != dataset.data_format:
            raise ValueError("data format mismatch")
        # Recspans
        our_recspan_id_base = len(self._recspans)
        for recspan_info in dataset.recspan_infos:
            self._add_recspan_info(recspan_info.ticks, dict(recspan_info))
        self._recspans += dataset._recspans
        self._lazy_recspans += dataset._lazy_recspans
        self._lazy_transforms += dataset._lazy_transforms
        # Events
        for their_event in dataset.events_query():
            self.add_event(their_event.recspan_id + our_recspan_id_base,
                           their_event.start_tick,
                           their_event.stop_tick,
                           dict(their_event))

    # We act like a sequence of recspan data objects
    def __len__(self):
        return len(self._recspans)

    def raw_slice(self, recspan_id, start_tick, stop_tick):
        if start_tick < 0 or stop_tick < 0:
            raise IndexError("only positive indexes allowed")
        ticks = stop_tick - start_tick
        recspan = self._recspans[recspan_id]
        if recspan is not None:
            result = np.asarray(recspan.iloc[start_tick:stop_tick, :])
        else:
            lr = self._lazy_recspans[recspan_id]
            lazy_data = lr.get_slice(start_tick, stop_tick)
            transform = self._lazy_transforms[recspan_id]
            if transform is not None:
                lazy_data = np.dot(lazy_data, transform.T)
            result = lazy_data
        if result.shape[0] != ticks:
            raise IndexError("slice spans missing data")
        return result

    def __getitem__(self, key):
        if not isinstance(key, int) and hasattr(key, "__index__"):
            key = key.__index__()
        if not isinstance(key, int):
            raise TypeError("Dataset indexing allows only a single integer "
                            "(no slicing or other fanciness!)")
        # May raise IndexError, which is what we want:
        recspan = self._recspans[key]
        if recspan is None:
            ticks = self.recspan_infos[key].ticks
            raw = self.raw_slice(key, 0, ticks)
            recspan = self._decorate_recspan(raw)
        return recspan

    def __iter__(self):
        for i in xrange(len(self)):
            yield self[i]

    def __repr__(self):
        return ("<%s with %s events in %s total ticks over %s recspans>"
                % (self.__class__.__name__,
                   len(self.events_query()),
                   sum([ri.ticks for ri in self.recspan_infos]),
                   len(self),
                   ))

    ################################################################
    # Event handling methods (mostly delegated to ._events)
    ################################################################

    def add_events(self, recspan_ids, start_ticks, stop_ticks, attributes_df):
        return self._events.add_events(recspan_ids, start_ticks, stop_ticks,
                                       attributes_df)
    add_events.__doc__ = rerpy.events.Events.add_events.__doc__

    def add_event(self, recspan_id, start_tick, stop_tick, attributes):
        return self._events.add_event(recspan_id, start_tick, stop_tick,
                                      attributes)
    add_event.__doc__ = rerpy.events.Events.add_event.__doc__

    def placeholder_event(self):
        return self._events.placeholder_event()
    placeholder_event.__doc__ = rerpy.events.Events.placeholder_event.__doc__

    def events_query(self, restrict=None):
        return self._events.events_query(restrict)
    events_query.__doc__ = rerpy.events.Events.events_query.__doc__

    def events(self, restrict=None):
        return list(self.events_query(restrict))

    def events_at_query(self, recspan_id, start_tick, stop_tick=None,
                        restrict=None):
        if stop_tick is None:
            stop_tick = start_tick + 1
        p = self.placeholder_event()

        q = p.overlaps(recspan_id, start_tick, stop_tick)
        q &= self.events_query(restrict)
        return q

    def events_at(self, recspan_id, start_tick, stop_tick=None,
                  restrict=None):
        return list(self.events_at_query(recspan_id, start_tick,
                                         stop_tick, restrict))

    ################################################################
    # rERP!
    ################################################################

    def rerp(self,
             # rERPRequest arguments
             event_query, start_time, stop_time, formula,
             name=None, eval_env=0, bad_event_query=None,
             all_or_nothing=False,
             # multi_rerp arguments
             artifact_query="has _ARTIFACT_TYPE",
             artifact_type_field="_ARTIFACT_TYPE",
             overlap_correction=True,
             regression_strategy="auto",
             verbose=True):
        eval_env = EvalEnvironment.capture(eval_env, reference=1)
        request = rERPRequest(event_query, start_time, stop_time, formula,
                              name=name, eval_env=eval_env,
                              bad_event_query=bad_event_query,
                              all_or_nothing=all_or_nothing)
        rerps = self.multi_rerp([request],
                                artifact_query=artifact_query,
                                artifact_type_field=artifact_type_field,
                                overlap_correction=overlap_correction,
                                regression_strategy=regression_strategy,
                                verbose=verbose)
        assert len(rerps) == 1
        return rerps[0]

    # regression_strategy can be "continuous", "by-epoch", or "auto". If
    # "continuous", we always build one giant regression model, treating the
    # data as continuous. If "auto", we use the (much faster) approach of
    # generating a single regression model and then applying it to each
    # latency separately -- but *only* if this will produce the same result as
    # doing the full regression. If "epoch", then we either use the fast
    # method, or else error out. Changing this argument never affects the
    # actual output of this function. If it does, that's a bug! In general, we
    # can do the fast thing if:
    # -- any artifacts affect either all or none of each
    #    epoch, and
    # -- either, overlap_correction=False,
    # -- or, overlap_correction=True and there are in fact no
    #    overlaps.
    #
    # WARNING: if you modify this function's arguments in any way, you must
    # also update rerp() to match!
    def multi_rerp(self, rerp_requests,
                   artifact_query="has _ARTIFACT_TYPE",
                   artifact_type_field="_ARTIFACT_TYPE",
                   overlap_correction=True,
                   regression_strategy="auto",
                   verbose=True):
        return multi_rerp_impl(self, rerp_requests,
                               artifact_query=artifact_query,
                               artifact_type_field=artifact_type_field,
                               overlap_correction=overlap_correction,
                               regression_strategy=regression_strategy,
                               verbose=verbose)

    ################################################################
    # Convenience methods
    ################################################################

    def epochs(self, event_query, start_time, stop_time,
               incomplete_action="raise"):
        start_tick, stop_tick = self.data_format.ms_span_to_ticks(
            start_time, stop_time)
        return self.epochs_ticks(event_query, start_tick, stop_tick,
                                 incomplete_action=incomplete_action)

    def epochs_ticks(self, event_query, start_tick, stop_tick,
                     incomplete_action="raise"):
        events = self.events(event_query)
        good_events = []
        good_epochs = []
        for i, event in enumerate(events):
            recspan = self[event.recspan_id]
            s = slice(event.start_tick + start_tick,
                      event.start_tick + stop_tick)
            try:
                data = recspan.iloc[s, :]
            except IndexError:
                if incomplete_action == "raise":
                    raise ValueError("only part of epoch #%s was actually "
                                     "recorded; if you want to discard "
                                     "such epochs instead, use "
                                     "incomplete_action=\"drop\""
                                     % (i,))
                elif incomplete_action == "drop":
                    pass
                else:
                    raise ValueError("invalid incomplete_action= argument")
            else:
                good_epochs.append(np.asarray(data)[np.newaxis, ...])
                good_events.append(i)
        result = np.concatenate(good_epochs, axis=0)
        tick_array = np.arange(start_tick, stop_tick)
        time_array = self.data_format.ticks_to_ms(tick_array)
        return pandas.Panel(result,
                            items=good_events,
                            major_axis=time_array,
                            minor_axis=self.data_format.channel_names)

    # This could possibly be made substantially faster by loading the df into
    # the database as a temporary table and then letting sqlite do the joins.
    def merge_df(self, df, on, restrict=None):
        # 'on' is like {df_colname: event_key}
        # or just [colname]
        # or just colname
        if isinstance(on, basestring):
            on = [on]
        if not isinstance(on, dict):
            on = dict([(key, key) for key in on])
        p = self.placeholder_event()
        query = self.events_query(restrict)
        NOTHING = object()
        for _, row in df.iterrows():
            this_query = query
            for df_key, db_key in on.iteritems():
                this_query &= (p[db_key] == row.loc[df_key])
            for ev in this_query:
                for df_key in row.index:
                    if df_key not in on:
                        current_value = ev.get(df_key, NOTHING)
                        if current_value is NOTHING:
                            ev[df_key] = row.loc[df_key]
                        else:
                            if current_value != row.loc[df_key]:
                                raise ValueError(
                                    "event already has a value for key %r, "
                                    "%r, which does not match new value %r"
                                    % (df_key, current_value, row[df_key]))

    def merge_csv(self, path, on, restrict=None, **kwargs):
        df = pandas.read_csv(path, **kwargs)
        self.merge_df(df, on, restrict=restrict)
