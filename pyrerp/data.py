# This file is part of pyrerp
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

import cPickle
from collections import OrderedDict, namedtuple
from itertools import groupby, izip
import abc
import csv

import numpy as np
import pandas
from patsy import DesignInfo

import pyrerp.events

# This is just a stub for now. (There's some code that may be resurrectable
# from my old stuff.) Notes on formats:
# http://robertoostenveld.nl/?p=5
# https://sccn.ucsd.edu/svn/software/eeglab/functions/sigprocfunc/readlocs.m
# http://sccn.ucsd.edu/eeglab/channellocation.html
# kutaslab: topo.1, topofiles.5 (this latter has the actual data embedded in it)
# spherical griddata (use s=0): http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectSphereBivariateSpline.html
# some MEG sensors:
#   https://wiki.umd.edu/meglab/images/6/61/KIT_sensor_pos.txt
class SensorInfo(object):
    def __init__(self):
        pass

    def update(self, sensor_info):
        pass

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

    assert df == df
    assert not (df != df)

    tr = df.compute_symbolic_transform("-A2/2", exclude=["rle"])
    assert np.allclose(tr, [[1, -0.5, 0],
                            [0,  0.5, 0],
                            [0,    0, 1]])

    from nose.tools import assert_raises
    assert_raises(ValueError, df.compute_symbolic_transform, "A2/2, A2/3")
    assert_raises(ValueError, df.compute_symbolic_transform, "A2/2 + 1")

class DataSet(object):
    def __init__(self, data_format):
        self.data_format = data_format
        self.sensor_info = SensorInfo()
        self._events = pyrerp.events.Events()
        self._recspans = []
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
        for i, recspan in enumerate(self._recspans):
            new_data = np.dot(recspan, matrix.T)
            self._recspans[i] = pandas.DataFrame(new_data,
                                                 columns=recspan.columns,
                                                 index=recspan.index)

    def add_recspan(self, data, metadata):
        data = np.asarray(data, dtype=np.float64)
        if data.shape[1] != self.data_format.num_channels:
            raise ValueError("wrong number of channels, array should have "
                             "shape (ticks, %s)"
                             % (self.data_format.num_channels,))
        ticks = data.shape[0]
        recspan_id = len(self._recspans)
        recspan_info = self._events.add_recspan_info(recspan_id,
                                                     ticks,
                                                     metadata)
        self.recspan_infos.append(recspan_info)
        index = np.arange(ticks) * float(self.data_format.approx_sample_period_ms)
        df = pandas.DataFrame(data,
                              columns=self.data_format.channel_names,
                              index=index)
        self._recspans.append(df)

    # We act like a sequence of recspan data objects
    def __len__(self):
        return len(self._recspans)

    def __getitem__(self, key):
        if not isinstance(key, int) and hasattr(key, "__index__"):
            key = key.__index__()
        if not isinstance(key, int):
            raise TypeError("DataSet indexing allows only a single integer "
                            "(no slicing or other fanciness!)")
        # May raise IndexError, which is what we want:
        return self._recspans[key]

    def __iter__(self):
        for i in xrange(len(self)):
            yield self[i]

    def __repr__(self):
        return ("<%s with %s recspans, %s events, and %s frames>"
                % (self.__class__.__name__,
                   len(self),
                   len(self.events_query()),
                   sum([ri.ticks for ri in self.recspan_infos])))

    ################################################################
    # Event handling methods (mostly delegated to ._events)
    ################################################################

    def add_event(self, recspan_id, start_tick, stop_tick, attributes):
        return self._events.add_event(recspan_id, start_tick, stop_tick,
                                      attributes)

    def placeholder_event(self):
        return self._events.placeholder_event()

    def events_query(self, subset=None):
        return self._events.events_query(subset)

    def events(self, subset=None):
        return list(self.events_query(subset))

    def events_at_query(self, recspan_id, start_tick, stop_tick=None,
                        subset=None):
        if stop_tick is None:
            stop_tick = start_tick + 1
        p = self.placeholder_event()

        q = p.overlaps(recspan_id, start_tick, stop_tick)
        q &= self.events_query(subset)
        return q

    def events_at(self, recspan_id, start_tick, stop_tick=None,
                  subset=None):
        return list(self.events_at_query(recspan_id, start_tick,
                                         stop_tick, subset))

    ################################################################
    # Convenience methods
    ################################################################

    def add_dataset(self, dataset):
        # Metadata
        if self.data_format != dataset.data_format:
            raise ValueError("data format mismatch")
        self.sensor_info.update(dataset.sensor_info)
        # Recspans
        our_recspan_id_base = len(self._recspans)
        for recspan, recspan_info in zip(dataset, dataset.recspan_infos):
            self.add_recspan(recspan, dict(recspan_info))
        # Events
        for their_event in dataset.events_query():
            self.add_event(their_event.recspan_id + our_recspan_id_base,
                           their_event.start_tick,
                           their_event.stop_tick,
                           dict(their_event))

    def merge_df(self, df, on, subset=None):
        # 'on' is like {df_colname: event_key}
        # or just [colname]
        # or just colname
        if isinstance(on, basestring):
            on = [on]
        if not isinstance(on, dict):
            on = dict([(key, key) for key in on])
        p = self.placeholder_event()
        query = self.events_query(subset)
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
