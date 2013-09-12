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
from patsy import DesignInfo, EvalEnvironment

from pyrerp.util import unpack_pandas, pack_pandas
import pyrerp.events
from pyrerp.rerp import multi_rerp_impl

class SensorInfo(object):
    def __init__(self, names, thetas, rs):
        self._names = {}
        for name, theta, r in zip(names, thetas, rs):
            # No unicode:
            name = str(name)
            self._names[name] = (theta, r)

    @classmethod
    def from_csv(cls, stream):
        r = csv.reader(stream)
        for header in r:
            assert header == ["theta", "r", "name"]
            break
        thetas = []
        rs = []
        names = []
        for (theta, r, name) in r:
            thetas.append(float(theta))
            rs.append(float(r))
            names.append(name)
        return cls(names, thetas, rs)

    def names(self):
        return self._names.keys()

    def xy_locations(self, names):
        known = []
        xy_locations = []
        for i in xrange(len(names)):
            name = names[i]
            if name in self._names:
                known.append(i)
                theta, r = self._names[name]
                xy_locations.append((np.sin(theta * np.pi / 180) * r,
                                     np.cos(theta * np.pi / 180) * r))
        xy_locations = np.array(xy_locations)
        return known, xy_locations

    def head_xy_center_radius_nose_direction(self):
        return ((0, 0), 0.5, (0, 1))

    def update(self, sensor_info):
        for name, values in sensor_info._names.iteritems():
            if name in self._names:
                if values != self._names[name]:
                    raise ValueError, "inconsistent values for %s" % (name,)
            else:
                self._names[name] = values

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

    def ms_to_ticks(self, ms):
        return int(round(ms * self.exact_sample_rate_hz / 1000.0))

    def ticks_to_ms(self, ticks):
        return ticks * 1000.0 / self.exact_sample_rate_hz

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

    assert df == df
    assert not (df != df)

    tr = df.compute_symbolic_transform("-A2/2", exclude=["rle"])
    assert np.allclose(tr, [[1, -0.5, 0],
                            [0,  0.5, 0],
                            [0,    0, 1]])

# XX FIXME: add bad_event_query and all_or_nothing arguments
class rERP(object):
    def __init__(self, event_query, start_time, stop_time,
                 formula="~ 1", name=None):
        if name is None:
            name = "%s: %s" % (event_query, formula)
        self.event_query = event_query
        self.start_time = start_time
        self.stop_time = stop_time
        self.formula = formula
        self.name = name

class DataSet(object):
    def __init__(self, data_format):
        self.data_format = data_format
        self.sensor_info = SensorInfo([], [], [])
        self._events = pyrerp.events.Events()
        # XX FIXME:
        # Eventually, add a way to load these on the fly, and to apply
        # transformations on the fly.
        self._recspans = []
        self.recspan_infos = []

    def add_recspan(self, data, metadata):
        if not isinstance(data, pandas.DataFrame):
            raise ValueError("data must be a DataFrame")
        if list(data.columns) != self.data_format.channel_names:
            raise ValueError("data columns don't match channel names")
        recspan_id = len(self._recspans)
        recspan_info = self._events.add_recspan_info(recspan_id,
                                                     data.shape[0],
                                                     metadata)
        self.recspan_infos.append(recspan_info)
        # Be careful to ensure this is always a floating point dtype; that way
        # we can be sure that when indexing, floats will be treated as times,
        # and integers will be treated as ticks.
        data.index = (np.arange(data.shape[0], dtype=float)
                         * self.data_format.approx_sample_period_ms)
        data.recspan_info = recspan_info
        self._recspans.append(data)

    # We act like a sequence of recspan data objects
    def __len__(self):
        return len(self._recspans)

    def __getitem__(self, key):
        #return self.iter(subset=[key]).next()
        return self._recspans[key]

    # To keep ourselves honest, though, here's a sequential-access interface
    # that can be efficiently implemented even if we don't have random access
    # to the recspan data.
    def iter(self, recspans=None):
        if recspans is None:
            return iter(self._recspans)
        else:
            return imap(self._recspans.__getitem__, recspans)

    def __iter__(self):
        return self.iter()

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
        assert False

    def merge_df(self, df, on, subset=None):
        # 'on' is like {df_colname: event_key}
        # or just [colname]
        # or just colname
        if isinstance(on, basestring):
            on = [on]
        if not isinstance(on, dict):
            on = dict([(key, key) for key in on])
        p = self.placeholder_event()
        query = self.events(subset)
        NOTHING = object()
        for _, row in df.iterrows():
            this_query = query
            for df_key, db_key in on.iteritems():
                this_query &= (p[db_key] == row[df_key])
            for ev in this_query:
                for df_key in row.index:
                    if df_key not in on:
                        current_value = ev.get(df_key, NOTHING)
                        if current_value is NOTHING:
                            ev[df_key] = row[df_key]
                        else:
                            if current_value != row[df_key]:
                                raise EventsError(
                                    "event already has a value for key %r, "
                                    "%r, which does not match new value %r"
                                    % (df_key, current_value, row[df_key]))

    ################################################################
    # rERP
    ################################################################

    # let's make the return value a list of rerp objects
    # where each has a ref to an analysis-global-info object
    # which itself has some weak record of all the results (rerp specs, but
    # not a circular link)

    def multi_rerp(self, rerp_specs,
                   artifact_query="has _ARTIFACT_TYPE",
                   artifact_type_field="_ARTIFACT_TYPE",
                   overlap_correction=True,
                   # This can be "continuous", "by-epoch", or "auto". If
                   # "continuous", we always build one giant regression model,
                   # treating the data as continuous. If "auto", we use the
                   # (much faster) approach of generating a single regression
                   # model and then applying it to each latency separately --
                   # but *only* if this will produce the same result as doing
                   # the full regression. If "epoch", then we either use the
                   # fast method, or else error out. Changing this argument
                   # never affects the actual output of this function -- if it
                   # does, that's a bug! In general, we can do the fast thing
                   # if:
                   # -- any artifacts affect either all or none of each
                   #    epoch, and
                   # -- either, overlap_correction=False,
                   # -- or, overlap_correction=True and there are in fact no
                   #    overlaps.
                   regression_strategy="auto",
                   eval_env=0):
        eval_env = EvalEnvironment.capture(eval_env, reference=1)
        return multi_rerp_impl(self, rerp_specs,
                               artifact_query, artifact_type_field,
                               overlap_correction, regression_strategy,
                               eval_env)
