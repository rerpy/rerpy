# This file is part of pyrerp
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

import cPickle
import numpy as np
import pandas
from patsy import DesignInfo
from pyrerp.util import unpack_pandas, pack_pandas
import pyrerp.events
from collections import OrderedDict
import abc

class ElectrodeInfo(object):
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

    def update(self, electrode_info):
        for name, values in electrode_info._names.iteritems():
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
            raise ValueError("electrode names must be distinct")

    def __eq__(self, other):
        return (self.exact_sample_rate_hz == other.exact_sample_rate_hz
                and self.units == other.units
                and np.all(self.channel_names == other.channel_names))

    def __ne__(self, other):
        return not (self == other)

    def compute_symbolic_transform(self, expression, exclude=[]):
        # This converts symbolic expressions like "-A1/2" into
        # matrices which perform that transformation. (Actually it is a bit of
        # a hack. The parser we have actually converts arbitrary
        # *combinations* of linear *constraints* into matrices, and is
        # designed to interpret strings like:
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

    assert df == df
    assert not (df != df)

    tr = df.compute_symbolic_transform("-A2/2", exclude=["rle"])
    assert np.allclose(tr, [[1, -0.5, 0],
                            [0,  0.5, 0],
                            [0,    0, 1]])

class Recording(object):
    """An object representing a single recording file. You shouldn't use this
    directly unless you are implementing a new file reader. Instead, put your
    Recordings into a DataSet and use that."""

    # This is an abstract base class
    __metaclass__ = abc.ABCMeta

    # It's expected that implementations of this interface will provide
    # definitions for anything set to None here. These should be thought of as
    # abstract properties. The problem is that if we made e.g. 'name' be an
    # @abstractproperty, then it would be required that any children define
    # the .name property using a @property; this way, it's also possible for
    # children to just assign a value to self.name in the usual Python way
    # inside __init__, and save on pointless boilerplate code.
    name = None
    "A name for this recording."

    metadata = {}
    "dict containing arbitrary metadata specific to this Recording object."

    # Can't default this to an ElectrodeInfo object, because those are mutable
    # and if the base class default ever got modified it would be a nightmare
    # to debug.
    electrode_info = None
    "ElectrodeInfo object (if available, can be None)"


    # Subclasses that represent an on-disk file should probably provide some
    # sort of pathname attribute as well, but this can get more complicated
    # (e.g. kutaslab recordings consist of *two* files, a .log and a
    # .raw/.crw), so we don't standardize it at this level.

    data_format = None
    "A DataFormat object holding metadata about the data format."

    span_lengths = None
    "A list giving the length of each span (in order by span id)"

    @abc.abstractmethod
    def span_data(self):
        "Iterator over the data in each span (in order by span id)"

    @abc.abstractmethod
    def event_iter(self):
        """Iterator over (span_id, ev_start_idx, ev_stop_idx, {ev_attributes}).

        For point events, stop_idx should be one greater than start_idx
        (following the usual Python half-open interval convention)."""

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__name__, self.name)

class DataSet(object):
    def __init__(self, recordings=[]):
        self._recordings = []
        self._data_format = None
        self._transform = None
        self.electrode_info = ElectrodeInfo([], [], [])
        self.events = pyrerp.events.Events()
        for recording in recordings:
            self.add(recording)

    def add(self, recording):
        if self._data_format is None:
            self._data_format = recording.data_format
        if self._data_format != recording.data_format:
            raise ValueError("incompatible data formats")
        if recording.electrode_info is not None:
            self.electrode_info.update(recording.electrode_info)
        self._recordings.append(recording)
        for (span_id, start_idx, stop_idx, attrs) in recording.event_iter():
            self.events.add_event(recording, span_id,
                                  start_idx, stop_idx,
                                  attrs)

    def _transform_data(self, data):
        if self._transform is None:
            return data
        else:
            return np.dot(self._transform, data)

    def transform(self, transformation, exclude=[]):
        if self._transform is None:
            self._transform = np.eye(self.data_format.num_channels)
        if isinstance(transformation, basestring):
            transformation = self.data_format.compute_symbolic_transform(
                transformation, exclude)
        self._transform = np.dot(transformation, self._transform)

    @property
    def data_format(self):
        if self._data_format is None:
            raise ValueError("add a Recording before trying to "
                             "access data format")
        return self._data_format

    @property
    def span_lengths(self):
        "An OrderedDict mapping (recording, span_id) to lengths (in samples)."
        info = OrderedDict()
        for recording in self._recordings:
            for (span_id, length) in enumerate(recording.span_lengths):
                info[(recording, span_id)] = length
        return info

    def span_items(self):
        for recording in self._recordings:
            for (span_id, in_data) in recording.span_items():
                yield (recording, span_id), self._transform_data(data)

    def span_values(self):
        for _, data in self.span_items():
            yield data

    # For each event type, need:
    #   name/tag
    #   query
    #   formula
    #   start time, stop time
    def rerp(self, event_query, start_time, stop_time, formula,
             artifact_query="has _ARTIFACT_TYPE",
             artifact_type_field="_ARTIFACT_TYPE",
             overlap_correction=True,
             # This can be True, False, or "auto". If False, we always build
             # one giant regression model, treating the data as continuous. If
             # "auto", we use the (much faster) approach of generating a
             # single regression model and then applying it to each latency
             # separately -- but *only* if this will produce the same result
             # as doing the full regression. If True, then we either use the
             # fast method, or else error out. Changing this argument never
             # affects the actual output of this function -- if it does,
             # that's a bug! In general, we can do the fast thing if:
             #   -- overlap_correction=False, or
             #   -- overlap_correction=True and there are in fact no overlaps,
             #      and
             #   -- any artifacts affect either all or none of each epoch.
             regress_by_epoch="auto"):
        rerp_specs = [("", event_query, start_time, stop_time, formula)]
        return self.multi_rerp(rerp_specs,
                               artifact_query=artifact_query,
                               artifact_type_field=artifact_type_field,
                               overlap_correction=overlap_correction,
                               regress_by_epoch="auto")

    def multi_rerp(self, rerp_specs,
                   artifact_query="has _ARTIFACT_TYPE",
                   artifact_type_field="_ARTIFACT_TYPE",
                   overlap_correction="if-needed"):
        pass
