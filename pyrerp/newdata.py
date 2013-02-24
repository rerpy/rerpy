# This file is part of pyrerp
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

import cPickle
import numpy as np
import pandas
from patsy import DesignInfo
from pyrerp.util import unpack_pandas, pack_pandas
import pyrerp.newevents
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
    def __init__(self, exact_srate_hz, units, channel_names):
        self.exact_sample_rate_hz = exact_srate_hz
        sample_period_ms = 1. / exact_srate_hz * 1000
        # If sample period is exactly an integer, use an integer type to store
        # it
        if 1000. / int(sample_period_ms) == exact_srate_hz:
            sample_period_ms = int(sample_period_ms)
        self.approx_sample_period_ms = sample_period_ms
        self.units = units
        self.channel_names = np.asarray(channel_names)
        self.num_channels = self.channel_names.shape[0]
        if not len(self.channel_names) == len(set(self.channel_names)):
            raise ValueError("electrode names must be distinct")

    def __eq__(self, other):
        return (self.exact_srate_hz == other.exact_srate_hz
                and self.units == other.units
                and np.all(self.channel_names == other.channel_names))

    def __ne__(self, other):
        return not (self == other)

    def compute_symbolic_transform(self, expression, exclude=[]):
        # This converts symbolic expressions like "-A1/2" into
        # matrices. (Actually it is a bit of a hack, and actually converts
        # arbitrary *combinations* of linear constraints into matrices,
        # like
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
            raise ValueError, "only one expression allowed!"
        if np.any(lc.coefs.constants != 0):
            raise ValueError, "no affine transformations allowed!"
        for i, channel_name in enumerate(self.channel_names):
            if channel_name not in exclude:
                transform[i, :] += lc.coefs[0, :]
        return transform

class Recording(object):
    __metaclass__ = abc.ABCMeta

    # It's expected that implementations of this interface will provide
    # definitions for anything here that is marked @abstract or that raises a
    # NotImplementedError. The reason for the NotImplementedErrors is that if
    # this were defined using @abc.abstractproperty, then it would be required
    # that any children define the .name property using a @property; this way,
    # it's also possible for children to just assign a value to self.name
    # in the usual Python way, and avoid that fuss.
    @property
    def name(self):
        "A name for this recording."
        raise NotImplementedError

    @property
    def metadata(self):
        "dict containing arbitrary metadata specific to this Recording object."
        return {}

    @property
    def electrode_info(self):
        "ElectrodeInfo object (if available, empty by default)"
        return ElectrodeInfo([], [], [])

    # Subclasses that represent an on-disk file should probably provide some
    # sort of pathname attribute as well, but this can get more complicated
    # (e.g. kutaslab recordings consist of *two* files, a .log and a
    # .raw/.crw), so we don't standardize it at this level.

    @property
    def data_format(self):
        "A DataFormat object holding metadata about the data format."
        raise NotImplementedError

    @property
    def span_info(self):
        "An ordered dictionary mapping span ids to lengths."
        raise NotImplementedError

    @abc.abstractmethod
    def span_items(self):
        "Iterator over (span_id, data)"

    def span_values(self):
        "Iterator over data in each span."
        for (_, data) in self.span_items():
            yield data

    @abc.abstractmethod
    def event_iter(self):
        """Yield (span_id, start_idx, stop_idx, {attributes}).

        For point events, stop_idx should be one greater than start_idx
        (following the usual Python half-open interval convention)."""

class DataSet(object):
    def __init__(self, recordings):
        self._recordings = []
        self._data_format = None
        self._transform = None
        self.electrode_info = ElectrodeInfo([], [], [])
        self.events = pyrerp.newevents.Events()
        for recording in recordings:
            self.add_recording(recording)
        assert self._data_format is not None

    def add_recording(self, recording):
        if self._data_format is None:
            self._data_format = recording.data_format
        if self._data_format != recording.data_format:
            raise ValueError, "incompatible data formats"
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
            transformation = self.design_info.compute_symbolic_transform(
                transformation, exclude)
        self._transform = np.dot(transformation, self._transform)

    @property
    def data_format(self):
        if self._output_data_format is None:
            raise ValueError("add a Recording before trying to "
                             "access data format")
        return self._output_data_format

    @property
    def span_info(self):
        info = OrderedDict()
        for recording in self._recordings:
            for (span_id, length) in recording.span_info:
                info[(recording, span_id)] = length
        return info

    def span_items(self):
        for recording in self._recordings:
            for (span_id, in_data) in recording.span_items():
                yield (recording, span_id), data

    def span_values(self):
        for _, data in self.span_items():
            yield data

    def rerp(self,
             artifact_query="has _ARTIFACT_TYPE",
             artifact_type_field="_ARTIFACT_TYPE"):
        pass
