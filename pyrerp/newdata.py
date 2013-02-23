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
    def __init__(self, exact_srate_hz, units, electrode_names,
                 reference_name, reference_matrix=None):
        self.exact_sample_rate_hz = exact_srate_hz
        sample_period_ms = 1. / exact_srate_hz * 1000
        # If sample period is exactly an integer, use an integer type to store
        # it
        if 1000. / int(sample_period_ms) == exact_srate_hz:
            sample_period_ms = int(sample_period_ms)
        self.approx_sample_period_ms = sample_period_ms
        self.units = units
        self.electrode_names = electrode_names
        self.reference_name = reference_name
        if reference_matrix is None:
            reference_matrix = np.eye(len(electrode_names),
                                      len(electrode_names) + 1)
            reference_matrix[:, -1] = -1
        self.reference_matrix = reference_matrix

    def __eq__(self, other):
        return (self.exact_srate_hz == other.exact_srate_hz
                and self.units == other.units
                and self.electrode_names == other.electrode_names
                and self.reference_name == other.reference_name
                and np.allclose(self.reference_matrix, other.reference_matrix))

    def __ne__(self, other):
        return not (self == other)

class Recording(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def name(self):
        "A name for this recording."

    @abc.abstractproperty
    def metadata(self):
        "dict containing arbitrary metadata specific to this Recording object."

    @property
    def electrode_info(self):
        "ElectrodeInfo object (if available, empty by default)"
        return ElectrodeInfo([], [], [])

    # Subclasses that represent an on-disk file should probably provide some
    # sort of pathname attribute as well, but this can get more complicated
    # (e.g. kutaslab recordings consist of *two* files, a .log and a
    # .raw/.crw), so we don't standardize it at this level.

    @abc.abstractproperty
    def data_format(self):
        "A DataFormat object holding metadata about the data format."

    @abc.abstractproperty
    def span_info(self):
        "An ordered dictionary mapping span ids to lengths."

    @abc.abstractmethod
    def span_items(self):
        "Iterator over (span_id, data)"

    def span_values(self):
        "Iterator over data in each span."
        for (_, data) in self.span_items():
            yield data

    @abc.abstractmethod
    def event_iter(self):
        "Yield (span_id, start_idx, stop_idx, {attributes})"

class DataSet(object):
    def __init__(self, recordings):
        self._recordings = []
        self._transform = None
        self._recording_data_format = None
        self._output_data_format = None
        self.electrode_info = ElectrodeInfo([], [], [])
        self.events = pyrerp.newevents.Events()
        for recording in recordings:
            self.add_recording(recording)

    def add_recording(self, recording):
        if self._recording_data_format is None:
            self._recording_data_format = recording.data_format
        if self._recording_data_format != recording.data_format:
            raise ValueError, "incompatible data formats"
        # Default our output data format to match the input data format
        if self._output_data_format is None:
            self._output_data_format = self._recording_data_format
        self.electrode_info.update(recording.electrode_info)
        self._recordings.append(recording)
        for (span_id, start_idx, stop_idx, attributes) in recording.event_iter():
            self.events.add_event(recording,
                                  span_id,
                                  start_idx,
                                  stop_idx,
                                  attributes)

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
            for (span_id, data) in recording.span_items():
                yield (recording, span_id), data

    def span_values(self):
        for _, data in self.span_items():
            yield data
