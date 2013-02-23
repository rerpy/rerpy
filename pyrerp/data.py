# This file is part of pyrerp
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

import cPickle
import numpy as np
import pandas
from patsy import DesignInfo
from pyrerp.util import unpack_pandas, pack_pandas
import pyrerp.events

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

class RecordingInfo(object):
    # Remember to update .clone() below if changing this:
    def __init__(self, exact_srate_hz, units, electrodes=None, metadata={}):
        self.exact_sample_rate_hz = exact_srate_hz
        sample_period_ms = 1. / exact_srate_hz * 1000
        # If sample period is exactly an integer, use an integer type to store
        # it
        if 1000. / int(sample_period_ms) == exact_srate_hz:
            sample_period_ms = int(sample_period_ms)
        self.approx_sample_period_ms = sample_period_ms
        self.units = units
        if electrodes is None:
            electrodes = ElectrodeInfo([], [], [])
        self.electrodes = electrodes
        self.metadata = dict(metadata)

    def clone(self):
        return self.__class__(self.exact_srate_hz,
                              self.units, self.electrodes, self.metadata)

class DataBase(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError, "virtual base class"

    def to_mmapable(self, path):
        f = open(path, "wb")
        data = self.data
        data_arr, pandas_metadata = unpack_pandas(data)
        try:
            self.data = None
            cPickle.dump(0, f, -1)
            cPickle.dump((self, data_arr.dtype, data_arr.shape,
                          pandas_metadata), f, -1)
        finally:
            self.data = data
        offset = f.tell()
        f.close()
        data_map = np.memmap(path, mode="r+", dtype=data_arr.dtype,
                             offset=offset, shape=data_arr.shape)
        data_map[...] = data_arr
        data_map.flush()
        data_map.close()

    @classmethod
    def from_mmapable(cls, path):
        f = open(path, "rb")
        version = cPickle.load(f)
        if version != 0:
            raise ValueError, "unknown mmapable file format version"
        self, dtype, shape, pandas_metadata = cPickle.load(f)
        offset = f.tell()
        f.close()
        data_map = np.memmap(path, mode="r",
                             dtype=dtype, offset=offset, shape=shape)
        self.data = pack_pandas(data_map, pandas_metadata)
        return self

    def reref(self, new_ref, ref_name="_REF_", skip=[]):
        # Theory: let the (notional) absolute potentials be a vector D, which
        # has n+1 entries (all the recorded electrodes, plus the unrecorded
        # reference electrode stuck on at the end). Then what we record is the
        # vector RD, where R is an n x (n+1) matrix that looks like:
        #   [1 0 0 -1]
        #   [0 1 0 -1]
        #   [0 0 1 -1]
        # Notionally, this matrix R is (np.eye(n+1) - ref)[:-1, :]. The
        # np.eye() part alone would return all the absolute potentials, and
        # 'ref' is a convex combination of electrodes that picks out the
        # reference. We assume that the current data used a reference of
        #   [[0 0 0 1]]
        # i.e., just the unmeasured electrode. This produces the R matrix
        # shown above. Notice that the last row of (np.eye(n+1) - ref) is
        # all-zeros (the reference electrode measured against itself), so that
        # "channel" is discarded.
        # 
        # Now our problem is, given a new reference vector -- like say
        #   [[0 0 0.5 0.5]]
        # for an average-of-mastoids reference, or
        #   [[0.25 0.25 0.25 0.25]]
        # for an average-of-everything reference, we want to end up with new
        # data that looks like SD where S is (np.eye(n+1) - newref)[:-1, :].
        #
        # The constraint on these matrices R, S is that each row sums to
        # zero, because that means that we are only comparing differences
        # between potentials, not measuring absolute potential. That means
        # that the ref vector needs to sum to 1.
        # 
        # So we need a matrix A such that
        #   AR = S
        # which when multiplied by our data will give us ARD = SD.
        #
        # R has very simple structure, so we just set A = S[:, :-1]. This
        # works because the left part of R is just the identity matrix, and
        # then the rightmost column is all -1, which means that the rightmost
        # column of AR will be whatever it needs to be so that each row sums
        # to 0... which means that it will match whatever's in S, since S also
        # has this property.
        #
        # FIXME: Maybe it would be useful to keep track of what our reference
        # actually _is_...
        # FIXME: it'd be nice to support transforming from arbitrary
        # references to arbitrary references. (This is pretty easy from the
        # equation AR = S above.)
        names = list(set(self.columns).difference(skip)) + [refname]
        constraint = DesignInfo(names).linear_constraint(new_ref, names)
        if constraint.coefs.shape[0] != 1:
            raise ValueError, "only one reference please!"
        if np.any(constraint.constants != 0):
            raise ValueError, "no constants in references please!"
        if (np.abs(np.sum(constraint.coefs) - 1) > 1e-6
            or np.any(constraint.coefs < 0)):
            raise ValueError("a reference must be a convex combination of "
                             "electrodes")
        full_S = np.eye(n + 1) - contraint.coefs
        assert np.all(np.abs(np.sum(full_S, axis=0) - 1) < 1e-6)
        all_zero_rows = np.all(full_S == 0, axis=0)
        channel_names = list(self.channels)
        if np.any(all_zero_rows):
            # re-referencing to a single existing channel -- we drop the
            # channel that is becoming the reference (since it would be
            # all-zero), and add in the old reference channel at the end
            # (adjusting column names to suit).
            assert np.sum(all_zero_rows) == 1
            new_ref_channel = all_zero_rows.nonzero()[0][0]
            # get rid of that row, instead of dropping the last row
            S = np.row_stack((full_S[:new_ref_channel, :],
                              full_S[new_ref_channel + 1:]))
            channel_names = (channel_names[:new_ref_channel]
                             + channel_names[new_ref_channel + 1:]
                             + [ref_name])
        else:
            S = full_S[:-1, :]
        A = S[:, :-1]
        arr, metadata = unpack_pandas(self.data)
        np.dot(arr, A.T, out=arr)
        self.data = pack_pandas(arr, metadata)
        self.data.columns = channel_names

def index_plus_latency(index, latency, direction):
    return index[:-1] + (index[-1] + latency,)

def index_sub(index1, index2):
    if index1[:-1] != index2[:-1]:
        raise ValueError, "incomparable indices"
    return index1[-1] - index2[-1]

def latency_normalize(approx_latency, sample_period):
    # Aligns approx_latency exactly to a value of the form int * sample_period
    count = int(round(approx_latency * 1.0 / sample_period))
    return count * sample_period

class ContinuousData(DataBase):
    def __init__(self, name, data, events, recording_info):
        self.name = name
        self.data = data
        self.events = events
        self.recording_info = recording_info

    def _get_channels(self):
        return self.data.columns

    def _set_channels(self, new_channels):
        self.data.columns = new_channels

    channels = property(_get_channels, _set_channels)

    @property
    def num_channels(self):
        return self.data.shape[1]

    @property
    def num_samples(self):
        return self.data.shape[0]

    # This does epochs_at_times, while returning a mask saying which items
    # were deleted for incompleteness
    def _epochs_at_times(self, timelock_indices, start_latency, end_latency,
                         name, metadata, on_incomplete):
        timelock_indices = np.asarray(timelock_indices,
                                      dtype=self.data.index.dtype)
        if on_incomplete not in ("error", "skip", "fill_NA"):
            raise ValueError("on_incomplete must be one of "
                             "\"error\", \"skip\", or \"fill_NA\"")

        start_offset = int(round(start_latency
                                 * self.recording_info.exact_srate_hz))
        end_offset = int(round(end_latency
                               * self.recording_info.exact_srate_hz))
        epoch_npoints = end_offset + 1 - start_offset
        sample_period = self.recording_info.approx_sample_period_ms
        epoch_index = pandas.Series(np.arange(start_offset, end_offset + 1)
                                    * sample_period)
        # We put the index bounds mid-way between sample points, to avoid any
        # issues of inconsistent floating-point rounding. (The actual bounds
        # of each epoch may vary by +/- 1 ulp.)
        start_latency_bound = ((start_offset - 0.5)
                               * self.recording_info.approx_sample_period_ms)
        end_latency_bound = ((end_offset + 0.5)
                             * self.recording_info.approx_sample_period_ms)

        truncated = np.zeros(timelock_indices.shape, dtype=bool)
        epochs = []
        for i, timelock_idx in enumerate(timelock_indices):
            # By using pandas range-based indexing, we can straightforwardly
            # isolate the portion of the data that falls into this epoch, even
            # if it is incomplete.
            start_idx = index_plus_latency(timelock_idx, start_latency_bound)
            end_idx = index_plus_latency(timelock_idx, end_latency_bound)
            epoch = self.data[start_idx:end_idx]
            if len(epoch) != epoch_npoints:
                if on_incomplete == "error":
                    raise ValueError("truncated epoch at %s "
                                     "with on_incomplete=\"error\""
                                     % (timelock_idx,))
                truncated[i] = True
            # The indices in our "ideal" epoch are all of the form:
            #   integer * sample period
            # we'll match each offset in our data to an index of this form
            # (thus making sure that the indices from different epochs match,
            # even in the presence of floating point rounding)
            approx_latencies = [index_sub(idx, timelock_idx)
                                for idx in epoch.index]
            exact_latencies = [latency_normalize(approx_latency, sample_period)
                               for approx_latency in approx_latencies]
            epoch.index = exact_latencies
            epochs.append(epoch)
        # pandas.Panel will align our single-epoch DataFrames, filling in NAs
        # where necessary (and re-sort them into time order after dict()
        # throws away the ordering):
        data = pandas.Panel(dict(zip(timelock_indices, epochs)))
        assert np.all(sorted(data.items) == data.items)
        recording_info = self.recording_info.clone()
        recording_info.metadata.update(metadata)
        recording_info.metadata["epoched-orig-indices"] = timelock_indices
        reject_counts = {}
        if on_incomplete == "skip" and np.any(truncated):
            data = data.ix[~truncated, : ,:]
            reject_counts["incomplete_epochs"] = np.sum(truncated)
        return truncated, EpochedData(name, data, None,
                                      recording_info, reject_counts)

    def epochs_at_times(self, timelock_indices, start_latency, end_latency,
                        name=None, metadata={}, on_incomplete="error"):
        res = self._epochs_at_times(timelock_indices, start_latency,
                                    end_latency, name, metadata)
        truncated, epoched = res
        return epoched

    def epochs(self, which_events, start_latency, end_latency,
               reject=None, name=None, metadata={}, on_incomplete="error"):
        # FIXME: is name is None, make up a name from which_events. (This
        # requires a way to stringify Events queries.)
        query = self.events.as_query(which_events)
        if reject is not None:
            reject_query = self.events.as_query(reject)
            num_rejected = len(query & reject_query)
            query &= ~reject_query
        else:
            num_rejected = 0
        event_set = self.events.find(query)
        events = list(event_set)
        timelock_indices = [ev.index for ev in events]
        res = self._epochs_at_times(timelock_indices, start_latency,
                                    end_latency, name, metadata)
        truncated, epoched = res
        if on_incomplete == "skip" and np.any(truncated):
            for i in np.nonzero(truncated)[0]:
                event_set.remove(events[i])
        epoched.event_set = event_set
        assert len(event_set) == epoched.num_epochs
        epoched.reject_counts["rejected_events"] = num_rejected
        return epoched

    def rerp(self, which_events, start_latency, end_latency, formula,
             reject=None, on_incomplete="error", name=None, correct_overlap=False):
        if correct_overlap:
            raise NotImplementedError
        else:
            epoched = self.epochs(which_events, start_latency, end_latency,
                                  reject=reject)
            return epoched.rerp(formula, name=name, eval_env=1)

class EpochedData(DataBase):
    def __init__(self, name, data, event_set, recording_info, reject_counts={}):
        self.name = name
        self.data = data
        self.event_set = event_set
        self.recording_info = recording_info
        self.reject_counts = dict(reject_counts)

    def _get_channels(self):
        return self.data.minor_axis

    def _set_channels(self, new_channels):
        self.data.minor_axis = new_channels

    channels = property(_get_channels, _set_channels)

    @property
    def num_channels(self):
        return self.data.shape[2]

    @property
    def num_samples(self):
        return self.data.shape[1]

    @property
    def num_epochs(self):
        return self.data.shape[0]

    def baseline(self):
        for epoch_idx in self.data.items:
            epoch = self.data[epoch_idx]
            self.data[epoch_idx] -= np.mean(epoch[epoch.index <= 0], axis=1)

    def erp(self, combine=np.mean, name=None):
        if name is None:
            method_name = "%s-epoch ERP" % (self.num_epochs,)
            if combine is not np.mean:
                method_name += "/%s" % (combine.__name__)
            if self.name is not None:
                name = "%s (%s)" % (self.name, method_name)
            else:
                name = method_name
        metadata = {
            "ERP_combine_method": combine.__name__,
            "ERP_combine_fn": combine,
            "ERP_num_used_trials": self.data.shape[0],
            "rejected_counts": dict(self.rejected_counts),
            }
        recording_info = self.recording_info.clone()
        recording_info.metadata.update(metadata)
        erp = combine(self.data, axis=0)
        data = pandas.DataFrame(erp,
                                columns=self.columns,
                                index=self.major_axis)
        return ContinuousData(name, data, None, recording_info)

    def rerp(self, formula_like, name=None, eval_env=0):
        from patsy import EvalEnvironment, dmatrix
        from pyrerp.incremental_ls import QRIncrementalLS
        eval_env = EvalEnvironment.capture(eval_env, reference=1)
        X = dmatrix(formula_like, self.event_set, eval_env=eval_env,
                    return_type="dataframe")
        if X.index.equals(self.data.index):
            in_data = self.data
        else:
            in_data = self.data[X.index]
        assert in_data.shape[0] == X.shape[0]
        model = QRIncrementalLS()
        model.append(np.asarray(X),
                     np.asarray(in_data).reshape((in_data.shape[0], -1)))
        result = model.fit()
        coef = result.coef()
        coef.resize((X.shape[1], self.num_samples, self.num_channels))
        out_data = pandas.Panel(data,
                                entries=X.design_info.column_names,
                                major_axis=self.data.major_axis,
                                minor_axis=self.data.minor_axis)
        metadata = {
            "rERP_formula_like": formula_like,
            "rERP_design_info": X.design_info,
            "rERP_num_used_trials": X.shape[0],
            "rejected_counts": dict(self.rejected_counts),
            }
        recording_info = self.recording_info.clone()
        recording_info.metadata.update(metadata)
        return EpochedData(name, out_data, None, recording_info)
