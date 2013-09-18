# This file is part of pyrerp
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

import os.path
import struct
import os
import string
from collections import OrderedDict
import bisect

import numpy as np
import pandas

from pyrerp.data import SensorInfo, DataFormat
from pyrerp.util import maybe_open
from pyrerp._kutaslab import _decompress_crw_chunk

PAUSE_CODE = 49152

# There are also read_avg and write_erp_as_avg functions in here, but their
# API probably needs another look before anyone should use them.
__all__ = ["KutaslabRecording", "KutaslabError"]

class KutaslabError(Exception):
    pass

# Derived from erp/include/64header.h:
_header_dtype = np.dtype([
    ("magic", "<u2"),
    ("epoch_len", "<i2"), # epoch length in msec
    ("nchans", "<i2"),
    ("sums", "<i2"), # 0 = ERP, 1 = single trial
    # ^^ 8 bytes
    ("tpfuncs", "<i2"), # number of processing funcs
    ("pp10uv", "<i2"), # points / 10 uV
    ("verpos", "<i2"), # 1 normally, -1 for sign inversion (I think?)
    ("odelay", "<i2"), # ms from trigger to stim (usually 8)
    # ^^ 16 bytes
    ("totevnt", "<i2"), # "total log events" (0 in mima217.avg)
    ("10usec_per_tick", "<i2"),
    ("time", "<i4"), # "time in sample clock ticks" (0 in mima217.avg)
    # ^^ 24 bytes
    ("cond_code", "<i2"), # (0 in mima217.avg)
    ("presam", "<i2"), # pre-event time in epoch in msec
    ("trfuncs", "<i2"), # "number of rejection functions"
    ("totrr", "<i2"), # "total raw records including rejects" (0 in mima217.avg)
    # ^^ 32 bytes
    ("totrej", "<i2"), # "total raw rejects" (0 in mima217.avg) (0 in mima217.avg)
    ("sbcode", "<i2"), # "subcondition number ( bin number )" (0 in mima217.avg)
    ("cprecis", "<i2"), # Our average contains cprecis * 256 samples
    ("dummy1", "<i2"),
    # ^^ 40 bytes
    ("decfact", "<i2"), # "decimation factor used in processing"
    ("dh_flag", "<i2"), # "see defines - sets time resolution" (0 in mima217.avg)
    ("dh_item", "<i4"), # "sequential item #" (0 in mima217.avg)
    # ^^ 48 bytes
    ("rfcnts", "<i2", (8,)), # "individual rejection counts 8 poss. rfs"
    ("rftypes", "S8", (8,)), # "8 char. descs for 8 poss. rfs"
    ("chndes", "S128"),
    ("subdes", "S40"),
    ("sbcdes", "S40"),
    ("condes", "S40"),
    ("expdes", "S40"),
    ("pftypes", "S24"),
    ("chndes2", "S40"),
    ("flags", "<u2"), # "see flag values below" (0 in mima217.avg)
    ("nrawrecs", "<u2"), # "# raw records if this is a raw file header"
                             # (0 in mima217.avg)
    ("idxofflow", "<u2"), # (0 in mima217.avg)
    ("idxoffhi", "<u2"), # (0 in mima217.avg)
    ("chndes3", "S24"),
    ])

# If, say, chndes has trailing null bytes, then rec["chndes"] will give us a
# less-than-128-byte string back. But this function always gives us the full
# 128 byte string, trailing nuls and all.
def _get_full_string(record, key):
    val = record[key]
    desired_len = record.dtype.fields[key][0].itemsize
    return val + (desired_len - len(val)) * "\x00"

# Translation tables for the ad hoc 6-bit character encoding used to encode
# electrode names in the 64-channel format:
_char2code = {}
for i, char in enumerate(string.lowercase):
    _char2code[char] = i + 1
for i, char in enumerate(string.uppercase):
    _char2code[char] = i + 27
for i, char in enumerate(string.digits):
    _char2code[char] = i + 53
_code2char = dict([(v, k) for (k, v) in _char2code.iteritems()])

def _read_header(stream):
    header_str = stream.read(512)
    header = np.fromstring(header_str, dtype=_header_dtype)[0]
    if header["magic"] == 0x17a5:
        # Raw file magic number:
        reader = _read_raw_chunk
    elif header["magic"] == 0x97a5:
        # Compressed file magic number:
        reader = _read_compressed_chunk
    else:
        assert False, "Unrecognized file type"
    hz = 1 / (header["10usec_per_tick"] / 100000.0)
    if abs(hz - int(hz)) > 1e-6:
        raise KutaslabError("file claims weird non-integer sample rate %shz"
                            % hz)
    hz = int(hz)

    channel_names = _channel_names_from_header(header)

    # Also read out the various free-form informational strings:
    info = {}
    info["subject"] = header["subdes"]
    info["experiment"] = header["expdes"]
    info["kutaslab_raw_header"] = header

    return (reader, header["nchans"], hz, channel_names, info, header)

def _channel_names_from_header(header):
    if header["nchans"] <= 16:
        # For small montages, each channel gets 8 bytes of ascii, smushed
        # together into a single array:
        return np.fromstring(_get_full_string(header, "chndes"), dtype="S8")
    elif header["nchans"] <= 32:
        # For mid-size montages, each channel gets 4 bytes:
        return np.fromstring(_get_full_string(header, "chndes"), dtype="S4")
    else:
        # And for large montages, a complicated scheme is used.
        # First, pull out and combine all the relevant buffers:
        chan_buf = (_get_full_string(header, "chndes")
                    + _get_full_string(header, "chndes2")
                    + _get_full_string(header, "chndes3"))
        # Then, each 3 byte chunk represents 4 characters, each coded in 6
        # bits and packed together:
        channel_names_l = []
        for i in xrange(header["nchans"]):
            chunk = np.fromstring(chan_buf[3*i : 3*i+3], dtype=np.uint8)
            codes = [
                (chunk[0] >> 2) & 0x3f,
                (chunk[0] & 0x03) << 4 | (chunk[1] >> 4) & 0x0f,
                (chunk[1] & 0x0f) << 2 | (chunk[2] >> 6) & 0x03,
                (chunk[2] & 0x3f),
                ]
            chars = [_code2char[code] for code in codes if code != 0]
            channel_names_l.append("".join(chars))
        return np.array(channel_names_l)

def _channel_names_to_header(channel_names, header):
    header["nchan"] = len(channel_names)
    if len(erp.channel_names) <= 16:
        header["chndes"] = np.asarray(channel_names, dtype="S8").tostring()
    elif len(erp.channel_names) <= 32:
        header["chndes"] = np.asarray(channel_names, dtype="S4").tostring()
    else:
        encoded_names = []
        for channel_name in channel_names:
            if len(channel_name) > 4:
                raise KutaslabError("can't store channel names with >4 chars")
            codes = [_char2code[char] for char in channel_name]
            codes += [0 * (4 - len(codes))]
            assert len(codes) == 4
            char0 = ((codes[0] << 2) | (codes[1] >> 4)) & 0xff
            char1 = ((codes[1] << 4) | (codes[2] >> 2)) & 0xff
            char2 = ((codes[2] << 6) | codes[3]) & 0xff
            encoded_names += [chr(char0), chr(char1), chr(char2)]
        concat_buf = "".join(encoded_names)
        header["chndes"] = concat_buf[:128]
        header["chndes2"] = concat_buf[128:128 + 40]
        header["chndes3"] = concat_buf[128 + 40:]
    assert np.all(_channel_names_from_header(header) == channel_names)

def read_raw(stream, dtype):
    (reader, nchans, hz, channel_names, info, header) = _read_header(stream)
    # Data is stored in a series of "chunks" -- each chunk contains 256 s16
    # samples from each channel (the 32/64/whatever analog channels, plus 1
    # channel for codes -- that channel being first.).  The code channel
    # contains a "record number" as its first entry in each chunk, which
    # simply increments by 1 each time.
    chunkno = 0
    code_chunks = []
    data_chunks = []
    while True:
        read = reader(stream, nchans)
        if read is None:
            break
        (code_chunk, data_chunk) = read
        assert len(code_chunk) == 256
        assert data_chunk.shape == (256 * nchans,)
        assert code_chunk[0] == chunkno
        code_chunk[0] = 0
        code_chunk = np.asarray(code_chunk, dtype=np.uint16)
        data_chunk.resize((256, nchans))
        data_chunk = np.asarray(data_chunk, dtype=dtype)
        code_chunks.append(code_chunk)
        data_chunks.append(data_chunk)
        chunkno += 1
    return (hz, channel_names,
            np.concatenate(code_chunks),
            np.row_stack(data_chunks),
            info)

def _read_raw_chunk(stream, nchans):
    chunk_bytes = (nchans + 1) * 512
    buf = stream.read(chunk_bytes)
    # Check for EOF:
    if not buf:
        return None
    codes_list = list(struct.unpack("<256H", buf[:512]))
    data_chunk = np.fromstring(buf[512:], dtype="<i2")
    return (codes_list, data_chunk)

def _read_compressed_chunk(stream, nchans):
    # Check for EOF:
    ncode_records_minus_one_buf = stream.read(1)
    if not ncode_records_minus_one_buf:
        return None
    # Code track (run length encoded):
    (ncode_records_minus_one,) = struct.unpack("<B",
                                               ncode_records_minus_one_buf)
    ncode_records = ncode_records_minus_one + 1
    code_records = []
    for i in xrange(ncode_records):
        code_records.append(struct.unpack("<BH", stream.read(3)))
    codes_list = []
    for (repeat_minus_one, code) in code_records:
        codes_list += [code] * (repeat_minus_one + 1)
    assert len(codes_list) == 256
    # Data bytes (delta encoded and packed into variable-length integers):
    (ncompressed_words,) = struct.unpack("<H", stream.read(2))
    compressed_data = stream.read(ncompressed_words * 2)
    data_chunk = _decompress_crw_chunk(compressed_data, ncompressed_words,
                                       nchans)
    return (codes_list, data_chunk)

def assert_files_match(p1, p2):
    (hz1, channames1, codes1, data1, info1) = read_raw(open(p1), "u2")
    (hz2, channames2, codes2, data2, info2) = read_raw(open(p2), "u2")
    assert hz1 == hz2
    assert (channames1 == channames2).all()
    assert (codes1 == codes2).all()
    assert (data1 == data2).all()
    for k in set(info1.keys() + info2.keys()):
        if k != "kutaslab_raw_header":
            assert info1[k] == info2[k]

def test_read_raw_on_test_data():
    import glob
    from pyrerp.test import test_data_path
    tested = 0
    for rawp in glob.glob(test_data_path("*.raw")):
        crwp = rawp[:-3] + "crw"
        print rawp, crwp
        assert_files_match(rawp, crwp)
        tested += 1
    # Cross-check, to make sure is actually finding the files... (bump up this
    # number if you add more test files):
    assert tested == 4

def test_64bit_channel_names():
    from pyrerp.test import test_data_path
    stream = open(test_data_path("two-chunks-64chan.raw"))
    (hz, channel_names, codes, data, info) = read_raw(stream, int)
    # "Correct" channel names as listed by headinfo(1):
    assert (channel_names ==
            ["LOPf", "ROPf", "LMPf", "RMPf", "LTPf", "RTPf", "LLPf", "RLPf",
             "LPrA",  "RPrA", "LTFr", "RTFr", "LLFr", "RLFr", "LDPf", "RDPf",
             "LTOc", "RTOc", "LTCe", "RTCe", "LLCe", "RLCe", "LDFr", "RDFr",
             "LMFr", "RMFr", "MiFo", "MiPf", "MiFr", "A2",   "LHEy", "RHEy",
             "LIOc", "RIOc", "LLOc", "RLOc", "LLPP", "RLPP", "LLPa", "RLPa",
             "LDCe", "RDCe", "LMCe", "RMCe", "LDOc", "RDOc", "LDPP", "RDPP",
             "LDPa", "RDPa", "LCer", "RCer", "LMOc", "RMOc", "LMPP", "RMPP",
             "LMPa", "RMPa", "MiCe", "MiPa", "MiPP", "MiOc", "LLEy", "RLEy"]
            ).all()

# For debugging:
def compare_raw_to_crw(raw_stream, crw_stream):
    raw_reader, raw_nchans, raw_hz, raw_names, raw_l, header = _read_header(raw_stream)
    crw_reader, crw_nchans, crw_hz, crw_names, crw_l, header = _read_header(crw_stream)
    assert raw_reader is _read_raw_chunk
    assert crw_reader is _read_compressed_chunk
    assert raw_nchans == crw_nchans
    assert raw_hz == crw_hz
    assert raw_names == crw_names
    assert crw_l is None
    while True:
        raw_start = raw_stream.tell()
        raw_chunk = _read_raw_chunk(raw_stream, raw_nchans)
        raw_end = raw_stream.tell()
        crw_start = crw_stream.tell()
        crw_chunk = _read_compressed_chunk(crw_stream, crw_nchans)
        crw_end = crw_stream.tell()
        assert (raw_chunk is None) == (crw_chunk is None)
        if raw_chunk is None:
            break
        (raw_codes, raw_data) = raw_chunk
        (crw_codes, crw_data) = crw_chunk
        problems = []
        if raw_codes != crw_codes:
            problems.append("codes")
        if tuple(raw_data) != tuple(crw_data):
            problems.append("data")
        if problems:
            print ("Bad %s! raw: [%s, %s], crw: [%s, %s]"
                   % (problems, raw_start, raw_end, crw_start, crw_end))
            assert False

def read_log(file_like):
    fo = maybe_open(file_like)
    ticks = []
    events = []
    while True:
        event = fo.read(8)
        if not event:
            break
        (code, tick_hi, tick_lo, condition, flag) \
               = struct.unpack("<HHHBB", event)
        ticks.append(tick_hi << 16 | tick_lo)
        events.append((code, condition, flag))
    df = pandas.DataFrame(events, columns=["code", "condition", "flag"],
                          index=ticks)
    df["flag_data_error"] = np.asarray(df["flag"] & 0o100, dtype=bool)
    df["flag_rejected"] = np.asarray(df["flag"] & 0o40, dtype=bool)
    df["flag_polinv"] = np.asarray(df["flag"] & 0o20, dtype=bool)
    return df

# XX no idea what this "loc" file thing is
# really should learn to read a topofile (see topofiles(5) and lib/topo)
def read_loc(file_like):
    fo = maybe_open(file_like)
    names = []
    thetas = []
    rs = []
    for line in fo:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        (_, theta, r, name) = line.split()
        # Strip trailing periods
        while name.endswith("."):
            name = name[:-1]
        names.append(name)
        thetas.append(theta)
        rs.append(r)
    return SensorInfo(names, thetas, rs)

# XX someday should fix this so that it delays reading the actual data until
# needed (to avoid the giant memory overhead of loading in lots of data sets
# together). It really wouldn't be hard -- the log file tells you where all
# the events are, and the length of each span. (You can even tell how many
# samples there are total, because the last event in the log file is always a
# pause code placed at the last sample of the data.)
class KutaslabRecording(Recording):
    def __init__(self, f_raw, f_log, name=None, f_loc=None,
                 dtype=np.float64, calibration_condition=0):
        if isinstance(f_raw, basestring) and name is None:
            name = os.path.basename(f_raw)
        if name is None:
            raise ValueError("recording name must be supplied when loading "
                             "from file object")
        self.name = name
        f_raw = maybe_open(f_raw)
        f_log = maybe_open(f_log)
        (hz, channel_names, raw_codes, data, metadata) = read_raw(f_raw, dtype)
        self.metadata = metadata
        # XX FIXME: build in the 26 and 64 channel cap info
        if f_loc is not None:
            self.electrode_info = read_loc(f_loc)
        else:
            self.electrode_info = SensorInfo([], [], [])
        self.data_format = DataFormat(hz, "RAW", channel_names)

        raw_log_events = read_log(f_log)
        expanded_log_codes = np.zeros(raw_codes.shape, dtype=int)
        expanded_log_codes[raw_log_events.index] = raw_log_events["code"]
        discrepancies = (expanded_log_codes != raw_codes)
        if (not (expanded_log_codes[discrepancies] == 0).all()
            or not (raw_codes[discrepancies] == 65535).all()):
            raise KutaslabError("raw and log files have mismatched codes")
        del raw_codes
        del expanded_log_codes

        pause_events = (raw_log_events["code"] == PAUSE_CODE)
        pause_ticks = raw_log_events.index[pause_events]
        # The pause code appears at the last sample of the old era, so if used
        # directly, adjacent pause ticks give contiguous spans of recording as
        # (pause1, pause2]. (Confirmed by checking by hand in a real recording
        # that the data associated with the sample that has the pause code is
        # contiguous with the sample before, but not the sample after.)
        # Adding +1 to each of them then converts this to Python style
        # [pause1, pause2) intervals. There is a pause code at the last record
        # of the file, but not one at the first, so we add that in explicitly.
        pause_ticks += 1
        span_edges = np.concatenate(([0], pause_ticks))

        self._span_slices = [(span_edges[i], span_edges[i + 1])
                             for i in xrange(len(span_edges) - 1)]
        self.span_lengths = [(stop - start)
                             for (start, stop) in self._span_slices]

        self._data = data
        self._raw_log_events = raw_log_events
        self._calibration_condition = calibration_condition

    def span_data(self, span_ids):
        for span_id in span_ids:
            start, stop = self._span_slices[span_id]
            yield self._data[start:stop, :]

    def event_iter(self):
        span_starts = [start for (start, stop) in self._span_slices]
        for tick, row in self._raw_log_events.iterrows():
            if row["condition"] == self._calibration_condition:
                attrs = {"calibration_pulse": True}
            else:
                attrs = row.to_dict()
            attrs["subject"] = self.metadata["subject"]
            attrs["experiment"] = self.metadata["experiment"]

            span_id = bisect.bisect(span_starts, tick) - 1
            span_start, span_stop = self._span_slices[span_id]
            assert span_start <= tick < span_stop

            yield (span_id, tick - span_start, tick - span_start + 1, attrs)

# To read multiple bins, call this repeatedly on the same stream
def read_avg(stream, dtype):
    # Ignore the 'reader' field -- avg files always have the same magic number
    # as whatever type of raw file was used to create them -- and they're
    # always in their own format, regardless.
    (_, nchans, hz, channel_names, info, header) = _read_header(stream)
    assert header["cprecis"] > 0
    data_chunks = []
    for i in xrange(header["cprecis"]):
        data_bytes = stream.read(512 * nchans)
        data_chunk = np.fromstring(data_bytes, dtype="<i2")
        data_chunk.resize((256, nchans))
        data_chunks.append(np.asarray(data_chunk, dtype=dtype))
    return np.vstack(data_chunks), hz, channel_names, info, header

# You can write multiple bins to the same file by calling this function
# repeatedly on the same open file handle.
def write_epoched_as_avg(epoched_data, stream, allow_resample=False):

    assert False, ("this code won't work; the data structures have been "
                   "rewritten multiple times since it was last used. "
                   "But it could be fixed up without too much work...")

    stream = maybe_open(stream, "ab")
    data_array = np.asarray(epoched_data.data)
    # One avg record is always exactly 256 * cprecis samples long, with
    # cprecis = 1, 2, 3 (a limitation of the data format).  So we pick the
    # smallest cprecis that is >= our actual number of samples (maximum 3),
    # and then we resample to have that many samples exactly.  (I.e., we try
    # to resample up when possible.)
    if epoched_data.num_samples <= 1 * 256:
        cprecis = 1
    elif epoched_data.num_samples <= 2 * 256:
        cprecis = 2
    else:
        cprecis = 3
    samples = cprecis * 256
    if epoched_data.num_samples != samples:
        if allow_resample:
            import scipy.signal
            data_array = scipy.signal.resample(data_array, samples, axis=1)
        else:
            raise KutaslabError("kutaslab avg files must contain exactly "
                                "256, 512, or 768 samples, but your data "
                                "has %s. Use allow_resample=True if you "
                                "want me to automatically resample your "
                                "data to %s samples"
                                % (epoched_data.num_samples, samples,))
    assert data_array.shape[1] == samples
    times = epoched_data.data.major_axis
    actual_length = times.max() - times.min()
    presam = 0 - times.min()
    # Compute closest internally-consistent approximations to the timing
    # information that can be represented in this data format. The two things
    # we get to store are:
    #   the sampling period measured as an integer multiple of 10us
    #   the epoch length measured as an integer number of ms
    sample_period_in_10us = int(round(actual_length * 100. / samples))
    epoch_len_ms = int(round(samples * sample_period_in_10us / 100.))
    # Need to convert data from floats to s16's. To preserve as much
    # resolution as possible, we use the full s16 range, minus a bit to make
    # sure we don't run into any overflow issues.
    s16_max = 2 ** 15 - 10
    # Same as np.abs(data).max(), but without copying the whole array:
    data_max = max(data_array.max(), np.abs(data_array.min()))
    # We have to write the conversion factor as an integer, so we round it
    # *down* here (to avoid overflow), and then use the *rounded* version to
    # actually convert the data.
    s16_per_10uV = int(s16_max / (data_max / 10))
    # Except that if our conversion factor itself overflows, then we have to
    # truncate it back down (and lose a bit of resolution in the process, oh
    # well):
    if s16_per_10uV > s16_max:
        s16_per_10uV = s16_max
    metadata = epoched_data.recording_info.metadata
    for i in xrange(epoched_data.num_epochs):
        header = np.zeros(1, dtype=_header_dtype)[0]
        header["magic"] = 0x97a5
        header["verpos"] = 1
        integer_data = np.asarray(np.round(s16_per_10uV
                                           * data_array[i, :, :] / 10.),
                                  dtype="<i2")

        header["epoch_len"] = epoch_len_ms
        header["nchans"] = integer_data.shape[1]
        # "pf" = "processing function", i.e., something like "averaging" or
        # "standard error" that describes how raw data was analyzed to create
        # this curve.
        header["tpfuncs"] = 1
        header["pftypes"] = "pyrERP"
        header["pp10uv"] = s16_per_10uV
        header["10usec_per_tick"] = sample_period_in_10us
        header["presam"] = presam
        header["cprecis"] = cprecis
        # Supposedly this should be used to write down resampling information.
        # The kutaslab tools only do integer-factor downsampling (decimation),
        # and they write the decimation factor to the file here.  I don't see
        # how it matters for the .avg file to retain the decimation
        # information, and the file won't let us write down upsampling
        # (especially non-integer upsampling!), so we just pretend our
        # decimation factor is 1 and be done with it.
        header["decfact"] = 1

        used_trials = metadata.get("ERP_num_used_trials",
                                   metadata.get("rERP_num_used_trials",
                                                0))
        rejected_counts = metadata.get("rejected_counts", {})
        total_rejected = np.sum(rejected_counts.values())
        header["totrr"] = used_trials + total_rejected
        header["totrej"] = total_rejected
        header["trfuncs"] = min(8, len(rejected_counts))
        for i, (name, count) in enumerate(rejected_counts.iteritems()):
            if i >= header["trfuncs"]:
                break
            header["rftypes"][i] = name[:8]
            header["rfcnts"][i] = count

        if "kutaslab_raw_header" in metadata:
            header["odelay"] = metadata["kutaslab_raw_header"]["odelay"]

        _channel_names_to_header(epoched_data.channels, header)

        if "experiment" in metadata:
            header["expdes"] = metadata["experiment"]
        if "subject" in metadata:
            header["subdes"] = metadata["subject"]
        header["condes"] = str(epoched_data.data.entries[i])

        header.tofile(stream)
        # avg files omit the mark track.  And, all the data for a single
        # channel goes together in a single chunk, rather than interleaving
        # all channels.  THIS IS TOTALLY DIFFERENT FROM RAW FILES, DON'T GET
        # CONFUSED!
        for i in xrange(integer_data.shape[1]):
            integer_data[:, i].tofile(stream)

# From looking at:
#   plot(data[:,
#             np.add.outer(np.arange(-100, 100),
#                          (marks == 1).nonzero()[0]).squeeze()].mean(axis=2).transpose())
# It looks like the high part of the cal is about code_idx+15:code_idx+45, and
# the low part is -60:-10

# class BadChannels(Exception):
#     pass

# def calibrate_in_place(data, codes,
#                        before=np.arange(-40, -10), after=np.arange(15, 45),
#                        pulse_size=10, # true pulse size measured in uV
#                        stddev_limit=3,
#                        cal_code=1):
#     assert len(before) == len(after)
#     cal_codes = (codes == cal_code).nonzero()[0]
#     # Trim off the first and last few cals, because there may be truncation
#     # effects:
#     cal_codes = cal_codes[2:-2]
#     before_cals = data[np.add.outer(cal_codes, before).ravel(), :]
#     after_cals = data[np.add.outer(cal_codes, after).ravel(), :]
#     deltas = before_cals - after_cals
#     bad_channels = (deltas.std(axis=0) > stddev_limit)
#     if bad_channels.any():
#         raise BadChannels, bad_channels.nonzero()[0]
#     delta = deltas.mean(axis=0)
#     assert (delta > 0).all() or (delta < 0).all()
#     delta = np.abs(delta)
#     data *= pulse_size / delta

if __name__ == "__main__":
    import nose
    nose.runmodule()
