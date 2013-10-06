# This file is part of rERPy
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

import os.path
import struct
import os
import string
from collections import OrderedDict
import bisect
import sys

import numpy as np
import pandas

from rerpy.data import DataFormat, Dataset
from rerpy.util import maybe_open
from rerpy.io._erpss import _decompress_crw_chunk

PAUSE_CODE = 49152
DELETE_CODE = 57344

# There are also read_avg and write_erp_as_avg functions in here, but their
# API probably needs another look before anyone should use them.
__all__ = ["load_erpss"]

# Derived from erp/include/64header.h:
_header_dtype = np.dtype([
    ("magic", "<u2"),
    ("epoch_len", "<i2"), # epoch length in msec
    ("nchans", "<i2"),
    ("sums", "<i2"), # 0 = ERP, 1 = single trial
    # -- 8 bytes --
    ("tpfuncs", "<i2"), # number of processing funcs
    ("pp10uv", "<i2"), # points / 10 uV
    ("verpos", "<i2"), # 1 normally, -1 for sign inversion (I think?)
    ("odelay", "<i2"), # ms from trigger to stim (usually 8)
    # -- 16 bytes --
    ("totevnt", "<i2"), # "total log events" (0 in mima217.avg)
    ("10usec_per_tick", "<i2"),
    ("time", "<i4"), # "time in sample clock ticks" (0 in mima217.avg)
    # -- 24 bytes --
    ("cond_code", "<i2"), # (0 in mima217.avg)
    ("presam", "<i2"), # pre-event time in epoch in msec
    ("trfuncs", "<i2"), # "number of rejection functions"
    ("totrr", "<i2"), # "total raw records including rejects" (0 in mima217.avg)
    # -- 32 bytes --
    ("totrej", "<i2"), # "total raw rejects" (0 in mima217.avg) (0 in mima217.avg)
    ("sbcode", "<i2"), # "subcondition number ( bin number )" (0 in mima217.avg)
    ("cprecis", "<i2"), # Our average contains cprecis * 256 samples
    ("dummy1", "<i2"),
    # -- 40 bytes --
    ("decfact", "<i2"), # "decimation factor used in processing"
    ("dh_flag", "<i2"), # "see defines - sets time resolution" (0 in mima217.avg)
    ("dh_item", "<i4"), # "sequential item #" (0 in mima217.avg)
    # -- 48 bytes --
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
    # -- 512 bytes --
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
        fetcher = RawChunkFetcher(stream, header["nchans"])
    elif header["magic"] == 0x97a5:
        # Compressed file magic number:
        fetcher = CrwChunkFetcher(stream, header["nchans"])
    else: # pragma: no cover
        assert False, "Unrecognized file type"
    hz = 1 / (header["10usec_per_tick"] / 100000.0)
    if abs(hz - int(round(hz))) > 1e-6:
        raise ValueError("file claims weird non-integer sample rate %shz"
                         % hz)
    hz = int(round(hz))

    channel_names = _channel_names_from_header(header)

    # Also read out the various general informational bits:
    info = {}
    info["subject"] = header["subdes"]
    info["experiment"] = header["expdes"]
    info["odelay"] = header["odelay"]
    # And save the raw header in case anyone wants it later (you never know)
    info["erpss_raw_header"] = header_str

    return (fetcher, header["nchans"], hz, channel_names, info, header)

def _channel_names_from_header(header):
    if header["nchans"] <= 16:
        # For small montages, each channel gets 8 bytes of ascii, smushed
        # together into a single array:
        return np.fromstring(_get_full_string(header, "chndes"),
                             dtype="S8")[:header["nchans"]]
    elif header["nchans"] <= 32:
        # For mid-size montages, each channel gets 4 bytes:
        return np.fromstring(_get_full_string(header, "chndes"),
                             dtype="S4")[:header["nchans"]]
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
        return np.array(channel_names_l[:header["nchans"]])

def _channel_names_to_header(channel_names, header):
    header["nchans"] = len(channel_names)
    if len(channel_names) <= 16:
        header["chndes"] = np.asarray(channel_names, dtype="S8").tostring()
    elif len(channel_names) <= 32:
        header["chndes"] = np.asarray(channel_names, dtype="S4").tostring()
    else:
        encoded_names = []
        for channel_name in channel_names:
            codes = [_char2code[char] for char in channel_name]
            codes += [0] * (4 - len(codes))
            char0 = ((codes[0] << 2) | (codes[1] >> 4)) & 0xff
            char1 = ((codes[1] << 4) | (codes[2] >> 2)) & 0xff
            char2 = ((codes[2] << 6) | codes[3]) & 0xff
            encoded_names += [chr(char0), chr(char1), chr(char2)]
        concat_buf = "".join(encoded_names)
        header["chndes"] = concat_buf[:128]
        header["chndes2"] = concat_buf[128:128 + 40]
        header["chndes3"] = concat_buf[128 + 40:]
    if not np.all(_channel_names_from_header(header) == channel_names):
        raise ValueError("failed to encode channel names in header -- maybe "
                         "some names are too long?")

def test_channel_names_roundtrip():
    # Try 1 char, 2 char, 3 char, 4 char names
    # Try all letters in 6-bit character set (digits, lowercase, uppercase)
    names = ["A", "a", "1", "Aa", "Aa1", "Aa1A"]
    import itertools
    for char, digit in itertools.izip(itertools.cycle(string.uppercase),
                                      itertools.cycle(string.digits)):
        names.append(char + char.lower() + digit)
        if len(names) == 64:
            break
    def t(test_names):
        header = np.zeros(1, dtype=_header_dtype)[0]
        _channel_names_to_header(test_names, header)
        got_names = _channel_names_from_header(header)
        assert np.all(got_names == test_names)
    # skip names == [], b/c we hit https://github.com/numpy/numpy/issues/3764
    # and anyway, who cares about the nchans=0 case
    for i in xrange(1, len(names)):
        # Try all lengths
        t(names[:i])
    # Also try some long names for small headers where they're allowed
    long_names = ["a" * i for i in xrange(8)] * 2
    t(long_names)
    from nose.tools import assert_raises
    header = np.zeros(1, dtype=_header_dtype)[0]
    # But even for small headers, only 8 chars are allowed
    assert_raises(ValueError, _channel_names_to_header, ["a" * 9], header)
    # And for larger headers, only 4 chars are allowed
    for i in xrange(17, 64):
        assert_raises(ValueError,
                      _channel_names_to_header, ["a" * 5] * i, header)

def read_raw(stream, dtype, load_data):
    (fetcher, nchans, hz, channel_names, info, header) = _read_header(stream)
    # Data is stored in a series of "chunks" -- each chunk contains 256 s16
    # samples from each channel (the 32/64/whatever analog channels, plus 1
    # channel for codes -- that channel being first.).  The code channel
    # contains a "record number" as its first entry in each chunk, which
    # simply increments by 1 each time.
    chunkno = 0
    code_chunks = []
    data_chunks = []
    while True:
        read = fetcher.read_next_chunk(load_data)
        if read is None:
            break
        (code_chunk, data_chunk) = read
        assert len(code_chunk) == 256
        assert code_chunk[0] == chunkno
        code_chunk[0] = 0
        code_chunks.append(code_chunk)
        if load_data:
            assert data_chunk.shape == (256 * nchans,)
            data_chunk.resize((256, nchans))
            data_chunk = np.asarray(data_chunk, dtype=dtype)
            data_chunks.append(data_chunk)
        chunkno += 1
    codes = np.concatenate(code_chunks)
    if load_data:
        data = np.row_stack(data_chunks)
    else:
        data = None
    return (fetcher, hz, channel_names, codes, data, info)

# These two classes have slightly weird invariants. They have one of two life
# cycles:
# Option 1:
# - __init__
# - read_next_chunk(True) called repeatedly to load all codes and data
# Option 2:
# - __init__
# - read_next_chunk(False) called repeatedly to load all codes
# - get_chunk(chunk_number) called repeatedly to load random pieces of data
# read_next_chunk has the invariants that at entry, the stream will always be
# pointing to the beginning of the wanted chunk, and then on exit, the stream
# will always be pointing to beginning of the next chunk.
class RawChunkFetcher(object):
    def __init__(self, stream, nchans):
        self._stream = stream
        self._nchans = nchans
        self._chunk_size_bytes = (nchans + 1) * 256 * 2

    def read_next_chunk(self, return_data):
        buf = self._stream.read(self._chunk_size_bytes)
        # Check for EOF:
        if not buf:
            return None
        codes = np.fromstring(buf[:512], dtype=np.uint16)
        if return_data:
            data_chunk = np.fromstring(buf[512:], dtype="<i2")
        else:
            data_chunk = None
        return codes, data_chunk

    def get_chunk(self, chunk_number):
        offset = 512 + chunk * self._chunk_size_bytes
        self._stream.seek(offset)
        chunk_bytes = self._stream.read(self._chunk_size_bytes)
        data = np.fromstring(buf[512:], dtype="<i2")
        return data

class CrwChunkFetcher(object):
    def __init__(self, stream, nchans):
        self._stream = stream
        self._nchans = nchans
        self._offsets = []

    def read_next_chunk(self, return_data):
        # Check for EOF:
        ncode_records_minus_one_buf = self._stream.read(1)
        if not ncode_records_minus_one_buf:
            return None
        # Code track (run length encoded):
        (ncode_records_minus_one,) = struct.unpack("<B",
                                                   ncode_records_minus_one_buf)
        ncode_records = ncode_records_minus_one + 1
        codes = np.empty(256, np.uint16)
        cursor = 0
        for i in xrange(ncode_records):
            repeat_minus_one, code = struct.unpack("<BH", self._stream.read(3))
            codes[cursor:cursor + repeat_minus_one + 1] = code
            cursor += repeat_minus_one + 1
        assert cursor == 256
        # Data bytes (delta encoded and packed into variable-length integers):
        # Record where these start so we can find it again in get_chunk().
        self._offsets.append(self._stream.tell())
        (ncompressed_words,) = struct.unpack("<H", self._stream.read(2))
        compressed_data = self._stream.read(ncompressed_words * 2)
        if return_data:
            # This is the slow part of loading data:
            data_chunk = _decompress_crw_chunk(compressed_data,
                                               ncompressed_words,
                                               self._nchans)
        else:
            data_chunk = None
        return codes, data_chunk

    def get_chunk(self, chunk_number):
        self._stream.seek(self._offsets[chunk_number])
        (ncompressed_words,) = struct.unpack("<H", self._stream.read(2))
        return _decompress_crw_chunk(self._stream.read(ncompressed_words * 2),
                                     ncompressed_words,
                                     self._nchans)

class DemandLoader(object):
    def __init__(self, chunk_fetcher, dtype, nchans):
        self._chunk_fetcher = chunk_fetcher
        self._dtype = dtype
        self._nchans = nchans

    def get_slice(self, start_tick, stop_tick):
        output = np.empty((stop_tick - start_tick, self._nchans),
                          dtype=self._dtype)
        cursor = 0
        chunk_number = start_tick // 256
        while True:
            tick = chunk_number * 256
            if tick >= stop_tick:
                break
            data = self._chunk_fetcher.get_chunk(chunk_number)
            data.resize((256, self._nchans))
            low = max(tick, start_tick)
            high = min(tick + 256, stop_tick)
            next_cursor = cursor + (high - low)
            output[cursor:next_cursor, :] = data[low - tick:high - tick]
            cursor = next_cursor
            chunk_number += 1
        return output

def assert_files_match(p1, p2):
    (_, hz1, channames1, codes1, data1, info1) = read_raw(open(p1), "u2", True)
    for (p, load_data) in [(p1, False), (p2, True), (p2, False)]:
        (fetcher2, hz2, channames2, codes2, data2, info2
         ) = read_raw(open(p), "u2", load_data)
    assert hz1 == hz2
    assert (channames1 == channames2).all()
    assert (codes1 == codes2).all()
    if not load_data:
        assert data2 is None
        loader = DemandLoader(fetcher2, "u2", len(channames2))
        data2 = loader.get_slice(0, len(codes2))
    assert (data1 == data2).all()
    for k in set(info1.keys() + info2.keys()):
        if k != "erpss_raw_header":
            assert info1[k] == info2[k]

def test_read_raw_on_test_data():
    import glob
    from rerpy.test import test_data_path
    tested = 0
    for rawp in glob.glob(test_data_path("erpss/*.raw")):
        crwp = rawp[:-3] + "crw"
        print rawp, crwp
        assert_files_match(rawp, crwp)
        tested += 1
    # Cross-check, to make sure is actually finding the files... (bump up this
    # number if you add more test files):
    assert tested == 5

def test_64bit_channel_names():
    from rerpy.test import test_data_path
    stream = open(test_data_path("erpss/two-chunks-64chan.raw"))
    (_, hz, channel_names, codes, data, info) = read_raw(stream, int, False)
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

# Little hack useful for testing. AFAIK this is identical to the erpss
# 'makelog' program, except that:
# - 'makelog' throws away some events from the end of the file, including the
# very helpful final "pause" marker
# - 'makelog' "cooks" the log file, i.e., toggles the high bit of all events
# that occur in a span ended by a "delete mark" (see logfile.5). We don't
# bother. (Though could, I guess.)
def make_log(raw, condition=64): # pragma: no cover
    import warnings; warnings.warn("This code is not tested!")
    codes = read_raw(maybe_open(raw), np.float64)[3]
    log = []
    for i in codes.nonzero()[0]:
        log.append(struct.pack("<HHHBB",
                               codes[i], (i & 0xffff0000) >> 16, i & 0xffff,
                               condition,
                               0))
        if codes[i] in (PAUSE_CODE, DELETE_CODE):
            condition += 1
    return "".join(log)

def test_read_log():
    def t(data, expected):
        from cStringIO import StringIO
        got = read_log(StringIO(data))
        # .sort() is a trick to make sure columns line up
        from pandas.util.testing import assert_frame_equal
        assert_frame_equal(expected.sort(axis=1), got.sort(axis=1))

    # The first 80 bytes of arquan25.log (from Delong, Urbach & Kutas 2005)
    data = "01000000ec01010001000000e103010001000000f50601004b00000044070100010000007b0701004b000000ca07010001000000010801004b0000004f08010001000000860801004b000000d5080100".decode("hex")
    # From 'logexam arquan25.log 1' (1 means, measure time in ticks)
    # then 'l 0 9'
    expected = pandas.DataFrame(
        {"code": [1, 1, 1, 75, 1, 75, 1, 75, 1, 75],
         "condition": [1] * 10,
         "flag": [0] * 10,
         "flag_data_error": [False] * 10,
         "flag_rejected": [False] * 10,
         "flag_polinv": [False] * 10,
         },
        index=[492, 993, 1781, 1860, 1915, 1994, 2049, 2127, 2182, 2261],
        )
    t(data, expected)

    # 80 bytes from arquan25.log, starting at 8080*8 bytes into the file
    data = "01000e00d39b010000c00e00ff9e010023010e005a9f000023010e00dc9f000023010e005da0000023010e00dea0000023010e005fa1000023010e00e1a1000023010e0062a2000023010e00e3a20000".decode("hex")
    # from logexam, 'l 8080 8089'
    expected = pandas.DataFrame(
        {"code": [1, 49152, 291, 291, 291, 291, 291, 291, 291, 291],
         "condition": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         "flag": [0] * 10,
         "flag_data_error": [False] * 10,
         "flag_rejected": [False] * 10,
         "flag_polinv": [False] * 10,
         },
        index=[957395, 958207, 958298, 958428, 958557,
               958686, 958815, 958945, 959074, 959203],
        )
    t(data, expected)

# XX someday should fix this so that it has the option to delay reading the
# actual data until needed (to avoid the giant memory overhead of loading in
# lots of data sets together). The way to do it for crw files is just to read
# through the file without decompressing to find where each block is located
# on disk, and then we can do random access after we know that.
def load_erpss(raw, log, calibration_events="condition == 0",
               calibrate=False,
               calibrate_half_width_ticks=5,
               calibrate_low_cursor_time=None,
               calibrate_high_cursor_time=None,
               calibrate_pulse_size=None,
               calibrate_polarity=1):
    dtype = np.float64

    metadata = {}
    if isinstance(raw, basestring):
        metadata["raw_file"] = os.path.abspath(raw)
    if isinstance(log, basestring):
        metadata["log_file"] = os.path.abspath(log)
    metadata["calibration_events"] = str(calibration_events)

    raw = maybe_open(raw)
    log = maybe_open(log)

    (fetcher, hz, channel_names, raw_codes, data, header_metadata
     ) = read_raw(raw, dtype, True)
    metadata.update(header_metadata)
    if calibrate:
        units = "uV"
    else:
        units = "RAW"
    data_format = DataFormat(hz, units, channel_names)

    raw_log_events = read_log(log)
    expanded_log_codes = np.zeros(raw_codes.shape, dtype=int)
    try:
        expanded_log_codes[raw_log_events.index] = raw_log_events["code"]
    except IndexError as e:
        raise ValueError("log file claims event at position where there is "
                         "no data: %s" % (e,))
    # Sometimes people "delete" events by setting the high (sign) bit of the
    # code in the log file (e.g. with 'logpoke'). So we ignore this bit when
    # comparing log codes to raw codes -- mismatches here do not indicate an
    # error -- and then are careful to use the log codes, rather than the
    # raw codes, below.
    if np.any((expanded_log_codes & ~0x8000) != (raw_codes & ~0x8000)):
        raise ValueError("raw and log files have mismatched codes")
    del raw_codes
    del expanded_log_codes

    pause_events = (raw_log_events["code"] == PAUSE_CODE)
    delete_events = (raw_log_events["code"] == DELETE_CODE)
    break_events = pause_events | delete_events
    break_ticks = raw_log_events.index[break_events]
    # The pause/delete code appears at the last sample of the old era, so if
    # used directly, adjacent pause ticks give contiguous spans of recording
    # as (pause1, pause2]. (Confirmed by checking by hand in a real recording
    # that the data associated with the sample that has the pause code is
    # contiguous with the sample before, but not the sample after.)  Adding +1
    # to each of them then converts this to Python style [pause1, pause2)
    # intervals. There is a pause code at the last record of the file, but not
    # one at the first, so we add that in explicitly.
    break_ticks += 1
    span_edges = np.concatenate(([0], break_ticks))
    assert span_edges[0] == 0
    assert span_edges[-1] == data.shape[0]

    span_slices = [slice(span_edges[i], span_edges[i + 1])
                   for i in xrange(len(span_edges) - 1)]

    dataset = Dataset(data_format)
    for span_slice in span_slices:
        dataset.add_recspan(data[span_slice, :], metadata)

    span_starts = [s.start for s in span_slices]
    recspan_ids = []
    start_ticks = []
    for tick in raw_log_events.index:
        recspan_id = bisect.bisect(span_starts, tick) - 1
        span_slice = span_slices[recspan_id]
        span_start = span_slice.start
        span_stop = span_slice.stop
        assert span_start <= tick < span_stop
        recspan_ids.append(recspan_id)
        start_ticks.append(tick - span_start)
    stop_ticks = [tick + 1 for tick in start_ticks]
    dataset.add_events(recspan_ids, start_ticks, stop_ticks,
                       raw_log_events)

    for delete_event in dataset.events_query({"code": DELETE_CODE}):
        delete_event.recspan_info["deleted"] = True

    for cal_event in dataset.events_query(calibration_events):
        for key in list(cal_event):
            del cal_event[key]
        cal_event["calibration_pulse"] = True

    if calibrate:
        for kwarg in ["calibrate_low_cursor_time",
                      "calibrate_high_cursor_time",
                      "calibrate_pulse_size"]:
            if locals()[kwarg] is None:
                raise ValueError("when calibrating, %s= argument must be "
                                 "specified" % (kwarg,))
        half_width = dataset.data_format.ticks_to_ms(calibrate_half_width_ticks)
        cal_vals = {}
        for which, cursor_time in [("low", calibrate_low_cursor_time),
                                   ("high", calibrate_high_cursor_time)]:
            # Round cursor to nearest tick
            cursor_tick = dataset.data_format.ms_to_ticks(cursor_time)
            cursor_time = dataset.data_format.ticks_to_ms(cursor_tick)
            erp = dataset.rerp("calibration_pulse",
                               cursor_time - half_width,
                               cursor_time + half_width,
                               "1",
                               all_or_nothing=True,
                               overlap_correction=False,
                               verbose=False)
            cal_vals[which] = erp.betas["Intercept"].mean()
        cal_diffs = cal_vals["high"] - cal_vals["low"]
        calibrate_pulse_size *= calibrate_polarity
        # For each channel, we want to multiply by a factor with units uV/raw
        # We have calibrate_pulse_size uV = cal_diffs raw
        cal_scaler = calibrate_pulse_size / cal_diffs
        dataset.transform(np.diagflat(cal_scaler))

    return dataset

def test_load_erpss():
    from rerpy.test import test_data_path
    # This crw/log file is constructed to have a few features:
    # - it only has 3 records, so it's tiny
    # - the first two records are in one recspan, the last is in a second, so
    #   we test the recspan splitting code
    # - the first recspan ends in a PAUSE event, the second ends in a DELETE
    #   event, so we test the deleted event handling.
    # There are some weird things about it too:
    # - several events in the first recspan have condition 0, to test
    #   calibration pulse stuff. In a normal ERPSS file all events within a
    #   single recspan would have the same condition number.
    # - most of the event codes are >32767. In a normal ERPSS file such events
    #   are supposed to be reserved for special stuff and deleted events, but
    #   it happens the file I was using as a basis violated this rule. Oh
    #   well.
    dataset = load_erpss(test_data_path("erpss/tiny-complete.crw"),
                          test_data_path("erpss/tiny-complete.log"))
    assert len(dataset) == 2
    assert dataset[0].shape == (512, 32)
    assert dataset[1].shape == (256, 32)

    assert dataset.data_format.exact_sample_rate_hz == 250
    assert dataset.data_format.units == "RAW"
    assert list(dataset.data_format.channel_names) == [
        "lle", "lhz", "MiPf", "LLPf", "RLPf", "LMPf", "RMPf", "LDFr", "RDFr",
        "LLFr", "RLFr", "LMFr", "RMFr", "LMCe", "RMCe", "MiCe", "MiPa", "LDCe",
        "RDCe", "LDPa", "RDPa", "LMOc", "RMOc", "LLTe", "RLTe", "LLOc", "RLOc",
        "MiOc", "A2", "HEOG", "rle", "rhz",
        ]

    for recspan_info in dataset.recspan_infos:
        assert recspan_info["raw_file"].endswith("tiny-complete.crw")
        assert recspan_info["log_file"].endswith("tiny-complete.log")
        assert recspan_info["experiment"] == "brown-1"
        assert recspan_info["subject"] == "Subject p3 2008-08-20"
        assert recspan_info["odelay"] == 8
        assert len(recspan_info["erpss_raw_header"]) == 512

    assert dataset.recspan_infos[0].ticks == 512
    assert dataset.recspan_infos[1].ticks == 256
    assert dataset.recspan_infos[1]["deleted"]

    assert len(dataset.events()) == 14
    # 2 are calibration events
    assert len(dataset.events("has code")) == 12
    for ev in dataset.events("has code"):
        assert ev["condition"] in (64, 65)
        assert ev["flag"] == 0
        assert not ev["flag_data_error"]
        assert not ev["flag_polinv"]
        assert not ev["flag_rejected"]
    for ev in dataset.events("calibration_pulse"):
        assert dict(ev) == {"calibration_pulse": True}
    def check_ticks(query, recspan_ids, start_ticks):
        events = dataset.events(query)
        assert len(events) == len(recspan_ids) == len(start_ticks)
        for ev, recspan_id, start_tick in zip(events, recspan_ids, start_ticks):
            assert ev.recspan_id == recspan_id
            assert ev.start_tick == start_tick
            assert ev.stop_tick == start_tick + 1

    check_ticks("condition == 64",
                [0] * 8, [21, 221, 304, 329, 379, 458, 483, 511])
    check_ticks("condition == 65",
                [1] * 4,
                [533 - 512, 733 - 512, 762 - 512, 767 - 512])
    check_ticks("calibration_pulse", [0, 0], [250, 408])

    # check calibration_events option
    dataset2 = load_erpss(test_data_path("erpss/tiny-complete.crw"),
                          test_data_path("erpss/tiny-complete.log"),
                          calibration_events="condition == 65")
    assert len(dataset2.events("condition == 65")) == 0
    assert len(dataset2.events("condition == 0")) == 2
    assert len(dataset2.events("calibration_pulse")) == 4

    # check calibration
    # idea: if calibration works, then the "calibration erp" will have been
    # set to be the same size as whatever we told it to be.
    dataset_cal = load_erpss(test_data_path("erpss/tiny-complete.crw"),
                             test_data_path("erpss/tiny-complete.log"),
                             calibration_events="condition == 65",
                             calibrate=True,
                             calibrate_half_width_ticks=2,
                             calibrate_low_cursor_time=-16,
                             calibrate_high_cursor_time=21,
                             calibrate_pulse_size=12.34,
                             calibrate_polarity=-1)
    assert dataset_cal.data_format.units == "uV"
    # -16 ms +/-2 ticks = -24 to -8 ms
    low_cal = dataset_cal.rerp("calibration_pulse", -24, -8, "1",
                               all_or_nothing=True,
                               overlap_correction=False)
    # 21 ms rounds to 20 ms, +/-2 ticks for the window = 12 to 28 ms
    high_cal = dataset_cal.rerp("calibration_pulse", 12, 28, "1",
                                all_or_nothing=True,
                                overlap_correction=False)
    low = low_cal.betas["Intercept"].mean(axis=0)
    high = high_cal.betas["Intercept"].mean(axis=0)
    assert np.allclose(high - low, -1 * 12.34)

    # check that we can load from file handles (not sure if anyone cares but
    # hey you never know...)
    assert len(load_erpss(open(test_data_path("erpss/tiny-complete.crw")),
                          open(test_data_path("erpss/tiny-complete.log")))) == 2

    # check that code/raw mismatch is detected
    from nose.tools import assert_raises
    for bad in ["bad-code", "bad-tick", "bad-tick2"]:
        assert_raises(ValueError,
                      load_erpss,
                      test_data_path("erpss/tiny-complete.crw"),
                      test_data_path("erpss/tiny-complete.%s.log" % (bad,)))
    # But if the only mismatch is an event that is "deleted" (sign bit set) in
    # the log file, but not in the raw file, then that is okay:
    load_erpss(test_data_path("erpss/tiny-complete.crw"),
               test_data_path("erpss/tiny-complete.code-deleted.log"))
