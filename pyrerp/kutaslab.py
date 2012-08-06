import struct
import numpy as np
from cStringIO import StringIO
import gzip
import os
import string

import pandas

from pyrerp.events import Events
from pyrerp._kutaslab import _decompress_crw_chunk

PAUSE_CODE = 49152

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

_char2code = {}
for i, char in enumerate(string.lowercase):
    _char2code[char] = i + 1
for i, char in enumerate(string.uppercase):
    _char2code[char] = i + 27
for i, char in enumerate(string.digits):
    _char2code[char] = i + 53
_code2char = dict([(v, k) for (k, v) in _char2code.iteritems()])

def _read_header(magic, stream):
    header_str = magic + stream.read(512 - 2)
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
        raise KutaslabError, "file claims weird non-integer sample rate %shz" % hz
    hz = int(hz)

    if header["nchans"] <= 16:
        # For small montages, each channel gets 8 bytes of ascii, smushed
        # together into a single array:
        channel_names = np.fromstring(_get_full_string(header, "chndes"),
                                      dtype="S8")
    elif header["nchans"] <= 32:
        # For mid-size montages, each channel gets 4 bytes:
        channel_names = np.fromstring(_get_full_string(header, "chndes"),
                                      dtype="S4")
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
        channel_names = np.array(channel_names_l)

    # Also read out the various free-form informational strings:
    info = {}
    info["subject"] = header["subdes"]
    info["experiment"] = header["expdes"]
    info["raw_header"] = header

    return (reader, header["nchans"], hz, channel_names, info, header)

# To read multiple bins, call this repeatedly on the same stream
def read_avg(stream, dtype):
    magic = stream.read(2)
    # Ignore the 'reader' field -- avg files always have the same magic number
    # as whatever type of raw file was used to create them -- and they're
    # always in their own format, regardless.
    (_, nchans, hz, channel_names, info, header) = _read_header(magic, stream)
    assert header["cprecis"] > 0
    data_chunks = []
    for i in xrange(header["cprecis"]):
        data_bytes = stream.read(512 * nchans)
        data_chunk = np.fromstring(data_bytes, dtype="<i2")
        data_chunk.resize((256, nchans))
        data_chunks.append(np.asarray(data_chunk, dtype=dtype))
    return np.vstack(data_chunks), hz, channel_names, info, header

def read_raw(stream, dtype):
    magic = stream.read(2)
    if magic == "\037\213":
        stream.seek(-2, os.SEEK_CUR)
        stream = gzip.GzipFile(mode="r", fileobj=stream)
        magic = stream.read(2)
    (reader, nchans, hz, channel_names, info, header) = _read_header(magic, stream)
    # Data is stored in a series of "chunks" -- each chunk contains 256 s16
    # samples from each channel (the 32/64/whatever analog channels, plus 1
    # channel for codes -- that channel being first.).  The code channel
    # contains a "record number" as its first entry in each chunk, which
    # simply increments by 1 each time.
    all_codes = []
    data_chunks = []
    chunk_bytes = (nchans + 1) * 512
    chunkno = 0
    while True:
        read = reader(stream, nchans)
        if read is None:
            break
        (codes_chunk, data_chunk) = read
        assert len(codes_chunk) == 256
        assert data_chunk.shape == (256 * nchans,)
        assert codes_chunk[0] == chunkno
        codes_chunk[0] = 65535
        all_codes += codes_chunk
        data_chunk.resize((256, nchans))
        data_chunks.append(np.asarray(data_chunk, dtype=dtype))
        chunkno += 1
    final_data = np.vstack(data_chunks)
    return (hz, channel_names,
            np.array(all_codes, dtype=np.uint16), final_data,
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
        if k != "raw_header":
            assert info1[k] == info2[k]

import os.path
_test_dir = os.path.join(os.path.dirname(__file__), "test-data")

def test_read_raw_on_test_data():
    import os.path, glob
    tested = 0
    for rawp in glob.glob(os.path.join(_test_dir, "*.raw")):
        crwp = rawp[:-3] + "crw"
        print rawp, crwp
        assert_files_match(rawp, crwp)
        if os.path.exists(rawp + ".gz"):
            print rawp, rawp + ".gz"
            assert_files_match(rawp, rawp + ".gz")
        tested += 1
    # Cross-check, to make sure is actually finding the files... (bump up this
    # number if you add more test files):
    assert tested == 4

def test_64bit_channel_names():
    stream = open(os.path.join(_test_dir, "two-chunks-64chan.raw"))
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

def read_log(fo):
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
    return pandas.DataFrame(events, columns=["code", "condition", "flag"],
                            index=ticks)

def load_kutaslab(f_raw, f_log, name=None, dtype=np.float64):
    (hz, channel_names, raw_codes, data, info) = read_raw(f_raw, dtype)
    raw_log_events = read_log(f_log)
    expanded_log_codes = np.zeros(raw_codes.shape, dtype=int)
    expanded_log_codes[raw_log_events.index] = raw_log_events["code"]
    discrepancies = (expanded_log_codes != raw_codes)
    if (not (expanded_log_codes[discrepancies] == 0).all()
        or not (raw_codes[discrepancies] == 65535).all()):
        raise KutaslabError, "raw and log files have mismatched codes"
    del raw_codes
    del expanded_log_codes
    sample_period_ms = int(1. / hz * 1000)
    if (1000. / sample_period_ms) != hz:
        raise KutaslabError, "sampling period in milliseconds is not an integer"
    pause_ticks = raw_log_events.index[raw_log_events["code"] == PAUSE_CODE]
    # I *think* the pause code appears at the last sample of the era, rather
    # than the first sample of the new era. Convert them to the Python-style
    # [start, end) convention:
    pause_ticks += 1
    name_index = [name] * data.shape[0]
    era_index = np.empty(data.shape[0], dtype=np.int16)
    time_index = np.empty(data.shape[0], dtype=np.int32)
    era_starts = np.concatenate(([0], pause_ticks))
    era_ends = np.concatenate((pause_ticks, [data.shape[0]]))
    for i, (era_start, era_end) in enumerate(zip(era_starts, era_ends)):
        era_index[era_start:era_end] = i
        times = np.arange(era_end - era_start) * sample_period_ms
        time_index[era_start:era_end] = times
    index_arrays = [name_index, era_index, time_index]
    index_names = ["name", "era", "time"]
    data_index = pandas.MultiIndex.from_arrays(index_arrays,
                                               names=index_names)
    ev = Events((str, int, int))
    for i in xrange(raw_log_events.shape[0]):
        tick = raw_log_events.index[i]
        ev.add_event(data_index[tick],
                     dict(zip(raw_log_events.columns,
                              raw_log_events.xs(tick))))
    return (pandas.DataFrame(data,
                             columns=channel_names,
                             index=data_index),
            ev)

def load(f_raw, f_log, dtype=np.float64,
         delete_channels=[], calibrate=True, **kwargs):
    (hz, channel_names, raw_codes, data, info) = read_raw(f_raw, dtype)
    codes_from_log = np.zeros(raw_codes.shape, dtype=raw_codes.dtype)
    for (code, tick, condition, flag) in read_log(f_log):
        codes_from_log[tick] = code
    discrepancies = (codes_from_log != raw_codes)
    assert (codes_from_log[discrepancies] == 0).all()
    assert (raw_codes[discrepancies] == 65535).all()
    if delete_channels: # fast-path: no need to do a copy if nothing to delete
        keep_channels = []
        for i in xrange(len(channel_names)):
            if channel_names[i] not in delete_channels:
                keep_channels.append(i)
        assert len(keep_channels) + len(delete_channels) == len(channel_names)
        data = data[:, keep_channels]
        channel_names = channel_names[keep_channels]
    if calibrate:
        calibrate_in_place(data, raw_codes, **kwargs)
    return (hz, channel_names, raw_codes, data, info)

# To write multiple "bins" to the same file, just call this function
# repeatedly on the same stream.
def write_erp_as_avg(erp, stream):
    header = np.zeros(1, dtype=_header_dtype)[0]
    header["magic"] = 0x97a5
    header["verpos"] = 1
    # One avg record is always exactly 256 * cprecis samples long, with
    # cprecis = 1, 2, 3 (limitation of the data format).  So we pick the
    # smallest cprecis that is <= our actual number of samples (maximum 3),
    # and then we resample to have that many samples exactly.  (I.e., we try
    # to resample up when possible.)  The kutaslab tools only do
    # integer-factor downsampling (decimation), and they write the decimation
    # factor to the file.  I don't see how it matters for the .avg file to
    # retain the decimation information, and the file won't let us write down
    # upsampling (especially non-integer upsampling!), so we just set our
    # decimation factor to 1 and be done with it.
    if erp.data.shape[0] <= 1 * 256:
        cprecis = 1
    elif erp.data.shape[0] <= 2 * 256:
        cprecis = 2
    else:
        cprecis = 3
    samples = cprecis * 256
    if erp.data.shape[0] != samples:
        import scipy.signal
        resampled_data = scipy.signal.resample(erp.data, samples)
    else:
        resampled_data = erp.data
    assert resampled_data.shape == (samples, erp.data.shape[1])
    resampled_sp_10us = int(round((erp.times.max() - erp.times.min())
                                  * 100. / samples))
    epoch_len_ms = int(round(samples * resampled_sp_10us / 100.))

    # Need to convert to s16's. To preserve as much resolution as possible,
    # we use the full s16 range, minus a bit to make sure we don't run into
    # any overflow issues.
    s16_max = 2 ** 15 - 10
    # Same as np.abs(data).max(), but without copying the whole array:
    data_max = max(resampled_data.max(), np.abs(resampled_data.min()))
    # We have to write the conversion factor as an integer, so we round it
    # down here, and then use the *rounded* version to actually convert the
    # data.
    s16_per_10uV = int(s16_max / (data_max / 10))
    # Except that if our conversion factor itself overflows, then we have to
    # truncate it back down (and lose a bit of resolution in the process, oh
    # well):
    if s16_per_10uV > s16_max:
        s16_per_10uV = s16_max
    integer_data = np.array(np.round(s16_per_10uV * resampled_data / 10.),
                            dtype="<i2")

    header["epoch_len"] = epoch_len_ms
    header["nchans"] = integer_data.shape[1]
    header["sums"] = 0 # ERP
    header["tpfuncs"] = 1 # processing function of "averaging"
    header["pftypes"] = "average"
    header["pp10uv"] = s16_per_10uV
    header["10usec_per_tick"] = resampled_sp_10us
    header["presam"] = 0 - erp.times.min()
    header["cprecis"] = cprecis
    header["decfact"] = 1
    
    if "num_combined_trials" in erp.metadata:
        header["totrr"] = erp.metadata["num_combined_trials"]

    if len(erp.channel_names) <= 16:
        header["chndes"] = np.asarray(erp.channel_names, dtype="S8").tostring()
    elif len(erp.channel_names) <= 32:
        header["chndes"] = np.asarray(erp.channel_names, dtype="S4").tostring()
    else:
        assert False, "Channel name writing for large montages not yet supported"
    if "experiment" in erp.metadata:
        header["expdes"] = erp.metadata["experiment"]
    if "subject" in erp.metadata:
        header["subdes"] = erp.metadata["subject"]
    if erp.name is not None:
        header["condes"] = erp.name

    header.tofile(stream)
    # avg files omit the mark track.  And, all the data for a single channel
    # goes together in a single chunk, rather than interleaving all channels.
    # THIS IS DIFFERENT FROM RAW FILES!
    for i in xrange(integer_data.shape[1]):
        integer_data[:, i].tofile(stream)

# From looking at:
#   plot(data[:,
#             np.add.outer(np.arange(-100, 100),
#                          (marks == 1).nonzero()[0]).squeeze()].mean(axis=2).transpose())
# It looks like the high part of the cal is about code_idx+15:code_idx+45, and
# the low part is -60:-10

class BadChannels(Exception):
    pass

def calibrate_in_place(data, codes,
                       before=np.arange(-40, -10), after=np.arange(15, 45),
                       pulse_size=10, # true pulse size measured in uV
                       stddev_limit=3,
                       cal_code=1):
    assert len(before) == len(after)
    cal_codes = (codes == cal_code).nonzero()[0]
    # Trim off the first and last few cals, because there may be truncation
    # effects:
    cal_codes = cal_codes[2:-2]
    before_cals = data[np.add.outer(cal_codes, before).ravel(), :]
    after_cals = data[np.add.outer(cal_codes, after).ravel(), :]
    deltas = before_cals - after_cals
    bad_channels = (deltas.std(axis=0) > stddev_limit)
    if bad_channels.any():
        raise BadChannels, bad_channels.nonzero()[0]
    delta = deltas.mean(axis=0)
    assert (delta > 0).all() or (delta < 0).all()
    delta = np.abs(delta)
    data *= pulse_size / delta

if __name__ == "__main__":
    import nose
    nose.runmodule()
