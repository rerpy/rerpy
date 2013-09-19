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
            raise ValueError("erpss avg files must contain exactly "
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
        # The erpss tools only do integer-factor downsampling (decimation),
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

        # We don't write out odelay -- you're expected to have dealt with that
        # already via .move_event() etc.

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
