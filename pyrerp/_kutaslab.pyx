import numpy as np
cimport numpy as np

# Each 16-bit integer contains 4 nibbles, where the most-significant-nibble
# comes 'first', and the least-significant nibble comes 'last'.
cdef unsigned char _nibble_at(np.uint16_t * data, int i):
    cdef int word_offset = i // 4
    cdef int in_word_offset = i % 4
    return (data[word_offset] >> ((3 - in_word_offset) * 4)) & 0x0f
        
def test__nibble_at():
    cdef np.ndarray[np.uint16_t] data = np.array([0x3412, 0x659f],
                                                 dtype=np.uint16)
    for i, nibble in enumerate([0x3, 0x4, 0x1, 0x2, 0x6, 0x5, 0x9, 0xf]):
        got = _nibble_at(<np.uint16_t *> data.data, i)
        print "%s: wanted %s, got %s" % (i, nibble, got)
        assert nibble == got

def _decompress_crw_chunk(compressed_data, int ncompressed_words,
                          int nchans,
                          int chunk_samples=256):
    # On big-endian systems, this copies the data, but not on little-endian
    # systems:
    cdef np.ndarray[np.uint16_t, ndim=1] cd_array
    cd_array = np.asarray(np.fromstring(compressed_data, dtype="<u2"),
                          dtype=np.uint16)
    # For some reason Cython doesn't want me passing a type ndarray directly
    # to the cdef function above, so oh well, have a pointer instead...:
    cdef np.uint16_t * cd = <np.uint16_t *> cd_array.data

    cdef np.ndarray[np.int16_t, ndim=1] data_chunk
    data_chunk = np.empty(nchans * chunk_samples, dtype=np.int16)

    cdef int out_i, nibble_i
    cdef unsigned char nibble
    cdef int delta, bits
    nibble_i = 0
    for out_i in xrange(nchans * chunk_samples):
        assert nibble_i < 4 * ncompressed_words
        nibble = _nibble_at(cd, nibble_i)
        # Deltas are encoded (vaguely UTF-8 (or UTF-4?) style) using one of
        # these templates (1 character = 1 bit):
        #   0???
        #   10?? ????
        #   110? ???? ????
        #   1110 ???? ???? ????
        # And then you sign-extend the ?-bits, and add to the value extracted
        # nchans back.
        # EXCEPT: if using the last form, the 12-bit form, that's wide enough
        # to contain the whole value... so it's *not* a delta, it's just the
        # raw value.
        if nibble & 0x8 == 0:
            # 0???
            delta = nibble & 0x7
            bits = 3
        elif nibble & 0x4 == 0:
            # 10?? ????
            delta = (((nibble & 0x3) << 4)
                     | _nibble_at(cd, nibble_i + 1))
            bits = 6
        elif nibble & 0x2 == 0:
            # 110? ???? ????
            delta = (((nibble & 0x1) << 8)
                     | _nibble_at(cd, nibble_i + 1) << 4
                     | _nibble_at(cd, nibble_i + 2))
            bits = 9
        elif nibble & 0x1 == 0:
            # 1110 ???? ???? ????
            delta = (_nibble_at(cd, nibble_i + 1) << 8
                     | _nibble_at(cd, nibble_i + 2) << 4
                     | _nibble_at(cd, nibble_i + 3))
            bits = 12
        else:
            assert False, "Bad leading nibble: %s" % hex(nibble)
        nibble_i += bits // 3
        # Sign-extend (i.e., if the high bit is set, then set all the higher
        # bits too:
        if delta & (1 << (bits - 1)):
            delta |= (-1 << bits)
        # Then apply the delta:
        if bits == 12:
            # Not a delta:
            data_chunk[out_i] = delta
        else:
            data_chunk[out_i] = data_chunk[out_i - nchans] + delta
    return data_chunk

def test__decompress_crw_chunk():
    import struct
    # Smoke test:
    test_vector = struct.pack("<" + "H" * 32,
                              *[x | 0xe000 for x in xrange(32)])
    print _decompress_crw_chunk(test_vector, 32, 32, chunk_samples=1)
    assert (_decompress_crw_chunk(test_vector, 32, 32, chunk_samples=1)
            == np.arange(32)).all()
    # Check all the different encoding lengths:
    test_12 = 0xe000 | 4
    test_9 =  0x0c00 | 3
    test_6 =  0x0080 | 2
    test_4 =  0x0000 | 1
    test_w1 = (test_12) & 0xffff
    test_w2 = (test_9 << 4 | test_6 >> 4) & 0xffff
    test_w3 = (test_6 << 12 | test_4 << 8) & 0xffff
    test_vector = struct.pack("<HHH", test_w1, test_w2, test_w3)
    print "vector:",  [ hex(ord(c)) for c in test_vector]
    output = _decompress_crw_chunk(test_vector, 3, 1, chunk_samples=4)
    assert (output == [4, 7, 9, 10]).all()
    # Check negative numbers:
    test_12 = 0xe000 | (-4 & 0x0fff)
    print "test_12:", hex(test_12)
    test_9 =  0x0c00 | (-3 & 0x01ff)
    print "test_9:", hex(test_9)
    test_6 =  0x0080 | (-2 & 0x003f)
    print "test_6:", hex(test_6)
    test_4 =  0x0000 | (-1 & 0x0007)
    print "test_4:", hex(test_4)
    test_w1 = (test_12) & 0xffff
    test_w2 = (test_9 << 4 | test_6 >> 4) & 0xffff
    test_w3 = (test_6 << 12 | test_4 << 8) & 0xffff
    test_vector = struct.pack("<HHH", test_w1, test_w2, test_w3)
    print "vector:", [hex(ord(c)) for c in test_vector]
    output = _decompress_crw_chunk(test_vector, 3, 1, chunk_samples=4)
    assert (output == np.array([-4, -7, -9, -10])).all()
    # Check that 12-bit entries are not considered deltas, but raw values:
    test_vector = struct.pack("<HH",
                              0xe000 | (-1 & 0x0fff),
                              0xe000 | (-1 & 0x0fff))
    assert (_decompress_crw_chunk(test_vector, 2, 1, chunk_samples=2) == [-1, -1]).all()

