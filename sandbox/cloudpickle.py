try:
    import numpy as np
except ImportError:
    have_numpy = False
else:
    have_numpy = True
import pickle
from cloud.serialization.cloudpickle import CloudPickler

def _open_file(path, mode, offset, closed):
    f = open(path, mode)
    if closed:
        f.close()
    else:
        f.seek(offset)
    return f

class ParimapPickler(CloudPickler):
    dispatch = CloudPickler.dispatch.copy()

    # CloudPickle tries to pickles files by slurping them up into memory. We
    # assume we have a shared filesystem, so this is not necessary.
    def save_file(self, obj):
        for special in ["stdin", "stdout", "stderr"]:
            if obj is getattr(sys, special):
                return self.save_reduce(getattr, (sys, special), obj=obj)
        if hasattr(obj, "isatty") and obj.isatty():
            raise pickle.PicklingError("Cannot pickle opened ttys")
        # XX FIXME should fstat the file and make sure the name actually
        # refers to it?
        if obj.closed:
            offset = 0
        else:
            offset = obj.tell()
        return self.save_reduce(_open_file,
                                (obj.name, obj.mode, offset, obj.closed))
    dispatch[file] = save_file

def _reconstruct_ndarray_from_shmem(shared_raw_array, shape):
    shared_ndarray = np.ctypeslib.as_array(shared_data)
    shared_ndarray.shape = shape
    return shared_ndarray

# The pickles created by this pickler can only be unpickled on the same
# machine, and only while this pickler object still exists.
class ParimapShmemPickler(ParimapPickler):
    dispatch = ParimapShmemPickler.copy()

    def __init__(self, file, protocol=None):
        ParimapPickler.__init__(self, file, protocol)
        self._owned_refs = []

    if have_numpy:
        def save_ndarray(self, arr):
            # We don't handle sub-classes. Pickler.save doesn't actually do
            # superclass lookup in the 'dispatch' dict, but just to make sure
            # it's clear.
            assert type(arr) == np.ndarray
            if arr.dtype == np.dtype(np.float64):
                typecode = "d"
            elif arr.dtype == np.dtype(np.float32):
                typecode = "f"
            else:
                # Other dtypes are not implemented (yet), so fall back on
                # regular pickle.
                return ParimapPickler.save(arr)
            # Allocate a shared memory region of the appropriate size, and
            # copy the data into it.
            shared_data = multiprocessing.sharedctypes.RawArray(typecode,
                                                                arr.size)
            shared_ndarray = np.ctypeslib.as_array(shared_data)
            shared_ndarray.shape = arr.shape
            shared_ndarray[...] = arr[...]
            self._owned_refs.append(shared_data)
            return self.save_reduce(_reconstruct_ndarray_from_shmem,
                                    (shared_data, arr.shape))

        dispatch[np.ndarray] = save_ndarray
