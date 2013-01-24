# This file is part of pyrerp
# Copyright (C) 2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

import os.path
import numpy as np
from scipy.io import loadmat
import pandas

from pyrerp.data import ElectrodeInfo, RecordingInfo, ContinuousData
from pyrerp.events import Events

__all__ = ["EEGLABError", "load_eeglab"]

class EEGLABError(Exception):
    pass

# Format documentation:
#   http://sccn.ucsd.edu/wiki/A05:_Data_Structures
#   https://sccn.ucsd.edu/svn/software/eeglab/functions/adminfunc/eeg_checkset.m
#   https://sccn.ucsd.edu/svn/software/eeglab/functions/popfunc/pop_loadset.m
# Events: boundaries, durations, epochs, etc.:
#   http://sccn.ucsd.edu/wiki/Chapter_03:_Event_Processing
#
# pop_loadset does mat-file reading, sanity checking, and data pathname
#   manipulation
# eeg_checkset(EEG, 'loaddata') is used to actually load the data
#   which calls eeg_getdatact(EEG)

def chanlocs_array(EEG, key):
    return [entry.squeeze()[()]
            for entry in EEG["chanlocs"][key].squeeze()]

def extract_electrode_info(EEG):    
    # str() ensures that we get ascii on python2, unicode on python3
    channel_names = [str(label) for label in chanlocs_array(EEG, "labels")]
    thetas = np.asarray(chanlocs_array(EEG, "theta"), dtype=float)
    rs = np.asarray(chanlocs_array(EEG, "radius"), dtype=float)
    electrodes = ElectrodeInfo(channel_names, thetas, rs)

def extract_data_matrix(EEG, set_filename, dtype):
    # Right, so, EEGLAB. It's... fun.
    # 
    # .data is an attribute which can have different values:
    #   -- "EEGDATA" (case insensitive): means that the data's in the .EEGDATA
    #      attribute
    #   -- any other string: means that the data is in a file whose name is
    #      exactly like the name of *the .set file*, except that the last few
    #      characters of the .data attribute (either .fdt or .dat, case
    #      insensitive again) determine the order the data is stored in.
    #   -- or it can just contain the actual data
    # 
    # EEGLAB's data loading routines represent the data as a matrix with
    # dimensions
    #   [nbchan, pnts * trials]
    # which really means
    #   [nbchan, pnts, trials]
    # (where "trial" means "epoch", and trials==1 means continuous).
    #
    # Now, there are two things that make handling the data layout
    # complicated. First, EEGLAB sort of makes a mess of it. MATLAB uses
    # "Fortran" (column-major) ordering by default, and EEGLAB follows this
    # convention. So when going from
    #   [nbchan, pnts * trials]
    # to
    #   [nbchan, pnts, trials]
    # we have to remember that we're using fortran order to divide up the
    # second component. When it comes to actually storing the files, EEGLAB
    # uses two conventions:
    #   FDT files:
    #     [nbchan, pnts * trials] stored in memory (fortran) order
    #   DAT files:
    #     [nbchan, pnts * trials] stored in transposed (C) order
    # (But remember that even in DAT files, the pnts * trials part is in
    # fortran order, i.e., this files have *mixed* order when considered as a
    # 3-d [nbchan, pnts, trials] array.)
    #
    # The second complication is that numpy by default uses the opposite
    # convention for storing arrays -- the default is "C" (row-major)
    # ordering. Fortunately though it is very flexible and can handle
    # arbitrary orderings, even mixed ones; we just have to be careful to
    # specify what we're talking about when we set things up in the first
    # place.
    #
    # This function always returns an array with shape
    #   (nbchan, pnts, trials)
    # If you request a little-endian float32 dtype, then the returned value
    # will be a view onto a read-only memory mapping of the file. Otherwise it
    # will be loaded into memory and converted to the requested dtype.
    nbchan = EEG["nbchan"].item()
    pnts = EEG["pnts"].item()
    trials = EEG["trials"].item()

    if np.issubdtype(EEG["data"], np.character):
        data_str = str(EEG["data"].item())
        if data_str.lower() == "eegdata":
            data = EEG["EEGDATA"]
        else:
            base_path, _ = os.path.splitext(set_filename)
            _, ext = os.path.splitext(data_str)
            data_path = base_path + ext
            if ext.lower() == ".dat":
                order = "C"
            else:
                order = "F"
            data = np.memmap(data_path, "r", dtype="<f4", order=order,
                             shape=(nbchan, pnts * trials))
    else:
        data = EEG["data"]

    # Now 'data' is an array with unknown dtype and shape
    #   (nbchan, pnts * trials)
    # We now want to reshape this to an array with shape
    #   (nbchan, pnts, trials)
    # *using Fortran rules* to determine the relationship between these array
    # layouts, *regardless* of the actual underlying memory layout.
    # Fortunately numpy makes this easy.
    data.resize((nbchan, pnts, trials), order="F")

    # And finally, require that the data be of the specified type. This passes
    # through compatible arrays unchanged, or otherwise creates a new array of
    # the specified type and casts the data into it:
    data = np.asarray(data, dtype=dtype)

    return data
    
def load_eeglab(set_filename, dtype=np.float64):
    # Read the .mat file
    contents = loadmat(set_filename)
    if "EEG" not in contents:
        if "ALLEEG" in contents:
            raise EEGLABError("reading of multi-set files is not implemented "
                              " -- patches gratefully accepted")
        else:
            raise EEGLABError("no 'EEG' variable found in matlab file")
    EEG = contents["EEG"][0, 0]

    srate = EEG["srate"][0, 0]
    units = "??"
    electrodes = extract_electrode_info(EEG)
    # General metadata:
    metadata = {}
    for key in ["setname", "filename", "filepath", "comments", "etc",
                "subject", "group", "condition", "session", "ref",
                "icasphere", "icaweights", "icawinv"]:
        if key in EEG.dtype.names and EEG[key].size > 0:
            if np.issubdtype(EEG[key], np.character_):
                metadata[key] = EEG[key][0]
            else:
                metadata[key] = EEG[key]
    recording_info = RecordingInfo(srate, units, electrodes, metadata)

    data = extract_data_matrix(EEG, set_filename, dtype)
    
    data = data.T
    (num_epochs, num_channels, num_samples) = data.shape
    if num_epochs != 1:
        raise EEGLABError("reading of epoched data is not implemented "
                          " -- patches gratefully accepted")
    assert num_epochs == 1
    data.resize(data.shape[1:])

    # Events
    # type (string) and latency (int) are the main ones
    # type == "boundary" is special
    #   duration is length of removed data if data was removed (generally 0,
    #   or NaN for breaks between concatenated datasets)
    #   usually only for type == "boundary"
    # "latency" is a floating point number, which refers to a 1-based index in
    # the data array. It's floating point because for some events, like
    # "boundary" events, they actually place the event *in between* two frames
    # -- so if you have two data sets
    #   [10, 11, 12]
    # and
    #   [17, 18, 19]
    # then the concatenated set is
    #   [10, 11, 12, 17, 18, 19]
    # with the 'latency' of the boundary event set to 3.5.
    zero_based_boundaries = []
    for eeglab_event in EEG["event"].ravel():
        if eeglab_event["type"].item() == "boundary":
            zero_based_boundaries.append(eeglab_event["latency"] - 1)
    
    name_index = [metadata["setname"]] * num_samples
    era_index = np.empty(num_samples, dtype=int)
    time_index = np.empty(num_samples, dtype=
    ev = Events((str, int, int))

    if num_epochs == 1:
        # Continuous data
    else:
        # Epoched data
        # EEG.xmin, EEG.xmax = start/end latency of epoch in seconds
        # EEG.epoch == ?


