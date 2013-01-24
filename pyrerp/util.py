# This file is part of pyrerp
# Copyright (C) 2012 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

import pandas

def maybe_open(file_like, mode="rb"):
    # FIXME: have more formal checking of binary/text, given how important
    # that is in py3?
    if isinstance(file_like, basestring):
        return open(file_like, mode)
    else:
        return file_like

# For working on homogenous data frames in place. These are sometimes used
# over pickle, hence the version number.
def unpack_pandas(obj):
    if isinstance(obj, pandas.Series):
        info = {"type": "Series",
                "name": obj.name, "index": obj.index}
    elif isinstance(obj, pandas.DataFrame):
        info = {"type": "DataFrame",
                "index": obj.index, "columns": obj.columns}
    elif isinstance(obj, pandas.PanelData):
        info = {"type": "PanelData",
                "items": obj.items, "major_axis": obj.major_axis,
                "minor_axis": obj.minor_axis,
                }
    else:
        raise ValueError, "don't recognize %s object" % (obj.__class__)
    return (np.asarray(df), (0, info))

def pack_pandas(array, metadata):
    version, info = metadata
    if version != 0:
        raise ValueError, "unrecognized dataframe metadata version"
    obj_type = info.pop("type")
    if obj_type == "Series":
        return pandas.Series(array, **info)
    elif obj_type == "DataFrame":
        return pandas.DataFrame(array, **info)
    elif obj_type == "PanelData":
        return pandas.PanelData(array, **info)
    else:
        raise ValueError, "invalid pandas metadata"

