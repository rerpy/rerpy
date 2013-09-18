# This file is part of pyrerp
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

import numpy as np
import csv


# For ICA: given a transformation matrix in *channel* space + a ChannelInfo,
# need some way to convert it to a new ChannelInfo.

class Ref(object):
    def __init__(self, main, reference):
        if not isinstance(main, str):
            raise ValueError("main must be a string")
        if not isinstance(reference, str):
            raise ValueError("reference must be a string")
        self.main = main
        self.reference = reference

    def name(self):
        return self.main
