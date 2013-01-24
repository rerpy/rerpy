# This file is part of pyrerp
# Copyright (C) 2012 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

from patsy import dmatrix

def epoched_regress(epoched_eeg, formula_like):
    
