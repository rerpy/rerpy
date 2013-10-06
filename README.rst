rERPy is a Python toolkit for doing ERP analysis of brainwave data
(EEG, MEG), using both traditional averaging-based ERP estimation, and
a fancy new regression-based technique for ERP estimation, which we
call rERP for short. rERPs can do anything ERPs can do -- in fact,
ERPs are special cases of rERPs; every ERP is also a rERP. But, the
reverse is not true. rERPs make it straightforward to analyze
experimental designs that use a mix of categorical and continuous
manipulations, even when these manipulations are partially confounded
or produce non-linear effects, and they can separate out overlapping
waveforms timelocked to temporally adjacent events. They can even do
all these things at the same time.

.. image:: https://travis-ci.org/rerpy/rerpy.png?branch=master
   :target: https://travis-ci.org/rerpy/rerpy
.. image:: https://coveralls.io/repos/rerpy/rerpy/badge.png?branch=master
   :target: https://coveralls.io/r/rerpy/rerpy?branch=master

Documentation:
  not yet

Downloads:
  not yet

Dependencies:
  * Python 2.7 (not Python 3 yet, sorry -- patches accepted!)
  * numpy
  * scipy
  * pandas
  * patsy

Optional dependencies:
  * nose: needed to run tests

Install:
  probably not a great idea yet

Mailing list:
  not yet, but in the mean time you can hassle nathaniel.smith@ed.ac.uk

Code and bug tracker:
  https://github.com/rerpy/rerpy

License:
  GPLv2+, see LICENSE.txt for details.
