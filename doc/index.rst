.. rERPy documentation master file, created by
   sphinx-quickstart on Fri Jan 25 12:26:03 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

rERPy - a toolkit for regression-based ERP analysis
===================================================

rERPy is a Python toolkit for doing ERP analysis of brainwave data
(EEG or MEG), using both traditional averaging-based ERP estimation,
and a fancy new regression-based technique for ERP estimation, which
we call rERP for short. rERPs can do anything ERPs can do -- in fact,
ERPs are special cases of rERPs; every ERP is also a rERP. But, the
reverse is not true. rERPs make it straightforward to analyze
experimental designs that use a mix of categorical and continuous
manipulations, even when these manipulations are partially confounded
or produce non-linear effects, and they can separate out overlapping
waveforms timelocked to temporally adjacent events. They can even do
all these things at the same time.

Contents:

.. toctree::
   :maxdepth: 3

   logistics
   user-manual
   compatibility
   reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
