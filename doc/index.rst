.. rERPy documentation master file, created by
   sphinx-quickstart on Fri Jan 25 12:26:03 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

rERPy - a toolkit for regression-based ERP analysis
===================================================

rERPy is a Python toolkit for doing ERP/ERF analysis of brainwave data
(EEG or MEG), using both traditional averaging-based ERP/ERF
estimation, and a fancy new regression-based technique for estimating
ERP/ERF waveforms, which we call `rERP (or rERF)
<http://vorpus.org/rERP/>` for short. rERP analysis is a strict
generalization of ERP analysis -- every published ERP is also a
rERP. But with rERP analysis, you can estimate ERP waveforms
regardless of whether your design is factorial or continuous or both,
whether it is orthogonal or partially confounded, whether your
continuous covariates have linear or non-linear effects, and whether
your events of interest produce overlapping ERPs or not.

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
