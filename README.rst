rERPy is a Python toolkit for doing ERP/ERF analysis of brainwave data
(EEG, MEG), using both traditional averaging-based ERP/ERF estimation,
and a fancy new regression-based technique for ERP/ERF estimation,
which we call rERP/rERF. rERPs can do anything ERPs can do -- in fact,
ERPs are special cases of rERPs; every ERP is also a rERP. But rERPs
are much more powerful. rERPs make it straightforward to analyze
experimental designs that use a mix of categorical and continuous
manipulations, even when these manipulations are partially confounded
or produce non-linear effects, and they can separate out overlapping
waveforms timelocked to temporally adjacent events. They can even do
all these things at the same time. Nonetheless, they are relatively
simple to use.

For more details on rERP/rERF analysis, see: http://vorpus.org/rERP

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

If you're starting from scratch (not previously a Python user), then
we recommend `installing a scientific Python distribution
<http://www.scipy.org/install.html>`_.

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
