rERPy is a Python library for analyzing EEG data using both
traditional averaging-based ERP estimation, and fancy new
regression-based ERP estimation (known as "rERP" to its
friends). rERPs can do anything ERPs can do -- in fact, ERPs are
special cases or rERPs -- but rERPs can also handle mixes of
categorical and continuous manipulations (including non-linear effects
like "ERP images"), disentangle confounded effects, and separate out
overlapping waveforms timelocked to temporally adjacent events. They
can even do all these things at the same time.

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
