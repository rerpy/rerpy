User manual
===========

rERPy doesn't work quite like other ERP analysis tools. rERP analysis
allows for stimuli that vary in complex ways on multiple dimensions --
therefore, we need a way to describe such stimuli. rERP analysis is
flexible, allowing for many potentially different analyses of the same
data, which means that we may need to spend some time exploring
different options -- therefore, we would like the process of
specifying a new model to be as quick and simple as possible. And rERP
analysis requires us to set up predictors, which for categorical
variables requires the application of dummy, treatment, or other more
exotic coding schemes, and for continuous variables may involve
transformations like centering, log transformations, the use of spline
bases, and so forth. Therefore, we would like a simple way to describe
complex sets of predictors.

To accomplish these goals, rERPy has two key features. First,

Getting oriented
----------------

rERPy holds EEG/MEG data in :class:`Dataset` objects. These objects
store several types of information, all bundled up together:

First, they hold the actual EEG/MEG data, divided into contiguous
spans of recording. Since we need some way to refer to these spans, we
call them *recspans*. If you combine two subjects' data for an
analysis, you will have multiple recspans in one :class:`Dataset`. If
you record from a subject for a bit, and then pause the recording, and
then record some more, then you will have multiple recspans. We find
this easier and less error-prone then the other common technique of
storing all data concatenated into one long array, with a separate
record of where the "boundary points" are between subjects/pauses,
etc.; by having one array per recspan, we can never accidentally treat
non-contiguous data as if it were contiguous. If you have a
:class:`Dataset`, then ``data_set[0]`` gives you the first recspan
(represented as a :class:`pandas.DataFrame`), ``len(data_set)`` tells
you how many recspans there are, ``for recspan in data_set: ...`` lets
you iterate over all of them, etc.

Second, they hold basic metadata needed to interpret the EEG/MEG:
sampling rate, units, channel names, etc., represented as a
:class:`DataFormat` object. This can be accessed with
``data_set.data_format``. This provides various convenience methods;
e.g., you can conveniently convert between millisecond-based and
tick-based representations of time using
:meth:`DataFormat.ms_to_ticks` and :meth:`DataFormat.ticks_to_ms`.

Third, they hold a record of what events have occurred, when they
occurred (relative to the recording), and arbitrarily detailed
information about each event.

rERPy's system for storing event data is very different from that used
in other systems, and is extremely rich and powerful. For each event,
we record:
* In which recspan it occurs.
* Which tick it starts on.
* Which tick it ends on (to allow for temporally extended events,
  e.g., marking the extent of an artifact).
*



Loading data
------------

XX

Examining data
--------------

XX

Estimating rERPs
----------------

XX

Visualizing results
-------------------

TBD

Exporting results
-----------------

TBD
