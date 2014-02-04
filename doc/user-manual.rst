User manual
===========

While rERPy can be used to perform traditional ERP/ERF analyses, it
was designed for doing regression-based rERP/rERF analyses, and so if
you're used to other ERP/ERF tools then you may find its approach to
things somewhat unusual (and perhaps, I hope, better!). This manual
will assume a familiarity with the basic concepts of rERP analysis, as
described in:

  Smith, N. J., & Kutas, M. (submitted). Regression-based estimation
  of ERP waveforms: I. The rERP framework.

and

  Smith, N. J., & Kutas, M. (submitted). Regression-based estimation
  of ERP waveforms: II. Non-linear effects, overlap correction, and
  practical considerations.

which are available `here <http://vorpus.org/rERP/>`_. Henceforth
we'll refer to these as rERP-I and rERP-II for short.

The big picture
---------------

The three critical steps in performing an (r)ERP analysis with rERPy
are:

  1. Loading your data.

     rERPy provides built-in readers for some common data file
     formats; alternatively, if you have some way to get your data
     into a numpy array, or a file format that you can read into a
     numpy array (e.g., HDF5, Matlab .mat, etc.), then you can write
     your own minimal "loader" in a few lines of Python.

  2. Annotating your events.

     Most (all?) common EEG/MEG file formats provide only minimal
     information about events -- e.g., a single numeric code and
     nothing else. rERPy allows you to go further, and attach a set of
     arbitrary event properties to each event. For example, rERP-I
     presents data from a study by DeLong, Kutas, & Urbach (2005), in
     which sentences were presented in a word-by-word fashion, and
     examines the effect of word expectancy (measured between 0 and 1)
     on the ERP waveform. In this experiment, our initial
     representation of some word when we load our data might look
     like::

       {
         "event_code": 1234,
       }

     And once we're done annotating our events, it might instead look
     like::

       {
         "event_code": 1234,
         "type": "word",
         "word": "a",
         "expectancy": 0.12,
         "sentence_id": 5678,
         "beginning_of_sentence": False,
         "end_of_sentence": False,
       }

     Of course you can use whatever event descriptions make sense for
     your experiment, and rERPy provides convenient tools to e.g. load
     these event properties from a spreadsheet.

  3. Performing an rERP analysis.

     Now that we have our data loaded and our events annotated, it's
     easy to estimate some rERP waveforms. We just have to specify
     three things:

       1. Which events we want to include in our analysis. rERPy
          allows you to pick out events by referring to their
          properties using a rich query language. In this case, if we
          want to analyze just the articles, we might specify that we
          want to include events which match::

            word == 'a' or word == 'an'

       2. What latency window we want to include -- e.g., from -100 ms
          pre-stimulus to 1000 ms post-stimulus.

       3. What regression design we want to use, including any
          interactions, transformations, etc. In rERPy, this is
          specified using a high-level "formula" system derived
          from R. In our example, where we want to consider both the
          categorical effect of *a* versus *an*, and simultaneously
          the linear effect of expectancy, we might write our
          abstract regression formula as::

            word + expectancy

    Or, in concrete, Python-code terms::

       # Assuming we have our data stored in a variable called 'dataset':
       rerp = dataset.rerp("word == 'a' or word == 'an'",
                           -100, 1000,
                           "word + expectancy")

    (In fact, this code performs exactly the analysis shown in
    Figure 3 of rERP-I.) Now ``rerp.betas`` has the $\beta$ values
    from our regression, and other attributes of ``rerp`` can be used
    to determine predictions, statistics about artifacts and overlap,
    and so forth.

The next sections describe each of these steps in more detail.

Loading and working with data
-----------------------------


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



XX

Annotating event properties
---------------------------

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
