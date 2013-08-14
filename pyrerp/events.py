# This file is part of pyrerp
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

import sqlite3
import string
import re
import struct
from cStringIO import StringIO
import numpy as np
import pandas
from patsy import PatsyError
from patsy.origin import Origin
from patsy.parse_core import Token, Operator, parse
# We need isinstance(..., Recording), but we don't do 'from ... import
# Recording' because that would create a circular import:
import pyrerp.data

__all__ = ["Events", "EventsError"]

class EventsError(PatsyError):
    pass

# An injective map from arbitrary non-empty unicode strings to strings that
# match the regex [_a-zA-Z][_a-zA-Z0-9]+ (i.e., are valid SQL identifiers)
def _munge_name(key):
    assert key
    # If it's just letters and underscores, pass it through unchanged, to make
    # debugging easier:
    if all([char in string.ascii_letters
            or char == "_"
            for char in key]):
        return key
    else:
        # Otherwise, munge it in a way that is guaranteed to add numbers (and
        # thus be disjoint from the strings returned by the above branch)
        pieces = []
        for char in key:
            if (char in string.ascii_letters
                or (pieces and char in string.digits)):
                pieces.append(char)
            else:
                pieces += ["_", str(ord(char))]
        return "".join(pieces)

def test__munge_name():
    clean_names = ["foo", "bar", "foo_bar"]
    unclean_names = ["a1", u"ab\u1234c", "123abc", "  "]
    for name in clean_names:
        assert _munge_name(name) == name
    for name in unclean_names:
        munged = _munge_name(name)
        assert munged[0] == "_" or munged[0].isalpha()
        for char in munged[1:]:
            assert char == "_" or char.isalnum()
    munged_names = set([_munge_name(n) for n in clean_names + unclean_names])
    assert len(munged_names) == len(clean_names) + len(unclean_names)

# http://www.logarithmic.net/pfh/blog/01235197474
def approx_interval_magnitude(span):
    """Returns a number M such that:
      M <= span <= 2*M
    and so only a few distinct M values are used. In practice, M is chosen to
    either be 0, or a power of 2."""
    if span == 0:
        return 0
    magnitude = 1
    while magnitude * 2 < span:
        magnitude *= 2
    return magnitude

def test_approx_interval_magnitude():
    for span in [0, 1, 2, 3, 10, 100, 127, 128, 129, 5000, 65536]:
        M = approx_interval_magnitude(span)
        assert M <= span <= 2 * M

def _table_name(key):
    return "attr_" + _munge_name(key)

# Keys that have a special magical meaning in dict-queries and
# string-queries (plus the actual database field name they map to).
# In string queries these can be de-magicalified by quoting them with
# backquotes, i.e.,:
#   "_SPAN_ID == 1"   <-> placeholder.span_id == 1
#   "`_SPAN_ID` == 1" <-> placeholder["_SPAN_ID"] == 1
_magic_query_strings = set(["_RECORDING", "_SPAN_ID", "_START_IDX",
                            "_STOP_IDX", "_RECORDING_NAME"])
def _magic_query_string_to_query(events, name, origin=None):
    if name == "_RECORDING":
        return RecordingQuery(events, origin)
    else:
        return IndexFieldQuery(events, name[1:].lower(), origin)

class Events(object):
    NUMERIC = "numeric"
    BLOB = "text"
    BOOL = "bool"

    def __init__(self):
        # int -> Recording
        self._recordings = []
        # Recording -> int
        self._recordings_map = {}
        self._interval_magnitudes = set()
        self._connection = sqlite3.connect(":memory:")
        # Maps key names to value types (NUMERIC, BLOB, BOOL), or None if we
        # have yet to see any values for the given key and so don't know what
        # type it should have.
        self._key_types = {}

        # Every time 'op_count' passes 'analyze_threshold', we run ANALYZE and
        # double 'analyze_threshold'. Starting analyze_threshold as 256 or so
        # would make more sense, but the extra cost of doing it this way is
        # minimal, and this way we actually test this logic even on small
        # examples.
        self._op_count = 0
        self._analyze_threshold = 1

        c = self._connection.cursor()
        c.execute("PRAGMA case_sensitive_like = true;")
        c.execute("CREATE TABLE sys_events "
                  "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                  "recording_id INTEGER NOT NULL, "
                  "span_id INTEGER NOT NULL, "
                  "start_idx NUMERIC NOT NULL, "
                  "stop_idx NUMERIC NOT NULL, "
                  "interval_magnitude INTEGER NOT NULL, "
                  "recording_name);")
        c.execute("CREATE INDEX sys_events_by_start_idx "
                  "ON sys_events (recording_id, span_id, start_idx);")
        c.execute("CREATE INDEX sys_events_by_stop_idx "
                  "ON sys_events (recording_id, span_id, stop_idx);")
        # The special indices used to make overlaps queries fast
        c.execute("CREATE INDEX sys_events_interval_start_idx "
                  "ON sys_events (recording_id, span_id, "
                                 "interval_magnitude, start_idx);")
        c.execute("CREATE INDEX sys_events_interval_stop_idx "
                  "ON sys_events (recording_id, span_id, "
                                 "interval_magnitude, stop_idx);")

    def _intern_recording(self, recording):
        if recording in self._recordings_map:
            return self._recordings_map[recording]
        else:
            idx = len(self._recordings)
            self._recordings.append(recording)
            self._recordings_map[recording] = idx
            return idx

    def _try_intern_recording(self, recording):
        return self._recordings_map.get(recording)

    def _extern_recording(self, recording_idx):
        return self._recordings[recording_idx]

    # Convert str's to buffer objects before passing them into sqlite, because
    # that is how you tell the sqlite3 module to store them as
    # BLOBs. (sqlite3.Binary is an alias for 'buffer'.) This function also
    # handles converting numpy scalars into equivalents that are acceptable to
    # sqlite.
    def _encode_sql_value(self, val):
        if np.issubsctype(type(val), np.str_):
            return sqlite3.Binary(val)
        elif np.issubsctype(type(val), np.bool_):
            return bool(val)
        elif np.issubsctype(type(val), np.integer):
            return int(val)
        elif np.issubsctype(type(val), np.floating):
            return float(val)
        else:
            return val

    # And reverse the transformation on the way out.
    def _decode_sql_value(self, val):
        if isinstance(val, sqlite3.Binary):
            return str(val)
        else:
            return val

    def _execute(self, c, sql, args):
        return c.execute(sql, [self._encode_sql_value(arg) for arg in args])

    def _value_type_for_key(self, key):
        return self._key_types.get(key)

    def _value_type(self, value):
        # must come first, because issubclass(bool, int)
        if isinstance(value, (bool, np.bool_)):
            return self.BOOL
        elif isinstance(value, (int, long, float, np.number)):
            return self.NUMERIC
        elif isinstance(value, (str, unicode, np.character)):
            return self.BLOB
        elif value is None:
            return None
        else:
            raise ValueError, ("Invalid value %r: "
                               "must be a string, number, bool, or None"
                               % (value,))

    def _sql_value_to_value_type(self, sql_value, value_type):
        # SQLite's type system discards the distinction between bools and
        # ints, so we have to recover it on the way out.
        if value_type == self.BOOL:
            return bool(sql_value)
        else:
            return sql_value

    def _observe_value_for_key(self, key, value):
        value_type = self._value_type(value)
        if value_type is None:
            return
        wanted_type = self._value_type_for_key(key)
        if wanted_type is None:
            self._key_types[key] = value_type
        else:
            if wanted_type != value_type:
                err = ("Invalid value %r for key %s: wanted %s"
                       % (value, key, wanted_type))
                raise ValueError, err

    # NOTE: DDL statements like table creation cannot be performed inside a
    # transaction. So don't call this method inside a transaction.
    def _ensure_table_for_key(self, key):
        table = _table_name(key)
        self._connection.execute("CREATE TABLE IF NOT EXISTS %s ("
                                   "event_id INTEGER PRIMARY KEY, "
                                   "value);" % (table,))
        self._connection.execute("CREATE INDEX IF NOT EXISTS %s_idx "
                                   "ON %s (value);" % (table, table))
        self._key_types.setdefault(key, None)

    def _incr_op_count(self):
        self._op_count += 1
        if self._op_count >= self._analyze_threshold:
            self._connection.execute("ANALYZE;")
        self._analyze_threshold *= 2

    def add_event(self, recording, span_id, start_idx, stop_idx, attributes):
        # Create tables up front before entering transaction:
        for key in attributes:
            self._ensure_table_for_key(key)
        recording_id = self._intern_recording(recording)
        if not start_idx < stop_idx:
            raise ValueError, "start_idx must be < stop_idx"
        if not start_idx >= 0:
            raise ValueError, "start_idx must be >= 0"
        interval_magnitude = approx_interval_magnitude(stop_idx - start_idx)
        self._interval_magnitudes.add(interval_magnitude)
        with self._connection:
            c = self._connection.cursor()
            self._execute(c,
              "INSERT INTO sys_events "
              "  (recording_id, span_id, start_idx, stop_idx,"
              "   interval_magnitude, recording_name) "
              "values (?, ?, ?, ?, ?, ?)",
              [recording_id, span_id, start_idx, stop_idx, interval_magnitude,
               recording.name])
            event_id = c.lastrowid
            for key, value in attributes.iteritems():
                self._observe_value_for_key(key, value)
                self._execute(c,
                              "INSERT INTO %s (event_id, value) VALUES (?, ?);"
                                % (_table_name(key),),
                              (event_id, value))
        self._incr_op_count()
        return Event(self, event_id)

    def _move_event(self, id, offset):
        # Fortunately, this operation doesn't change the magnitude of the
        # span, so it's relatively simple to implement -- we can just update
        # the start_idx and stop_idx directly without having to worry about
        # the complexities of interval_magnitudes.
        with self._connection:
            c = self._connection.cursor()
            self._execute(c,
                          "UPDATE sys_events "
                          "SET start_idx = start_idx + ?, "
                          "    stop_idx = stop_idx + ? "
                          "WHERE id = ?",
                          [offset, offset, id])

    def _delete_event(self, id):
        with self._connection:
            c = self._connection.cursor()
            self._execute(c, "DELETE FROM sys_events WHERE id = ?;", (id,))
            for key in self._key_types:
                self._execute(c,
                              "DELETE FROM %s WHERE event_id = ?;"
                                % (_table_name(key),),
                              (id,))
        self._incr_op_count()

    def _event_index_field(self, field, id):
        c = self._connection.cursor()
        code = "SELECT %s FROM sys_events WHERE id = ?;" % (field,)
        self._execute(c, code, (id,))
        return self._decode_sql_value(c.fetchone()[0])

    def _event_getitem(self, id, key):
        c = self._connection.cursor()
        table = _table_name(key)
        code = "SELECT value FROM %s WHERE event_id = ?;" % (table,)
        try:
            self._execute(c, code, (id,))
            row = c.fetchone()
        except sqlite3.OperationalError:
            raise KeyError, key
        if row is None:
            raise KeyError, key
        return self._sql_value_to_value_type(self._decode_sql_value(row[0]),
                                             self._key_types[key])

    def _event_setitem(self, id, key, value):
        self._ensure_table_for_key(key)
        with self._connection:
            c = self._connection.cursor()
            table = _table_name(key)
            self._observe_value_for_key(key, value)
            self._execute(c,
                          "UPDATE %s SET value = ? WHERE event_id = ?;"
                            % (table,),
                          (value, id))
            if c.rowcount == 0:
                self._execute(c,
                              "INSERT INTO %s (event_id, value) VALUES (?, ?);"
                                % (table,),
                              (id, value))
        self._incr_op_count()

    def _event_delitem(self, id, key):
        # Raise a KeyError if it doesn't exist:
        self._event_getitem(id, key)
        # Okay, now delete it
        c = self._connection.cursor()
        table = _table_name(key)
        code = "DELETE FROM %s WHERE event_id = ?;" % (table,)
        self._execute(c, code, (id,))
        self._incr_op_count()

    @property
    def placeholder(self):
        return PlaceholderEvent(self)

    @property
    def ANY(self):
        return LiteralQuery(self, True)

    def as_query(self, query_like):
        if isinstance(query_like, dict):
            p = self.placeholder
            query = self.ANY
            equalities = []
            for query_name in _magic_query_strings.intersection(query_like):
                q = _magic_query_string_to_query(self, query_name)
                query &= (q == query_like.pop(query_name))
            for k, v in query_like.iteritems():
                query &= (p[k] == v)
        elif isinstance(query_like, basestring):
            query = query_from_string(self, query_like)
        else:
            query = query_like

        if not isinstance(query, Query):
            raise ValueError, "expected a query object, not %r" % (query,)
        if query._events is not self:
            raise ValueError("query object does not refer to this Events "
                             "object")
        return query

    def find(self, query_like={}):
        """find({"a": 1, "b": 2}) or find(query_obj) or find(\"string\") or
        find().
        """
        return self.as_query(query_like).run()

    def _query(self, sql_where, tables, query_vals):
        tables = set(sql_where.tables)
        tables.add("sys_events")
        joins = ["%s.event_id == sys_events.id" % (table,)
                 for table in tables if table != "sys_events"]
        code = ("SELECT %s FROM %s WHERE (%s) "
                % (", ".join(query_vals),
                   ", ".join(tables),
                   sql_where.code))
        if joins:
            code += " AND (%s)" % (" AND ".join(joins),)
        code += "ORDER BY sys_events.recording_id, sys_events.span_id, sys_events.start_idx"
        c = self._connection.cursor()
        self._execute(c, code, sql_where.args)
        return c

    def at(self, recording, span_id, start_idx, stop_idx=None):
        if stop_idx is None:
            stop_idx = start_idx + 1
        p = self.placeholder
        q = p.overlaps(recording, span_id, start_idx, stop_idx)
        return self.find(q)

    def __iter__(self):
        return iter(self.ANY.run())

    def __len__(self):
        c = self._connection.cursor()
        self._execute(c, "SELECT count(*) FROM sys_events;", [])
        return self._decode_sql_value(c.fetchone()[0])

    def __repr__(self):
        return "<%s object with %s entries>" % (self.__class__.__name__,
                                                len(self))

    def merge_df(self, df, on, subset=None):
        # 'on' is like {df_colname: event_key}
        # or just [colname]
        # or just colname
        if isinstance(on, basestring):
            on = [on]
        if not isinstance(on, dict):
            on = dict([(key, key) for key in on])
        p = self.placeholder
        query = subset
        if query is None:
            query = self.ANY
        for row_idx in df.index:
            row = df.xs(row_idx)
            this_query = query
            for df_key, db_key in on.iteritems():
                this_query &= (p[db_key] == row[df_key])
            for ev in this_query.run():
                for df_key in row.index:
                    if df_key not in on:
                        ev[df_key] = row[df_key]

    # Pickling
    def __getstate__(self):
        events = []
        for ev in self:
            events.append((ev.recording, ev.span_id, ev.start_idx, ev.stop_idx, dict(ev)))
        # 0 as an ad-hoc version number in case we need to change this later
        return (0, events)

    def __setstate__(self, state):
        if state[0] != 0:
            raise ValueError, "unrecognized pickle data version for Events object"
        self.__init__()
        _, events = state
        for recording, span_id, start_idx, stop_idx, attrs in events:
            self.add_event(recording, span_id, start_idx, stop_idx, attrs)

# This is like a "frozen" query -- it always refers to the same events even if
# the original set changes (though the values in those events may change!),
# can be mutated in place, and, crucially, it supports __getitem__ to get a
# Series of values. So you can pass it as a 'data' object to dmatrix and
# friends!
class EventSet(object):
    def __init__(self, events, event_ids):
        self._events = events
        self._event_ids = event_ids

    def remove(self, event):
        if event._events is not self._events:
            raise KeyError, event
        try:
            self._event_ids.remove(event.id)
        except ValueError:
            raise KeyError, event

    def __len__(self):
        return len(self._event_ids)

    def __iter__(self):
        for db_id in self._event_ids:
            yield Event(self._events, db_id)

    def __getitem__(self, idx):
        if hasattr(idx, "__index__") or isinstance(idx, (slice, type(Ellipsis))):
            return Event(self._events, self._event_ids[idx])
        elif isinstance(idx, basestring):
            # This will raise a KeyError for any events where the field is
            # just undefined, and will return None otherwise. This could be
            # done more efficiently by querying the database directly, but
            # let's not fret about that until it matters.
            values = [ev[idx] for ev in self]
            # We use pandas.Series here because it has much more sensible
            # NaN/None handling than raw numpy.
            #   np.asarray([None, 1, 2]) -> object (!) array
            #   np.asarray([np.nan, "a", "b"]) -> ["nan", "a", "b"] (!)
            # but
            #   pandas.Series([None, 1, 2]) -> [nan, 1, 2]
            #   pandas.Series([None, "a", "b"]) -> [None, "a", "b"]
            return pandas.Series(values)
        else:
            raise TypeError, "index must be an integer or string"

    # def __setitem__(self, key, value):
    #     # XX we should probably support Series and ndarrays and stuff here
    #     # XX we REALLY SHOULD support aligned assignment so we can write
    #     # things like
    #     #   my_set["foo"] = my_set.next(query)["foo"]
    #     # and make sure that nan gets mapped to deleting the value
    #     # ...though this would require that my_set.next() be indexed by the
    #     # same events in my_set, not the events in my_set.next(). Is that
    #     # good or bad?
    #     xx

    def update(self, d):
        for ev in self:
            ev.update(d)

    def __repr__(self):
        return "<%s with %s events>" % (self.__class__.__name__,
                                        len(self._event_ids))

    def _repr_pretty_(self, p, cycle):
        assert not cycle
        p.begin_group(2, "<%s with %s events:" % (self.__class__.__name__,
                                                  len(self._event_ids)))
        p.breakable()
        p.pretty(list(self))
        p.end_group(2, ">")

class Event(object):
    def __init__(self, events, id):
        self._events = events
        self._id = id

    def __getstate__(self):
        raise ValueError, "Event objects are not pickleable"

    @property
    def recording(self):
        rid = self._events._event_index_field("recording_id", self._id)
        return self._events._extern_recording(rid)

    @property
    def span_id(self):
        return self._events._event_index_field("span_id", self._id)

    @property
    def start_idx(self):
        return self._events._event_index_field("start_idx", self._id)

    @property
    def stop_idx(self):
        return self._events._event_index_field("stop_idx", self._id)

    def overlaps(self, *args):
        if len(args) == 1:
            ev = args[0]
            return self.overlaps(ev.recording, ev.span_id,
                                 ev.start_idx, ev.stop_idx)
        else:
            (recording, span_id, start_idx, stop_idx) = args
            return (self.recording == recording
                    and self.span_id == span_id
                    and self.start_idx < stop_idx
                    and start_idx < self.stop_idx)

    def __getitem__(self, key):
        return self._events._event_getitem(self._id, key)

    def __setitem__(self, key, value):
        self._events._event_setitem(self._id, key, value)

    def __delitem__(self, key):
        self._events._event_delitem(self._id, key)

    def delete(self):
        self._events._delete_event(self._id)

    def iteritems(self):
        for key in self._events._key_types:
            try:
                yield key, self[key]
            except KeyError:
                continue

    # Everything else is defined in terms of the above methods.

    def update(self, d):
        for k, v in d.iteritems():
            self[k] = v

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __iter__(self):
        for key, _ in self.iteritems():
            yield key

    iterkeys = __iter__

    def itervalues(self):
        for _, value in self.iteritems():
            yield value

    def keys(self):
        return list(self.iterkeys())

    def values(self):
        return list(self.itervalues())

    def items(self):
        return list(self.iteritems())

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    has_key = __contains__

    def __repr__(self):
        return "<%s %s at %s: %s>" % (
            self.__class__.__name__, self._id, self.index,
            repr(dict(self.iteritems())))

    # For the IPython pretty-printer:
    #   http://ipython.org/ipython-doc/dev/api/generated/IPython.lib.pretty.html
    def _repr_pretty_(self, p, cycle):
        assert not cycle
        p.begin_group(2,
                      "<%s in %r, span %s, ticks %s-%s:"
                      % (self.__class__.__name__, self.recording,
                         self.span_id, self.start_idx, self.stop_idx))
        p.breakable()
        p.pretty(dict(self.iteritems()))
        p.end_group(2, ">")

    def relative(self, count, query={}):
        """Counts 'count' events forward or back from the current event (or
        optionally, only events that match 'query'), and returns that. Use
        negative for backwards."""
        if count == 0:
            raise IndexError, "count must be non-zero"
        query = self._events.as_query(query)
        p = self._events.placeholder
        query &= (p.recording == self.recording)
        query &= (p.span_id == self.span_id)
        if count > 0:
            query &= (p.start_idx > self.start_idx)
            return query.run()[count - 1]
        else:
            query &= (p.start_idx < self.start_idx)
            return query.run()[count]

    def move(self, offset):
        """Shifts this event's timestamp by 'offset' samples (positive for
        forward, negative for backward)."""
        self._events._move_event(self._id, offset)

###############################################
#
# The query specification system (Python API)
#
###############################################

class PlaceholderEvent(object):
    def __init__(self, events):
        self._events = events

    def __getitem__(self, key):
        return AttrQuery(self._events, key)

    @property
    def recording(self):
        return RecordingQuery(self._events)

    @property
    def span_id(self):
        return IndexFieldQuery(self._events, "span_id")

    @property
    def start_idx(self):
        return IndexFieldQuery(self._events, "start_idx")

    @property
    def stop_idx(self):
        return IndexFieldQuery(self._events, "stop_idx")

    def has_key(self, key):
        return HasKeyQuery(self._events, key)

    def overlaps(self, *args):
        if len(args) == 1:
            ev = args[0]
            if isinstance(ev, PlaceholderEvent):
                raise ValueError, "cannot test for overlap between two placeholders"
            return self.overlaps(ev.recording, ev.span_id,
                                 ev.start_idx, ev.stop_idx)
        else:
            (recording, span_id, start_idx, stop_idx) = args
            return OverlapsQuery(self._events,
                                 self._events._try_intern_recording(recording),
                                 span_id,
                                 start_idx, stop_idx)

class SqlWhere(object):
    def __init__(self, code, tables, args):
        self.code = code
        self.tables = tables
        self.args = args

class Query(object):
    def __init__(self, events, origin=None):
        self._events = events
        self.origin = origin

    # Building queries:

    def _as_query(self, other):
        if isinstance(other, Query):
            assert other._events is self._events
            return other
        return LiteralQuery(self._events, other)

    def _make_op(self, op_name, sql_op, children, expected_type=None):
        children = [self._as_query(child) for child in children]
        # Otherwise, do type checking.
        types = []
        for child in children:
            type = child._value_type()
            if expected_type is not None and type != expected_type:
                raise EventsError("%r operator expected %s, got %s "
                                    "(check your parentheses?)"
                                    % (op_name, expected_type, type),
                                  child)
            types.append(type)
        if len(set(types).difference([None])) > 1:
            raise EventsError("mismatched types: %s" % (" vs ".join(types)),
                              Origin.combine(children))
        return QueryOperator(self._events, sql_op, children,
                             Origin.combine(children))

    def __eq__(self, other):
        # NB: IS is like == except that it treats NULL as just another
        # value. Which is what we want, since we use NULL to encode None, and
        # None == None should be True, not NULL.
        return self._make_op("==", "IS", [self, other])

    def __ne__(self, other):
        # And IS NOT is the NULL-handling equivalent to !=.
        return self._make_op("!=", "IS NOT", [self, other])

    def __lt__(self, other):
        return self._make_op("<", "<", [self, other])

    def __gt__(self, other):
        return self._make_op(">", ">", [self, other])

    def __le__(self, other):
        return self._make_op("<=", "<=", [self, other])

    def __ge__(self, other):
        return self._make_op(">=", ">=", [self, other])

    def __and__(self, other):
        return self._make_op("and", "AND", [self, other], Events.BOOL)

    def __or__(self, other):
        return self._make_op("or", "OR", [self, other], Events.BOOL)

    def __invert__(self):
        return self._make_op("not", "NOT", [self], Events.BOOL)

    def __nonzero__(self):
        raise TypeError("can't convert query directly to bool "
                        "(maybe you want a bitwise operator like & | ~ "
                        "instead of a logical operator like "
                        "'and' 'or' 'not')")

    def _value_type(self):
        assert False

    def _sql_where(self):
        assert False

    def __len__(self):
        if self._value_type() != Events.BOOL:
            raise EventsError("query must be boolean", query)
        c = self._events._query(self._sql_where(), [], ["count(*)"])
        for (count,) in c:
            return count

    def _db_ids(self):
        if self._value_type() != Events.BOOL:
            raise EventsError("top-level query must be boolean", self)
        c = self._events._query(self._sql_where(),
                                ["sys_events"],
                                ["sys_events.id"])
        for (db_id,) in c:
            yield self._events._decode_sql_value(db_id)

    def run(self):
        return EventSet(self._events, list(self._db_ids()))

class LiteralQuery(Query):
    def __init__(self, events, value, origin=None):
        Query.__init__(self, events, origin)
        if isinstance(value, pyrerp.data.Recording):
            self._value = self._events._try_intern_recording(value)
            self._saved_value_type = "RECORDING"
        else:
            self._value = value
            try:
                self._saved_value_type = self._events._value_type(self._value)
            except ValueError:
                raise EventsError("literals must be boolean, string, numeric, "
                                  "or None",
                                  self)

    def _sql_where(self):
        return SqlWhere("?", set(), [self._value])

    def _value_type(self):
        return self._saved_value_type

    def __repr__(self):
        return "<%s: %r>" % (self.__class__.__name__, self._value)

class AttrQuery(Query):
    def __init__(self, events, key, origin=None):
        Query.__init__(self, events, origin)
        self._key = key
        # Go ahead and create any tables that are being queried -- if someone
        # makes a typo then this will create an empty table, but creating a
        # table by itself is a total no-op at the data level, so whatever.
        self._events._ensure_table_for_key(self._key)

    def _sql_where(self):
        table_id = _table_name(self._key)
        return SqlWhere(table_id + ".value", set([table_id]), [])

    def _value_type(self):
        return self._events._value_type_for_key(self._key)

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__name__, self._key)

class HasKeyQuery(Query):
    def __init__(self, events, key, origin=None):
        Query.__init__(self, events, origin)
        self._key = key
        # Go ahead and create any tables that are being queried -- if someone
        # makes a typo then this will create an empty table, but creating a
        # table by itself is a total no-op at the data level, so whatever.
        self._events._ensure_table_for_key(self._key)

    def _sql_where(self):
        table_id = _table_name(self._key)
        return SqlWhere("EXISTS (SELECT 1 FROM %s inner_table "
                        "WHERE sys_events.id == inner_table.event_id)"
                        % (table_id,),
                        set(), [])

    def _value_type(self):
        return self._events.BOOL

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__name__, self._key)

class RecordingQuery(Query):
    def _sql_where(self):
        # sys_events is always included in the join, so no need to add it to
        # the tables.
        return SqlWhere("sys_events.recording_id", set(), [])

    def _value_type(self):
        return "RECORDING"

    def __repr__(self):
        return "<%s>" % (self.__class__.__name__,)

    @property
    def name(self):
        return IndexFieldQuery(self._events, "recording_name")

class IndexFieldQuery(Query):
    def __init__(self, events, field, origin=None):
        Query.__init__(self, events, origin)
        self._field = field

    def _sql_where(self):
        # sys_events is always included in the join, so no need to add it to
        # the tables.
        return SqlWhere("sys_events.%s" % (self._field,), set(), [])

    def _value_type(self):
        if self._field == "recording_name":
            return self._events.BLOB
        else:
            return self._events.NUMERIC

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__name__, self._field)

class OverlapsQuery(Query):
    def __init__(self, events,
                 recording_id, span_id, start_idx, stop_idx, origin=None):
        Query.__init__(self, events, origin)
        self._recording_id = recording_id
        self._span_id = span_id
        self._start_idx = start_idx
        self._stop_idx = stop_idx

    def _sql_where(self):
        args = []
        possibilities = []
        for magnitude in self._events._interval_magnitudes:
            constraints = [
                "sys_events.recording_id == ?",
                "sys_events.span_id == ?",
                "sys_events.interval_magnitude == ?",
                "((sys_events.start_idx < ? AND (? - ?) < sys_events.start_idx)"
                  " OR (sys_events.stop_idx < (? + ?) AND ? < sys_events.stop_idx))",
                ]
            args += [self._recording_id, self._span_id,
                     magnitude,
                     self._stop_idx, self._start_idx, magnitude,
                     self._stop_idx, magnitude, self._start_idx]
            possibilities.append(" AND ".join(constraints))
        return SqlWhere(" OR ".join(possibilities), set(), args)

    def _value_type(self):
        return self._events.BOOL

    def __repr__(self):
        return ("<%s %r %r [%r, %r)>"
                % (self.__class__.__name__, self._recording_id, self._span_id,
                   self._start_idx, self._stop_idx))

# This is only used internally
class IdQuery(Query):
    def _sql_where(self):
        # sys_events is always included in the join, so no need to add it to
        # the tables.
        return SqlWhere("sys_events.id", set(), [])

class QueryOperator(Query):
    def __init__(self, events, sql_op, children, origin=None):
        Query.__init__(self, events, origin)
        self._sql_op = sql_op
        assert 1 <= len(children) <= 2
        self._children = children

    def _sql_where(self):
        lhs_sqlwhere = self._children[0]._sql_where()
        if len(self._children) == 1:
            # Special case for NOT
            rhs_sqlwhere = lhs_sqlwhere
            lhs_sqlwhere = SqlWhere("", set(), [])
        else:
            rhs_sqlwhere = self._children[1]._sql_where()
        return SqlWhere("(%s %s %s)"
                       % (lhs_sqlwhere.code, self._sql_op, rhs_sqlwhere.code),
                       lhs_sqlwhere.tables.union(rhs_sqlwhere.tables),
                       lhs_sqlwhere.args + rhs_sqlwhere.args)

    def _value_type(self):
        return Events.BOOL

    def __repr__(self):
        return "<%s (%s %r)>" % (self.__class__.__name__,
                                 self._sql_op,
                                 self._children)

########################################
#
# A string-based query language
#
########################################

_punct_ops = [
    Operator("==", 2, 100),
    Operator("!=", 2, 100),
    Operator("<", 2, 100),
    Operator(">", 2, 100),
    Operator("<=", 2, 100),
    Operator(">=", 2, 100),
    ]
_text_ops = [
    Operator("not", 1, 0),
    Operator("has", 1, 0),
    Operator("and", 2, 0),
    Operator("or", 2, 0),
    ]
_ops = _punct_ops + _text_ops

_atomic = ["ATTR", "LITERAL", "MAGIC_FIELD"]

def _read_quoted_string(string, i):
    start = i
    quote_type = string[i]
    assert quote_type in "\"'`"
    chars = []
    i += 1
    while i < len(string):
        char = string[i]
        if char == quote_type:
            break
        elif char == "\\":
            # Consume the backslash
            i += 1
            if i >= len(string):
                break
            escaped_char = string[i]
            if escaped_char in "\"'`\\":
                chars.append(escaped_char)
            else:
                raise EventsError("unrecognized escape sequence \\%s"
                                  % (escaped_char,),
                                  Origin(string, i - 1, i))
        else:
            chars.append(string[i])
        i += 1
    if i >= len(string):
        raise EventsError("unclosed string",
                          Origin(string, start, i))
    assert string[i] == quote_type
    i += 1
    if quote_type == "`":
        type = "ATTR"
    else:
        type = "LITERAL"
    return Token(type, Origin(string, start, i), "".join(chars)), i

def _tokenize(string):
    punct_op_tokens = [op.token_type for op in _punct_ops]
    punct_op_tokens.sort(key=len, reverse=True)
    text_op_tokens = [op.token_type for op in _text_ops]

    num_re = re.compile(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?")
    # This works because \w matches underscore, letters, and digits.
    # But if a token starts with a digit, then it'll be caught by num_re above
    # first, so in fact this works like "[_a-z][_a-z0-9]*" except for being
    # unicode-enabled.
    ident_re = re.compile(r"\w+", re.IGNORECASE | re.UNICODE)
    whitespace_re = re.compile(r"\s+")

    i = 0
    while i < len(string):
        if "(" == string[i]:
            yield Token(Token.LPAREN, Origin(string, i, i + 1))
            i += 1
            continue
        if ")" == string[i]:
            yield Token(Token.RPAREN, Origin(string, i, i + 1))
            i += 1
            continue
        if string[i] in "\"'`":
            token, i = _read_quoted_string(string, i)
            yield token
            continue
        match = num_re.match(string, i)
        if match is not None:
            try:
                value = int(match.group())
            except ValueError:
                value = float(match.group())
            yield Token("LITERAL", Origin(string, *match.span()), value)
            i = match.end()
            continue
        match = ident_re.match(string, i)
        if match is not None:
            token = match.group()
            origin = Origin(string, *match.span())
            if token in text_op_tokens:
                yield Token(token, origin)
            elif token.lower() == "true":
                yield Token("LITERAL", origin, True)
            elif token.lower() == "false":
                yield Token("LITERAL", origin, False)
            elif token.lower() == "none":
                yield Token("LITERAL", origin, None)
            elif token in _magic_query_strings:
                yield Token("MAGIC_FIELD", origin, token)
            else:
                yield Token("ATTR", origin, token)
            i = match.end()
            continue
        match = whitespace_re.match(string, i)
        if match is not None:
            i = match.end()
            continue
        for punct_token in punct_op_tokens:
            if string[i:i + len(punct_token)] == punct_token:
                yield Token(punct_token,
                            Origin(string, i, i + len(punct_token)))
                i += len(punct_token)
                break
        else:
            raise EventsError("unrecognized token",
                              Origin(string, i, i + 1))

_op_to_pymethod = {"==": "__eq__",
                   "!=": "__ne__",
                   "<": "__lt__",
                   ">": "__gt__",
                   "<=": "__le__",
                   ">=": "__ge__",
                   "and": "__and__",
                   "or": "__or__",
                   "not": "__invert__",
                   }
def _eval(events, tree):
    if tree.type in _op_to_pymethod:
        eval_args = [_eval(events, arg) for arg in tree.args]
        return getattr(eval_args[0], _op_to_pymethod[tree.type])(*eval_args[1:])
    elif tree.type == "has":
        assert len(tree.args) == 1
        arg = tree.args[0]
        if arg.type != "ATTR":
            raise EventsError("'has' expects an attribute name", arg)
        return HasKeyQuery(events, arg.token.extra, tree.origin)
    elif tree.type == "LITERAL":
        return LiteralQuery(events, tree.token.extra, tree.origin)
    elif tree.type == "ATTR":
        return AttrQuery(events, tree.token.extra, tree.origin)
    elif tree.type == "MAGIC_FIELD":
        return _magic_query_string_to_query(events, tree.token.extra,
                                              tree.origin)
    else:
        assert False

def query_from_string(events, string):
    return _eval(events, parse(_tokenize(string), _ops, _atomic))
