# This file is part of rERPy
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# Need to put this somewhere, so...
#
# NOTES ON SQLITE3 TRANSACTION HANDLING
# -------------------------------------
#
# The Python sqlite3 module has really confusing transaction behaviour which
# is substantially more annoying than sqlite3's own built-in transaction
# behaviour.
#
# Basically:
# - Anything that looks like SELECT will leave the transaction state
#   unchanged.
# - Anything that looks like an UPDATE/DELETE/INSERT/REPLACE will silently
#   create a new transaction.
# - Anything else (e.g. DDL statements) will silently commit any existing
#   transaction (!!!!)
# - 'commit' will also commit any existing transaction.
# - 'with con: ...' is just a way to putting a rollback/commit at the end of
#   the block; it does nothing at the beginning.

import sqlite3
import string
import re
from collections import namedtuple

import numpy as np
import pandas
from patsy import PatsyError, Origin
# XX FIXME: these aren't actually exposed from patsy yet, should fix that at
# some point...
from patsy.infix_parser import Token, Operator, infix_parse
from patsy.util import repr_pretty_delegate

from rerpy.util import memoized_method

__all__ = ["Events", "EventsError"]

class EventsError(PatsyError):
    pass

################################################################
## Name munging (because it's easier than dealing with SQL quoting)
################################################################

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

################################################################
## A simple type system, and tools to get them in and out of sqlite
################################################################

# General idea:
# For each key, there is a single type: numeric, text, or bool. These are
# stored as numeric, blob, and bool in sqlite respectively.
#
# For each event, a key can be:
#   missing: no row in the db, ev["foo"] raises KeyError
#   None: db row with NULL value, ev["foo"] returns None
#   a value of the given type: db row with that value, ev["foo"] gives that value

_NUMERIC = "numeric"
_BLOB = "text"
_BOOL = "bool"

def _value_type(value):
    # must come before numeric, because issubclass(bool, int)
    if isinstance(value, (bool, np.bool_)):
        return _BOOL
    elif isinstance(value, (int, long, float, np.number)):
        return _NUMERIC
    elif isinstance(value, (str, unicode, np.character)):
        return _BLOB
    elif value is None:
        return None
    else:
        raise ValueError, ("Invalid value %r: "
                           "must be a string, number, bool, or None"
                           % (value,))

def _sql_value_to_value_type(sql_value, value_type):
    # SQLite's type system discards the distinction between bools and
    # ints, so we have to recover it on the way out.
    if value_type is _BOOL and sql_value is not None:
        return bool(sql_value)
    else:
        return sql_value

# Convert str's to buffer objects before passing them into sqlite, because
# that is how you tell the sqlite3 module to store them as
# BLOBs. (sqlite3.Binary is an alias for 'buffer'.) This function also
# handles converting numpy scalars into equivalents that are acceptable to
# sqlite.
ok_types = set([int, bool, float])
def _encode_sql_value(val):
    # This is a non-trivial speedup for bulk inserts (e.g. creating thousands
    # of events while loading a file):
    if type(val) in ok_types:
        return val
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

def _encode_seq_to_sql_values(seq):
    dtype = pandas.Series(seq).dtype
    if np.issubdtype(dtype, np.bool_):
        type_ = bool
    elif np.issubdtype(dtype, np.integer):
        type_ = int
    elif np.issubdtype(dtype, np.floating):
        type_ = float
    else:
        # Series store strings as 'object' dtype, and that's the only valid
        # thing that gets stored using an object dtype.
        type_ = sqlite3.Binary
    def convert(value):
        if value is None:
            return None
        else:
            return type_(value)
    return [convert(value) for value in seq]

# And reverse the transformation on the way out.
def _decode_sql_value(val):
    if isinstance(val, sqlite3.Binary):
        return str(val)
    else:
        return val

################################################################
## A clever trick for supporting efficient overlap queries
##    (see http://www.logarithmic.net/pfh/blog/01235197474)
################################################################

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

################################################################
## The core event stuff.
################################################################

# This is used to allow for lots of code to be shared between events and
# recspans, both of which can have arbitrary key/value pairs associated with
# them.
class ObjType(object):
    def __init__(self, name, sys_table, event_join_field):
        # Maps key names to value types (NUMERIC, BLOB, BOOL), or None if we
        # have yet to see any values for the given key and so don't know what
        # type it should have. Used not just for type checking, but also to
        # know which tables we've inserted values into (needed for iterating
        # over all keys, etc.).
        self.key_types = {}
        self.name = name
        self.sys_table = sys_table
        self.event_join_field = event_join_field

    def table_name(self, key):
        return self.name + "_attr_" + _munge_name(key)

    def value_type_for_key(self, key):
        return self.key_types.get(key)

    def observe_value_for_key(self, key, value):
        self.key_types.setdefault(key, None)
        value_type = _value_type(value)
        if value_type is None:
            return
        wanted_type = self.value_type_for_key(key)
        if wanted_type is None:
            self.key_types[key] = value_type
        else:
            if wanted_type != value_type:
                err = ("Invalid value %r for key %s: wanted %s"
                       % (value, key, wanted_type))
                raise TypeError(err)

class Events(object):
    def __init__(self):
        self._interval_magnitudes = set()
        self._connection = sqlite3.connect(":memory:")

        # Every time 'op_count' passes 'analyze_threshold', we run ANALYZE and
        # double 'analyze_threshold'. Starting analyze_threshold as 256 or so
        # would make more sense, but the extra cost of doing it this way is
        # minimal, and this way we actually test this logic even on small
        # examples.
        self._op_count = 0
        self._analyze_threshold = 1

        self._objtypes = {}

        # Allocating ids ourselves is better than letting sqlite do it,
        # because it allows us to do bulk inserts via executemany(), which is
        # must faster than calling execute() repeatedly.
        self._next_id = 0

        c = self._connection.cursor()
        c.execute("PRAGMA case_sensitive_like = true;")
        c.execute("PRAGMA foreign_keys = on;")
        self._objtypes["recspan_info"] = ObjType("recspan_info",
                                                 "sys_recspan_infos",
                                                 "recspan_id")
        c.execute("CREATE TABLE sys_recspan_infos "
                  "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                  "ticks INTEGER NOT NULL)"
                  );
        self._objtypes["event"] = ObjType("event", "sys_events", "id")
        c.execute("CREATE TABLE sys_events "
                  "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                  "recspan_id INTEGER NOT NULL, "
                  "start_tick NUMERIC NOT NULL, "
                  "stop_tick NUMERIC NOT NULL, "
                  "interval_magnitude INTEGER NOT NULL, "
                  "FOREIGN KEY(recspan_id) REFERENCES sys_recspan_infos(id))"
                  );
        c.execute("CREATE INDEX sys_events_by_start_tick "
                  "ON sys_events (recspan_id, start_tick);")
        c.execute("CREATE INDEX sys_events_by_stop_tick "
                  "ON sys_events (recspan_id, stop_tick);")
        # The special indices used to make overlaps queries fast
        c.execute("CREATE INDEX sys_events_interval_start_tick "
                  "ON sys_events (recspan_id, "
                                 "interval_magnitude, start_tick);")
        c.execute("CREATE INDEX sys_events_interval_stop_tick "
                  "ON sys_events (recspan_id, "
                                 "interval_magnitude, stop_tick);")

    def _execute(self, sql, args):
        c = self._connection.cursor()
        c.execute(sql, [_encode_sql_value(arg) for arg in args])
        # Weird things can happen to a cursor when other changes are made to
        # the db; e.g., merge_df sets new event attributes while iterating
        # over a query result, and it when this happened the cursor just
        # mysteriously stopped returning results. I don't understand the exact
        # situations under which this occurs, so to be safe, we just always
        # read out all db data immediately instead of letting true iterators
        # escape.
        return list(c)

    # WARNING: this should not be called inside a transaction, it might commit
    # it.
    def _incr_op_count(self, count=1):
        self._op_count += count
        if self._op_count >= self._analyze_threshold:
            self._connection.execute("ANALYZE;")
        self._analyze_threshold = 2 * self._op_count

    # WARNING: this will commit any ongoing transaction!
    def _ensure_table_for_key(self, objtype, key):
        # Non-trivial optimization when doing bulk inserts (e.g. adding
        # thousands of events when loading a file)
        if key in objtype.key_types:
            return
        table = objtype.table_name(key)
        self._connection.execute("CREATE TABLE IF NOT EXISTS %s ("
                                 "obj_id INTEGER PRIMARY KEY, "
                                 "value, "
                                 "FOREIGN KEY(obj_id) REFERENCES %s(id));"
                                 % (table, objtype.sys_table))
        self._connection.execute("CREATE INDEX IF NOT EXISTS %s_idx "
                                 "ON %s (value);" % (table, table))

    # WARNING: this neither commits nor creates the table if it doesn't exist,
    # it's your job to call _ensure_table_for_key and start a transaction
    # before calling this, and maybe call _incr_op_count after
    def _obj_setitem_core(self, objtype, id, key, value):
        table = objtype.table_name(key)
        objtype.observe_value_for_key(key, value)
        self._execute("INSERT OR REPLACE INTO %s (obj_id, value) "
                      "VALUES (?, ?);"
                      % (table,), (id, value))

    def add_events(self, recspan_ids, start_ticks, stop_ticks,
                   attributes):
        """Add a set of events in bulk.

        This method is substantially faster than repeated calls to
        add_event(), but it requires that all the events have the same types
        of attribute (though the actual attribute values may differ).

        Example::

          add_events([0, 1], [10, 10], [11, 11], {"x": [1, 2], "y": [3, 4]})

        :arg recspan_ids: An iterable of recspan ids, one per event.
        :arg start_ticks: An iterable of start sticks, one per event.
        :arg stop_ticks: An iterable of stop sticks, one per event.
        :arg attributes: A dict-like object who values are each an iterable of
          attribute values, one per event. (In particular, a pandas DataFrame
          will work well here.)
        """
        objtype = self._objtypes["event"]
        event_ids = range(self._next_id, self._next_id + len(recspan_ids))
        self._next_id += len(recspan_ids)
        interval_magnitudes = []
        for start_tick, stop_tick in zip(start_ticks, stop_ticks):
            if not start_tick < stop_tick:
                raise ValueError("start_tick must be < stop_tick")
            if not start_tick >= 0:
                raise ValueError("start_tick must be >= 0")
            interval_magnitude = approx_interval_magnitude(stop_tick - start_tick)
            interval_magnitudes.append(interval_magnitude)
        self._interval_magnitudes.update(interval_magnitudes)
        # Make sure everything is an int (skips _encode_sql_value)
        recspan_ids = [int(recspan_id) for recspan_id in recspan_ids]
        start_ticks = [int(tick) for tick in start_ticks]
        stop_ticks = [int(tick) for tick in stop_ticks]
        # Create tables up front before entering transaction:
        for key in attributes:
            self._ensure_table_for_key(objtype, key)
        with self._connection:
            try:
                self._connection.executemany(
                    "INSERT INTO sys_events "
                    "  (id, recspan_id, start_tick, stop_tick, "
                    "   interval_magnitude) "
                    "values (?, ?, ?, ?, ?)",
                    zip(event_ids, recspan_ids, start_ticks, stop_ticks,
                        interval_magnitudes))
            except sqlite3.IntegrityError:
                raise EventsError("undefined recspan")
            for column in attributes:
                sql_values = _encode_seq_to_sql_values(attributes[column])
                table = objtype.table_name(column)
                for value in attributes[column]:
                    objtype.observe_value_for_key(column, value)
                self._connection.executemany(
                    "INSERT INTO %s (obj_id, value) VALUES (?, ?)" % (table,),
                    zip(event_ids, sql_values))
        self._incr_op_count(len(recspan_ids))

    def add_event(self, recspan_id, start_tick, stop_tick, attributes):
        objtype = self._objtypes["event"]
        event_id = self._next_id
        self.add_events([recspan_id], [start_tick], [stop_tick],
                        {key: [val] for (key, val) in attributes.iteritems()})
        return Event(self, event_id)

    def add_recspan_info(self, recspan_id, ticks, attributes):
        objtype = self._objtypes["recspan_info"]
        # Create tables up front before entering transaction:
        for key in attributes:
            self._ensure_table_for_key(objtype, key)
        with self._connection:
            self._execute(
              "INSERT INTO sys_recspan_infos (id, ticks) values (?, ?)",
              [recspan_id, ticks])
            for key, value in attributes.iteritems():
                self._obj_setitem_core(objtype, recspan_id, key, value)
        self._incr_op_count()
        return RecspanInfo(self, recspan_id)

    def _delete_obj(self, objtype, obj_id):
        with self._connection:
            for key in objtype.key_types:
                self._execute("DELETE FROM %s WHERE obj_id = ?;"
                              % (objtype.table_name(key),),
                              (obj_id,))
            self._execute("DELETE FROM %s WHERE id = ?;"
                          % (objtype.sys_table,), (obj_id,))
        self._incr_op_count()

    def _obj_index_field(self, objtype, id, field):
        code = "SELECT %s FROM %s WHERE id = ?;" % (field, objtype.sys_table)
        results = self._execute(code, (id,))
        assert len(results) == 1
        return _decode_sql_value(results[0][0])

    def _obj_setitem(self, objtype, id, key, value):
        self._ensure_table_for_key(objtype, key)
        with self._connection:
            try:
                self._obj_setitem_core(objtype, id, key, value)
            except sqlite3.IntegrityError:
                raise EventsError("event no longer exists")
        self._incr_op_count()

    def _obj_getitem(self, objtype, id, key):
        table = objtype.table_name(key)
        code = "SELECT value FROM %s WHERE obj_id = ?;" % (table,)
        try:
            results = self._execute(code, (id,))
        except sqlite3.OperationalError:
            # The table doesn't exist:
            raise KeyError, key
        # The table exists, but has no entry with the given obj_id:
        if not results:
            raise KeyError, key
        return _sql_value_to_value_type(_decode_sql_value(results[0][0]),
                                        objtype.key_types[key])

    def _obj_delitem(self, objtype, id, key):
        # Raise a KeyError if it doesn't exist:
        self._obj_getitem(objtype, id, key)
        # Okay, now delete it
        table = objtype.table_name(key)
        code = "DELETE FROM %s WHERE obj_id = ?;" % (table,)
        with self._connection:
            self._execute(code, (id,))
        self._incr_op_count()

    def _obj_exists(self, objtype, id):
        code = "SELECT COUNT(*) FROM %s WHERE id = ?" % (objtype.sys_table,)
        results = self._execute(code, (id,))
        return bool(results[0][0])

    def _move_event(self, id, offset):
        # Fortunately, this operation doesn't change the magnitude of the
        # span, so it's relatively simple to implement -- we can just update
        # the start_tick and stop_tick directly without having to worry about
        # the complexities of interval_magnitudes.
        with self._connection:
            self._execute("UPDATE sys_events "
                          "SET start_tick = start_tick + ?, "
                          "    stop_tick = stop_tick + ? "
                          "WHERE id = ?",
                          [offset, offset, id])

    def placeholder_event(self):
        return PlaceholderEvent(self)

    def events_query(self, restrict=None):
        """restrict can be {"a": 1, "b": 2} or a Query or a string or a bool
        or None to mean "all"
        """
        if isinstance(restrict, Query):
            if restrict._events is not self:
                raise ValueError("query object does not refer to this Events "
                                 "object")
            return restrict
        elif isinstance(restrict, dict):
            p = self.placeholder_event()
            query = LiteralQuery(self, True)
            equalities = []
            for query_name in _magic_query_strings.intersection(restrict):
                q = _magic_query_string_to_query(self, query_name)
                query &= (q == restrict.pop(query_name))
            for k, v in restrict.iteritems():
                query &= (p[k] == v)
            return query
        elif isinstance(restrict, basestring):
            return _query_from_string(self, restrict)
        elif isinstance(restrict, bool):
            return LiteralQuery(self, restrict)
        elif restrict is None:
            return LiteralQuery(self, True)
        else:
            raise ValueError("I don't know how to interpret %r as an event "
                             "query" % (restrict,))

    def _query(self, sql_where, query_tables, query_vals):
        tables = set(["sys_events"])
        tables.update(query_tables)
        joins = []
        for objtype, objtype_tables in sql_where.attr_tables.iteritems():
            for table in objtype_tables:
                tables.add(table)
                joins.append("%s.obj_id == sys_events.%s"
                             % (table, objtype.event_join_field))
        code = ("SELECT %s FROM %s WHERE (%s) "
                % (", ".join(query_vals),
                   ", ".join(tables),
                   sql_where.code))
        if joins:
            code += " AND (%s)" % (" AND ".join(joins),)
        code += "ORDER BY sys_events.recspan_id, sys_events.start_tick"
        return self._execute(code, sql_where.args)

    # This is called directly by the test code, but is not really public.
    def _all_recspan_infos(self):
        for row in self._execute("SELECT id FROM sys_recspan_infos ORDER BY id",
                                 ()):
            yield RecspanInfo(self, _decode_sql_value(row[0]))

    def __repr__(self):
        return "<%s object with %s entries>" % (self.__class__.__name__,
                                                len(self))

    # Pickling
    def __getstate__(self):
        recspans = []
        for recspan in self._all_recspan_infos():
            recspans.append((recspan.id, recspan.ticks, dict(recspan)))
        events = []
        for ev in self.events_query(True):
            events.append((ev.recspan_id, ev.start_tick, ev.stop_tick, dict(ev)))
        # 0 as an ad-hoc version number in case we need to change this later
        return (0, recspans, events)

    def __setstate__(self, state):
        if state[0] != 0:
            raise ValueError, "unrecognized pickle data version for Events object"
        self.__init__()
        _, recspans, events = state
        for recspan_id, ticks, attrs in recspans:
            self.add_recspan_info(recspan_id, ticks, attrs)
        for recspan_id, start_tick, stop_tick, attrs in events:
            self.add_event(recspan_id, start_tick, stop_tick, attrs)

################################################################
## Objects representing single events/recspans
################################################################

class _Obj(object):
    def __init__(self, events, objtype, obj_id):
        self._events = events
        self._objtype = objtype
        self._obj_id = obj_id

    def __eq__(self, other):
        return (type(self) is type(other)
                and self._events is other._events
                and self._objtype is other._objtype
                and self._obj_id is other._obj_id)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((id(self._events), id(self._objtype), self._obj_id))

    def __getstate__(self):
        raise ValueError("%s objects are not pickleable"
                         % (self.__class__.__name__,))

    def _index_field(self, field):
        return self._events._obj_index_field(self._objtype,
                                             self._obj_id, field)

    def __getitem__(self, key):
        return self._events._obj_getitem(self._objtype, self._obj_id, key)

    def __setitem__(self, key, value):
        self._events._obj_setitem(self._objtype, self._obj_id, key, value)

    def __delitem__(self, key):
        self._events._obj_delitem(self._objtype, self._obj_id, key)

    def delete(self):
        self._events._delete_obj(self._objtype, self._obj_id)

    def iteritems(self):
        for key in self._objtype.key_types:
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

    __repr__ = repr_pretty_delegate
    # For the IPython pretty-printer:
    #   http://ipython.org/ipython-doc/dev/api/generated/IPython.lib.pretty.html
    def _repr_pretty_(self, p, cycle):
        assert not cycle
        if not self._events._obj_exists(self._objtype, self._obj_id):
            p.text("<%s (deleted)>" % (self.__class__.__name__,))
        else:
            p.begin_group(2,
                          "<%s %s:"
                          % (self.__class__.__name__, self._repr_fragment()))
            p.breakable()
            p.pretty(dict(self.iteritems()))
            p.end_group(2, ">")

class Event(_Obj):
    def __init__(self, events, id):
        _Obj.__init__(self, events, events._objtypes["event"], id)

    @property
    def recspan_id(self):
        return self._index_field("recspan_id")

    @property
    def recspan_info(self):
        return RecspanInfo(self._events, self._index_field("recspan_id"))

    @property
    def start_tick(self):
        return self._index_field("start_tick")

    @property
    def stop_tick(self):
        return self._index_field("stop_tick")

    def overlaps(self, *args):
        if len(args) == 1:
            ev = args[0]
            return self.overlaps(ev.recspan_id,
                                 ev.start_tick, ev.stop_tick)
        else:
            (recspan_id, start_tick, stop_tick) = args
            return (self.recspan_id == recspan_id
                    and self.start_tick < stop_tick
                    and start_tick < self.stop_tick)

    def matches(self, query):
        # To find out whether this event is matched by the given query, we
        # constrain the query to return only this event, and then see if it
        # returns any events.
        query = self._events.events_query(query)
        query &= (IndexFieldQuery(self._events, "id") == self._obj_id)
        return bool(len(query))

    def relative(self, count, restrict=None):
        """Counts 'count' events forward or back from the current event (or
        optionally, only events that match 'restrict'), and returns that. Use
        negative for backwards."""
        if count == 0:
            raise IndexError, "count must be non-zero"
        query = self._events.events_query(restrict)
        p = self._events.placeholder_event()
        query &= (p.recspan_id == self.recspan_id)
        if count > 0:
            query &= (p.start_tick > self.start_tick)
            return list(query)[count - 1]
        else:
            query &= (p.start_tick < self.start_tick)
            return list(query)[count]

    def move(self, offset):
        """Shifts this event's timestamp by 'offset' samples (positive for
        forward, negative for backward)."""
        self._events._move_event(self._obj_id, offset)

    def _repr_fragment(self):
        return ("in recspan %s, ticks %s-%s"
                % (self.recspan_id, self.start_tick, self.stop_tick))

class RecspanInfo(_Obj):
    def __init__(self, events, id):
        _Obj.__init__(self, events, events._objtypes["recspan_info"], id)

    @property
    def id(self):
        return self._index_field("id")

    @property
    def ticks(self):
        return self._index_field("ticks")

    def _repr_fragment(self):
        return "%s with %s ticks" % (self._obj_id, self.ticks)

###############################################
#
# The query specification system (Python API)
#
###############################################

# Keys that have a special magical meaning in dict-queries and
# string-queries (plus the actual database field name they map to).
# In string queries these can be de-magicalified by quoting them with
# backquotes, i.e.,:
#   "_RECSPAN_ID == 1"   <-> placeholder.recspan_id == 1
#   "`_RECSPAN_ID` == 1" <-> placeholder["_RECSPAN_ID"] == 1
_magic_query_strings = set(["_RECSPAN_ID", "_START_TICK", "_STOP_TICK"])
def _magic_query_string_to_query(events, name, origin=None):
    return IndexFieldQuery(events, name[1:].lower(), origin)

class _PlaceholderObj(object):
    def __init__(self, events, objtype):
        self._events = events
        self._objtype = objtype

    def __getitem__(self, key):
        return AttrQuery(self._events, self._objtype, key)

    def has_key(self, key):
        return HasKeyQuery(self._events, self._objtype, key)

    def __repr__(self):
        return "<%s>" % (self.__class__.__name__,)

class PlaceholderEvent(_PlaceholderObj):
    def __init__(self, events):
        _PlaceholderObj.__init__(self, events, events._objtypes["event"])

    @property
    def recspan_id(self):
        return IndexFieldQuery(self._events, "recspan_id")

    @property
    def start_tick(self):
        return IndexFieldQuery(self._events, "start_tick")

    @property
    def stop_tick(self):
        return IndexFieldQuery(self._events, "stop_tick")

    @property
    def recspan_info(self):
        return PlaceholderRecspanInfo(self._events)

    def overlaps(self, *args):
        if len(args) == 1:
            ev = args[0]
            if isinstance(ev, PlaceholderEvent):
                raise ValueError, "cannot test for overlap between two placeholders"
            return self.overlaps(ev.recspan_id,
                                 ev.start_tick, ev.stop_tick)
        else:
            (recspan_id, start_tick, stop_tick) = args
            return OverlapsQuery(self._events,
                                 recspan_id, start_tick, stop_tick)

    def matches(self, query):
        # Not terribly useful but included for completeness.
        return self._events.events_query(query)

class PlaceholderRecspanInfo(_PlaceholderObj):
    def __init__(self, events):
        _PlaceholderObj.__init__(self, events, events._objtypes["recspan_info"])

SqlWhere = namedtuple("SqlWhere", ["code", "attr_tables", "args"])

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
        return self._make_op("and", "AND", [self, other], _BOOL)

    def __or__(self, other):
        return self._make_op("or", "OR", [self, other], _BOOL)

    def __invert__(self):
        return self._make_op("not", "NOT", [self], _BOOL)

    def __nonzero__(self):
        raise TypeError("can't convert query directly to bool "
                        "(maybe you want a bitwise operator like & | ~ "
                        "instead of a logical operator like "
                        "'and' 'or' 'not')")

    def _value_type(self): # pragma: no cover
        assert False

    def _sql_where(self): # pragma: no cover
        assert False

    def __len__(self):
        if self._value_type() is not _BOOL:
            raise EventsError("top-level query must be boolean", self)
        c = self._events._query(self._sql_where(), [], ["count(*)"])
        for (count,) in c:
            return count

    def __iter__(self):
        if self._value_type() is not _BOOL:
            raise EventsError("top-level query must be boolean", self)
        db_ids = self._events._query(self._sql_where(),
                                     ["sys_events"],
                                     ["sys_events.id"])
        for (db_id,) in db_ids:
            yield Event(self._events, _decode_sql_value(db_id))

class LiteralQuery(Query):
    def __init__(self, events, value, origin=None):
        Query.__init__(self, events, origin)
        self._value = value
        try:
            self._saved_value_type = _value_type(self._value)
        except ValueError:
            raise EventsError("literals must be boolean, string, numeric, "
                              "or None",
                              self)

    def _sql_where(self):
        return SqlWhere("?", {}, [self._value])

    def _value_type(self):
        return self._saved_value_type

    def __repr__(self):
        return "<%s: %r>" % (self.__class__.__name__, self._value)

class AttrQuery(Query):
    def __init__(self, events, objtype, key, origin=None):
        Query.__init__(self, events, origin)
        self._key = key
        self._objtype = objtype
        # Go ahead and create any tables that are being queried -- if someone
        # makes a typo then this will create an empty table, but creating a
        # table by itself is a total no-op at the semantic level, so whatever.
        self._events._ensure_table_for_key(objtype, self._key)

    def exists(self):
        return HasKeyQuery(self._events, self._objtype,
                           self._key, self.origin)

    @memoized_method
    def _sql_where(self):
        table_id = self._objtype.table_name(self._key)
        return SqlWhere(table_id + ".value",
                        {self._objtype: frozenset([table_id])},
                        [])

    def _value_type(self):
        return self._objtype.value_type_for_key(self._key)

    def __repr__(self):
        return "<%s %s %r>" % (self.__class__.__name__,
                               self._objtype.name,
                               self._key)

class HasKeyQuery(Query):
    def __init__(self, events, objtype, key, origin=None):
        Query.__init__(self, events, origin)
        self._key = key
        self._objtype = objtype
        # Go ahead and create any tables that are being queried -- if someone
        # makes a typo then this will create an empty table, but creating a
        # table by itself is a total no-op at the semantic level, so whatever.
        self._events._ensure_table_for_key(objtype, self._key)

    @memoized_method
    def _sql_where(self):
        table = self._objtype.table_name(self._key)
        return SqlWhere("EXISTS (SELECT 1 FROM %s inner_table "
                        "WHERE sys_events.%s == inner_table.obj_id)"
                        % (table, self._objtype.event_join_field),
                        {}, [])

    def _value_type(self):
        return _BOOL

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__name__, self._key)

# For now we only support queries on Event index fields, not RecspanInfo index
# fields. The only RecspanInfo index field is 'ticks', and can't really see
# why that would be useful, so whatever.
class IndexFieldQuery(Query):
    def __init__(self, events, field, origin=None):
        Query.__init__(self, events, origin)
        self._field = field

    def _sql_where(self):
        # sys_events is always included in the join, so no need to add it to
        # the tables.
        return SqlWhere("sys_events.%s" % (self._field,), {}, [])

    def _value_type(self):
        return _NUMERIC

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__name__, self._field)

class OverlapsQuery(Query):
    def __init__(self, events,
                 recspan_id, start_tick, stop_tick, origin=None):
        Query.__init__(self, events, origin)
        self._recspan_id = recspan_id
        self._start_tick = start_tick
        self._stop_tick = stop_tick

    @memoized_method
    def _sql_where(self):
        args = []
        possibilities = []
        for magnitude in self._events._interval_magnitudes:
            constraints = [
                "sys_events.recspan_id == ?",
                "sys_events.interval_magnitude == ?",
                "((sys_events.start_tick < ? AND (? - ?) < sys_events.start_tick)"
                  " OR (sys_events.stop_tick < (? + ?) AND ? < sys_events.stop_tick))",
                ]
            args += [self._recspan_id,
                     magnitude,
                     self._stop_tick, self._start_tick, magnitude,
                     self._stop_tick, magnitude, self._start_tick]
            possibilities.append(" AND ".join(constraints))
        return SqlWhere(" OR ".join(possibilities), {}, args)

    def _value_type(self):
        return _BOOL

    def __repr__(self):
        return ("<%s %r %r [%r, %r)>"
                % (self.__class__.__name__, self._recspan_id,
                   self._start_tick, self._stop_tick))

class QueryOperator(Query):
    def __init__(self, events, sql_op, children, origin=None):
        Query.__init__(self, events, origin)
        self._sql_op = sql_op
        assert 1 <= len(children) <= 2
        self._children = children

    @memoized_method
    def _sql_where(self):
        lhs_sqlwhere = self._children[0]._sql_where()
        if len(self._children) == 1:
            # Special case for NOT
            rhs_sqlwhere = lhs_sqlwhere
            lhs_sqlwhere = SqlWhere("", {}, [])
        else:
            rhs_sqlwhere = self._children[1]._sql_where()
        new_attr_tables = {}
        for attr_tables in [lhs_sqlwhere.attr_tables,
                            rhs_sqlwhere.attr_tables]:
            for objtype, table_set in attr_tables.iteritems():
                new_attr_tables[objtype] = table_set.union(
                    new_attr_tables.get(objtype, []))
        return SqlWhere("(%s %s %s)"
                       % (lhs_sqlwhere.code, self._sql_op, rhs_sqlwhere.code),
                        new_attr_tables,
                        lhs_sqlwhere.args + rhs_sqlwhere.args)

    def _value_type(self):
        return _BOOL

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
    Operator(".", 2, 200),
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

_atomic = ["ATTR", "LITERAL", "MAGIC_FIELD", "_RECSPAN_INFO"]

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
            elif token == "_RECSPAN_INFO":
                yield Token("_RECSPAN_INFO", origin, token)
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
    elif tree.type == ".":
        if tree.args[0].type != "_RECSPAN_INFO":
            raise EventsError("left argument of '.' must be _RECSPAN_INFO",
                              tree.args[0].origin)
        if tree.args[1].type != "ATTR":
            raise EventsError("right arguments of '.' must be attribute",
                              tree.args[1].origin)
        return AttrQuery(events, events._objtypes["recspan_info"],
                         tree.args[1].token.extra, tree.origin)
    elif tree.type == "has":
        assert len(tree.args) == 1
        eval_arg = _eval(events, tree.args[0])
        if not isinstance(eval_arg, AttrQuery):
            raise EventsError("argument of 'has' must be an attribute",
                              tree.args[0].origin)
        return eval_arg.exists()
    elif tree.type == "LITERAL":
        return LiteralQuery(events, tree.token.extra, tree.origin)
    elif tree.type == "ATTR":
        return AttrQuery(events, events._objtypes["event"],
                         tree.token.extra, tree.origin)
    elif tree.type == "MAGIC_FIELD":
        return _magic_query_string_to_query(events, tree.token.extra,
                                            tree.origin)
    elif tree.type == "_RECSPAN_INFO":
        raise EventsError("_RECSPAN_INFO must appear on the left side of '.'",
                          tree.origin)
    else: # pragma: no cover
        assert False

def _query_from_string(events, string):
    return _eval(events, infix_parse(_tokenize(string), _ops, _atomic))
