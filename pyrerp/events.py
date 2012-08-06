import sqlite3
import string
import re
import struct
from cStringIO import StringIO
import numpy as np
from patsy import PatsyError
from patsy.origin import Origin
from patsy.parse_core import Token, Operator, parse

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

def _table_name(key):
    return "attr_" + _munge_name(key)

class Events(object):
    NUMERIC = "numeric"
    BLOB = "text"
    BOOL = "bool"

    # Index type is like (int, int) or (str, int, int) or int, etc.
    def __init__(self, index_type):
        self._index_type = index_type
        self._connection = sqlite3.connect(":memory:")
        self._keys = set()

        c = self._connection.cursor()
        c.execute("PRAGMA case_sensitive_like = true;")
        c.execute("CREATE TABLE sys_events "
                  "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                  "event_index BLOB NOT NULL);")
        c.execute("CREATE TABLE sys_key_types (key PRIMARY KEY, type)");
        c.execute("CREATE INDEX sys_events_index_idx "
                  "ON sys_events (event_index);")

    # The idea here is to encode/decode the index structures we care about
    # (tuples of ints and strings) in a way that is (1) unique, (2) sorts
    # properly in the database (so >, < comparisons work).
    def _encode_index(self, idx, origin=None):
        buf = StringIO()
        index_type = self._index_type
        if not isinstance(index_type, tuple):
            index_type = (index_type,)
            idx = (idx,)
        if not isinstance(idx, tuple) or len(idx) != len(index_type):
            raise EventsError("wrong length index %r" % (idx,), origin)
        for (obj, type) in zip(idx, index_type):
            if not isinstance(obj, type):
                raise EventsError("index object %r not of type %s"
                                  % (obj, type), origin)
            try:
                if type is int:
                    # inverted sign bit goes first so that negatives will
                    # compare smaller than positives
                    buf.write(struct.pack(">Bq", obj >= 0, obj))
                elif type is str:
                    buf.write(struct.pack(">I", len(obj)))
                    buf.write(obj)
            except struct.error:
                raise EventsError("bad index value: wanted %s, got %r"
                                  % (type, obj), origin)
        return buf.getvalue()

    def _decode_index(self, idx_str):
        idx = []
        offset = 0
        index_type = self._index_type
        if not isinstance(index_type, tuple):
            index_type = (index_type,)
        for type in index_type:
            if type is int:
                idx.append(struct.unpack_from(">Bq", idx_str, offset)[1])
                offset += struct.calcsize(">Bq")
            elif type is str:
                length = struct.unpack_from(">I", idx_str, offset)[0]
                offset += struct.calcsize(">I")
                idx.append(idx_str[offset:offset + length])
                offset += length
        if not isinstance(self._index_type, tuple):
            return idx[0]
        else:
            return tuple(idx)

    # Convert str's to buffer objects before passing them into sqlite, because
    # that is how you tell the sqlite3 module to store them as
    # BLOBs. (sqlite3.Binary is an alias for 'buffer'.) This encoding/decoding
    # is layered *on top* of index encoding/decoding, if that is in use. This
    # function also handles converting numpy scalars into equivalents that are
    # acceptable to sqlite.
    def _encode_value(self, val):
        if np.issubsctype(type(val), np.str_):
            return sqlite3.Binary(val)
        elif np.issubsctype(type(val), np.integer):
            return int(val)
        elif np.issubsctype(type(val), np.floating):
            return float(val)
        else:
            return val

    # And reverse the transformation on the way out.
    def _decode_value(self, val):
        if isinstance(val, sqlite3.Binary):
            return str(val)
        else:
            return val

    def _execute(self, c, sql, args):
        return c.execute(sql, [self._encode_value(arg) for arg in args])

    def _value_type_for_key(self, key):
        c = self._connection.cursor()
        code = "SELECT type FROM sys_key_types WHERE key = ?;"
        self._execute(c, code, (key,))
        row = c.fetchone()
        if row is None:
            return None
        return self._decode_value(row[0])

    def _value_type(self, value):
        # must come first, because issubclass(bool, int)
        if isinstance(value, bool):
            return self.BOOL
        elif isinstance(value, (int, long, float)):
            return self.NUMERIC
        elif isinstance(value, (str, unicode)):
            return self.BLOB
        elif value is None:
            return None
        else:
            raise ValueError, ("Invalid value %r: "
                               "must be a string, number, bool, or None"
                               % (value,))

    def _observe_value_for_key(self, key, value):
        value_type = self._value_type(value)
        if value_type is None:
            return
        wanted_type = self._value_type_for_key(key)
        if wanted_type is None:
            c = self._connection.cursor()
            self._execute(c,
                          "INSERT INTO sys_key_types (key, type) "
                            "VALUES (?, ?);",
                          (key, value_type))
        else:
            if wanted_type != value_type:
                err = ("Invalid value %r for key %s: wanted %s"
                       % (value, key, wanted_type))
                raise ValueError, err

    # NOTE: DDL statements like table creation cannot be performed inside a
    # transaction. So don't call this method inside a transaction.
    def _ensure_table(self, key):
        table = _table_name(key)
        self._connection.execute("CREATE TABLE IF NOT EXISTS %s ("
                                   "event_id INTEGER PRIMARY KEY, "
                                   "value);" % (table,))
        self._connection.execute("CREATE INDEX IF NOT EXISTS %s_idx "
                                   "ON %s (value);" % (table, table))
        self._keys.add(key)

    def add_event(self, index, info):
        # Create tables up front before entering transaction:
        for key in info:
            self._ensure_table(key)
        with self._connection:
            c = self._connection.cursor()
            self._execute(c, "INSERT INTO sys_events (event_index) values (?)",
                          [self._encode_index(index)])
            event_id = c.lastrowid
            for key, value in info.iteritems():
                self._observe_value_for_key(key, value)
                self._execute(c,
                              "INSERT INTO %s (event_id, value) VALUES (?, ?);"
                                % (_table_name(key),),
                              (event_id, value))
            return Event(self, event_id)

    def _delete_event(self, id):
        with self._connection:
            c = self._connection.cursor()
            self._execute(c, "DELETE FROM sys_events WHERE id = ?;", (id,))
            for key in self._keys:
                self._execute(c,
                              "DELETE FROM %s WHERE event_id = ?;"
                                % (_table_name(key),),
                              (id,))

    def _event_index(self, id):
        c = self._connection.cursor()
        code = "SELECT event_index FROM sys_events WHERE id = ?;"
        self._execute(c, code, (id,))
        return self._decode_index(self._decode_value(c.fetchone()[0]))

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
        return self._decode_value(row[0])

    def _event_setitem(self, id, key, value):
        self._ensure_table(key)
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

    def _event_delitem(self, id, key):
        # Raise a KeyError if it doesn't exist:
        self._event_getitem(id, key)
        # Okay, now delete it
        c = self._connection.cursor()
        table = _table_name(key)
        code = "DELETE FROM %s WHERE event_id = ?;" % (table,)
        self._execute(c, code, (id,))

    @property
    def placeholder(self):
        return PlaceholderEvent(self)

    @property
    def ANY(self):
        return LiteralQuery(self, True)

    def find(self, *args, **kwargs):
        "Usage: find(a=1, b=2) or find(query_obj) or find(\"string\") or find()"
        if not args and not kwargs:
            args = (self.ANY,)
        if len(args) == 0 and kwargs:
            p = self.placeholder
            equalities = []
            if "INDEX" in kwargs:
                equalities.append(p.index == kwargs.pop("INDEX"))
            for k, v in kwargs.iteritems():
                equalities.append(p[k] == v)
            query = reduce(lambda a, b: a & b, equalities)
            args = [query]
            kwargs = {}
        if not (len(args) == 1 and not kwargs):
            raise TypeError("Usage: find(a=1, b=2) or find(query_obj) "
                            "or find(\"query string\")")
        query = args[0]
        if isinstance(query, basestring):
            query = query_from_string(self, query)
        if not isinstance(query, Query):
            raise ValueError, "expected a query object, not %r" % (query,)
        if query._value_type() != self.BOOL:
            raise EventsError("query object must be boolean", query)
        if query._events is not self:
            raise ValueError("query object does not refer to this Events "
                             "object")
        sql_expr = query._sql_expr()
        sql_expr.tables.add("sys_events")
        joins = ["%s.event_id == sys_events.id" % (table,)
                 for table in sql_expr.tables if table != "sys_events"]
        c = self._connection.cursor()
        code = ("SELECT sys_events.id FROM %s WHERE (%s) "
                % (", ".join(sql_expr.tables), sql_expr.code))
        if joins:
            code += " AND (%s)" % (" AND ".join(joins))
        code += "ORDER BY sys_events.event_index"
        #print code, sql_expr.args
        self._execute(c, code, sql_expr.args)
        for (db_id,) in c:
            yield Event(self, self._decode_value(db_id))

    def at(self, index):
        return list(self.find(INDEX=index))

    def __iter__(self):
        return self.find(self.ANY)

    def __len__(self):
        c = self._connection.cursor()
        self._execute(c, "SELECT count(*) FROM sys_events;", [])
        return self._decode_value(c.fetchone()[0])

    def __repr__(self):
        return "<%s object with %s entries>" % (self.__class__.__name__,
                                                len(self))

    # Pickling
    def __getstate__(self):
        events = []
        for ev in self:
            events.append((ev.index, dict(ev)))
        # 0 as an ad-hoc version number in case we need to change this later
        return (0, self._index_type, events)

    def __setstate__(self, state):
        if state[0] != 0:
            raise ValueError, "unrecognized pickle data version for Events object"
        _, index_type, events = state
        self.__init__(index_type)
        for index, attrs in events:
            self.add_event(index, attrs)

class Event(object):
    def __init__(self, events, id):
        self._events = events
        self._id = id

    def __getstate__(self):
        raise ValueError, "Event objects are not pickleable"

    @property
    def index(self):
        return self._events._event_index(self._id)

    def __getitem__(self, key):
        return self._events._event_getitem(self._id, key)

    def __setitem__(self, key, value):
        self._events._event_setitem(self._id, key, value)

    def __delitem__(self, key):
        self._events._event_delitem(self._id, key)

    def delete(self):
        self._events._delete_event(self._id)

    def iteritems(self):
        for key in self._events._keys:
            try:
                yield key, self[key]
            except KeyError:
                continue
            
    # Everything else is defined in terms of the above methods.

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

    def __repr__(self):
        return "<%s %s at %s: %s>" % (
            self.__class__.__name__, self._id, self.index,
            repr(dict(self.iteritems())))

    # For the IPython pretty-printer:
    #   http://ipython.org/ipython-doc/dev/api/generated/IPython.lib.pretty.html
    def _repr_pretty_(self, p, cycle):
        assert not cycle
        p.begin_group(2, "<%s at %s:" % (self.__class__.__name__, self.index))
        p.breakable()
        p.pretty(dict(self.iteritems()))
        p.end_group(2, ">")
            

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
    def index(self):
        return IndexQuery(self._events)

class SqlExpr(object):
    def __init__(self, code, tables, args):
        self.code = code
        self.tables = tables
        self.args = args

class Query(object):
    def __init__(self, events, origin=None):
        self._events = events
        self.origin = origin

    def _as_constraint(self, other):
        if isinstance(other, Query):
            assert other._events is self._events
            return other
        return LiteralQuery(self._events, other)

    def _make_op(self, op_name, sql_op, children, expected_type=None):
        children = [self._as_constraint(child) for child in children]
        if any([isinstance(child, IndexQuery) for child in children]):
            # Special case: in operations involving a LiteralQuery plus an
            # IndexQuery, convert the LiteralQuery into a LiteralIndexQuery:
            for i, child in enumerate(children):
                if isinstance(child, LiteralQuery):
                    children[i] = LiteralIndexQuery(child._events,
                                                    child._value,
                                                    child.origin)
        else:
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

    def _value_type(self):
        assert False

    def _sql_expr(self):
        assert False

class LiteralQuery(Query):
    def __init__(self, events, value, origin=None):
        Query.__init__(self, events, origin)
        self._value = value

    def _sql_expr(self):
        if (self._value is not None
            and not isinstance(self._value, (bool, basestring, int, float))):
            raise EventsError("literals must be boolean, string, numeric, "
                                "or None",
                              self)
        return SqlExpr("?", set(), [self._value])

    def _value_type(self):
        return self._events._value_type(self._value)

    def __repr__(self):
        return "<%s: %r>" % (self.__class__.__name__, self._value)

class LiteralIndexQuery(LiteralQuery):
    def _sql_expr(self):
        return SqlExpr("?", set(),
                       [self._events._encode_index(self._value, self.origin)])

    def _value_type(self):
        return None

class AttrQuery(Query):
    def __init__(self, events, name, origin=None):
        Query.__init__(self, events, origin)
        self._name = name

    def _sql_expr(self):
        table_id = _table_name(self._name)
        return SqlExpr(table_id + ".value", set([table_id]), [])

    def _value_type(self):
        return self._events._value_type_for_key(self._name)

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__name__, self._name)

class IndexQuery(Query):
    def _sql_expr(self):
        # sys_events is always included in the join, so no need to add it to
        # the tables.
        return SqlExpr("sys_events.event_index", set(), [])

    def _value_type(self):
        return None

    def __repr__(self):
        return "<%s>" % (self.__class__.__name__,)

class QueryOperator(Query):
    def __init__(self, events, sql_op, children, origin=None):
        Query.__init__(self, events, origin)
        self._sql_op = sql_op
        assert 1 <= len(children) <= 2
        self._children = children
        
    def _sql_expr(self):
        lhs_sqlexpr = self._children[0]._sql_expr()
        if len(self._children) == 1:
            # Special case for NOT
            rhs_sqlexpr = lhs_sqlexpr
            lhs_sqlexpr = SqlExpr("", set(), [])
        else:
            rhs_sqlexpr = self._children[1]._sql_expr()
        return SqlExpr("(%s %s %s)"
                       % (lhs_sqlexpr.code, self._sql_op, rhs_sqlexpr.code),
                       lhs_sqlexpr.tables.union(rhs_sqlexpr.tables),
                       lhs_sqlexpr.args + rhs_sqlexpr.args)

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
    Operator(",", 2, 200),
    ]
_text_ops = [
    Operator("not", 1, 0),
    Operator("and", 2, 0),
    Operator("or", 2, 0),
    ]
_ops = _punct_ops + _text_ops

_atomic = ["ATTR", "LITERAL", "INDEX"]

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
            elif token == "INDEX":
                yield Token("INDEX", origin)
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
    elif tree.type == "LITERAL":
        return LiteralQuery(events, tree.token.extra, tree.origin)
    elif tree.type == "ATTR":
        return AttrQuery(events, tree.token.extra, tree.origin)
    elif tree.type == "INDEX":
        return IndexQuery(events, tree.origin)
    elif tree.type == ",":
        eval_args = [_eval(events, arg) for arg in tree.args]
        for arg in eval_args:
            if not isinstance(arg, LiteralQuery):
                raise EventsError("comma operator can only be applied "
                                  "to literals, not %s" % (arg,), arg)
        # Flatten out nested calls like "1, 2, 3":
        def tuplify(v):
            if isinstance(v, tuple):
                return v
            return (v,)
        twople = tuplify(eval_args[0]._value) + tuplify(eval_args[1]._value)
        return LiteralQuery(events, twople)
    else:
        assert False

def query_from_string(events, string):
    return _eval(events, parse(_tokenize(string), _ops, _atomic))
