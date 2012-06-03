import sqlite3
import string
import re
from charlton import CharltonError
from charlton.origin import Origin
from charlton.parse_core import Token, Operator, parse

__all__ = ["Events"]

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

def _index(obj, origin=None):
    if isinstance(obj, (int, long)):
        return (0, obj)
    elif (isinstance(obj, tuple)
          and len(obj) == 2
          and isinstance(obj[0], (int, long))
          and isinstance(obj[1], (int, long))):
        return obj
    else:
        raise CharltonError("bad index: %s" % (obj,), origin)

def test__index():
    assert _index(1) == (0, 1)
    assert _index(10) == (0, 10)
    assert _index((10, 20)) == (10, 20)

class Events(object):
    NUMERIC = "numeric"
    BLOB = "text"
    BOOL = "bool"

    def __init__(self, era_lens):
        self._era_lens = era_lens
        self._connection = sqlite3.connect(":memory:")
        self._keys = set()

        c = self._connection.cursor()
        c.execute("PRAGMA case_sensitive_like = true;")
        c.execute("CREATE TABLE sys_events "
                  "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                  "era INTEGER NOT NULL, offset INTEGER NOT NULL);")
        c.execute("CREATE TABLE sys_key_types (key PRIMARY KEY, type)");
        c.execute("CREATE INDEX sys_events_era_offset_idx ON sys_events "
                  "(era, offset);")

    def _value_type(self, key, value):
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
            raise ValueError, ("Invalid value %r for key %s: "
                               "must be a string, number, bool, or None"
                               % (value, key))

    def _value_type_for_key(self, key):
        c = self._connection.cursor()
        c.execute("SELECT type FROM sys_key_types WHERE key = ?;", (key,))
        row = c.fetchone()
        if row is None:
            return None
        return row[0]

    def _observe_value_for_key(self, key, value):
        value_type = self._value_type(key, value)
        if value_type is None:
            return
        wanted_type = self._value_type_for_key(key)
        if wanted_type is None:
            c = self._connection.cursor()
            c.execute("INSERT INTO sys_key_types (key, type) VALUES (?, ?);",
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
        era, offset = _index(index)
        if era >= len(self._era_lens):
            raise ValueError, "Bad era index: %s" % (era,)
        if offset >= self._era_lens[era]:
            raise ValueError, "Offset %s is past end of era" % (offset,)
        # Create tables up front before entering transaction:
        for key in info:
            self._ensure_table(key)
        with self._connection:
            c = self._connection.cursor()
            assert isinstance(era, (int, long))
            assert isinstance(offset, (int, long))
            c.execute("INSERT INTO sys_events (era, offset) values (?, ?)",
                      (era, offset))
            event_id = c.lastrowid
            for key, value in info.iteritems():
                value_type = self._value_type(key, value)
                self._observe_value_for_key(key, value)
                c.execute("INSERT INTO %s (event_id, value) VALUES (?, ?);"
                          % (_table_name(key),), (event_id, value))
            return Event(self, event_id)

    def _delete_event(self, id):
        with self._connection:
            c = self._connection.cursor()
            c.execute("DELETE FROM sys_events WHERE id = ?;", (id,))
            for key in self._keys:
                c.execute("DELETE FROM %s WHERE event_id = ?;"
                          % (_table_name(key),), (id,))

    def _event_index(self, id):
        c = self._connection.cursor()
        c.execute("SELECT era, offset FROM sys_events WHERE id = ?;", (id,))
        return c.fetchone()

    def _event_select_attr(self, id, key):
        c = self._connection.cursor()
        table = _table_name(key)
        c.execute("SELECT value FROM %s WHERE event_id = ?;" % (table,), (id,))
        return tuple(c.fetchone())

    def _event_set_attr(self, id, key, value):
        self._ensure_table(key)
        with self._connection:
            c = self._connection.cursor()
            table = _table_name(key)
            self._observe_value_for_key(key, value)        
            c.execute("UPDATE %s SET value = ? WHERE event_id = ?;"
                      % (table,), (value, id))
            if c.rowcount == 0:
                c.execute("INSERT INTO %s (event_id, value) VALUES (?, ?);"
                          % (table,), (id, value))

    @property
    def placeholder(self):
        return PlaceholderEvent(self)

    @property
    def ANY(self):
        return LiteralQuery(self, True)

    def find(self, *args, **kwargs):
        "Usage: either find(a=1, b=2) or find(query_obj) or find(\"string\")"
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
        if not query._is_bool():
            raise CharltonError("query object must be boolean", query)
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
        code += "ORDER BY sys_events.era, sys_events.offset"
        #print code, sql_expr.args
        c.execute(code, sql_expr.args)
        for (id,) in c:
            yield Event(self, id)

    def at(self, index):
        return self.find(INDEX=index)

    def __iter__(self):
        return self.find(self.ANY)

    def __len__(self):
        c = self._connection.cursor()
        c.execute("SELECT count(*) FROM sys_events;")
        return c.fetchone()[0]

    def __repr__(self):
        return "<%s object with %s entries>" % (self.__class__.__name__,
                                                len(self))

class Event(object):
    def __init__(self, events, id):
        self._events = events
        self._id = id

    @property
    def index(self):
        return self._events._event_index(self._id)

    def __getitem__(self, key):
        try:
            row = self._events._event_select_attr(self._id, key)
        except sqlite3.OperationalError:
            raise KeyError, key
        if row is None:
            raise KeyError, key
        return row[0]

    def __setitem__(self, key, value):
        self._events._event_set_attr(self._id, key, value)

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

    def __contains__(self):
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
            return other
        return LiteralQuery(self._events, other)

    def _make_op(self, sql_op, children, children_bool):
        children = [self._as_constraint(child) for child in children]
        for child in children:
            if child._is_bool() != children_bool:
                if children_bool:
                    expected = "boolean"
                else:
                    expected = "non-boolean"
                msg = ("%s operator expected %s arguments "
                       "(check your parentheses?)" % (sql_op, expected))
                raise CharltonError(msg, child)
            assert child._events is self._events
        return QueryOperator(self._events, sql_op, children,
                                  Origin.combine(children))

    def __eq__(self, other):
        return self._make_op("==", [self, other], False)

    def __ne__(self, other):
        return self._make_op("!=", [self, other], False)

    def __lt__(self, other):
        return self._make_op("<", [self, other], False)

    def __gt__(self, other):
        return self._make_op(">", [self, other], False)

    def __le__(self, other):
        return self._make_op("<=", [self, other], False)

    def __ge__(self, other):
        return self._make_op(">=", [self, other], False)

    def __and__(self, other):
        return self._make_op("AND", [self, other], True)

    def __or__(self, other):
        return self._make_op("OR", [self, other], True)

    def __invert__(self, other):
        return self._make_op("NOT", [self], True)

    def _is_bool(self):
        assert False

    def _sql_expr(self):
        assert False

class LiteralQuery(Query):
    def __init__(self, events, value, origin=None):
        Query.__init__(self, events, origin)
        self._value = value

    # So that "10 == index" will get processed by Index._make_op instead of
    # us:
    def _make_op(self, sql_op, children, children_bool):
        for child in children:
            if isinstance(child, Index):
                return NotImplemented
        return Query._make_op(self, sql_op, children, children_bool)

    def _sql_expr(self):
        if not isinstance(self._value, (bool, basestring, int, float)):
            raise CharltonError("literals must be boolean, strings, or numbers",
                                self)
        return SqlExpr("?", set(), [self._value])

    def _is_bool(self):
        return isinstance(self._value, bool)

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__name__, self._value)

class AttrQuery(Query):
    def __init__(self, events, name, origin=None):
        Query.__init__(self, events, origin)
        self._name = name

    def _sql_expr(self):
        table_id = _table_name(self._name)
        return SqlExpr(table_id + ".value", set([table_id]), [])

    def _is_bool(self):
        known_type = self._events._value_type_for_key(self._name)
        if known_type is None:
            raise CharltonError("unknown attribute %r" % (self._name), self)
        return known_type == Events.BOOL

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__name__, self._name)

class SysFieldQuery(Query):
    def _sql_expr(self):
        return SqlExpr("sys_events.%s" % (self._sys_field), set(), [])

    def _is_bool(self):
        return False

    def __repr__(self):
        return "<%s>" % (self.__class__.__name__,)

class EraQuery(SysFieldQuery):
    _sys_field = "era"

class OffsetQuery(SysFieldQuery):
    _sys_field = "offset"

class IndexQuery(Query):
    def _make_op(self, sql_op, children, children_bool):
        if children[0] is self:
            other_idx = 1
        else:
            assert children[1] is self
            other_idx = 0
        other = self._as_constraint(children[other_idx])
        if not isinstance(other, LiteralQuery):
            raise CharltonError("index can only be compared with literal "
                                "values", other)
        index = _index(other._value, other.origin)
        era = EraQuery(self._events, self.origin)
        era_args = [era, index[0]]
        if sql_op == "<":
            era_op = "<="
        elif sql_op == ">":
            era_op = ">="
        else:
            era_op = sql_op
        offset = OffsetQuery(self._events, self.origin)
        offset_args = [offset, index[1]]
        if other_idx == 0:
            era_args = era_args[::-1]
            offset_args = offset_args[::-1]
        return (era._make_op(era_op, era_args, children_bool)
                & offset._make_op(sql_op, offset_args, children_bool))

    def _is_bool(self):
        return False

class QueryOperator(Query):
    def __init__(self, events, sql_op, children, origin=None):
        Query.__init__(self, events, origin)
        self._sql_op = sql_op
        assert 1 <= len(children) <= 2
        self._children = children
        
    def _sql_expr(self):
        if len(self._children) == 1:
            lhs_sqlexpr = SqlExpr("", set(), [])
        else:
            lhs_sqlexpr = self._children[0]._sql_expr()
        rhs_sqlexpr = self._children[1]._sql_expr()
        return SqlExpr("(%s %s %s)"
                       % (lhs_sqlexpr.code, self._sql_op, rhs_sqlexpr.code),
                       lhs_sqlexpr.tables.union(rhs_sqlexpr.tables),
                       lhs_sqlexpr.args + rhs_sqlexpr.args)

    def _is_bool(self):
        return True

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

_atomic = ["ATTR", "LITERAL"]

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
                raise CharltonError("unrecognized escape sequence \\%s"
                                    % (escaped_char,),
                                    Origin(string, i - 1, i))
        else:
            chars.append(string[i])
        i += 1
    if i >= len(string):
        raise CharltonError("unclosed string",
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
            raise CharltonError("unrecognized token",
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
        if tree.token.extra == "INDEX":
            return IndexQuery(events, tree.origin)
        else:
            return AttrQuery(events, tree.token.extra, tree.origin)
    elif tree.type == ",":
        eval_args = [_eval(events, arg) for arg in tree.args]
        for arg in eval_args:
            if not isinstance(arg, LiteralQuery):
                raise CharltonError("comma operator can only be applied "
                                    "to literals, not %s" % (arg,), arg)
        twople = (eval_args[0]._value, eval_args[1]._value)
        return LiteralQuery(events, twople)
    else:
        assert False

def query_from_string(events, string):
    return _eval(events, parse(_tokenize(string), _ops, _atomic))
