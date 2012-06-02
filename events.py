import sqlite3
import string
import pprint

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

def _index(obj):
    if isinstance(obj, tuple):
        return obj
    else:
        return (0, obj)

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

    # def _munge_key(self, key):
    #     new_key = []
    #     for i in xrange(len(key)):
    #         if key[i] in string.letters:
    #             new_key.append(key[i])
    #             break
    #     for char in key[i + 1:]:
    #         if char in string.letters + "_" + string.digits:
    #             new_key.append(char)
    #     return "".join(new_key)

    # def _table_name(self, key):
    #     munged = self._munge_key(key)
    #     if munged != key:
    #         err = ("Illegal event attribute name %r. Names must begin with a "
    #                "letter, and contain only letters, digits, and underscores."
    #                " For example: %r" % (key, munged))
    #         raise ValueError, err
    #     return "attr_%s" % (key,)

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
            raise ValueError, "Invalid value %r for key %s: must be a string, number, bool, or None" % (value, key)

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

    def at(self, index):
        return self.in_range(era, start=index, count=1)

    # XX this is all wrong
    def in_range(self, era, start=0, stop=None, stop_incr=None):
        if start < 0:
            start += self._era_lens[era]
        if stop is None:
            if stop_incr is None:
                stop = self._era_lens[era]
            else:
                stop = start + stop_incr
        else:
            if stop_incr is not None:
                raise ValueError, "can't specify stop= and stop_incr= together"
        if stop < 0:
            stop += self._era_lens[era]
        c = self._connection.cursor()
        c.execute("SELECT id FROM sys_events"
                  " WHERE era = ?"
                  " AND offset >= ? AND offset < ?"
                  " ORDER BY era, offset;",
                  (era, start, stop))
        for (id,) in c:
            yield Event(self, id)

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

class Event(object):
    def __init__(self, events, id):
        self._events = events
        self._id = id

    @property
    def index(self):
        return self._events._event_index(self._id)

    def __getitem__(self, key):
        row = self._events._event_select_attr(self._id, key)
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
            
    # Everything else is defined in terms of the above methods:
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError, name
        try:
            return self[name]
        except KeyError:
            raise AttributeError, name

    # We intentionally don't define __setattr__ -- too dangerous! __getattr__
    # support is just a convenience -- use __getitem__/__setitem__ support
    # instead.

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __iter__(self):
        for key, _ in self.iteritems():
            yield key

    def __contains__(self):
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def __repr__(self):
        return "<%s %s at (%s): %s>" % (
            self.__class__.__name__, self._id, self.index,
            #pprint.pformat(dict(self.iteritems()), indent=4)
            repr(dict(self.iteritems())))

class SqlExpr(object):
    def __init__(self, code, tables, args):
        self.code = code
        self.tables = tables
        self.args = args

class Constraint(object):
    origin = None

    def __init__(self, events):
        self._events = events

    @classmethod
    def _as_constraint(cls, other):
        if isinstance(other, Constraint):
            return other
        return Literal(other)

    def __eq__(self, other):
        return Operator(self._events,
                        "==",
                        [self, other],
                        False)

class Operator(object):
    def __init__(self, events, sql_op, children, children_bool):
        self.sql_op = sql_op
        if INDEX in 
        children = [Constraint._as_constraint(child) for child in children]
        for child in children:
            if child.is_bool(events) != children_bool:
                raise CharltonError("operator '%s' %s boolean arguments"
                                    % (sql_op, ["did not expect",
                                                "expected"][children_bool]),
                                    child)
        self.children = children
        
    def is_bool(self, events):
        return True

class Attr(object):
    def __init__(self, name, origin=None):
        self._name = name
        self.origin = origin

    def to_sql(self):
        table_id = _table_name(self._name)
        return SqlExpr(table_id + ".value",
                       set([table_id]),
                       [])

    def is_bool(self, events):
        wanted_type = events._value_type_for_key(self._name)
        if wanted_type is None:
            raise CharltonError("unknown attribute %r" % (self._name), self)
        return wanted_type == Events.BOOL

class SysField(object):
    def __init__(self, origin=None):
        self.origin = origin

    def to_sql(self):
        return SqlExpr("sys_events.%s" % (self._sys_field),
                       set(["sys_events"]),
                       [])

    def is_bool(self, events):
        return False

class Era(SysField):
    _sys_field = "era"

class Offset(SysField):
    _sys_field = "offset"

OFFSET = object()
class Offset(object):
    def __init__(self, origin=None):
        self.origin = origin

    def to_sql(self):
        

    def is_bool(self, events):
        return False

class Literal(object):
    def __init__(self, value, origin=None):
        self._value = value
        self.origin = origin

    def to_sql(self):
        return SqlExpr("?", set(), [self._value])

    def is_bool(self, events):
        return isinstance(self._value, bool)

class BinOp(object):
    sql_op = NotImplemented

    def __init__(self, lhs, rhs, origin=None):
        self._lhs = lhs
        self._rhs = rhs
        self.origin = origin

    def to_sql(self):
        lhs_sqlexpr = self._lhs.to_sql()
        rhs_sqlexpr = self._rhs.to_sql()
        return SqlExpr("(%s %s %s)"
                       % (lhs_sqlexpr.code, self.sql_op, rhs_sqlexpr.code),
                       lhs_sqlexpr.tables.union(rhs_sqlexpr.tables),
                       lhs_sqlexpr.args + rhs_sqlexpr.args)

    def is_bool(self, events):
        return True

class AND(BinOp):
    sql_op = "AND"
    boolean = True

class OR(BinOp):
    sql_op = "OR"
    boolean = True

class NOT(object):
    boolean = True

    def __init__(self, arg, origin=None):
        self._arg = arg
        self.origin = origin

    def to_sql(self):
        sqlexpr = self._arg.to_sql()
        return SqlExpr("(NOT %s)" % (sqlexpr.code,), sqlexpr.tables, sqlexpr.args)

    def is_bool(self, events):
        return True

class EQ(BinOp):
    sql_op = "=="

class NEQ(BinOp):
    sql_op = "!="

class LT(BinOp):
    sql_op = "<"

class GT(BinOp):
    sql_op = ">"

class GTE(BinOp):
    sql_op = ">="

class LTE(BinOp):
    sql_op = "<="

def _eval_comma(events, tree):
    # Two children, both literals, returns a twople

def _eval_comparison(events, tree):
    comparisons = {"==": EQ,
                   "!=": NEQ,
                   "<": LT,
                   ">": GT,
                   ">=": GTE,
                   "<=": LTE}
    # check for list values and OFFSET magic
    # if either side is OFFSET, then the other side gets evaluated and can be
    # a pair. We are the only the place that actually dispatches to
    # _eval_comma.

def _eval(events, tree, want_bool):
    # This never lets pairs out

def _eval_literal(events, tree):
    return Literal(tree.extra)

def _eval_attr(events, tree, want_bool):
    return 

from charlton import CharltonError
from charlton.origin import Origin
from charlton.parse_core import Token, Operator, parse

_open_paren = Operator("(", -1, -9999999)
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

import re

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

# class SyntaxOperator(object):
#     def __init__(self, token, arity, precedence, filter, domain):
#         self.token = token
#         self.arity = arity
#         self.precedence = precedence
#         self.filter = filter
#         self.want_type = domain

#     def __repr__(self):
#         return "<Op %r>" % (self.token,)

# class Token(object):
#     def __init__(self, type, token, s, span):
#         self.type = type
#         self.token = token
#         self.string = s
#         self.span = span

#     def __repr__(self):
#         return "%s(%r, %r from %r)" % (self.__class__.__name__,
#                                        self.type, self.token,
#                                        self.string[self.span[0]:self.span[1]])

# class CharStream(object):
#     def __init__(self, s):
#         self._i = 0
#         self.string = s

#     def peek(self):
#         if self.done():
#             return None
#         return self.string[self._i]

#     def tell(self):
#         return self._i

#     def done(self):
#         return self._i >= len(self.string)

#     def next(self):
#         char = self.peek()
#         if char is not None:
#             self._i += 1
#         return char

#     def match_re(self, regex):
#         if self.done():
#             return None
#         match = regex.match(self.string, self._i)
#         if match is None:
#             return None
#         self._i = match.end()
#         return (match.span(), match.group())

#     def match(self, s):
#         if self.done():
#             return None
#         if self.string[self._i:].startswith(s):
#             span = (self._i, self._i + len(s))
#             self._i += len(s)
#             return (span, s)
#         else:
#             return None

# class TokenizerError(Exception):
#     def __init__(self, message, s, offset):
#         Exception.__init__(self, message)
#         self.message = message
#         self.string = s
#         self.offset = offset

#     def __str__(self):
#         return ("%s\n    %s\n    %s^"
#                 % (self.message,
#                    self.string,
#                    " " * self.offset))
# import re
# whitespace_re = re.compile(r"\s+", re.UNICODE)
# num_re = re.compile(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?")
# int_re = re.compile(r"[-+]?[0-9]+$")
# # This works because \w matches underscore, letters, and digits.
# # But if a token starts with a digit, then it'll be caught by num_re above
# # first, so in fact this works like "[_a-z][_a-z0-9]*" except for being
# # unicode-enabled.
# ident_re = re.compile(r"\w+", re.IGNORECASE | re.UNICODE)

# _punct_tokens = [op.token for op in _punct_ops]
# _punct_tokens.sort(key=len, reverse=True)
# _text_tokens = set([op.token for op in _text_ops])

# def _read_quoted_string(stream):
#     start = stream.tell()
#     quote_type = stream.next()
#     assert quote_type in "\"'`"
#     chars = []
#     while not stream.done():
#         char = stream.peek()
#         if char == quote_type:
#             break
#         elif char == "\\":
#             # Consume the backslash
#             stream.next()
#             if stream.done():
#                 break
#             escaped_char = stream.next()
#             if escaped_char in "\"'`\\":
#                 chars.append(escaped_char)
#             else:
#                 raise TokenizerError("unrecognized escape sequence \\%s"
#                                      % (escaped_char,),
#                                      stream.string, stream.tell() - 2)
#         else:
#             chars.append(stream.next())
#     if stream.done():
#         raise TokenizerError("unclosed string", stream.string, start)
#     assert stream.next() == quote_type
#     span = (start, stream.tell())
#     if quote_type == "`":
#         type = "ATTR"
#     else:
#         type = "LITERAL"
#     return Token(type, "".join(chars), stream.string, span)

# def tokenize(s):
#     stream = CharStream(s)
#     while not stream.done():
#         match = stream.match_re(whitespace_re)
#         if match is not None:
#             continue
#         match = stream.match_re(num_re)
#         if match is not None:
#             (span, num_str) = match
#             if int_re.match(num_str):
#                 num = int(num_str)
#             else:
#                 num = float(num_str)
#             yield Token("LITERAL", num, s, span)
#             continue
#         if stream.peek() in "\"'`":
#             yield _read_quoted_string(stream)
#             continue
#         if stream.peek() == "(":
#             yield Token("OPEN-PAREN", "(", s,
#                         (stream.tell(), stream.tell() + 1))
#             stream.next()
#             continue
#         if stream.peek() == ")":
#             yield Token("CLOSE-PAREN", ")", s,
#                         (stream.tell(), stream.tell() + 1))
#             stream.next()
#             continue
#         for punct_token in _punct_tokens:
#             match = stream.match(punct_token)
#             if match is not None:
#                 yield Token("OP", match[1], s, match[0])
#                 break
#         else:
#             match = stream.match_re(ident_re)
#             if match is None:
#                 raise TokenizerError("unknown token", s, stream.tell())
#             span, token = match
#             if token.lower() in _text_tokens:
#                 yield Token("OP", token.lower(), s, span)
#             else:
#                 yield Token("ATTR", token, s, span)
#     yield Token("END", None, s, (len(s), len(s)))


# class ParseError(Exception):
#     pass

# def parse(s):
#     unary_ops = {}
#     binary_ops = {}
#     for op in _ops:
#         if op.arity == 1:
#             unary_ops[op.token] = op
#         elif op.arity == 2:
#             binary_ops[op.token] = op
#         else:
#             assert False

#     op_stack = []
#     noun_stack = []
    
#     want_noun = True
#     for token in tokenize(s):
#         if want_noun:
#             if token.type == "OPEN-PAREN":
#                 op_stack.append(_open_paren)
#             elif token.type == "OP":
#                 if token.token in unary_ops:
#                     op_stack.append(unary_ops[token])
#                 else:
#                     raise ParseError("
                   
