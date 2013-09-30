# This file is part of rERPy
# Copyright (C) 2012 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

import os.path
import functools
import types

def maybe_open(file_like, mode="rb"):
    # FIXME: have more formal checking of binary/text, given how important
    # that is in py3?
    # FIXME: handle URLs? not sure how useful that would be given how huge
    # data files are though.
    if isinstance(file_like, basestring):
        return open(file_like, mode)
    else:
        return file_like

def memoized_method(meth):
    attr_name = "_memoized_method_cache_%s" % (meth.__name__,)
    @functools.wraps(meth)
    def memoized_wrapper(self, *args):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, {})
        if args not in getattr(self, attr_name):
            value = meth(self, *args)
            assert not isinstance(value, types.GeneratorType)
            getattr(self, attr_name)[args] = value
        return getattr(self, attr_name)[args]
    return memoized_wrapper

class _MemoizedTest(object):
    def __init__(self):
        self.x = 1

    @memoized_method
    def return_x(self):
        return self.x

    @memoized_method
    def multiply_by_x(self, y):
        return y * self.x

def test_memoized_method():
    t = _MemoizedTest()
    assert t.return_x() == 1
    assert t.multiply_by_x(3) == 3
    t.x = 2
    assert t.return_x() == 1
    assert t.multiply_by_x(3) == 3
    t2 = _MemoizedTest()
    t2.x = 2
    assert t2.return_x() == 2
    assert t2.multiply_by_x(3) == 6
    t2.x = 1
    assert t2.return_x() == 2
    assert t2.multiply_by_x(3) == 6

def indent(string, chars, indent_first=True):
    lines = string.split("\n")
    indented = "\n".join([" " * chars + line for line in lines])
    if not indent_first:
        indented = indented[chars:]
    return indented

def test_indent():
    assert indent("a\nb", 4) == "    a\n    b"
    assert indent("a\nb", 2) == "  a\n  b"
    assert indent("a\nb", 4, indent_first=False) == "a\n    b"
    assert indent("a\nb", 2, indent_first=False) == "a\n  b"
