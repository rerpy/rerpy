# This file is part of pyrerp
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

import cPickle
import numpy as np
from pyrerp.events import Events, EventsError
from nose.tools import assert_raises

def test_Events_basic():
    # load/store of different types
    # errors on type mismatches
    # storing None
    e = Events()
    ev1 = e.add_event(0, 10, 11, {"a": 1, "b": "hello", "c": True})
    assert ev1.recspan_id == 0
    assert ev1.start_idx == 10
    assert ev1.stop_idx == 11
    assert ev1["a"] == 1
    assert ev1["b"] == "hello"
    assert ev1["c"] == True
    assert type(ev1["a"]) == int
    assert type(ev1["b"]) == str
    assert type(ev1["c"]) == bool
    assert dict(ev1) == {"a": 1, "b": "hello", "c": True}
    ev1["a"] = 2
    assert ev1["a"] == 2
    assert_raises(ValueError, e.add_event, 0, 20, 21, {"a": "string"})
    assert_raises(ValueError, e.add_event, 0, 20, 21, {"a": True})
    assert_raises(ValueError, e.add_event, 0, 20, 21, {"b": 10})
    assert_raises(ValueError, e.add_event, 0, 20, 21, {"b": True})
    assert_raises(ValueError, e.add_event, 0, 20, 21, {"c": 10})
    assert_raises(ValueError, e.add_event, 0, 20, 21, {"c": "string"})
    ev1["a"] = None
    assert ev1["a"] is None
    ev1["xxx"] = None
    ev2 = e.add_event(0, 11, 12, {"xxx": 3})
    assert_raises(ValueError, e.add_event, 0, 20, 21, {"xxx": "string"})
    assert_raises(ValueError, e.add_event, 0, 20, 21, {"xxx": True})
    assert ev2["xxx"] == 3
    assert ev2.start_idx == 11
    assert ev1["xxx"] is None

    e_pick = cPickle.loads(cPickle.dumps(e))
    ev1_pick, ev2_pick = list(e_pick)
    assert ev1_pick.recspan_id == 0
    assert ev1_pick.start_idx == 10
    assert ev1_pick.stop_idx == 11
    assert ev1_pick.items() == ev1.items()
    assert ev2_pick.recspan_id == 0
    assert ev2_pick.start_idx == 11
    assert ev2_pick.stop_idx == 12
    assert ev2_pick.items() == ev2.items()

def test_Event():
    # set/get/del, index
    # dict methods
    e = Events()
    d = {"a": 1, "b": "hello", "c": True}
    ev1 = e.add_event(1, 10, 12, d)
    assert ev1.recspan_id == 1
    assert ev1.start_idx == 10
    assert ev1.stop_idx == 12
    assert dict(ev1.iteritems()) == d
    ev1["added"] = True
    assert ev1["added"] == True
    assert_raises(KeyError, ev1.__getitem__, "notadded")
    assert_raises(KeyError, ev1.__delitem__, "notadded")
    del ev1["added"]
    assert_raises(KeyError, ev1.__getitem__, "added")
    assert_raises(KeyError, ev1.__delitem__, "added")
    assert "a" in ev1
    assert "added" not in ev1
    assert_raises(ValueError, ev1.__setitem__, "c", "asdf")
    assert ev1["c"] == True

    assert ev1.get("c") == True
    assert ev1.get("notadded") is None
    assert ev1.get("notadded", 10) == 10
    assert sorted(list(ev1)) == ["a", "b", "c"]
    assert sorted(ev1.iterkeys()) == ["a", "b", "c"]
    assert sorted(ev1.itervalues()) == sorted(d.itervalues())
    assert list(ev1.iterkeys()) == ev1.keys()
    assert list(ev1.itervalues()) == ev1.values()
    assert list(ev1.iteritems()) == ev1.items()

    assert ev1.has_key("a")
    assert not ev1.has_key("z")

    assert ev1.overlaps(ev1)
    assert not ev1.overlaps(1, 9, 11)
    assert not ev1.overlaps(0, 9, 11)
    assert ev1.overlaps(1, 9, 11)
    assert ev1.overlaps(1, 10, 12)
    assert ev1.overlaps(1, 10, 11)
    assert ev1.overlaps(1, 11, 12)
    assert not ev1.overlaps(1, 0, 2)
    assert not ev1.overlaps(1, 100, 102)
    # Check half-openness
    assert not ev1.overlaps(0, 9, 10)
    assert not ev1.overlaps(0, 12, 15)
    # Nothing overlaps an empty interval
    assert not ev1.overlaps(0, 11, 11)

    assert len(e) == 1
    ev1.delete()
    assert len(e) == 0
    # XX: maybe this should be a ValueError or something, but checking for
    # whether the event object itself exists in __getitem__ is extra work that
    # doesn't seem worth bothering with.
    assert_raises(KeyError, ev1.__getitem__, "a")

def test_misc_queries():
    # ANY, at, __iter__, __len__
    e = Events()
    e.add_event(0, 20, 21, {"a": -1})
    e.add_event(0, 10, 11, {"a": 1})
    e.add_event(0, 30, 31, {"a": 100})
    assert len(e) == 3
    # Always sorted by index:
    assert [ev.start_idx for ev in e] == [10, 20, 30]
    assert [ev.start_idx for ev in e.find(e.ANY)] == [10, 20, 30]
    assert len(e.at(0, 10)) == 1
    assert e.at(0, 10)[0]["a"] == 1
    assert len(e.at(0, 20)) == 1
    assert e.at(0, 20)[0]["a"] == -1
    assert len(e.at(0, 15)) == 0
    assert len(e.at(0, 15, 40)) == 2

def test_Event_relative():
    e = Events()
    ev20 = e.add_event(0, 20, 21, {"a": 20, "extra": True})
    ev10 = e.add_event(0, 10, 11, {"a": 10})
    ev30 = e.add_event(0, 30, 31, {"a": 30})
    ev40 = e.add_event(0, 40, 41, {"a": 40, "extra": True})

    assert ev20.relative(1)["a"] == 30
    assert_raises(IndexError, ev20.relative, 0)
    assert ev20.relative(-1)["a"] == 10
    assert ev10.relative(2)["a"] == 30
    assert ev30.relative(-2)["a"] == 10
    assert ev10.relative(1, "extra")["a"] == 20
    assert ev10.relative(2, "extra")["a"] == 40

def test_Event_move():
    e = Events()
    ev20 = e.add_event(0, 20, 21, {"a": 20, "extra": True})
    ev10 = e.add_event(0, 10, 15, {"a": 10})
    assert ev20.start_idx == 20
    assert ev20.stop_idx == 21
    assert ev20.relative(-1)["a"] == 10
    ev20.move(-15)
    assert ev20.start_idx == 5
    assert ev20.stop_idx == 6
    assert ev20.relative(1)["a"] == 10

    assert ev10.start_idx == 10
    assert ev10.stop_idx == 15
    ev10.move(5)
    assert ev10.start_idx == 15
    assert ev10.stop_idx == 20

def test_find():
    # all the different calling conventions
    e = Events()
    e.add_event(0, 10, 11, {"a": 1, "b": True})
    e.add_event(0, 20, 21, {"a": -1, "b": True})

    assert [ev.start_idx for ev in e.find()] == [10, 20]

    assert [ev.start_idx for ev in e.find({"a": 1})] == [10]
    assert [ev.start_idx for ev in e.find({"a": 1, "b": True})] == [10]
    assert [ev.start_idx for ev in e.find({"a": 1, "b": False})] == []
    assert [ev.start_idx for ev in e.find({"b": True})] == [10, 20]
    assert [ev.start_idx for ev in e.find({"_RECSPAN_ID": 0})] == [10, 20]
    assert [ev.start_idx for ev in e.find({"_RECSPAN_ID": 1})] == []
    assert [ev.start_idx for ev in e.find({"_START_IDX": 10})] == [10]
    assert [ev.start_idx for ev in e.find({"_STOP_IDX": 11})] == [10]

    assert [ev.start_idx for ev in e.find(e.placeholder["a"] == 1)] == [10]

def test_EventSet():
    e = Events()
    e.add_event(0, 10, 11, {"a": 1, "b": True, "c": None})
    e.add_event(0, 20, 21, {"a": -1, "b": True})
    e.add_event(0, 15, 16, {"a": -1, "b": False, "c": 1})
    es = e.find()
    assert es[0].start_idx == 10
    assert es[1].start_idx == 15
    assert es[2].start_idx == 20
    assert es[-1].start_idx == 20
    assert_raises(IndexError, es.__getitem__, 3)
    a_series = es["a"]
    assert np.all(a_series == [1, -1, -1])
    assert a_series.dtype == np.dtype(int)
    b_series = es["b"]
    assert np.all(b_series == [True, False, True])
    assert b_series.dtype == np.dtype(bool)
    assert_raises(KeyError, es.__getitem__, "c")
    # Make sure that both None and flat-out missing values are handled
    # correctly:

def test_python_query():
    # all operators
    # types (esp. including None)
    # index
    e = Events()
    ev10 = e.add_event(0, 10, 11,
                       {"a": 1, "b": "asdf", "c": True, "d": 1.5, "e": None})
    ev20 = e.add_event(1, 20, 25,
                       {"a": -1, "b": "fdsa", "c": False, "d": -3.14, "e": None})
    ev21 = e.add_event(0, 21, 26,
                       {"a": 1, "b": "asdf", "c": False, "d": -3.14, "e": 123})

    p = e.placeholder
    def t(q, expected):
        assert [ev.start_idx for ev in e.find(q)] == expected

    t(p.start_idx == 10, [10])
    t(p.start_idx != 10, [20, 21])
    t(p.start_idx < 10, [])
    t(p.start_idx > 10, [20, 21])
    t(p.start_idx >= 20, [20, 21])
    t(p.start_idx <= 20, [10, 20])
    t(~(p.start_idx == 20), [10, 21])
    t(~(p.start_idx != 20), [20])

    t(p["a"] == 1, [10, 21])
    t(p["a"] != 1, [20])
    t(p["a"] < 1, [20])
    t(p["a"] > 1, [])
    t(p["a"] >= 1, [10, 21])
    t(p["a"] <= 1, [10, 20, 21])
    t(~(p["a"] == 1), [20])
    t(~(p["a"] != 1), [10, 21])
    t(p["a"] == 1.5, [])
    t(p["a"] < 1.5, [10, 20, 21])
    t(p["a"] < 0.5, [20])
    t(p["a"] > 0.5, [10, 21])

    t(p["b"] == "asdf", [10, 21])
    t(p["b"] != "asdf", [20])
    t(p["b"] < "asdf", [])
    t(p["b"] > "asdf", [20])
    t(p["b"] >= "asdf", [10, 20, 21])
    t(p["b"] <= "asdf", [10, 21])
    t(p["b"] <= "b", [10, 21])
    t(p["b"] >= "b", [20])
    t(~(p["b"] == "asdf"), [20])
    t(~(p["b"] != "asdf"), [10, 21])

    t(p["c"] == True, [10])
    t(p["c"] != True, [20, 21])
    t(p["c"] == False, [20, 21])
    t(p["c"] != False, [10])
    t(p["c"], [10])
    t(~p["c"], [20, 21])
    t(p["c"] < True, [20, 21])
    t(p["c"] <= True, [10, 20, 21])
    t(p["c"] > True, [])
    t(p["c"] > False, [10])
    t(p["c"] >= False, [10, 20, 21])
    t(~(p["c"] == True), [20, 21])
    t(~(p["c"] != True), [10])

    t(p["d"] == 1.5, [10])
    t(p["d"] != 1.5, [20, 21])
    t(p["d"] < 1.5, [20, 21])
    t(p["d"] > 1.5, [])
    t(p["d"] >= 1.5, [10])
    t(p["d"] <= 1.5, [10, 20, 21])
    t(~(p["d"] == 1.5), [20, 21])
    t(~(p["d"] != 1.5), [10])
    t(p["d"] == 1, [])
    t(p["d"] < 10, [10, 20, 21])
    t(p["d"] < 1, [20, 21])
    t(p["d"] > 1, [10])

    t(p["e"] == None, [10, 20])
    t(p["e"] != None, [21])

    t(p.has_key("e"), [10, 20, 21])
    t(~p.has_key("e"), [])

    t(p["nonexistent"] == 10, [])
    t(p["nonexistent"] != 10, [])
    t(p["nonexistent"] == None, [])
    t(p.has_key("nonexistent"), [])
    t(~p.has_key("nonexistent"), [10, 20, 21])

    t(p["a"] > p["d"], [20, 21])
    t(p["a"] < p["d"], [10])

    t((p.start_idx < 21) & (p["a"] == 1), [10])
    t((p.start_idx < 21) | (p["a"] == 1), [10, 20, 21])
    t(~((p.start_idx < 21) & (p["a"] == 1)), [20, 21])
    t(~((p.start_idx < 21) | (p["a"] == 1)), [])

    t(p.overlaps(ev10), [10])
    t(p.overlaps(ev20), [20])
    t(p.overlaps(ev21), [21])
    t(p.overlaps(0, 5, 10), [])
    t(p.overlaps(0, 5, 11), [10])
    t(p.overlaps(0, 11, 20), [])
    t(p.overlaps(0, 11, 100), [])
    for i in xrange(20, 25):
        t(p.overlaps(1, i, i + 1), [20])

def test_python_query_typechecking():
    e = Events()
    e.add_event(0, 10, 11,
                {"a": 1, "b": "asdf", "c": True, "d": 1.5, "e": None})

    p = e.placeholder

    assert list(e.find(p["e"] == 1)) == []
    assert len(list(e.find(p["e"] == None))) == 1

    for bad in (True, "asdf", r):
        assert_raises(EventsError, p["a"].__eq__, bad)
        assert_raises(EventsError, p["a"].__gt__, bad)
        assert_raises(EventsError, p["a"].__lt__, bad)
        assert_raises(EventsError, p["a"].__le__, bad)
        assert_raises(EventsError, p["a"].__ge__, bad)

    assert_raises(EventsError, p["a"].__and__, p["c"])
    assert_raises(EventsError, p["a"].__and__, p["c"])
    assert_raises(EventsError, p["a"].__invert__)

    assert_raises(EventsError, e.find, p["e"])

def test_string_query():
    e = Events()
    e.add_event(0, 10, 12,
                {"a": 1, "b": "asdf", "c": True, "d": 1.5, "e": None})
    e.add_event(1, 20, 25,
                {"a": 2, "b": "fdsa", "c": False, "d": 5.1, "e": 22,
                 "f": "stuff", "_START_IDX": 10,
                 "and": 33})

    def t(s, expected_start_indices):
        result = e.find(s)
        start_indices = [ev.start_idx for ev in result]
        assert start_indices == expected_start_indices

    # all operators
    t("a == 2", [20])
    t("a != 2", [10])
    t("a < 2", [10])
    t("a > 1", [20])
    t("a <= 2", [10, 20])
    t("a >= 1", [10, 20])
    t("not (a > 1)", [10])
    t("c", [10])
    t("not c", [20])
    t("has f", [20])
    t("a == 1 and d > 1", [10])
    t("a == 1 and d > 2", [])
    t("a == 1 or d > 2", [10, 20])
    t("1 == 1", [10, 20])

    # quoting
    t("b == \"asdf\"", [10])
    t("b == \'asdf\'", [10])
    t("`a` == 1", [10])
    assert_raises(EventsError, e.find, "a == \"1\"")

    # _RECSPAN_ID and friends
    t("_RECSPAN_ID == 1", [20])
    t("_START_IDX < 15", [10])
    t("_STOP_IDX > 22", [20])

    # backquotes
    t("`_START_IDX` == 10", [20])
    t("`and` == 33", [20])
    t("not has `and`", [10])
