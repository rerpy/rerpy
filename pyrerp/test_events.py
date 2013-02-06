# This file is part of pyrerp
# Copyright (C) 2012 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

import cPickle
from pyrerp.events import Events, EventsError
from nose.tools import assert_raises

def test_Events_basic():
    # load/store of different types
    # errors on type mismatches
    # storing None
    e = Events(int)
    ev1 = e.add_event(10, {"a": 1, "b": "hello", "c": True})
    assert ev1.index == 10
    assert ev1["a"] == 1
    assert ev1["b"] == "hello"
    ev1["a"] = 2
    assert ev1["a"] == 2
    assert_raises(ValueError, e.add_event, 20, {"a": "string"})
    assert_raises(ValueError, e.add_event, 20, {"a": True})
    assert_raises(ValueError, e.add_event, 20, {"b": 10})
    assert_raises(ValueError, e.add_event, 20, {"b": True})
    assert_raises(ValueError, e.add_event, 20, {"c": 10})
    assert_raises(ValueError, e.add_event, 20, {"c": "string"})
    ev1["a"] = None
    assert ev1["a"] is None
    ev1["xxx"] = None
    ev2 = e.add_event(11, {"xxx": 3})
    assert_raises(ValueError, e.add_event, 20, {"xxx": "string"})
    assert_raises(ValueError, e.add_event, 20, {"xxx": True})
    assert ev2["xxx"] == 3
    assert ev2.index == 11
    assert ev1["xxx"] is None

    e_pick = cPickle.loads(cPickle.dumps(e))
    ev1_pick, ev2_pick = list(e_pick)
    assert ev1_pick.index == ev1.index
    assert ev1_pick.items() == ev1.items()
    assert ev2_pick.index == ev2.index
    assert ev2_pick.items() == ev2.items()

def test_index_encoding():
    e = Events((int, int))
    tests = [(0, 0), (10, 10), (0, -1), (-1, 10),
             (0, 1000), (1000, 0), (2**32, 2**31)]
    for t in tests:
        assert e._decode_index(e._encode_index(t)) == t
    encoded_tests = [e._encode_index(t) for t in tests]
    assert [e._decode_index(t) for t in sorted(encoded_tests)] == sorted(tests)

    e2 = Events((str, int))
    tests = [("a", 0), ("aa", 10), ("ba", 1000), ("bb", 0), ("zz", 2**30)]
    for t in tests:
        assert e2._decode_index(e2._encode_index(t)) == t
    encoded_tests = [e2._encode_index(t) for t in tests]
    assert [e2._decode_index(t) for t in sorted(encoded_tests)] == sorted(tests)

def test_Event():
    # set/get/del, index
    # dict methods
    e = Events(int)
    d = {"a": 1, "b": "hello", "c": True}
    ev1 = e.add_event(10, d)
    assert ev1.index == 10
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

    assert len(e) == 1
    ev1.delete()
    assert len(e) == 0
    # XX: maybe this should be a ValueError or something, but checking for
    # whether the event object itself exists in __getitem__ is extra work that
    # doesn't seem worth bothering with.
    assert_raises(KeyError, ev1.__getitem__, "a")

def test_misc_queries():
    # ANY, at, __iter__, __len__
    e = Events(int)
    e.add_event(20, {"a": -1})
    e.add_event(10, {"a": 1})
    e.add_event(30, {"a": 100})
    assert len(e) == 3
    # Always sorted by index:
    assert [ev.index for ev in e] == [10, 20, 30]
    assert [ev.index for ev in e.find(e.ANY)] == [10, 20, 30]
    assert len(e.at(10)) == 1
    assert e.at(10)[0]["a"] == 1
    assert len(e.at(20)) == 1
    assert e.at(20)[0]["a"] == -1
    assert e.at(15) == []

def test_Event_relative():
    e = Events(int)
    e.add_event(20, {"a": 20, "extra": True})
    e.add_event(10, {"a": 10})
    e.add_event(30, {"a": 30})
    e.add_event(40, {"a": 40, "extra": True})
    
    ev20 = e.at(20)[0]
    assert ev20.relative(1)["a"] == 30
    assert_raises(IndexError, ev20.relative, 0)
    assert ev20.relative(-1)["a"] == 10
    ev10 = e.at(10)[0]
    assert ev10.relative(2)["a"] == 30
    ev30 = e.at(30)[0]
    assert ev30.relative(-2)["a"] == 10
    assert ev10.relative(1, "extra")["a"] == 20
    assert ev10.relative(2, "extra")["a"] == 40

def test_find():
    # all the different calling conventions
    e = Events(int)
    e.add_event(10, {"a": 1, "b": True})
    e.add_event(20, {"a": -1, "b": True})

    assert [ev.index for ev in e.find()] == [10, 20]

    assert [ev.index for ev in e.find({"a": 1})] == [10]
    assert [ev.index for ev in e.find({"a": 1, "b": True})] == [10]
    assert [ev.index for ev in e.find({"a": 1, "b": False})] == []
    assert [ev.index for ev in e.find({"b": True})] == [10, 20]
    
    assert [ev["a"] for ev in e.find({"INDEX": 10})] == [1]
    assert [ev["a"] for ev in e.find({"INDEX": 20})] == [-1]
    assert [ev["a"] for ev in e.find({"INDEX": 20, "b": False})] == []

def test_python_query():
    # all operators
    # types (esp. including None)
    # index
    e = Events(int)
    e.add_event(10, {"a": 1, "b": "asdf", "c": True, "d": 1.5, "e": None})
    e.add_event(-10, {"a": -1, "b": "fdsa", "c": False, "d": -3.14, "e": None})
    e.add_event(20, {"a": 1, "b": "asdf", "c": False, "d": -3.14, "e": 123,
                     "INDEX": 10})

    p = e.placeholder
    def t(q, expected):
        assert [ev.index for ev in e.find(q)] == expected

    t(p.index == 10, [10])
    t(p.index != 10, [-10, 20])
    t(p.index < 10, [-10])
    t(p.index > 10, [20])
    t(p.index >= 10, [10, 20])
    t(p.index <= 10, [-10, 10])
    t(~(p.index == 10), [-10, 20])
    t(~(p.index != 10), [10])

    t(p["a"] == 1, [10, 20])
    t(p["a"] != 1, [-10])
    t(p["a"] < 1, [-10])
    t(p["a"] > 1, [])
    t(p["a"] >= 1, [10, 20])
    t(p["a"] <= 1, [-10, 10, 20])
    t(~(p["a"] == 1), [-10])
    t(~(p["a"] != 1), [10, 20])
    t(p["a"] == 1.5, [])
    t(p["a"] < 1.5, [-10, 10, 20])
    t(p["a"] < 0.5, [-10])
    t(p["a"] > 0.5, [10, 20])

    t(p["b"] == "asdf", [10, 20])
    t(p["b"] != "asdf", [-10])
    t(p["b"] < "asdf", [])
    t(p["b"] > "asdf", [-10])
    t(p["b"] >= "asdf", [-10, 10, 20])
    t(p["b"] <= "asdf", [10, 20])
    t(p["b"] <= "b", [10, 20])
    t(p["b"] >= "b", [-10])
    t(~(p["b"] == "asdf"), [-10])
    t(~(p["b"] != "asdf"), [10, 20])

    t(p["c"] == True, [10])
    t(p["c"] != True, [-10, 20])
    t(p["c"] == False, [-10, 20])
    t(p["c"] != False, [10])
    t(p["c"], [10])
    t(~p["c"], [-10, 20])
    t(p["c"] < True, [-10, 20])
    t(p["c"] <= True, [-10, 10, 20])
    t(p["c"] > True, [])
    t(p["c"] > False, [10])
    t(p["c"] >= False, [-10, 10, 20])
    t(~(p["c"] == True), [-10, 20])
    t(~(p["c"] != True), [10])

    t(p["d"] == 1.5, [10])
    t(p["d"] != 1.5, [-10, 20])
    t(p["d"] < 1.5, [-10, 20])
    t(p["d"] > 1.5, [])
    t(p["d"] >= 1.5, [10])
    t(p["d"] <= 1.5, [-10, 10, 20])
    t(~(p["d"] == 1.5), [-10, 20])
    t(~(p["d"] != 1.5), [10])
    t(p["d"] == 1, [])
    t(p["d"] < 10, [-10, 10, 20])
    t(p["d"] < 1, [-10, 20])
    t(p["d"] > 1, [10])

    t(p["e"] == None, [-10, 10])
    t(p["e"] != None, [20])

    t(p["INDEX"] == 10, [20])
    t(p["INDEX"] != 10, [])

    t(p["a"] > p["d"], [-10, 20])
    t(p["a"] < p["d"], [10])

    t((p.index < 20) & (p["a"] == 1), [10])
    t((p.index < 20) | (p["a"] == 1), [-10, 10, 20])
    t(~((p.index < 20) & (p["a"] == 1)), [-10, 20])
    t(~((p.index < 20) | (p["a"] == 1)), [])

def test_python_query_typechecking():
    e = Events(int)
    e.add_event(10, {"a": 1, "b": "asdf", "c": True, "d": 1.5, "e": None})
    
    p = e.placeholder

    assert list(e.find(p["e"] == 1)) == []
    assert len(list(e.find(p["e"] == None))) == 1

    for bad in (True, "asdf"):
        assert_raises(EventsError, p["a"].__eq__, bad)
        assert_raises(EventsError, p["a"].__gt__, bad)
        assert_raises(EventsError, p["a"].__lt__, bad)
        assert_raises(EventsError, p["a"].__le__, bad)
        assert_raises(EventsError, p["a"].__ge__, bad)

    assert_raises(EventsError, p["a"].__and__, p["c"])
    assert_raises(EventsError, p["a"].__and__, p["c"])
    assert_raises(EventsError, p["a"].__invert__)

    assert_raises(EventsError, e.find, p["e"])

# def test_string_query():
#     # all operators
#     # quoting
#     # INDEX versus `INDEX`
#     # `and`
#     # comma operator
#     assert False

def test_index():
    def t(type, good_value, bad_value):
        e = Events(type)
        ev = e.add_event(good_value, {"a": 1})
        assert ev.index == good_value
        assert len(list(e.find(e.placeholder.index == good_value))) == 1
        assert_raises(EventsError, e.add_event, bad_value, {"a": 1})
        assert_raises(EventsError,
                      e.find, e.placeholder.index == bad_value)
    t(int, 10, "asdf")
    t((int,), (10,), 10)
    t(str, "asdf", 10)
    t((int, int), (10, 20), ("10", 20))
    t((int, int), (10, 20), (10, "20"))
    t((int, int), (10, 20), 10)
    t((str, int), ("subj1", 10), (10, "subj1"))
    t((str, int, int), ("subj1", 1, 10), ("subj1", 10))
    
