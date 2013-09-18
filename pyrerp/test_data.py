# This file is part of pyrerp
# Copyright (C) 2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

import numpy as np
import pandas
from nose.tools import assert_raises

from pyrerp.data import DataSet, DataFormat, mock_dataset

class MockDataSource(object):
    def __init__(self, tick_lengths):
        self._tick_lengths = tick_lengths
        self._transform = np.eye(2)

    def __getitem__(self, local_recspan_id):
        ticks = self._tick_lengths[local_recspan_id]
        return np.dot(np.ones((ticks, 2)) * local_recspan_id,
                      self._transform.T)

    def transform(self, matrix):
        self._transform = np.dot(matrix, self._transform)

    def copy(self):
        new = MockDataSource(self._tick_lengths)
        new.transform(self._transform)
        return new

def test_DataSet():
    data_format = DataFormat(250, "uV", ["MOCK1", "MOCK2"])
    data_set = DataSet(data_format)

    assert len(data_set) == 0
    assert_raises(IndexError, data_set.__getitem__, 0)
    assert_raises(TypeError, data_set.__getitem__, slice(0, 0))
    assert list(data_set) == []

    # Mismatch between tick_lengths and metadatas
    assert_raises(ValueError, data_set.add_recspan_source,
                  MockDataSource([10, 20]), [10, 20], [{}])
    data_set.add_recspan_source(MockDataSource([10, 20]),
                                [10, 20],
                                [{"a": 0}, {"a": 1}])
    data_set.add_recspan_source(MockDataSource([30, 40]),
                                [30, 40],
                                [{"a": 2}, {"a": 3}])
    assert len(data_set) == 4
    assert_raises(IndexError, data_set.__getitem__, 4)

    data_set.add_event(1, 10, 11, {"foo": "bar"})

    def t(ds, recspan_id, expected_values=None):
        recspan = ds[recspan_id]
        assert isinstance(recspan, pandas.DataFrame)
        expected_ticks = 10 * (recspan_id + 1)
        assert recspan.shape == (expected_ticks, 2)
        # 1/250 Hz = 4.0 ms
        assert np.all(recspan.index == np.arange(expected_ticks) * 4.0)
        # index is supposed to be floats. Not sure if that means float or
        # np.float64, but this check should work for either:
        assert isinstance(recspan.index[0], float)
        assert np.all(recspan.columns == ["MOCK1", "MOCK2"])
        # Values are supposed to be floating point as well.
        assert type(recspan.iloc[0, 0]) is np.float64
        if expected_values is None:
            local_recspan_id = recspan_id % 2
            expected_values = local_recspan_id
        assert np.allclose(recspan, expected_values)

        assert ds.recspan_infos[recspan_id]["a"] == recspan_id
        assert ds.recspan_infos[recspan_id].ticks == expected_ticks

    for i in xrange(4):
        t(data_set, i)

    # DataFormat mismatch
    diff_data_set = DataSet(DataFormat(500, "uV", ["MOCK1", "MOCK2"]))
    assert_raises(ValueError, diff_data_set.add_dataset, data_set)

    data_set_copy = DataSet(data_format)
    data_set_copy.add_dataset(data_set)
    assert len(data_set_copy) == 4
    for i in xrange(4):
        t(data_set_copy, i)
    assert len(data_set_copy.events()) == 1
    assert dict(data_set_copy.events()[0]) == {"foo": "bar"}
    assert data_set_copy.events()[0].recspan_id == 1

    assert_raises(ValueError, data_set.transform, np.eye(2), exclude=["MOCK1"])
    data_set.transform([[2, 0], [0, 3]])
    # Transforming the first data set doesn't affect the second
    for i in xrange(4):
        t(data_set_copy, i)
    # But it does affect the first!
    for i in xrange(4):
        t(data_set, i, expected_values=[[2 * (i % 2), 3 * (i % 2)]])
    # Try a symbolic transform too -- it should stack with the previous
    # transform.
    data_set.transform("-MOCK1/3", exclude=["MOCK1"])
    for i in xrange(4):
        t(data_set, i, expected_values=[[2 * (i % 2),
                                         3 * (i % 2) - (2./3) * (i % 2)]])

    # Also check that changing one DataSet's metadata doesn't affect the copy.
    data_set.recspan_infos[0]["a"] = 100
    assert data_set.recspan_infos[0]["a"] == 100
    assert data_set_copy.recspan_infos[0]["a"] == 0
    # Set it back to avoid any confusion later in the test
    data_set.recspan_infos[0]["a"] = 0

    # Check __iter__
    recspans = list(data_set)
    assert len(recspans) == 4
    for i in xrange(4):
        assert np.all(recspans[i] == data_set[i])

def test_DataSet_add_recspan():
    # Check the add_recspan convenience method
    data_set = mock_dataset(num_channels=2, num_recspans=4)
    data_set.add_recspan([[1, 2], [3, 4], [5, 6]], {"a": 31337})
    assert len(data_set) == 5
    assert np.all(data_set[4].columns == ["MOCK0", "MOCK1"])
    assert np.all(data_set[4].index == [0.0, 4.0, 8.0])
    assert np.all(np.asarray(data_set[4]) == [[1, 2], [3, 4], [5, 6]])
    assert type(data_set[4].iloc[0, 0]) is np.float64
    assert data_set.recspan_infos[4]["a"] == 31337

    # Wrong number of channels
    assert_raises(ValueError,
                  data_set.add_recspan, [[1, 2, 3], [4, 5, 6]], {})

def test_DataSet_events():
    # Thorough tests are in test_events; here we just make sure the basic API
    # is functioning.
    data_set = mock_dataset()
    e1 = data_set.add_event(1, 10, 15, {"a": 1, "b": "foo", "c": False})
    e2 = data_set.add_event(2, 12, 17, {"a": 2, "b": "foo", "c": True})

    assert isinstance(data_set.events(), list)

    for args, expected in [((), [e1, e2]),
                           (("_START_TICK == 10",), [e1]),
                           ]:
        assert data_set.events(*args) == expected
        assert len(data_set.events_query(*args)) == len(expected)
        assert list(data_set.events_query(*args)) == expected

    for args, kwargs, expected in [((1, 13), {}, [e1]),
                                   ((2, 13), {}, [e2]),
                                   ((1, 13), {"subset": "a == 2"}, []),
                                   ((1, 8, 12), {}, [e1]),
                                   ]:
        assert data_set.events_at(*args, **kwargs) == expected
        assert len(data_set.events_at_query(*args, **kwargs)) == len(expected)
        assert list(data_set.events_at_query(*args, **kwargs)) == expected

    p = data_set.placeholder_event()
    assert list(p["a"] == 2) == [e2]
    assert data_set.events(p["a"] == 2) == [e2]

def test_DataSet_merge_df():
    def make_events():
        ds = mock_dataset()
        ev1 = ds.add_event(0, 10, 11, {"code": 10, "code2": 20})
        ev2 = ds.add_event(0, 20, 21, {"code": 10, "code2": 21})
        ev3 = ds.add_event(0, 30, 31, {"code": 11, "code2": 20})
        return ds, ev1, ev2, ev3

    ds, ev1, ev2, ev3 = make_events()
    ds.merge_df(pandas.DataFrame({"code": [10, 11], "foo": ["a", "b"]}),
                on="code")
    assert dict(ev1) == {"code": 10, "code2": 20, "foo": "a"}
    assert dict(ev2) == {"code": 10, "code2": 21, "foo": "a"}
    assert dict(ev3) == {"code": 11, "code2": 20, "foo": "b"}

    ds, ev1, ev2, ev3 = make_events()
    ds.merge_df(pandas.DataFrame({"code": [10, 11], "foo": ["a", "b"]}),
                on=["code"])
    assert dict(ev1) == {"code": 10, "code2": 20, "foo": "a"}
    assert dict(ev2) == {"code": 10, "code2": 21, "foo": "a"}
    assert dict(ev3) == {"code": 11, "code2": 20, "foo": "b"}

    ds, ev1, ev2, ev3 = make_events()
    ds.merge_df(pandas.DataFrame({"code": [10, 11], "foo": ["a", "b"]}),
                on={"code": "code"})
    assert dict(ev1) == {"code": 10, "code2": 20, "foo": "a"}
    assert dict(ev2) == {"code": 10, "code2": 21, "foo": "a"}
    assert dict(ev3) == {"code": 11, "code2": 20, "foo": "b"}

    ds, ev1, ev2, ev3 = make_events()
    ds.merge_df(pandas.DataFrame(
            {"code": [10, 11], "code2": [20, 20], "foo": ["a", "b"]}),
               on=["code", "code2"])
    assert dict(ev1) == {"code": 10, "code2": 20, "foo": "a"}
    assert dict(ev2) == {"code": 10, "code2": 21}
    assert dict(ev3) == {"code": 11, "code2": 20, "foo": "b"}

    # Trying to overwrite existing fields
    ds, ev1, ev2, ev3 = make_events()
    assert_raises(ValueError,
                  ds.merge_df,
                  pandas.DataFrame({"code": [10, 11],
                                    "code2": [20, 20],
                                    "foo": ["a", "b"]}),
                  on=["code"])

    ds, ev1, ev2, ev3 = make_events()
    ds.merge_df(pandas.DataFrame({"THECODE": [10, 11], "foo": ["a", "b"]}),
                on={"THECODE": "code"})
    assert dict(ev1) == {"code": 10, "code2": 20, "foo": "a"}
    assert dict(ev2) == {"code": 10, "code2": 21, "foo": "a"}
    assert dict(ev3) == {"code": 11, "code2": 20, "foo": "b"}

    ds, ev1, ev2, ev3 = make_events()
    ds.merge_df(
        pandas.DataFrame({"code": [20, 21, 20],
                          "code2": [10, 10, 11],
                          "foo": ["a", "b", "c"]}),
        on={"code": "code2", "code2": "code"})
    assert dict(ev1) == {"code": 10, "code2": 20, "foo": "a"}
    assert dict(ev2) == {"code": 10, "code2": 21, "foo": "b"}
    assert dict(ev3) == {"code": 11, "code2": 20, "foo": "c"}

    ds, ev1, ev2, ev3 = make_events()
    ds.merge_df(
        pandas.DataFrame({"code": [20, 21, 20],
                          "code2": [10, 10, 11],
                          "foo": ["a", "b", "c"]}),
        on={"code": "code2", "code2": "code"},
        subset="code == 10")
    assert dict(ev1) == {"code": 10, "code2": 20, "foo": "a"}
    assert dict(ev2) == {"code": 10, "code2": 21, "foo": "b"}
    assert dict(ev3) == {"code": 11, "code2": 20}
