# This file is part of rERPy
# Copyright (C) 2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

from itertools import product

import numpy as np
import pandas

from nose.tools import assert_raises

from rerpy.rerp import rERPRequest
from rerpy.test_data import mock_dataset

def test_multi_rerp():
    ds = mock_dataset(num_channels=2, hz=1000)
    ds.add_event(0, 10, 11, {"type": "standard"})
    ds.add_event(0, 20, 21, {"type": "standard"})
    ds.add_event(0, 30, 31, {"type": "target"})
    ds.add_event(0, 40, 41, {"type": "target"})
    ds.add_event(0, 50, 51, {"type": "target"})
    ds.add_event(0, 51, 53, {"maybe_artifact": True})

    for (regression_strategy, overlap_correction) in product(
        ["auto", "by-epoch", "continuous"],
        [True, False],
        ):
        assert ds.multi_rerp([],
                             regression_strategy=regression_strategy,
                             overlap_correction=overlap_correction) == []

        standard_req = rERPRequest("type == 'standard'", -2, 4, "~ 1")
        target_req = rERPRequest("type == 'target'", -2, 4, "~ 1")
        erps = ds.multi_rerp([standard_req, target_req],
                             regression_strategy=regression_strategy,
                             overlap_correction=overlap_correction)
        standard_erp, target_erp = erps
        assert standard_erp.event_query == "type == 'standard'"
        assert target_erp.event_query == "type == 'target'"
        for i, erp in enumerate([standard_erp, target_erp]):
            assert erp.start_time == -2
            assert erp.stop_time == 4
            assert erp.formula == "~ 1"
            assert erp.bad_event_query is None
            assert erp.all_or_nothing == False
            assert erp.data_format is ds.data_format
            assert erp.design_info.column_names == ["Intercept"]
            assert erp.start_tick == -2
            assert erp.stop_tick == 5
            assert erp.ticks == 7
            assert erp.this_rerp_index == i
            assert erp.total_rerps == 2
            assert erp.global_stats.epochs.fully_accepted == 5
            assert erp.global_stats.ticks.requested == 5 * 7
            assert erp.this_rerp_stats.epochs.fully_accepted in (2, 3)
            if regression_strategy == "auto":
                assert erp.regression_strategy == "by-epoch"
            else:
                assert erp.regression_strategy == regression_strategy
            assert erp.overlap_correction == overlap_correction
            assert isinstance(erp.betas, pandas.Panel)
            assert erp.betas.shape == (1, 7, 2)
            assert np.all(erp.betas.items == ["Intercept"])
            assert np.all(erp.betas.major_axis == [-2, -1, 0, 1, 2, 3, 4])
            assert np.all(erp.betas.minor_axis == ["MOCK0", "MOCK1"])
        standard_epoch0 = np.asarray(ds[0].iloc[10 - 2:10 + 5, :])
        standard_epoch1 = np.asarray(ds[0].iloc[20 - 2:20 + 5, :])
        target_epoch0 = np.asarray(ds[0].iloc[30 - 2:30 + 5, :])
        target_epoch1 = np.asarray(ds[0].iloc[40 - 2:40 + 5, :])
        target_epoch2 = np.asarray(ds[0].iloc[50 - 2:50 + 5, :])
        assert np.allclose(standard_erp.betas["Intercept"],
                           (standard_epoch0 + standard_epoch1) / 2.0)
        assert np.allclose(target_erp.betas["Intercept"],
                           (target_epoch0 + target_epoch1 + target_epoch2)
                           / 3.0)

        ################

        both_req = rERPRequest("has type", -2, 4, formula="~ type")
        erps = ds.multi_rerp([both_req],
                             regression_strategy=regression_strategy,
                             overlap_correction=overlap_correction)
        both_erp, = erps
        assert both_erp.event_query == "has type"
        assert both_erp.start_time == -2
        assert both_erp.stop_time == 4
        assert both_erp.formula == "~ type"
        assert both_erp.bad_event_query is None
        assert both_erp.all_or_nothing == False
        assert both_erp.data_format is ds.data_format
        assert both_erp.design_info.column_names == ["Intercept",
                                                     "type[T.target]"]
        assert both_erp.start_tick == -2
        assert both_erp.stop_tick == 5
        assert both_erp.ticks == 7
        assert both_erp.this_rerp_index == 0
        assert both_erp.total_rerps == 1
        assert both_erp.global_stats.epochs.fully_accepted == 5
        assert both_erp.global_stats.ticks.requested == 5 * 7
        assert both_erp.this_rerp_stats.epochs.fully_accepted == 5
        if regression_strategy == "auto":
            assert both_erp.regression_strategy == "by-epoch"
        else:
            assert both_erp.regression_strategy == regression_strategy
        assert both_erp.overlap_correction == overlap_correction
        assert isinstance(both_erp.betas, pandas.Panel)
        assert both_erp.betas.shape == (2, 7, 2)
        assert np.all(both_erp.betas.items == ["Intercept", "type[T.target]"])
        assert np.all(both_erp.betas.major_axis == [-2, -1, 0, 1, 2, 3, 4])
        assert np.all(both_erp.betas.minor_axis == ["MOCK0", "MOCK1"])
        standard_epoch0 = np.asarray(ds[0].iloc[10 - 2:10 + 5, :])
        standard_epoch1 = np.asarray(ds[0].iloc[20 - 2:20 + 5, :])
        target_epoch0 = np.asarray(ds[0].iloc[30 - 2:30 + 5, :])
        target_epoch1 = np.asarray(ds[0].iloc[40 - 2:40 + 5, :])
        target_epoch2 = np.asarray(ds[0].iloc[50 - 2:50 + 5, :])
        standard_avg = (standard_epoch0 + standard_epoch1) / 2.0
        target_avg = (target_epoch0 + target_epoch1 + target_epoch2) / 3.0
        assert np.allclose(both_erp.betas["Intercept"], standard_avg)
        assert np.allclose(both_erp.betas["type[T.target]"],
                           target_avg - standard_avg)

        ################

        both_req2 = rERPRequest("has type", -2, 4, formula="~ 0 + type")
        erps = ds.multi_rerp([both_req2],
                             regression_strategy=regression_strategy,
                             overlap_correction=overlap_correction)
        both_erp2, = erps
        assert both_erp2.design_info.column_names == ["type[standard]",
                                                      "type[target]"]
        assert np.allclose(both_erp2.betas["type[standard]"], standard_avg)
        assert np.allclose(both_erp2.betas["type[target]"], target_avg)

        ################
        # regular artifact (check accounting)
        if regression_strategy == "by-epoch":
            assert_raises(ValueError, ds.multi_rerp, [both_req2],
                          artifact_query="has maybe_artifact",
                          regression_strategy=regression_strategy,
                          overlap_correction=overlap_correction)
        else:
            both_erp3, = ds.multi_rerp([both_req2],
                                       artifact_query="has maybe_artifact",
                                       regression_strategy=regression_strategy,
                                       overlap_correction=overlap_correction)
            assert both_erp3.regression_strategy == "continuous"
            assert both_erp3.global_stats.epochs.requested == 5
            assert both_erp3.global_stats.epochs.fully_accepted == 4
            assert both_erp3.global_stats.epochs.partially_accepted == 1
            assert both_erp3.global_stats.ticks.rejected == 2
            assert np.allclose(both_erp3.betas["type[standard]"],
                               standard_avg)
            target_2avg = (target_epoch0 + target_epoch1) / 2.0
            target_art_avg = target_avg.copy()
            # starts 1 tick past timelock event, which itself is 2 ticks into
            # epoch, and continues for 2 ticks
            assert both_erp3.start_tick == -2
            art_span = slice(2 + 1, 2 + 1 + 2)
            target_art_avg[art_span, :] = target_2avg[art_span, :]
            assert np.allclose(both_erp3.betas["type[target]"],
                               target_art_avg)

        ################
        # all or nothing
        both_req4 = rERPRequest("has type", -2, 4, formula="~ 0 + type",
                                all_or_nothing=True)
        both_erp4, = ds.multi_rerp([both_req4],
                                   artifact_query="has maybe_artifact",
                                   regression_strategy=regression_strategy,
                                   overlap_correction=overlap_correction)
        if regression_strategy == "auto":
            assert both_erp4.regression_strategy == "by-epoch"
        assert np.allclose(both_erp4.betas["type[standard]"],
                           standard_avg)
        assert np.allclose(both_erp4.betas["type[target]"],
                           (target_epoch0 + target_epoch1) / 2.0)

        ################
        # bad_event_query
        both_req5 = rERPRequest("has type", -2, 4, formula="~ 0 + type",
                                bad_event_query="_START_TICK == 20")
        if regression_strategy == "by-epoch":
            assert_raises(ValueError, ds.multi_rerp, [both_req5],
                          artifact_query="has maybe_artifact",
                          regression_strategy=regression_strategy,
                          overlap_correction=overlap_correction)
        else:
            both_erp5, = ds.multi_rerp([both_req5],
                                       artifact_query="has maybe_artifact",
                                       regression_strategy=regression_strategy,
                                       overlap_correction=overlap_correction)
            assert both_erp5.regression_strategy == "continuous"
            # standard_epoch1 is knocked out by bad_event_query
            assert np.allclose(both_erp5.betas["type[standard]"],
                               standard_epoch0)
            # plus there's an artifact knocking out part of target_epoch2
            assert np.allclose(both_erp5.betas["type[target]"],
                               target_art_avg)

def test_rerp_overlap():
    # A very simple case where overlap correction can be worked out by hand:
    #  event type A: |-- 1 --|   |-- 2 --|
    #  event type B:                  |-- 3 --|
    # The rerp for event type A will be:
    #   the average of 1 & 2 in the part where the epoch 2 has no overlap
    #   just the values from 1 in the part where epoch 2 has overlap
    # The rerp for event type B will be:
    #   the difference between the values in 3 and 1 in the part where 2 and 3
    #     overlap
    #   just the values from 3 in the part where 3 does not overlap
    HALF_EPOCH = 1
    EPOCH = 2 * HALF_EPOCH
    ds = mock_dataset(num_channels=2, ticks_per_recspan=10 * EPOCH, hz=1000)
    ds.add_event(0, 0, 1, {"type": "A"})
    ds.add_event(0, 4 * HALF_EPOCH, 4 * HALF_EPOCH + 1, {"type": "A"})
    ds.add_event(0, 5 * HALF_EPOCH, 5 * HALF_EPOCH + 1, {"type": "B"})

    epoch1 = np.asarray(ds[0].iloc[0:EPOCH, :])
    epoch2 = np.asarray(ds[0].iloc[4 * HALF_EPOCH:6 * HALF_EPOCH, :])
    epoch3 = np.asarray(ds[0].iloc[5 * HALF_EPOCH:7 * HALF_EPOCH, :])

    expected_A = np.empty((EPOCH, 2))
    expected_A[:HALF_EPOCH, :] = ((epoch1 + epoch2) / 2)[:HALF_EPOCH, :]
    expected_A[HALF_EPOCH:, :] = epoch1[HALF_EPOCH:, :]
    expected_B = np.empty((EPOCH, 2))
    # Notice that the indexes here are different for the different arrays:
    expected_B[:HALF_EPOCH, :] = epoch3[:HALF_EPOCH, :] - epoch1[HALF_EPOCH:, :]
    expected_B[HALF_EPOCH:, :] = epoch3[HALF_EPOCH:, :]

    for (regression_strategy, overlap_correction) in product(
        ["auto", "by-epoch", "continuous"],
        [True, False],
        ):
        if overlap_correction and regression_strategy == "by-epoch":
            assert_raises(ValueError,
                          ds.rerp, "True", 0, EPOCH - 1, "0 + type",
                          regression_strategy=regression_strategy,
                          overlap_correction=overlap_correction)
        else:
            rerp = ds.rerp("True", 0, EPOCH - 1, "0 + type",
                           regression_strategy=regression_strategy,
                           overlap_correction=overlap_correction)
            if overlap_correction:
                assert np.allclose(rerp.betas["type[A]"], expected_A)
                assert np.allclose(rerp.betas["type[B]"], expected_B)
                assert rerp.regression_strategy == "continuous"
                s = rerp.global_stats
                assert s.ticks.requested == 5 * HALF_EPOCH
                assert s.ticks.accepted == 5 * HALF_EPOCH
                assert s.event_ticks.requested == 3 * EPOCH
                assert s.event_ticks.accepted == 3 * EPOCH
                # all of epoch 1, plus half of epoch 2 and half of epoch 3:
                assert s.no_overlap_ticks.requested == 4 * HALF_EPOCH
                assert s.no_overlap_ticks.accepted == 4 * HALF_EPOCH
            else:
                assert np.allclose(rerp.betas["type[A]"],
                                   (epoch1 + epoch2) / 2)
                assert np.allclose(rerp.betas["type[B]"], epoch3)
                if regression_strategy == "auto":
                    assert rerp.regression_strategy == "by-epoch"
                else:
                    assert rerp.regression_strategy == regression_strategy
                s = rerp.global_stats
                assert s.ticks.requested == 3 * EPOCH
                assert s.ticks.accepted == 3 * EPOCH
                assert s.event_ticks.requested == 3 * EPOCH
                assert s.event_ticks.accepted == 3 * EPOCH
                assert s.no_overlap_ticks.requested == 3 * EPOCH
                assert s.no_overlap_ticks.accepted == 3 * EPOCH

def test_predict():
    ds = mock_dataset(num_channels=2, hz=1000)
    ds.add_event(0, 10, 11, {"type": "standard", "x": 1})
    ds.add_event(0, 20, 21, {"type": "standard", "x": 2})
    ds.add_event(0, 30, 31, {"type": "target", "x": 3})
    ds.add_event(0, 40, 41, {"type": "target", "x": 4})

    rerp = ds.rerp("has type", 0, 9, "type + x")

    for predictors in [{"type": ["standard"], "x": [5]},
                       {"type": "standard", "x": 5},
                       pandas.DataFrame({"type": ["standard"], "x": [5]}),
                       ]:
        prediction = rerp.predict(predictors)
        assert isinstance(prediction, pandas.DataFrame)
        assert np.all(prediction.index == rerp.betas.major_axis)
        assert np.all(prediction.columns == rerp.betas.minor_axis)
        assert np.allclose(prediction,
                           rerp.betas["Intercept"] + 5 * rerp.betas["x"])

        prediction = rerp.predict(predictors, which_terms="0 + x")
        assert isinstance(prediction, pandas.DataFrame)
        assert np.all(prediction.index == rerp.betas.major_axis)
        assert np.all(prediction.columns == rerp.betas.minor_axis)
        assert np.allclose(prediction, 5 * rerp.betas["x"])

        prediction = rerp.predict(predictors, which_terms="0 + type + x")
        assert isinstance(prediction, pandas.DataFrame)
        assert np.allclose(prediction, 5 * rerp.betas["x"])

    assert_raises(ValueError, rerp.predict, {"type": ["standard", "standard"],
                                             "x": [1, 2]})

    prediction = rerp.predict_many({"type": ["standard", "target"],
                                    "x": [4, 5]})
    assert isinstance(prediction, pandas.Panel)
    assert np.all(prediction.items == [0, 1])
    assert np.all(prediction.major_axis == rerp.betas.major_axis)
    assert np.all(prediction.minor_axis == rerp.betas.minor_axis)
    assert np.allclose(prediction[0],
                       rerp.betas["Intercept"] + 4 * rerp.betas["x"])
    assert np.allclose(prediction[1],
                       rerp.betas["Intercept"]
                         + rerp.betas["type[T.target]"]
                         + 5 * rerp.betas["x"])
    # non-trivial index on input
    predictors = pandas.DataFrame({"type": ["standard", "target"],
                                   "x": [4, 5]},
                                  index=[10, 20])
    prediction2 = rerp.predict_many(predictors)
    assert np.all(prediction2.items == [10, 20])
    assert np.all(np.asarray(prediction2) == np.asarray(prediction))

    # broadcasting
    prediction = rerp.predict_many({"type": "standard", "x": [4, 5]})
    assert isinstance(prediction, pandas.Panel)
    assert np.all(prediction.items == [0, 1])
    assert np.all(prediction.major_axis == rerp.betas.major_axis)
    assert np.all(prediction.minor_axis == rerp.betas.minor_axis)
    assert np.allclose(prediction[0],
                       rerp.betas["Intercept"] + 4 * rerp.betas["x"])
    assert np.allclose(prediction[1],
                       rerp.betas["Intercept"] + 5 * rerp.betas["x"])

    # all scalars okay (well, this behaviour is kind of unfortunate in fact,
    # maybe we should return a dataframe in this case instead of a panel, see:
    #   https://github.com/pydata/patsy/issues/24
    # but in the mean time, oh well, let's test it):
    prediction = rerp.predict_many({"type": "standard", "x": 4})
    assert isinstance(prediction, pandas.Panel)
    assert np.all(prediction.items == [0])
    assert np.all(prediction.major_axis == rerp.betas.major_axis)
    assert np.all(prediction.minor_axis == rerp.betas.minor_axis)
    assert np.allclose(prediction[0],
                       rerp.betas["Intercept"] + 4 * rerp.betas["x"])

    # subsetting
    prediction = rerp.predict_many({"type": "standard", "x": [4, 5]},
                                   which_terms=["x"])
    assert np.all(prediction.items == [0, 1])
    assert np.all(prediction.major_axis == rerp.betas.major_axis)
    assert np.all(prediction.minor_axis == rerp.betas.minor_axis)
    assert np.allclose(prediction[0], 4 * rerp.betas["x"])
    assert np.allclose(prediction[1], 5 * rerp.betas["x"])

def test_diff_lengths():
    # Originally this caused by-epoch regression to blow up
    ds = mock_dataset(hz=1000)
    ds.add_event(0, 10, 11, {"type": "a"})
    ds.add_event(0, 20, 21, {"type": "a"})
    ds.add_event(0, 30, 31, {"type": "b"})
    ds.add_event(0, 40, 41, {"type": "b"})

    req_a = rERPRequest("type == 'a'", -1, 3, "1")
    req_b = rERPRequest("type == 'b'", -2, 4, "1")

    epoch_a1 = np.asarray(ds[0].iloc[10 - 1:10 + 4, :])
    epoch_a2 = np.asarray(ds[0].iloc[20 - 1:20 + 4, :])
    epoch_b1 = np.asarray(ds[0].iloc[30 - 2:30 + 5, :])
    epoch_b2 = np.asarray(ds[0].iloc[40 - 2:40 + 5, :])

    expected_a = (epoch_a1 + epoch_a2) / 2
    expected_b = (epoch_b1 + epoch_b2) / 2

    for regression_strategy in ["by-epoch", "continuous"]:
        a_alone, = ds.multi_rerp([req_a],
                                 regression_strategy=regression_strategy)
        b_alone, = ds.multi_rerp([req_b],
                                 regression_strategy=regression_strategy)
        a, b = ds.multi_rerp([req_a, req_b],
                             regression_strategy=regression_strategy)
        assert np.allclose(expected_a, a_alone.betas)
        assert np.allclose(expected_b, b_alone.betas)
        assert np.allclose(expected_a, a.betas)
        assert np.allclose(expected_b, b.betas)

def test_not_enough_data():
    ds = mock_dataset(hz=1000)
    # Values chosen to avoid perfect collinearity, that's tested separately
    ds.add_event(0, 10, 11, {"type": "a", "x1": 2, "x2": 3})
    ds.add_event(0, 20, 21, {"type": "a", "x1": 3, "x2": 9})
    ds.add_event(0, 12, 13, {"maybe_artifact": True})
    ds.add_event(0, 22, 23, {"maybe_artifact": True})

    for regression_strategy in ["by-epoch", "continuous"]:
        # No events match
        assert_raises(ValueError, ds.rerp, "type == 'b'", 0, 5, "1",
                      regression_strategy=regression_strategy)
        # No artifact-free data
        assert_raises(ValueError, ds.rerp, "type == 'a'", 0, 5, "1",
                      bad_event_query="type == 'a'",
                      regression_strategy=regression_strategy)
        # Some artifact-free data, but fewer data points than predictors
        assert_raises(ValueError, ds.rerp, "type == 'a'", 0, 5, "x1 + x2",
                      regression_strategy=regression_strategy)
        if regression_strategy == "continuous":
            # No artifact-free data at some latencies
            assert_raises(ValueError,
                          ds.rerp, "type == 'a'", 0, 5, "1",
                          artifact_query="maybe_artifact")

def test_perfect_collinearity():
    ds = mock_dataset(hz=1000)
    ds.add_event(0, 10, 11, {"x1": 1, "x2": 1, "type": "a"})
    ds.add_event(0, 20, 21, {"x1": 2, "x2": 2, "type": "a"})
    ds.add_event(0, 30, 31, {"x1": 3, "x2": 3, "type": "b"})
    ds.add_event(0, 40, 41, {"x1": 4, "x2": 4, "type": "b"})

    for regression_strategy in ["by-epoch", "continuous"]:
        # two predictors that are identical
        assert_raises(ValueError,
                      ds.rerp, "has x1", 0, 5, "x1 + x2",
                      regression_strategy=regression_strategy)
        # two predictors that are *almost* identical
        assert_raises(ValueError,
                      ds.multi_rerp,
                      # Call rERPRequest here to get correct evaluation
                      # environment for formula.
                      [rERPRequest("has x1", 0, 5,
                                   "x1 + I(x2 + np.finfo(float).eps)")],
                      regression_strategy=regression_strategy)
        # missing data in some cell
        assert_raises(ValueError,
                      ds.rerp, "has x1", 0, 5, "type",
                      bad_event_query="type == 'a'")
        assert_raises(ValueError,
                      ds.rerp, "has x1", 0, 5, "type",
                      bad_event_query="type == 'b'")
        assert_raises(ValueError,
                      ds.rerp, "has x1", 0, 5, "0 + type",
                      bad_event_query="type == 'a'")
        assert_raises(ValueError,
                      ds.rerp, "has x1", 0, 5, "0 + type",
                      bad_event_query="type == 'b'")

# XX TODO: make continuous do some amount of batching
