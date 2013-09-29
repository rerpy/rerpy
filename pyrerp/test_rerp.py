# This file is part of pyrerp
# Copyright (C) 2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

from itertools import product

import numpy as np
import pandas

from nose.tools import assert_raises

from pyrerp.rerp import rERPRequest, multi_rerp
from pyrerp.test_data import mock_dataset
import pyrerp.parimap

def test_rerp_simple():
    ds = mock_dataset(num_channels=2, hz=1000)
    ds.add_event(0, 10, 11, {"type": "standard"})
    ds.add_event(0, 20, 21, {"type": "standard"})
    ds.add_event(0, 30, 31, {"type": "target"})
    ds.add_event(0, 40, 41, {"type": "target"})
    ds.add_event(0, 50, 51, {"type": "target"})
    ds.add_event(0, 51, 53, {"maybe_artifact": True})

    for (regression_strategy, overlap_correction, parimap_mode) in product(
        ["auto", "by-epoch", "continuous"],
        [True, False],
        ["multiprocess", "serial"]):

        pyrerp.parimap.configure(mode=parimap_mode)

        assert multi_rerp(ds, [],
                          regression_strategy=regression_strategy,
                          overlap_correction=overlap_correction) == []

        standard_req = rERPRequest("type == 'standard'", -2, 4)
        target_req = rERPRequest("type == 'target'", -2, 4)
        erps = multi_rerp(ds, [standard_req, target_req],
                          regression_strategy=regression_strategy,
                          overlap_correction=overlap_correction)
        standard_erp, target_erp = erps
        assert standard_erp.event_query == "type == 'standard'"
        assert target_erp.event_query == "type == 'target'"
        for i, erp in enumerate([standard_erp, target_erp]):
            assert erp.start_time == -2
            assert erp.stop_time == 4
            assert erp.formula == "1"
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
        erps = multi_rerp(ds, [both_req],
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
        erps = multi_rerp(ds, [both_req2],
                          regression_strategy=regression_strategy,
                          overlap_correction=overlap_correction)
        both_erp2, = erps
        assert both_erp2.design_info.column_names == ["type[standard]",
                                                      "type[target]"]
        assert np.allclose(both_erp2.betas["type[standard]"], standard_avg)
        assert np.allclose(both_erp2.betas["type[target]"], target_avg)

        # regular artifact (check accounting)
        if regression_strategy == "by-epoch":
            assert_raises(ValueError, multi_rerp, ds, [both_req2],
                          artifact_query="has maybe_artifact",
                          regression_strategy=regression_strategy,
                          overlap_correction=overlap_correction)
        else:
            both_erp3, = multi_rerp(ds, [both_req2],
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

        # all or nothing
        both_req4 = rERPRequest("has type", -2, 4, formula="~ 0 + type",
                                all_or_nothing=True)
        both_erp4, = multi_rerp(ds, [both_req4],
                                artifact_query="has maybe_artifact",
                                regression_strategy=regression_strategy,
                                overlap_correction=overlap_correction)
        if regression_strategy == "auto":
            assert both_erp4.regression_strategy == "by-epoch"
        assert np.allclose(both_erp4.betas["type[standard]"],
                           standard_avg)
        assert np.allclose(both_erp4.betas["type[target]"],
                           (target_epoch0 + target_epoch1) / 2.0)

        # bad_event_query
        both_req5 = rERPRequest("has type", -2, 4, formula="~ 0 + type",
                                bad_event_query="_START_TICK == 20")
        if regression_strategy == "by-epoch":
            assert_raises(ValueError, multi_rerp, ds, [both_req5],
                          artifact_query="has maybe_artifact",
                          regression_strategy=regression_strategy,
                          overlap_correction=overlap_correction)
        else:
            both_erp5, = multi_rerp(ds, [both_req5],
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

        # overlap

        # predict
