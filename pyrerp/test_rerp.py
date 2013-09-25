# This file is part of pyrerp
# Copyright (C) 2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

import numpy as np
import pandas

from pyrerp.rerp import rERPRequest, multi_rerp
from pyrerp.test_data import mock_dataset

def test_rerp_simple():
    ds = mock_dataset(num_channels=2, hz=1000)
    ds.add_event(0, 10, 11, {"type": "standard"})
    ds.add_event(0, 20, 21, {"type": "standard"})
    ds.add_event(0, 30, 31, {"type": "target"})
    ds.add_event(0, 40, 41, {"type": "target"})
    ds.add_event(0, 50, 51, {"type": "target"})
    ds.add_event(0, 51, 55, {"maybe-artifact": True})

    standard_req = rERPRequest("type == 'standard'", -2, 4)
    target_req = rERPRequest("type == 'target'", -2, 4)
    for regression_strategy in ["auto", "by-epoch", "continuous"]:
        for overlap_correction in [True, False]:
            try:
                erps = multi_rerp(ds, [standard_req, target_req],
                                  regression_strategy=regression_strategy,
                                  overlap_correction=overlap_correction)
            except Exception, e:
                #import pdb; pdb.set_trace()
                #print str(e)
                raise
            standard_erp, target_erp = erps
            assert standard_erp.event_query == "type == 'standard'"
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
