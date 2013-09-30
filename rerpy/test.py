# This file is part of pyrerp
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

import os
import os.path

def test_data_path(path):
    test_data_dir = os.environ.get("PYRERP_TEST_DATA",
                                   # if unset, then maybe we're running from
                                   # the source directory
                                   os.path.join(os.path.dirname(__file__),
                                                "../test-data"))
    if not os.path.exists(test_data_dir):
        from nose.plugins.skip import SkipTest
        raise SkipTest("can't find test data!")
    return os.path.join(test_data_dir, path)

test_data_path.__test__ = False
