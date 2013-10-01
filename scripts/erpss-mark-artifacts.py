#!/usr/bin/env python

# This file is part of rERPy
# Copyright (C) 2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# In the ERPSS system, .arf files specify a set of hand-tuned parameters for
# detecting artifacts, and the actual artifacts aren't detecting until
# averaging. To let people use their existing files with rERPy, this
# script uses the ERPSS 'avg' tool (which must be available on the $PATH)
# to generate a .log file with artifacts marked. This file can then be loaded
# into rERPy, e.g.:
#   # Shell:
#   python erpss-mark-artifacts.py 100 foo.crw foo.log foo.arf foo-arfed.log
#   # Python:
#   dataset = load_erpss("foo.crw", "foo-arfed.log")
#   dataset.rerp(<your analysis here>, bad_event_query="flag_rejected")
# Note that you're pretty much stuck using whichever epoch you originally
# designed your .arf file for (though rERPy will not enforce this).
#
# For usage information:
#   python erpss-mark-artifacts.py --help

import sys
import argparse
from tempfile import mkdtemp
import os
import os.path
import shutil
import subprocess
import struct

parser = argparse.ArgumentParser(
    description="Given raw, log, and arf files, create a new .log file "
    "with artifacts flagged, suitable for loading into rERPy.")
parser.add_argument("presampling",
                    help="Presampling interval to pass to avg. Use whatever "
                    "you would use if you were actually using avg.")
parser.add_argument("raw",
                    help="The raw or compressed raw file containing data")
parser.add_argument("log",
                    help="The original log file containing events")
parser.add_argument("arf",
                    help="The artifact rejection file, as used by garv or avg")
parser.add_argument("output_log",
                    help="The new log file to create, with artifacts marked.")
parser.add_argument("--overwrite-really-i-mean-it",
                    action="store_true", dest="overwrite",
                    help="Blow away any existing file when writing out the "
                    "output_log. For those who like to live dangerously. "
                    "(Default: refuse to overwrite any existing file.)")
parser.add_argument("-c",
                    help="Passed to avg. Use whatever you would use if "
                    "you were actually using avg.")
parser.add_argument("-r",
                    help="Passed to avg. Use whatever you would use if "
                    "you were actually using avg.")
args = parser.parse_args()

# Get sampling rate
with open(args.raw, "rb") as raw_f:
    raw_f.seek(18)
    ten_usec_per_tick, = struct.unpack("<H", raw_f.read(2))
hz = 1 / (ten_usec_per_tick / 100000.0)
if abs(hz - int(round(hz))) > 1e-6:
    sys.exit("Can't read hz: got weird value %f" % (hz,))
hz = int(round(hz))
sys.stderr.write("Sampling rate: %s hz (auto-detected)\n" % (hz,))

# Figure out how many events are in the log
# Each log entry is 8 bytes long
log_entries = os.stat(args.log).st_size // 8
sys.stderr.write("# of log entries: %s (auto-detected)\n" % (log_entries,))

tmpdir = mkdtemp(prefix="erpss-mark-artifacts-tmp")
try:
    blf = os.path.join(tmpdir, "all.blf")
    sys.stderr.write("generating blf\n")
    with open(blf, "w") as blf_f:
        blf_f.write("1\n"
                    "CD 0\n"
                    "all\n"
                    "SD 0\n"
                    "all\n")
        for i in xrange(log_entries):
            blf_f.write("%s 0\n" % (i,))
    if not args.overwrite and os.path.exists(args.output_log):
        sys.exit("Output file %s already exists" % (args.output_log,))
    shutil.copyfile(args.log, args.output_log)
    avg = os.path.join(tmpdir, "all.avg")
    avg_cmd = ["avg", args.presampling, avg,
               "-x",
               "-a", args.arf,
               ]
    if args.c is not None:
        avg_cmd += ["-c", args.c]
    if args.r is not None:
        avg_cmd += ["-r", args.r]
    sys.stderr.write("Running: %s\n" % (" ".join(avg_cmd),))
    avg_proc = subprocess.Popen(avg_cmd,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
    (stdout, _) = avg_proc.communicate(
        "%s %s %s\n"
        "none\n" % (args.raw, args.output_log, blf))
    if avg_proc.returncode != 0:
        sys.stderr.write("avg failed with return code %s\n"
                         % (avg_proc.returncode,))
        sys.stderr.write("avg stdout:\n")
        sys.stderr.write(stdout)
        sys.stderr.write("FAILED\n")
        sys.exit(1)
    stdout_lines = stdout.split("\n")
    TAIL = 4
    sys.stderr.write("[... %s lines of output suppressed ...]\n"
                     % (len(stdout_lines) - TAIL,))
    sys.stderr.write("\n".join(stdout_lines[-TAIL:]))
    sys.stderr.write("\n")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
