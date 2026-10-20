"""Run the whole analysis from the parsed CAMS table to every figure and table, in order.

Parsing (parse_cams.py) is slow and changes only when CAMS-BD does, so it is skipped unless
--parse is given.  Independent analyses in the middle run in parallel.  Stops at the first
failure.  The paper itself is built afterwards by paper/build.py, which also verifies
every number in the text.

Usage (from 11_natexp):  python -I code/run_all.py [--parse]
"""
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
ROOT = os.path.join(BASE, "..")
PY = sys.executable
LOG = os.path.join(BASE, "out", "run_all.log")


def cmd(script, *args):
    return [PY, "-I", os.path.join(HERE, script), *args]


SEQUENCE = [
    [cmd("parse_cams.py", os.path.join(ROOT, "02_data", "raw", "cams_bd", "CAMS-BD"))] if "--parse" in sys.argv else [],
    [cmd("cams_qc.py")],
    [cmd("cams_validate.py", ROOT)],
    [cmd("screens.py")],
    # Google's mobility CSVs are not redistributed with the deposit; without them the
    # mobility results already in out/ are kept
    [cmd("mobility.py", os.path.join(ROOT, "02_data", "raw", "google_mobility"),
         os.path.join(ROOT, "Traffic", "data_availability", "02_event_calendars"))]
    if glob.glob(os.path.join(ROOT, "02_data", "raw", "google_mobility", "*_BD_*.csv")) else [],
    # independent analyses, in parallel
    [cmd("robustness.py", ROOT), cmd("events_hourly.py", ROOT), cmd("hourly_profiles.py", ROOT),
     cmd("tropomi_events.py", ROOT), cmd("narsingdi_eid.py", ROOT), cmd("cams_shocks.py", ROOT)],
    # depend on the analyses above
    [cmd("blh_weighting.py", ROOT), cmd("aqi_scenarios.py", ROOT), cmd("events_checks.py", ROOT),
     cmd("events_tracers.py", ROOT), cmd("reconstruct.py", ROOT), cmd("mechanism.py", ROOT),
     cmd("audit.py", ROOT), cmd("clock_check.py", ROOT), cmd("aqi_cap.py", ROOT)],
    # bootstrap intervals for the hourly profiles (parallel)
    [cmd("hourly_boot.py", ROOT)],
    [cmd("fig1_domain.py", ROOT), cmd("fig5_events.py"), cmd("figs_2_3.py"), cmd("figs_supp.py"),
     cmd("make_table3.py")],
]


def main():
    with open(LOG, "w", encoding="utf-8") as log:
        for group in SEQUENCE:
            if not group:
                continue
            t0 = time.time()
            procs = [(c, subprocess.Popen(c, cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, encoding="utf-8", errors="replace")) for c in group]
            failed = False
            for c, p in procs:
                out, _ = p.communicate()
                name = os.path.basename(c[2])
                log.write("===== %s (exit %d)\n%s\n" % (name, p.returncode, out))
                print("%-22s exit %d" % (name, p.returncode), flush=True)
                failed |= p.returncode != 0
            print("   group done in %.0f s" % (time.time() - t0), flush=True)
            if failed:
                print("stopping: see out/run_all.log")
                return 1
    print("all analyses done; now run paper/build.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
