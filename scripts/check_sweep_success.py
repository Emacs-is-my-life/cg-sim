#!/usr/bin/env python3
"""Scan sweep result JSONs for simulation.success == True.

Usage: python scripts/check_sweep_success.py [<results_dir>]
       Default results_dir is "tmp/results".
"""
import json
import sys
from pathlib import Path


def find_simulation_result(obj):
    """Locate the SIMULATION_RESULT event anywhere in the JSON tree."""
    if isinstance(obj, dict):
        if obj.get("name") == "SIMULATION_RESULT" and "args" in obj:
            return obj["args"]
        for v in obj.values():
            r = find_simulation_result(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_simulation_result(v)
            if r is not None:
                return r
    return None


def main(results_dir: Path) -> int:
    files = sorted(results_dir.glob("*.json"))
    if not files:
        print(f"No JSON files found in {results_dir}")
        return 1

    total = len(files)
    passed = []
    failed = []
    missing = []

    for f in files:
        try:
            with f.open() as fh:
                data = json.load(fh)
        except Exception as e:
            missing.append((f.name, f"parse error: {e}"))
            continue

        result = find_simulation_result(data)
        if result is None:
            missing.append((f.name, "no SIMULATION_RESULT event"))
            continue

        # Note: engine writes the boolean as str(...), so it's the string "True"/"False".
        success = result.get("simulation", {}).get("success")
        sim_time = result.get("simulation", {}).get("time")

        if success == "True" or success is True:
            passed.append((f.name, sim_time))
        else:
            failed.append((f.name, success, sim_time))

    print(f"Scanned {total} file(s) in {results_dir}")
    print(f"  passed : {len(passed)}")
    print(f"  failed : {len(failed)}")
    print(f"  missing: {len(missing)}")

    if failed:
        print("\n-- Failed runs --")
        for name, success, sim_time in failed:
            print(f"  {name}: success={success!r} time={sim_time}")

    if missing:
        print("\n-- Missing / unparsable --")
        for name, reason in missing:
            print(f"  {name}: {reason}")

    return 0 if (not failed and not missing) else 2


if __name__ == "__main__":
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tmp/results")
    sys.exit(main(results_dir))
