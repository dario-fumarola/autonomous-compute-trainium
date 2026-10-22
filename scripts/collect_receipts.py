"""Publish small aggregate receipts, never corpora, states or compiler caches."""

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, measured = [], 0.
    for path in sorted((args.root/"snapshots").iterdir()):
        name = path.name
        launch = path/"launch.json"
        closure = path/"closure.json"
        report = args.root/"runs"/name/"report.json"
        row = {"run": name, "supervised": launch.exists()}
        for label, source in [("launch", launch), ("closure", closure), ("report", report),
                              ("cpu-agreement", args.root/"runs"/name/"cpu-agreement.json")]:
            if source.exists():
                shutil.copyfile(source, args.output/(name+"-"+label+".json"))
        if closure.exists():
            row.update(json.loads(closure.read_text()))
            row["status"] = (json.loads(report.read_text())["status"] if row["exit_code"] == 0
                             else "PASS_HARDWARE_WRAPPER_FAILED") if report.exists() else "FAILED_CAPTURED"
            measured += row["assigned_chip_hours"]
        elif report.exists():
            body = json.loads(report.read_text())
            row.update(status=body["status"], body_wall_seconds=body.get("wall_seconds"),
                       timing_scope="Body only; supervisor instrumentation was not yet enabled.")
        else:
            row.update(status="RUNNING_NO_FINAL_REPORT" if launch.exists() else "FAILED_UNINSTRUMENTED", wall_seconds=None)
        rows.append(row)
    now = dt.datetime.now(dt.timezone.utc)
    # This elapsed node window includes setup, writing, data work, gaps, overlaps
    # and the uninstrumented mini-probes. Do not sum overlapping node windows.
    window_start = dt.datetime(2026, 10, 22, 16, 17, 8, tzinfo=dt.timezone.utc)
    summary = {"utc": now.isoformat(), "runs": rows,
               "predecessor_assigned_chip_hours": 11.863125911612538,
               "new_supervised_assigned_chip_hours": measured,
               "new_total_assigned_chip_hours_exact": None,
               "uninstrumented_checks": ["Initial native import/matrix setup probes", "First two tiny TorchTitan attempts", "Compiler-error reproducer"],
               "timing_limit": "Uninstrumented mini-check costs are not zero; exact assigned time was not captured. They are covered by the inclusive whole-node elapsed window.",
               "inclusive_whole_node_window_start_utc": window_start.isoformat(),
               "inclusive_whole_node_opportunity_chip_hours": (now-window_start).total_seconds()*16/3600,
               "window_scope": "Existing 16-chip node elapsed time since predecessor closure, including idle/code/setup time. Not additional cloud capacity or a dollar bill."}
    (args.output/"resource-ledger.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
