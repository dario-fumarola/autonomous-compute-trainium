"""Publish only aggregate experiment receipts, including failed attempts."""

import argparse
import datetime as dt
import json
import shutil
import time
from pathlib import Path


def collect(root, experiment, output, start_utc):
    specification = json.loads((experiment/"specification.json").read_text())
    identity = specification["identity"]
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    start = dt.datetime.fromisoformat(start_utc).timestamp()
    end = time.time()
    for snapshot in sorted((root/"snapshots").iterdir()):
        if not (snapshot.name.startswith(identity+"-") or snapshot.name == "learning-pad1536-qualification-001"):
            continue
        paths = {"launch": snapshot/"launch.json", "closure": snapshot/"closure.json",
                 "report": root/"runs"/snapshot.name/"report.json",
                 "cpu-probe": root/"runs"/snapshot.name/"cpu-probe.json",
                 "supervisor": root/"logs"/(snapshot.name+".supervisor.json")}
        body = {}
        for label, path in paths.items():
            if path.exists():
                body[label] = json.loads(path.read_text())
                shutil.copyfile(path, output/(snapshot.name+"-"+label+".json"))
        launch, closure = body.get("launch", {}), body.get("closure", {})
        supervisor = body.get("supervisor", {})
        clean = (closure.get("exit_code") == 0 and supervisor.get("worker_exit_code") == 0
                 and supervisor.get("supervisor_failure") is None and "report" in body)
        row = {"run": snapshot.name, "status": body["report"]["status"] if clean else "FAILED_OR_INCOMPLETE",
               "closure": closure, "supervisor": supervisor}
        monitor = root/"logs"/(snapshot.name+".monitor.jsonl")
        if monitor.exists():
            samples = [json.loads(line) for line in monitor.read_text().splitlines()]
            if samples:
                row["monitor"] = {"samples": len(samples), "interval_seconds": 2,
                    "minimum_host_available_bytes": min(s["host_available_bytes"] for s in samples),
                    "minimum_filesystem_free_bytes": min(min(s["filesystem_free_bytes"].values()) for s in samples),
                    "sampled_container_peak_bytes": max(s.get("container_current_bytes", 0) for s in samples)}
        if launch:
            start = min(start, launch["started_unix"])
        rows.append(row)
    if (experiment/"complete.json").exists():
        # End at the final worker closure, not the time a collector is rerun.
        end = max(json.loads((root/"snapshots"/r["run"]/"launch.json").read_text())["started_unix"]+
                  r["closure"]["wall_seconds"] for r in rows if r["closure"])
    data = json.loads((output/"corpus-manifest.json").read_text())
    for name in ("specification.json", "selected-lrs.json", "tuning.json", "launch-manifest.json", "complete.json"):
        path = experiment/name
        if path.exists():
            shutil.copyfile(path, output/name)
    analysis = experiment/"analysis"
    if analysis.exists():
        for name in ("summary.json", "learning-curve.csv", "learning-curve.svg", "learning-curve.png",
                     "domain-curves.svg", "domain-curves.png", "source-group-uncertainty.json"):
            if (analysis/name).exists():
                shutil.copyfile(analysis/name, output/name)
    ledger = {"campaign": identity, "start_utc": dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(),
              "last_worker_or_collection_utc": dt.datetime.fromtimestamp(end, dt.timezone.utc).isoformat(),
              "runs": rows, "assigned_chip_hours_including_failures": sum(r["closure"].get("assigned_chip_hours", 0) for r in rows),
              "inclusive_node_hours": (end-start)/3600,
              "inclusive_opportunity_chip_hours": (end-start)*16/3600,
              "data_preparation_wall_seconds": data["preparation_wall_seconds"],
              "scope": "This learning-curve campaign and its padding qualification only; corpus preparation overlaps this node window and is not added twice. Preserved earlier campaigns are separate. No new instance."}
    (output/"resource-ledger.json").write_text(json.dumps(ledger, indent=2)+"\n")
    return ledger


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start-utc", required=True, help="Inclusive data-preparation/campaign start, with timezone")
    args = p.parse_args()
    collect(args.root, args.experiment, args.output, args.start_utc)
