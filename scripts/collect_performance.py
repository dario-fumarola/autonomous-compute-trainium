"""Small receipts for the bounded performance/trainer follow-up only."""

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = dt.datetime(2026, 10, 22, 17, 50, tzinfo=dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((args.root/"snapshots").iterdir()):
        launch_file = path/"launch.json"
        if not launch_file.exists():
            continue
        launch = json.loads(launch_file.read_text())
        if launch["started_unix"] < start.timestamp():
            continue
        run = path.name
        paths = {"launch": launch_file, "closure": path/"closure.json",
                 "report": args.root/"runs"/run/"report.json",
                 "cpu-probe": args.root/"runs"/run/"cpu-probe.json",
                 "supervisor": args.root/"logs"/(run+".supervisor.json")}
        body = {}
        for label, source in paths.items():
            if source.exists():
                shutil.copyfile(source, args.output/f"{run}-{label}.json")
                body[label] = json.loads(source.read_text())
        closure, supervisor = body.get("closure", {}), body.get("supervisor", {})
        passed = (closure.get("exit_code") == 0 and supervisor.get("worker_exit_code") == 0
                  and supervisor.get("supervisor_failure") is None and "report" in body)
        row = {"run": run, "status": body["report"]["status"] if passed else "FAILED_OR_INCOMPLETE",
               "closure": closure, "supervisor": supervisor}
        monitor = args.root/"logs"/(run+".monitor.jsonl")
        if monitor.exists():
            samples = [json.loads(line) for line in monitor.read_text().splitlines()]
            row["monitor"] = {"samples": len(samples), "interval_seconds": 2,
                "minimum_host_available_bytes": min(s["host_available_bytes"] for s in samples),
                "minimum_filesystem_free_bytes": min(min(s["filesystem_free_bytes"].values()) for s in samples),
                "sampled_container_peak_bytes": max((s.get("container_current_bytes", 0) for s in samples), default=0)}
        rows.append(row)
    seen = {r["run"] for r in rows}
    for supervisor_file in sorted((args.root/"logs").glob("*.supervisor.json")):
        name = supervisor_file.name.removesuffix(".supervisor.json")
        if name in seen:
            continue
        log = args.root/"logs"/(name+".log")
        if not log.exists() or "reserve failed" not in log.read_text().lower():
            continue
        body = json.loads(supervisor_file.read_text())
        shutil.copyfile(supervisor_file, args.output/(name+"-supervisor.json"))
        rows.append({"run": name, "status": "PREFLIGHT_RESERVE_REJECTED",
                     "closure": {"assigned_chip_hours": 0}, "supervisor": body,
                     "scope": "Reserve rejected before docker exec; no accelerator worker launched. CPU/preflight time is in the inclusive node window."})
    ledger = {"utc": now.isoformat(), "campaign_start_utc": start.isoformat(), "runs": rows,
              "assigned_chip_hours_including_failures": sum(r["closure"].get("assigned_chip_hours", 0) for r in rows),
              "inclusive_campaign_opportunity_chip_hours": (now-start).total_seconds()*16/3600,
              "scope": "This follow-up only; do not add its opportunity window to an overlapping ledger. No new instance."}
    (args.output/"resource-ledger.json").write_text(json.dumps(ledger, indent=2)+"\n")
    print(json.dumps({k: v for k, v in ledger.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
