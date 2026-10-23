"""Host supervisor: reserve guards, bounded runtime and fail-closed compile logs.

Only the dedicated research container is stopped on failure. Receipt/log content
is local; publish aggregate summaries through the receipt collector.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


FATAL = re.compile(r"Compilation failed|Failed to compile|neuronx-cc[^\n]*(?:failed|error)|"
                   r"\[ERROR\][^\n]*(?:compil|neff)|Compiler failed|BUG_NO_EXCEPTION", re.I)


def main():
    root = Path(os.environ.get("RESEARCH_ROOT", "/run/user/1000/autonomous-compute"))
    run = os.environ["RESEARCH_RUN"]
    logs = root/"logs"
    logs.mkdir(exist_ok=True)
    log = logs/f"{run}.log"
    monitor = logs/f"{run}.monitor.jsonl"
    if log.exists():
        raise FileExistsError(log)
    started = time.monotonic()
    reason = None
    container_pid = subprocess.check_output(["docker", "inspect", "--format", "{{.State.Pid}}",
                                             "autonomous-compute-neuron233"], text=True).strip()
    cgroup_name = Path(f"/proc/{container_pid}/cgroup").read_text().strip().split("::", 1)[1]
    cgroup = Path("/sys/fs/cgroup")/cgroup_name.lstrip("/")
    last_oom = {"oom_kill": 0}
    def oom_events():
        nonlocal last_oom
        try:
            last_oom = {k: int(v) for k, v in (line.split() for line in (cgroup/"memory.events").read_text().splitlines())}
        except FileNotFoundError:
            # docker stop may remove the cgroup before the final receipt.
            pass
        return last_oom
    initial_oom = oom_events()
    peak_container = 0
    with log.open("w") as output, monitor.open("w") as telemetry:
        process = subprocess.Popen(["bash", str(Path(__file__).with_name("run_native.sh")), *sys.argv[1:]],
                                   stdout=output, stderr=subprocess.STDOUT)
        offset, tail = 0, ""
        while process.poll() is None:
            mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            available = int(mem["MemAvailable"].split()[0])*1024
            paths = [root, Path("/"), Path('/run/user/1000/autonomous-compute')]
            frees = {str(p): shutil.disk_usage(p).free for p in paths}
            with log.open() as reader:
                reader.seek(offset)
                new = reader.read()
                offset = reader.tell()
            fatal = FATAL.search(tail+new)
            tail = (tail+new)[-32000:]
            sample = {"elapsed": time.monotonic()-started, "host_available_bytes": available,
                      "filesystem_free_bytes": frees, "log_bytes": offset}
            current_container = int((cgroup/"memory.current").read_text())
            peak_container = max(peak_container, current_container)
            sample["container_current_bytes"] = current_container
            telemetry.write(json.dumps(sample)+"\n")
            telemetry.flush()
            if available < 512*1024**3:
                reason = "host_memory_reserve"
            elif min(frees.values()) < 35*1024**3:
                reason = "filesystem_reserve"
            elif fatal:
                reason = "compiler_error_in_log_fail_closed"
            elif oom_events()["oom_kill"] > initial_oom["oom_kill"]:
                reason = "container_oom_kill"
            elif time.monotonic()-started > int(os.environ.get("RESEARCH_TIMEOUT_SECONDS", "1200"))+30:
                reason = "supervisor_timeout"
            if reason:
                subprocess.run(["docker", "stop", "--time", "5", "autonomous-compute-neuron233"],
                               capture_output=True, timeout=25)
                break
            time.sleep(2)
        code = process.wait(timeout=30)
        # Errors printed just before process exit must also invalidate a report.
        if FATAL.search(log.read_text()):
            reason = reason or "compiler_error_in_log_fail_closed"
    final_oom = oom_events()
    if final_oom["oom_kill"] > initial_oom["oom_kill"]:
        reason = reason or "container_oom_kill"
    result = {"worker_exit_code": code, "supervisor_failure": reason,
              "sampled_container_peak_bytes": peak_container,
              "container_oom_kill_delta": final_oom["oom_kill"]-initial_oom["oom_kill"],
              "elapsed_seconds": time.monotonic()-started}
    (logs/f"{run}.supervisor.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    return 1 if reason else code


if __name__ == "__main__":
    sys.exit(main())
