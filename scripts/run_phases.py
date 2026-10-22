"""Monitored, process-isolated checkpoint/eval/late-switch controller.

The plan is explicit JSON, not an autonomous permission to launch a large fit.
Each phase commits a full checkpoint, exits, evaluates DEV in a fresh process,
and resumes with the same stream/optimizer. Changing pause count never keeps the
old compiled graph resident. A failed phase or evaluation stops the controller.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def run(plan, output):
    root = Path(os.environ.get("RESEARCH_ROOT", "/run/user/1000/autonomous-compute"))
    supervisor = Path(__file__).with_name("supervise.py")
    previous = plan.get("resume")
    events = []
    output.mkdir(parents=True, exist_ok=False)
    common = plan["training_arguments"]
    def memory_limit(gib):
        subprocess.run(["docker", "update", "--memory", str(gib*1024**3),
                        "autonomous-compute-neuron233"], check=True, capture_output=True)
    if not plan.get("qualification_only", True):
        manifest = json.loads(Path(plan["scientific_manifest"]).read_text())
        required = ("immutable_sources", "near_duplicate_audit", "frozen_order_sha256",
                    "lr_selected_at_this_batch", "measured_budget", "trainer_qualified")
        if not all(manifest.get(k) for k in required):
            raise ValueError("Scientific launch manifest gates incomplete")
        if "--qualification-repeat" in common:
            raise ValueError("Scientific training may not silently repeat the corpus")
        if "--packed-corpus" not in common:
            raise ValueError("Scientific training requires mapped packing, not per-rank Python corpora")
    for phase in plan["phases"]:
        run_id = phase["run"]
        memory_limit(plan.get("training_memory_gib", 256))
        env = dict(os.environ, RESEARCH_TRAIN="1", RESEARCH_EVAL="0", RESEARCH_RUN=run_id,
                   RESEARCH_DDP_RANKS=str(plan["world"]),
                   NEURON_RT_VISIBLE_CORES=",".join(map(str, range(plan["world"]))))
        command = ["python3", str(supervisor), *common, "--steps", str(phase["steps"]),
                   "--pauses", str(phase["pauses"]), "--save"]
        resume = phase.get("resume", previous)
        if resume:
            command += ["--resume", resume]
        if phase.get("decay_start") is not None:
            command += ["--decay-start", str(phase["decay_start"]),
                        "--decay-end", str(phase["decay_end"]), "--decay-branch"]
        subprocess.run(command, env=env, check=True)
        phase_report = json.loads((root/"runs"/run_id/"report.json").read_text())
        if phase_report["status"] != "PASS_PACKED_TRAINING_PHASE":
            raise AssertionError("Training phase did not pass")
        if phase.get("decay_end") and phase_report["total_real_targets"] < phase["decay_end"]:
            raise AssertionError("Decay endpoint is incomplete; cannot compare this branch")
        checkpoint = f"/work/runs/{run_id}/checkpoint.pt"
        if not phase.get("branch", False):
            previous = checkpoint
        evaluation = run_id+"-dev"
        memory_limit(plan.get("evaluation_memory_gib", 64))
        eval_env = dict(env, RESEARCH_TRAIN="0", RESEARCH_EVAL="1", RESEARCH_RUN=evaluation,
                        NEURON_RT_VISIBLE_CORES="0")
        evaluation_command = ["python3", str(supervisor), "--checkpoint", checkpoint,
                              "--max-documents-per-domain", str(plan.get("dev_documents_per_domain", 0))]
        if "--dev-corpus" in common:
            evaluation_command += ["--corpus", common[common.index("--dev-corpus")+1]]
        subprocess.run(evaluation_command, env=eval_env, check=True)
        eval_report = json.loads((root/"runs"/evaluation/"report.json").read_text())
        if eval_report["status"] != "PASS_CHECKPOINT_DEV_EVALUATION":
            raise AssertionError("DEV phase did not pass")
        events.append({"run": run_id, "checkpoint": checkpoint, "branch": phase.get("branch", False),
                       "real_tokens": phase_report["total_real_targets"],
                       "dev": eval_report["metrics"], "completed_unix": time.time()})
        (output/"progress.json").write_text(json.dumps(events, indent=2)+"\n")
    (output/"report.json").write_text(json.dumps({"status": "PASS_ISOLATED_PHASES", "phases": events}, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.plan.read_text()), args.output)
