"""Verify raw outputs, fixed protocol and preservation of earlier experiments."""
import argparse
import json
from pathlib import Path
import subprocess

from lab import MODEL_NAMES, ROOT, read_json, save_json
from workflows import FLOW_NAMES, metrics, verify_protocol


def verify(cpu_dir, mps_dir, offline_dir, output):
    protocol = verify_protocol()
    cases = read_json("workflow_cases.json")
    checked = {}
    for model in MODEL_NAMES:
        reports = [json.loads((Path(folder) / (model + ".json")).read_text())
                   for folder in (cpu_dir, mps_dir, offline_dir)]
        cpu, mps, offline = reports
        for report, device in zip(reports, ("cpu", "mps", "cpu")):
            assert report["protocol"] == protocol
            assert report["model_key"] == model
            assert report["model"] == read_json("models.lock.json")[model]
            assert report["device_requested"] == report["device_actual"] == device
            assert [{k: row[k] for k in case} for row, case in zip(report["rows"], cases)] == cases
            assert len(report["rows"]) == 36
            for flow in FLOW_NAMES:
                assert report["workflows"][flow] == metrics(flow, [r for r in report["rows"] if r["workflow"] == flow])
        assert offline["network_denial_proof"] == "OS sandbox denied IPv4 and IPv6 connect with EPERM/EACCES"
        assert [r["result"] for r in cpu["rows"]] == [r["result"] for r in offline["rows"]]
        assert [r["action"] for r in cpu["rows"]] == [r["action"] for r in mps["rows"]]
        checked[model] = {"full_cpu_offline_responses_identical": 36, "cpu_mps_actions_identical": 36,
                          "metrics_recomputed": True, "same_cases_labels_and_protocol": True,
                          "offline_network_denial": offline["network_denial_proof"]}
    previous = protocol["previous_commit"]
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", previous], cwd=ROOT).decode().splitlines()
    unchanged = []
    for name in files:
        if name in ("app.py", "README.md"):
            continue
        assert (ROOT / name).read_bytes() == subprocess.check_output(["git", "show", previous + ":" + name], cwd=ROOT), name
        unchanged.append(name)
    save_json(output, {"protocol_frozen_at_utc": protocol["frozen_at_utc"], "models": checked,
                       "previous_commit": previous, "earlier_files_unchanged": unchanged})
    print(f"Verified 108 complete offline responses, 108 CPU/MPS decisions and {len(unchanged)} earlier files.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu-dir", default="results/workflows-cpu")
    p.add_argument("--mps-dir", default="results/workflows-mps")
    p.add_argument("--offline-dir", default="results/workflows-offline")
    p.add_argument("--output", default="results/workflows-verification.json")
    args = p.parse_args()
    verify(args.cpu_dir, args.mps_dir, args.offline_dir, args.output)
