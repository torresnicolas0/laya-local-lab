"""Three bounded Laya experiments; predictions never execute external actions."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from lab import MODEL_NAMES, ROOT, load_agent, predict, read_json, save_json, sha256, verify_network_denied

FLOW_NAMES = ("guardrails", "rag", "routing")
SOURCE_FILES = ("workflow_cases.json", "workflows.json", "workflows.py", "lab.py", "models.lock.json", "uv.lock")


def verify_protocol():
    lock = read_json("workflow_protocol.lock.json")
    for name, expected in lock["hashes"].items():
        if sha256(name) != expected:
            raise ValueError(f"Frozen experiment changed: {name}. Preserve the original protocol and version a new experiment.")
    return lock


def validate_state(flow, state):
    fields = read_json("workflows.json")[flow]["fields"]
    if not isinstance(state, dict) or set(state) != set(fields):
        raise ValueError("Expected fields: " + ", ".join(fields))
    if any(not isinstance(state[field], str) or not state[field].strip() for field in fields):
        raise ValueError("Todos los campos deben contener texto.")


def decide(flow, answers):
    t = read_json("workflows.json")[flow]["thresholds"]
    if flow == "guardrails":
        if answers["attack"]["noul"] >= t["attack"]:
            return "BLOCK"
        return "REVIEW" if answers["sensitive_data"]["noul"] >= t["sensitive_data"] else "PASS"
    if flow == "rag":
        if answers["injection"]["noul"] >= t["injection"]:
            return "DROP_INJECTION"
        if answers["relevant"]["noul"] < t["relevant"]:
            return "DROP_IRRELEVANT"
        return "REVIEW_CONTRADICTION" if answers["contradicts"]["noul"] >= t["contradicts"] else "KEEP"
    if flow == "routing":
        if answers["needs_tools"]["noul"] >= t["needs_tools"]:
            return "TOOLS"
        return "LARGE" if answers["difficulty"]["score"] >= t["difficulty"] else "SMALL"
    raise ValueError(f"Unknown workflow: {flow}")


def run_case(agent, flow, state):
    validate_state(flow, state)
    result, elapsed = predict(agent, state, read_json("workflows.json")[flow]["questions"])
    return {"result": result, "action": decide(flow, result["answers"]), "latency_ms": elapsed}


def binary_metrics(pairs):
    tp = sum(p >= 0.5 and y for p, y in pairs)
    fp = sum(p >= 0.5 and not y for p, y in pairs)
    fn = sum(p < 0.5 and y for p, y in pairs)
    tn = len(pairs) - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy_at_0_5": (tp + tn) / len(pairs),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "brier": sum((p - int(y)) ** 2 for p, y in pairs) / len(pairs)}


def metrics(flow, rows):
    import numpy as np
    n = len(rows)
    correct = sum(r["action"] == r["expected"]["action"] for r in rows)
    confusion = {}
    for r in rows:
        key = r["expected"]["action"]
        confusion.setdefault(key, {})[r["action"]] = confusion.get(key, {}).get(r["action"], 0) + 1
    out = {"n": n, "action_correct": correct, "action_accuracy": correct / n,
           "majority_action_baseline": max(Counter(r["expected"]["action"] for r in rows).values()) / n,
           "confusion_expected_to_predicted": confusion,
           "latency_p50_ms": float(np.percentile([r["latency_ms"] for r in rows], 50)),
           "latency_p95_ms": float(np.percentile([r["latency_ms"] for r in rows], 95)),
           "errors": [r["id"] for r in rows if r["action"] != r["expected"]["action"]]}
    for q, spec in read_json("workflows.json")[flow]["questions"].items():
        if spec["type"] == "noul":
            out[q] = binary_metrics([(r["result"]["answers"][q]["noul"], r["expected"][q]) for r in rows])
        elif spec["type"] == "score":
            out[q + "_mae"] = sum(abs(r["result"]["answers"][q]["score"] - r["expected"][q]) for r in rows) / n
    if flow == "guardrails":
        out["missed_blocks"] = sum(r["expected"]["attack"] and r["action"] != "BLOCK" for r in rows)
        out["false_blocks"] = sum(not r["expected"]["attack"] and r["action"] == "BLOCK" for r in rows)
    elif flow == "rag":
        out["unsafe_keeps"] = sum((r["expected"]["injection"] or r["expected"]["contradicts"]) and r["action"] == "KEEP" for r in rows)
        positives = [r for r in rows if r["expected"]["action"] == "KEEP"]
        out["good_passage_retention"] = sum(r["action"] == "KEEP" for r in positives) / len(positives)
    else:
        out["missed_tool_routes"] = sum(r["expected"]["needs_tools"] and r["action"] != "TOOLS" for r in rows)
        out["false_tool_routes"] = sum(not r["expected"]["needs_tools"] and r["action"] == "TOOLS" for r in rows)
    return out


def evaluate(model, device, output, offline=False):
    if Path(output).exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    protocol = verify_protocol()
    proof = verify_network_denied() if offline else None
    start = time.perf_counter()
    agent = load_agent(device, model)
    load_seconds = time.perf_counter() - start
    all_rows, warmup = [], {}
    cases = read_json("workflow_cases.json")
    for flow in FLOW_NAMES:
        selected = [c for c in cases if c["workflow"] == flow]
        warmup[flow] = [run_case(agent, flow, selected[0]["state"])["latency_ms"] for _ in range(3)]
        for case in selected:
            row = {**case, **run_case(agent, flow, case["state"])}
            if str(agent.device) != device:
                raise RuntimeError("Model changed device.")
            all_rows.append(row)
            print(f"{model} {case['id']}: {row['action']}", flush=True)
    verify_protocol()
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "model_key": model,
              "model": read_json("models.lock.json")[model], "protocol": protocol,
              "device_requested": device, "device_actual": str(agent.device),
              "network_denial_proof": proof, "warmup_ms": warmup,
              "load_seconds_including_imports": load_seconds,
              "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1048576 if sys.platform == "darwin" else 1024),
              "environment": {"python": platform.python_version(), "os": platform.platform(),
                  "packages": {p: importlib.metadata.version(p) for p in ["laya", "torch", "transformers", "huggingface-hub", "numpy", "streamlit"]}},
              "workflows": {f: metrics(f, [r for r in all_rows if r["workflow"] == f]) for f in FLOW_NAMES},
              "rows": all_rows}
    save_json(output, report)


def compare(device, output_dir, offline=False):
    folder = Path(output_dir)
    if any((folder / (name + ".json")).exists() for name in (*MODEL_NAMES, "summary")):
        raise FileExistsError("Use a new output directory; results are never overwritten.")
    reports = {}
    for model in MODEL_NAMES:
        output = folder / (model + ".json")
        command = [sys.executable, str(ROOT / "workflows.py"), "evaluate", "--model", model,
                   "--device", device, "--output", str(output)]
        if offline:
            command.append("--verify-offline")
        subprocess.run(command, check=True)
        reports[model] = json.loads(output.read_text())
    if any(r["protocol"] != verify_protocol() for r in reports.values()):
        raise RuntimeError("Mixed experiment protocols.")
    save_json(folder / "summary.json", {"device": device, "language": "en", "inputs_per_model": 36,
        "model_order": list(MODEL_NAMES), "separate_sequential_processes": True,
        "models": {m: r["workflows"] for m, r in reports.items()}})
    print(json.dumps({m: {f: x["action_correct"] for f, x in r["workflows"].items()} for m, r in reports.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--model", choices=MODEL_NAMES, required=True)
    run.add_argument("--output", required=True)
    comp = sub.add_parser("compare")
    comp.add_argument("--output-dir", required=True)
    for p in (run, comp):
        p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
        p.add_argument("--verify-offline", action="store_true")
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args.model, args.device, args.output, args.verify_offline)
    else:
        compare(args.device, args.output_dir, args.verify_offline)


if __name__ == "__main__":
    main()
