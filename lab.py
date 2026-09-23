"""Small, fixed-protocol Laya experiment. See README.md for the recipe."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import socket
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def read_json(name):
    return json.loads((ROOT / name).read_text())


def sha256(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def model_path(download=False):
    from huggingface_hub import snapshot_download
    lock = read_json("model.lock.json")
    path = snapshot_download(
        lock["repository"], revision=lock["revision"],
        allow_patterns=[lock["subfolder"] + "/*"],
        local_files_only=not download, token=False,
    )
    return Path(path) / lock["subfolder"]


def load_agent(device="cpu"):
    import laya
    import torch
    torch.set_num_threads(4)
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; run --device cpu instead.")
    agent = laya.load(str(model_path()), device=device)
    if str(agent.device) != device:
        raise RuntimeError(f"Requested {device}, but Laya selected {agent.device}.")
    return agent


def validate_input(agent, state, questions):
    """Reject empty or over-budget inputs instead of upstream's silent truncation."""
    from laya.common import render_options, serialize_state
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("Input must be non-empty text, an object, or a list.")
    if isinstance(state, str) and not state.strip():
        raise ValueError("Input text must not be blank.")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Provide at least one typed question.")
    tok = agent.tok
    encode = lambda text: tok(text.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
    state_tokens = len(encode(serialize_state(state)))
    for qid, question in questions.items():
        agent._check_question(qid, question)
        q = agent._to_internal(question)
        head = len(encode(f"{q['t']} question: {q['ins']}"))
        options = [len(encode(" " + opt)) for opt in render_options(q)]
        if any(length > 48 for length in options):
            raise ValueError(f"{qid}: an option exceeds Laya's 48-token limit.")
        option_tokens = sum(length + 1 for length in options)
        head_max = agent.cfg.get("head_max_len", 192)
        if option_tokens > head_max - 16 or head > max(8, head_max - option_tokens):
            raise ValueError(f"{qid}: questions/options exceed the head token budget.")
        if head + option_tokens + state_tokens + 4 > agent.cfg.get("max_len", 512):
            raise ValueError(f"{qid}: input exceeds the model context; shorten it.")


def predict(agent, state, questions):
    import torch
    validate_input(agent, state, questions)
    if agent.device.type == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    result = agent.predict(state, questions)
    if agent.device.type == "mps":
        torch.mps.synchronize()
    if agent.device.type != "cpu" and str(agent.device) != "mps":
        raise RuntimeError(f"Unexpected device: {agent.device}")
    return result, (time.perf_counter() - started) * 1000


def cases():
    return [
        {"id": f"{case['id']}_{lang}", "scenario": case["id"], "language": lang,
         "tags": case["tags"], "state": {"body": text}, "expected": case["expected"]}
        for case in read_json("cases.json") for lang, text in case["texts"].items()
    ]


def metrics(rows):
    import numpy as np
    n = len(rows)
    correct = sum(r["result"]["answers"]["department"]["choice"] == r["expected"]["department"] for r in rows)
    urgencies = [abs(r["result"]["answers"]["urgency"]["score"] - r["expected"]["urgency"]) for r in rows]
    refund = [(r["result"]["answers"]["refund_requested"]["noul"], int(r["expected"]["refund_requested"])) for r in rows]
    counts = Counter(r["expected"]["department"] for r in rows)
    return {
        "n": n, "department_correct": correct, "department_accuracy": correct / n,
        "majority_department_baseline": max(counts.values()) / n,
        "urgency_mae_0_to_2": sum(urgencies) / n,
        "refund_accuracy_threshold_0_5": sum((p >= 0.5) == bool(y) for p, y in refund) / n,
        "refund_always_no_baseline": sum(y == 0 for _, y in refund) / n,
        "refund_brier": sum((p - y) ** 2 for p, y in refund) / n,
        "latency_p50_ms": float(np.percentile([r["latency_ms"] for r in rows], 50)),
        "latency_p95_ms": float(np.percentile([r["latency_ms"] for r in rows], 95)),
    }


def verify_network_denied():
    """Must be run inside the OS network-denying sandbox, not just HF offline mode."""
    import errno
    for family, address in [(socket.AF_INET, ("1.1.1.1", 443)), (socket.AF_INET6, ("2606:4700:4700::1111", 443))]:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            try:
                sock.connect(address)
            except OSError as error:
                if error.errno not in (errno.EPERM, errno.EACCES):
                    raise RuntimeError(f"Network denial not proven: {error}") from error
            else:
                raise RuntimeError("Network is available; refusing to label this offline.")
    return "OS sandbox denied IPv4 and IPv6 connect with EPERM/EACCES"


def evaluate(device, output, offline=False):
    if Path(output).exists():
        raise FileExistsError(f"Refusing to overwrite results: {output}")
    network_proof = verify_network_denied() if offline else None
    started = time.perf_counter()
    agent = load_agent(device)
    load_seconds = time.perf_counter() - started
    questions = read_json("questions.json")
    warmup_ms = []
    for _ in range(3):
        _, latency = predict(agent, {"body": "Please send information about your product."}, questions)
        warmup_ms.append(latency)
    rows = []
    for case in cases():
        result, latency = predict(agent, case["state"], questions)
        if str(agent.device) != device:
            raise RuntimeError("The model changed device during evaluation.")
        rows.append({**case, "result": result, "latency_ms": latency})
        print(f"{case['id']}: {result['answers']['department']['choice']} ({latency:.1f} ms)", flush=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"scenarios": 10, "translated_inputs": 30, "questions_per_request": 3,
                     "warmup_requests": 3, "repeats_per_input": 1, "torch_threads": 4,
                     "question_language": "en", "device_requested": device, "device_actual": str(agent.device),
                     "network_denial_proof": network_proof,
                     "timing_scope": "SDK predict for all three questions; prevalidation excluded; MPS synchronized"},
        "environment": {"python": platform.python_version(), "os": platform.platform(),
                        "machine": platform.machine(),
                        "packages": {p: importlib.metadata.version(p) for p in ["laya", "torch", "transformers", "huggingface-hub", "numpy", "streamlit"]}},
        "model": read_json("model.lock.json"),
        "hashes": {p: sha256(p) for p in ["cases.json", "questions.json", "model.lock.json", "uv.lock", "lab.py"]},
        "load_seconds_including_imports": load_seconds, "warmup_ms": warmup_ms,
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1048576 if sys.platform == "darwin" else 1024),
        "overall": metrics(rows),
        "by_language": {lang: metrics([r for r in rows if r["language"] == lang]) for lang in ["es", "pt", "en"]},
        "rows": rows,
    }
    save_json(output, report)
    print(json.dumps(report["overall"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Download only the pinned multilingual checkpoint")
    run = sub.add_parser("evaluate", help="Run the fixed 30-input evaluation")
    run.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    run.add_argument("--output", required=True)
    run.add_argument("--verify-offline", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(model_path(download=True))
    else:
        evaluate(args.device, args.output, args.verify_offline)


if __name__ == "__main__":
    main()
