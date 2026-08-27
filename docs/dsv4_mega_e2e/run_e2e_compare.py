#!/usr/bin/env python3
"""DSV4 e2e: normal vs Mega, or controlled four-kernel-front comparison.

Starts the RTP server twice and replays the same greedy queries. By default the
first leg has Mega switches off and the second enables CSA/HCA. With
``E2E_MOE_FRONT=1``, both legs instead enable CSA/HCA and MegaMoE-SE, and the
only changed execution switch is ``DSV4_MEGA_MOE_FRONT``.

Environment:
    E2E_CKPT    (required) checkpoint dir, e.g. a DeepSeek-V4-Flash checkout
    E2E_GPU     CUDA device list (default 0; use 0,1,2,3 for EP4)
    E2E_EP_SIZE expert-parallel size (default 1; native MoE front needs >1)
    E2E_TP_SIZE tensor-parallel size (default 1)
    E2E_DP_SIZE data-parallel size (default 1; EP4 recipe uses 4)
    E2E_WORLD_SIZE distributed world size (default E2E_EP_SIZE)
    E2E_LOCAL_WORLD_SIZE local ranks on this host (default E2E_WORLD_SIZE)
    E2E_MOE_FRONT set to 1 to enable the native four-kernel Pro/Flash MoE front
    E2E_PYTHON  python of a serving-capable venv (default: this interpreter)
    E2E_SOURCE_ROOT staged RTP source root (default: infer from this script)
    E2E_KERNEL_ROOT optional clean-wheel install root for extension overrides
    E2E_OUT     output dir for logs/results (default ./e2e_out)
    E2E_JIT_CACHE  base dir for the managed JIT caches (default ~/.cache/rtp_jit)
    E2E_CUDA_GRAPH set to 1 to capture and replay the decode path
    E2E_DECODE_CAPTURE_CONFIG graph batch buckets (default: 1 for this runner)
    E2E_PERF    set to 1 for a controlled 64-token warmup + 3x256-token run
    E2E_PERF_ONLY  skip functional queries and run only the controlled TPS case
    E2E_PERF_WARMUPS number of identical warmup requests (default 1)
    E2E_PERF_SAMPLES number of measured requests (default 3)
"""

import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

VENV_PY = os.environ.get("E2E_PYTHON", sys.executable)
CKPT = os.environ.get("E2E_CKPT")
if not CKPT:
    sys.exit("E2E_CKPT must point at a DSV4 checkpoint directory")


def validate_checkpoint(path: str) -> None:
    checkpoint = Path(path)
    if not checkpoint.is_dir():
        sys.exit(f"E2E_CKPT is not a directory: {checkpoint}")
    if not (checkpoint / "config.json").is_file():
        sys.exit(
            "E2E_CKPT is missing config.json; point it at a populated "
            f"DeepSeek-V4 snapshot, not an empty cache directory: {checkpoint}"
        )


validate_checkpoint(CKPT)
PORT = int(os.environ.get("E2E_PORT", "18901"))
GPU = os.environ.get("E2E_GPU", "0")
EP_SIZE = int(os.environ.get("E2E_EP_SIZE", "1"))
TP_SIZE = int(os.environ.get("E2E_TP_SIZE", "1"))
DP_SIZE = int(os.environ.get("E2E_DP_SIZE", "1"))
WORLD_SIZE = int(os.environ.get("E2E_WORLD_SIZE", str(EP_SIZE)))
LOCAL_WORLD_SIZE = int(
    os.environ.get("E2E_LOCAL_WORLD_SIZE", str(WORLD_SIZE))
)
MOE_FRONT = os.environ.get("E2E_MOE_FRONT", "0") not in (
    "0",
    "",
    "false",
    "False",
)
PERF = os.environ.get("E2E_PERF", "0") not in ("0", "", "false", "False")
PERF_ONLY = os.environ.get("E2E_PERF_ONLY", "0") not in (
    "0",
    "",
    "false",
    "False",
)
if PERF_ONLY:
    PERF = True
CUDA_GRAPH = os.environ.get("E2E_CUDA_GRAPH", "0") not in (
    "0",
    "",
    "false",
    "False",
)
DECODE_CAPTURE_CONFIG = os.environ.get(
    "E2E_DECODE_CAPTURE_CONFIG", "1" if CUDA_GRAPH else ""
)
if CUDA_GRAPH and not DECODE_CAPTURE_CONFIG:
    sys.exit("E2E_CUDA_GRAPH=1 requires E2E_DECODE_CAPTURE_CONFIG")
if MOE_FRONT and EP_SIZE <= 1:
    sys.exit("E2E_MOE_FRONT=1 requires E2E_EP_SIZE > 1")
OUT_DIR = Path(os.environ.get("E2E_OUT", "e2e_out"))
SERVER_ARGS = [
    "--start_port",
    str(PORT),
    "--load_method",
    "scratch",
    "--max_seq_len",
    "4096",
    "--enable_cuda_graph",
    "1" if CUDA_GRAPH else "0",
    "--act_type",
    "BF16",
    "--tp_size",
    str(TP_SIZE),
    "--dp_size",
    str(DP_SIZE),
    "--ep_size",
    str(EP_SIZE),
    "--world_size",
    str(WORLD_SIZE),
    "--local_world_size",
    str(LOCAL_WORLD_SIZE),
    "--seq_size_per_block",
    "256",
    "--kv_cache_mem_mb",
    "8192",
    "--concurrency_limit",
    "1",
    "--max_context_batch_size",
    "1",
    "--reserver_runtime_mem_mb",
    "4096",
    "--fp8_kv_cache",
    "1",
]
if DECODE_CAPTURE_CONFIG:
    SERVER_ARGS.extend(["--decode_capture_config", DECODE_CAPTURE_CONFIG])
QUERIES = [
    {"prompt": "What is the capital of France?", "max_new_tokens": 64},
    {"prompt": "2+2=", "max_new_tokens": 64},
    # Long generation: decode crosses the 128-token compression boundary, so
    # both the CSA and HCA boundary-compressor paths run inside mega decode.
    {
        "prompt": "Write a detailed step-by-step explanation of how paged "
        "attention works in LLM inference engines.",
        "max_new_tokens": 200,
    },
]
PERF_PROMPT = (
    "Write a detailed step-by-step explanation of how paged attention works "
    "in LLM inference engines. Include concrete implementation details and "
    "examples."
)
PERF_WARMUP_TOKENS = 64
PERF_SAMPLE_TOKENS = 256
PERF_WARMUPS = int(os.environ.get("E2E_PERF_WARMUPS", "1"))
PERF_SAMPLES = int(os.environ.get("E2E_PERF_SAMPLES", "3"))
if PERF_WARMUPS < 1 or PERF_SAMPLES < 1:
    sys.exit("E2E_PERF_WARMUPS and E2E_PERF_SAMPLES must be positive")


def assert_port_unused() -> None:
    """Reject stale servers before a new comparison leg is launched."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        if probe.connect_ex(("127.0.0.1", PORT)) == 0:
            raise RuntimeError(
                f"port {PORT} is already accepting connections; stop the "
                "stale server or choose a different E2E_PORT"
            )


def start_server(tag: str, extra_env: dict) -> subprocess.Popen:
    assert_port_unused()
    env = os.environ.copy()
    # The WebIDE image may export a PYTHONPATH for an older RTP checkout.  A
    # mixed source/native ABI makes ``rtp_llm.ops`` discover that old
    # libth_transformer_config.so, so the serving child must see only this
    # staged checkout (plus the normal site-packages supplied by VENV_PY).
    source_root_env = os.environ.get("E2E_SOURCE_ROOT")
    repo_root = (
        Path(source_root_env)
        if source_root_env
        else Path(__file__).resolve().parents[2]
    )
    python_roots = [str(repo_root)]
    kernel_root_env = os.environ.get("E2E_KERNEL_ROOT")
    if kernel_root_env:
        kernel_root = Path(kernel_root_env)
        if not kernel_root.is_dir():
            raise RuntimeError(
                f"E2E_KERNEL_ROOT is not a directory: {kernel_root}"
            )
        python_roots.append(str(kernel_root))
    env.update(
        {
            "MODEL_TYPE": "deepseek_v4",
            "CHECKPOINT_PATH": CKPT,
            "TOKENIZER_PATH": CKPT,
            "START_PORT": str(PORT),
            "CUDA_VISIBLE_DEVICES": GPU,
            "WORLD_RANK": "0",
            "DG_JIT_CPP_STANDARD": "20",
            "LOG_PATH": str(OUT_DIR / f"{tag}_logs"),
            "PYTHONPATH": os.pathsep.join(python_roots),
        }
    )
    # /tmp/rtp-llm belongs to another user in this container; preset every
    # managed JIT cache env so jit_cache_manager's setdefault keeps our dirs.
    jit_base = Path(
        os.environ.get("E2E_JIT_CACHE", str(Path.home() / ".cache/rtp_jit"))
    )
    for env_name, sub in (
        ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
        ("DG_JIT_CACHE_DIR", "deep_gemm"),
        ("TRTLLM_DG_CACHE_DIR", "trtllm_deep_gemm"),
        ("TILELANG_CACHE_DIR", "tilelang"),
        ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
        ("TVM_FFI_CACHE_DIR", "tvm_ffi"),
        ("CUTE_DSL_CACHE_DIR", "cute_dsl"),
        ("TRITON_CACHE_DIR", "triton"),
    ):
        target = jit_base / sub
        target.mkdir(parents=True, exist_ok=True)
        env.setdefault(env_name, str(target))
    env.update(extra_env)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = open(OUT_DIR / f"{tag}.server.log", "w")
    print(f"[{tag}] starting server (GPU {GPU}, port {PORT}) ...", flush=True)
    return subprocess.Popen(
        [VENV_PY, "-m", "rtp_llm.start_server"] + SERVER_ARGS,
        env=env,
        stdout=log,
        stderr=log,
        start_new_session=True,
        cwd=str(OUT_DIR),
    )


# Full V4-Flash (156GB) loads from NAS at ~4GB/min plus first-run JIT;
# 30 minutes is not enough.
def wait_ready(proc: subprocess.Popen, timeout: int = 5400) -> bool:
    source_root = os.environ.get("E2E_SOURCE_ROOT")
    if source_root:
        sys.path.insert(0, source_root)
    else:
        sys.path.insert(
            0, str(Path(VENV_PY).parents[1] / "lib/python3.10/site-packages")
        )
    from rtp_llm.utils.util import wait_sever_done

    ready = wait_sever_done(proc, PORT, timeout)
    return bool(ready and proc.poll() is None)


def query(prompt: str, max_new_tokens: int) -> dict:
    body = json.dumps(
        {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": max_new_tokens,
                "top_k": 1,
                "top_p": 0,
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def query_all(tag: str) -> list:
    results = []
    for item in QUERIES:
        payload = query(item["prompt"], item["max_new_tokens"])
        results.append(payload)
        text = payload.get("response", payload)
        print(
            f"[{tag}] Q: {item['prompt'][:40]!r}\n[{tag}] A: "
            f"{str(text)[:160]!r}",
            flush=True,
        )
    return results


def performance_sample(payload: dict) -> dict:
    aux = payload.get("aux_info")
    if not isinstance(aux, dict):
        raise RuntimeError("performance response is missing aux_info")
    output_len = int(aux.get("output_len", 0))
    cost_ms = float(aux.get("cost_time", 0.0))
    first_token_ms = float(aux.get("first_token_cost_time", 0.0))
    decode_ms = cost_ms - first_token_ms
    if output_len < 2 or cost_ms <= 0.0 or decode_ms <= 0.0:
        raise RuntimeError(
            "invalid performance response: "
            f"output_len={output_len}, cost_ms={cost_ms}, "
            f"first_token_ms={first_token_ms}"
        )
    return {
        "output_len": output_len,
        "cost_ms": cost_ms,
        "first_token_ms": first_token_ms,
        "e2e_tps": output_len * 1000.0 / cost_ms,
        "decode_tps": (output_len - 1) * 1000.0 / decode_ms,
    }


def query_performance(tag: str) -> dict:
    warmups = [
        query(PERF_PROMPT, PERF_WARMUP_TOKENS)
        for _ in range(PERF_WARMUPS)
    ]
    print(
        f"[{tag}] performance warmup complete: {PERF_WARMUPS}x"
        f"{PERF_WARMUP_TOKENS} tokens",
        flush=True,
    )
    samples = []
    raw_samples = []
    for index in range(PERF_SAMPLES):
        payload = query(PERF_PROMPT, PERF_SAMPLE_TOKENS)
        raw_samples.append(payload)
        sample = performance_sample(payload)
        samples.append(sample)
        print(f"[{tag}] performance sample {index}: {sample}", flush=True)
    summary = {
        "prompt": PERF_PROMPT,
        "cuda_graph": CUDA_GRAPH,
        "decode_capture_config": DECODE_CAPTURE_CONFIG,
        "warmup_tokens": PERF_WARMUP_TOKENS,
        "warmup_count": PERF_WARMUPS,
        "sample_tokens": PERF_SAMPLE_TOKENS,
        "sample_count": PERF_SAMPLES,
        "samples": samples,
        "median_cost_ms": statistics.median(x["cost_ms"] for x in samples),
        "median_first_token_ms": statistics.median(
            x["first_token_ms"] for x in samples
        ),
        "median_e2e_tps": statistics.median(x["e2e_tps"] for x in samples),
        "median_decode_tps": statistics.median(
            x["decode_tps"] for x in samples
        ),
    }
    (OUT_DIR / f"{tag}.perf.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    (OUT_DIR / f"{tag}.perf.raw.json").write_text(
        json.dumps(
            {"warmups": warmups, "samples": raw_samples},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"[{tag}] performance median: {summary}", flush=True)
    return summary


def stop_server(proc: subprocess.Popen) -> None:
    # SIGTERM first so rtp_llm's process manager can tear down its children
    # and free device memory; SIGKILL only as a last resort (a killed CUDA
    # process can leave driver-held memory behind in this container).
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(60):
        if proc.poll() is not None:
            break
        time.sleep(2)
    else:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait()
    # Give the driver a moment to reclaim device memory.
    time.sleep(20)


def run(tag: str, extra_env: dict) -> list:
    proc = start_server(tag, extra_env)
    try:
        if not wait_ready(proc):
            raise RuntimeError(
                f"[{tag}] server failed to become ready; see "
                f"{OUT_DIR / (tag + '.server.log')}"
            )
        print(f"[{tag}] server ready", flush=True)
        results = [] if PERF_ONLY else query_all(tag)
        if not PERF_ONLY:
            (OUT_DIR / f"{tag}.results.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2)
            )
        if PERF:
            query_performance(tag)
        return results
    finally:
        stop_server(proc)


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    runs = {}
    if only in (None, "baseline"):
        controlled_front_env = {
            "DSV4_MEGA_CSA": "1",
            "DSV4_MEGA_HCA": "1",
            "DSV4_MEGA_MOE_FRONT": "0",
            "DSV4_USE_MEGA_MOE_SE": "1",
            "DSV4_USE_MEGA_MOE": "1",
            "DSV4_USE_MEGA_MOE_FUSED": "0",
            "DSV4_MOE_STRATEGY": "",
        }
        runs["baseline"] = run(
            "baseline",
            controlled_front_env if MOE_FRONT else {
                "DSV4_MEGA_CSA": "0",
                "DSV4_MEGA_HCA": "0",
                "DSV4_MEGA_MOE_FRONT": "0",
                "DSV4_USE_MEGA_MOE_SE": "0",
                "DSV4_USE_MEGA_MOE": "1",
                "DSV4_USE_MEGA_MOE_FUSED": "0",
                "DSV4_MOE_STRATEGY": "",
            },
        )
    if only in (None, "mega"):
        runs["mega"] = run(
            "mega",
            {
                "DSV4_MEGA_CSA": "1",
                "DSV4_MEGA_HCA": "1",
                "DSV4_MEGA_MOE_FRONT": "1" if MOE_FRONT else "0",
                "DSV4_USE_MEGA_MOE_SE": "1" if MOE_FRONT else "0",
                "DSV4_USE_MEGA_MOE": "1",
                "DSV4_USE_MEGA_MOE_FUSED": "0",
                "DSV4_MOE_STRATEGY": "",
            },
        )
    if len(runs) < 2:
        for tag in ("baseline", "mega"):
            path = OUT_DIR / f"{tag}.results.json"
            if tag not in runs and path.exists():
                runs[tag] = json.loads(path.read_text())
    if len(runs) == 2 and any(runs.values()):
        print("\n========== COMPARISON ==========", flush=True)
        mismatches = 0
        for index, (base, mega) in enumerate(zip(runs["baseline"], runs["mega"])):
            base_text = base.get("response")
            mega_text = mega.get("response")
            same = base_text == mega_text
            mismatches += not same
            print(f"query {index}: {'IDENTICAL' if same else 'DIFFERENT'}")
            if not same:
                print(f"  baseline: {str(base_text)[:200]!r}")
                print(f"  mega    : {str(mega_text)[:200]!r}")
        print(
            f"\n{len(runs['baseline']) - mismatches}/"
            f"{len(runs['baseline'])} queries identical"
        )
    perf_runs = {}
    for tag in ("baseline", "mega"):
        path = OUT_DIR / f"{tag}.perf.json"
        if path.exists():
            perf_runs[tag] = json.loads(path.read_text())
    if len(perf_runs) == 2:
        baseline = perf_runs["baseline"]
        mega = perf_runs["mega"]
        print("\n========== PERFORMANCE COMPARISON ==========", flush=True)
        print(
            json.dumps(
                {
                    "baseline": baseline,
                    "mega": mega,
                    "mega_over_baseline_e2e_tps": (
                        mega["median_e2e_tps"] / baseline["median_e2e_tps"]
                    ),
                    "mega_over_baseline_decode_tps": (
                        mega["median_decode_tps"]
                        / baseline["median_decode_tps"]
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
