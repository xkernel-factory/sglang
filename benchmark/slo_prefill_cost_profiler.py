"""Generate an offline SLO prefill Cp/Cd cost profile.

This wrapper launches an SGLang server with SLO startup profiling enabled, waits for
that profiling path to write a JSON cost table, then terminates the server. Pass
normal server arguments after ``--``.

Example:
    python3 benchmark/slo_prefill_cost_profiler.py \
      --output /tmp/slo_prefill_cost_profile.json \
      -- \
      --model-path /path/to/model \
      --tp-size 4 \
      --chunked-prefill-size 32768 \
      --slo-prefill-min-chunk-size 4096 \
      --slo-prefill-profile-prefill-token-sizes 4096 32768 \
      --slo-prefill-profile-decode-context-lens 4096 8192 16384 \
      --slo-prefill-profile-decode-batch-sizes 1 2 4 8
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_ROOT = _REPO_ROOT / "python"
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))


_PROFILE_MODULE_PATH = (
    _PYTHON_ROOT / "sglang/srt/managers/slo_prefill_cost_profile.py"
)
_PROFILE_SPEC = importlib.util.spec_from_file_location(
    "slo_prefill_cost_profile", _PROFILE_MODULE_PATH
)
_PROFILE_MODULE = importlib.util.module_from_spec(_PROFILE_SPEC)
assert _PROFILE_SPEC.loader is not None
sys.modules[_PROFILE_SPEC.name] = _PROFILE_MODULE
_PROFILE_SPEC.loader.exec_module(_PROFILE_MODULE)
load_slo_prefill_cost_profile = _PROFILE_MODULE.load_slo_prefill_cost_profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an offline SLO prefill Cp/Cd cost profile JSON."
    )
    parser.add_argument("--output", required=True, help="Output JSON profile path.")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=1800.0,
        help="Maximum seconds to wait for profiling to finish.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output profile file.",
    )
    parser.add_argument(
        "--ttft-slo-ms",
        type=float,
        default=1000.0,
        help="Default TTFT SLO used only to initialize the SLO controller.",
    )
    parser.add_argument(
        "--tpot-slo-ms",
        type=float,
        default=100.0,
        help="Default TPOT SLO used only to initialize the SLO controller.",
    )
    parser.add_argument(
        "server_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to sglang.launch_server; prefix with --.",
    )
    args = parser.parse_args()

    server_args = list(args.server_args)
    if server_args and server_args[0] == "--":
        server_args = server_args[1:]

    output_path = Path(args.output).resolve()
    if output_path.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output profile already exists: {output_path}. Use --overwrite."
            )
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _has_flag(server_args, "--disable-slo-prefill-startup-profiling"):
        raise SystemExit(
            "Cannot generate a profile with --disable-slo-prefill-startup-profiling."
        )
    if _has_flag(server_args, "--slo-prefill-cost-profile-path"):
        raise SystemExit(
            "Do not pass --slo-prefill-cost-profile-path while generating a profile."
        )
    if _has_flag(server_args, "--slo-prefill-cost-profile-output-path"):
        raise SystemExit(
            "Do not pass --slo-prefill-cost-profile-output-path; use --output instead."
        )

    _append_flag(server_args, "--enable-slo-aware-prefill")
    _append_value_if_missing(
        server_args, "--slo-prefill-ttft-slo-ms", str(args.ttft_slo_ms)
    )
    _append_value_if_missing(
        server_args, "--slo-prefill-tpot-slo-ms", str(args.tpot_slo_ms)
    )
    _append_flag(server_args, "--enable-slo-prefill-startup-profiling")
    server_args.extend(["--slo-prefill-cost-profile-output-path", str(output_path)])

    command = [sys.executable, "-m", "sglang.launch_server", *server_args]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_PYTHON_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    print("Launching profiler server:")
    print(" ".join(command))
    process = subprocess.Popen(command, env=env, start_new_session=True)
    deadline = time.monotonic() + args.timeout_s
    try:
        while time.monotonic() < deadline:
            if output_path.exists():
                profile = load_slo_prefill_cost_profile(str(output_path))
                print(
                    "Generated SLO prefill cost profile: "
                    f"path={output_path}, "
                    f"Cp={len(profile.prefill_cost_ms)}, "
                    f"Cd={len(profile.decode_cost_by_context_ms)}"
                )
                return 0
            return_code = process.poll()
            if return_code is not None:
                raise SystemExit(
                    f"Profiler server exited with code {return_code} before writing "
                    f"{output_path}."
                )
            time.sleep(1.0)
        raise SystemExit(f"Timed out waiting for profile output: {output_path}")
    finally:
        _terminate_process_group(process)


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=10.0)


def _has_flag(args: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def _append_flag(args: list[str], flag: str) -> None:
    if not _has_flag(args, flag):
        args.append(flag)


def _append_value_if_missing(args: list[str], flag: str, value: str) -> None:
    if not _has_flag(args, flag):
        args.extend([flag, value])


if __name__ == "__main__":
    raise SystemExit(main())
