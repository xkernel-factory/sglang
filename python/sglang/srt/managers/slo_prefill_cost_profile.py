from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

SLO_PREFILL_COST_PROFILE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SloPrefillCostProfile:
    prefill_cost_ms: list[tuple[int, float]]
    decode_cost_ms: list[tuple[int, float]]
    decode_cost_by_context_ms: list[tuple[int, int, float]]
    metadata: dict[str, Any]


def load_slo_prefill_cost_profile(path: str) -> SloPrefillCostProfile:
    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as profile_file:
        data = json.load(profile_file)

    if not isinstance(data, dict):
        raise ValueError("SLO prefill cost profile must be a JSON object.")

    schema_version = data.get(
        "schema_version", SLO_PREFILL_COST_PROFILE_SCHEMA_VERSION
    )
    if schema_version != SLO_PREFILL_COST_PROFILE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported SLO prefill cost profile schema_version="
            f"{schema_version}; expected {SLO_PREFILL_COST_PROFILE_SCHEMA_VERSION}."
        )

    metadata = data.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("SLO prefill cost profile metadata must be an object.")

    prefill_cost_ms = _parse_cost_points(
        data.get("prefill_cost_ms", []),
        field_name="prefill_cost_ms",
        width=2,
    )
    decode_cost_ms = _parse_cost_points(
        data.get("decode_cost_ms", []),
        field_name="decode_cost_ms",
        width=2,
    )
    decode_cost_by_context_ms = _parse_cost_points(
        data.get("decode_cost_by_context_ms", []),
        field_name="decode_cost_by_context_ms",
        width=3,
    )

    if not prefill_cost_ms and not decode_cost_ms and not decode_cost_by_context_ms:
        raise ValueError(
            "SLO prefill cost profile must contain at least one cost point."
        )

    return SloPrefillCostProfile(
        prefill_cost_ms=[
            (int(tokens), float(cost_ms)) for tokens, cost_ms in prefill_cost_ms
        ],
        decode_cost_ms=[
            (int(batch_size), float(cost_ms))
            for batch_size, cost_ms in decode_cost_ms
        ],
        decode_cost_by_context_ms=[
            (int(context_len), int(batch_size), float(cost_ms))
            for context_len, batch_size, cost_ms in decode_cost_by_context_ms
        ],
        metadata=metadata,
    )


def write_slo_prefill_cost_profile(
    path: str,
    *,
    prefill_cost_ms: Sequence[tuple[int, float]],
    decode_cost_by_context_ms: Sequence[tuple[int, int, float]],
    decode_cost_ms: Sequence[tuple[int, float]] = (),
    metadata: dict[str, Any] | None = None,
) -> None:
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_data = {
        "schema_version": SLO_PREFILL_COST_PROFILE_SCHEMA_VERSION,
        "created_at_unix_s": time.time(),
        "metadata": _json_safe(metadata or {}),
        "prefill_cost_ms": [
            [int(tokens), float(cost_ms)] for tokens, cost_ms in prefill_cost_ms
        ],
        "decode_cost_ms": [
            [int(batch_size), float(cost_ms)]
            for batch_size, cost_ms in decode_cost_ms
        ],
        "decode_cost_by_context_ms": [
            [int(context_len), int(batch_size), float(cost_ms)]
            for context_len, batch_size, cost_ms in decode_cost_by_context_ms
        ],
    }

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(profile_path.parent),
        prefix=f".{profile_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        json.dump(profile_data, temp_file, indent=2, sort_keys=True)
        temp_file.write("\n")
        temp_name = temp_file.name
    os.replace(temp_name, profile_path)


def _parse_cost_points(
    raw_points: Any,
    *,
    field_name: str,
    width: int,
) -> list[tuple[Any, ...]]:
    if raw_points is None:
        return []
    if not isinstance(raw_points, list):
        raise ValueError(f"{field_name} must be a list of points.")

    parsed_points = []
    for index, raw_point in enumerate(raw_points):
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) != width:
            raise ValueError(
                f"{field_name}[{index}] must be a {width}-item list/tuple."
            )
        try:
            parsed_point = tuple(float(value) for value in raw_point)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}[{index}] contains a non-numeric value.") from exc

        integer_dims = parsed_point[:-1]
        cost_ms = parsed_point[-1]
        if any(dim <= 0 or int(dim) != dim for dim in integer_dims):
            raise ValueError(
                f"{field_name}[{index}] dimensions must be positive integers."
            )
        if cost_ms <= 0.0:
            raise ValueError(f"{field_name}[{index}] cost must be positive.")
        parsed_points.append(parsed_point)
    return parsed_points


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return str(value)
