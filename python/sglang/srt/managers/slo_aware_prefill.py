from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


@dataclass
class SloAwarePrefillDecision:
    chunked_prefill_size: Optional[int]
    max_prefill_requests: Optional[int]
    allow_prefill: bool
    objective: str
    ttft_pressure: float
    tpot_pressure: float
    has_decode_work: bool
    yield_prefill_to_decode: bool
    prefill_cost_per_token_s: float
    decode_cost_s: float
    decode_context_len: int
    ttft_future_prefill_cost_s: float
    ttft_future_miss_tokens: int
    ttft_future_hit_tokens: int
    ttft_future_io_cost_s: float
    ttft_cache_hit_rate: float
    ttft_slack_s: float
    yield_rhs_s: float
    yield_guard_s: float
    min_prefill_cost_s: float
    ttft_stat: str
    tpot_stat: str


@dataclass(frozen=True)
class SloAwarePrefillPressureState:
    ttft_pressure: float
    tpot_pressure: float
    has_decode_work: bool
    prefill_cost_per_token_s: float = 0.0
    decode_cost_s: float = 0.0
    decode_context_len: int = 0
    ttft_future_prefill_cost_s: float = 0.0
    ttft_future_miss_tokens: int = 0
    ttft_future_hit_tokens: int = 0
    ttft_future_io_cost_s: float = 0.0
    ttft_cache_hit_rate: float = 0.0


class SloAwarePrefillController:
    """A lightweight SOLA-inspired controller for prefill admission.

    This controller approximates SOLA's state-aware scheduling in SGLang's
    existing scheduler by changing prefill phase priority and workload size.
    """

    def __init__(
        self,
        *,
        ttft_slo_ms: float,
        tpot_slo_ms: float,
        base_chunked_prefill_size: Optional[int],
        max_prefill_tokens: int,
        page_size: int,
        min_chunk_size: Optional[int],
        ttft_stat: str = "max",
        tpot_stat: str = "max",
        initial_prefill_cost_ms_per_1k: Optional[float] = None,
        initial_decode_cost_ms: Optional[float] = None,
        yield_guard_ratio: float = 0.05,
        cache_hit_io_cost_ratio: float = 0.3,
    ) -> None:
        self.ttft_slo_s = ttft_slo_ms / 1000.0
        self.tpot_slo_s = tpot_slo_ms / 1000.0
        self.base_chunked_prefill_size = base_chunked_prefill_size
        self.max_prefill_tokens = max_prefill_tokens
        self.page_size = page_size
        self.min_chunk_size = self._resolve_min_chunk_size(
            min_chunk_size=min_chunk_size,
            base_chunked_prefill_size=base_chunked_prefill_size,
        )
        self.ttft_stat = self._normalize_pressure_stat(ttft_stat)
        self.tpot_stat = self._normalize_pressure_stat(tpot_stat)
        self.objective_margin = 0.10
        self.yield_guard_ratio = max(yield_guard_ratio, 0.0)
        self.cache_hit_io_cost_ratio = max(cache_hit_io_cost_ratio, 0.0)
        default_cost_tokens = max(
            self.base_chunked_prefill_size or self.max_prefill_tokens,
            self.min_chunk_size,
            1,
        )
        self.default_prefill_cost_per_token_s = max(
            self.tpot_slo_s / default_cost_tokens, 1e-6
        )
        self.default_decode_cost_s = max(min(self.tpot_slo_s, 0.05), 1e-4)
        if initial_prefill_cost_ms_per_1k is not None:
            self.default_prefill_cost_per_token_s = max(
                initial_prefill_cost_ms_per_1k / 1_000_000.0, 1e-9
            )
        if initial_decode_cost_ms is not None:
            self.default_decode_cost_s = max(initial_decode_cost_ms / 1000.0, 1e-9)
        self._prefill_cost_points_s: list[tuple[int, float]] = []
        self._decode_cost_points_s: list[tuple[int, float]] = []
        self._decode_cost_table_s: list[tuple[int, list[tuple[int, float]]]] = []
        self._prefill_cost_per_token_s = self.default_prefill_cost_per_token_s
        self._decode_cost_s = self.default_decode_cost_s
        self._active_prefill_cost_per_token_s = self._prefill_cost_per_token_s
        self._active_decode_cost_s = self._decode_cost_s
        self._last_objective = "ttft"

    def _resolve_min_chunk_size(
        self,
        *,
        min_chunk_size: Optional[int],
        base_chunked_prefill_size: Optional[int],
    ) -> int:
        if min_chunk_size is None:
            base_chunk = base_chunked_prefill_size or self.max_prefill_tokens
            min_chunk = base_chunk
        else:
            min_chunk = min_chunk_size
        return max(min_chunk, self.page_size, 1)

    def make_decision(
        self,
        *,
        waiting_queue: Sequence["Req"],
        running_batch: "ScheduleBatch",
        chunked_req: Optional["Req"],
        default_chunked_prefill_size: Optional[int],
        default_prefill_max_requests: Optional[int],
    ) -> SloAwarePrefillDecision:
        pressure_state = self.compute_pressure_state(
            waiting_queue=waiting_queue,
            running_batch=running_batch,
            chunked_req=chunked_req,
        )
        return self.make_decision_from_pressure_state(
            pressure_state=pressure_state,
            chunked_req=chunked_req,
            default_chunked_prefill_size=default_chunked_prefill_size,
            default_prefill_max_requests=default_prefill_max_requests,
        )

    def compute_pressure_state(
        self,
        *,
        waiting_queue: Sequence["Req"],
        running_batch: "ScheduleBatch",
        chunked_req: Optional["Req"],
    ) -> SloAwarePrefillPressureState:
        now = time.perf_counter()
        decode_reqs = self._decode_reqs(running_batch.reqs)
        decode_context_len = self._decode_context_len(decode_reqs)
        (
            ttft_pressure,
            ttft_future_prefill_cost_s,
            ttft_future_miss_tokens,
            ttft_future_hit_tokens,
            ttft_future_io_cost_s,
            ttft_cache_hit_rate,
        ) = self._ttft_pressure(now, waiting_queue, chunked_req)
        return SloAwarePrefillPressureState(
            ttft_pressure=ttft_pressure,
            tpot_pressure=self._tpot_pressure(now, decode_reqs),
            has_decode_work=len(decode_reqs) > 0,
            prefill_cost_per_token_s=self._prefill_cost_per_token_s,
            decode_cost_s=self._decode_cost_for_batch(
                len(decode_reqs), decode_context_len
            ),
            decode_context_len=decode_context_len,
            ttft_future_prefill_cost_s=ttft_future_prefill_cost_s,
            ttft_future_miss_tokens=ttft_future_miss_tokens,
            ttft_future_hit_tokens=ttft_future_hit_tokens,
            ttft_future_io_cost_s=ttft_future_io_cost_s,
            ttft_cache_hit_rate=ttft_cache_hit_rate,
        )

    def make_decision_from_pressure_state(
        self,
        *,
        pressure_state: SloAwarePrefillPressureState,
        chunked_req: Optional["Req"],
        default_chunked_prefill_size: Optional[int],
        default_prefill_max_requests: Optional[int],
    ) -> SloAwarePrefillDecision:
        ttft_pressure = pressure_state.ttft_pressure
        tpot_pressure = pressure_state.tpot_pressure
        has_decode_work = pressure_state.has_decode_work
        self._active_prefill_cost_per_token_s = (
            pressure_state.prefill_cost_per_token_s or self._prefill_cost_per_token_s
        )
        self._active_decode_cost_s = pressure_state.decode_cost_s or self._decode_cost_s
        objective = self._choose_objective(
            ttft_pressure, tpot_pressure, has_decode_work
        )
        chunked_prefill_size = None
        base_chunk = default_chunked_prefill_size or self.base_chunked_prefill_size
        if base_chunk is not None:
            base_chunk = max(1, min(base_chunk, self.max_prefill_tokens))
            chunked_prefill_size = self._select_chunk(
                base_chunk, objective, has_decode_work
            )

        allow_prefill = True
        yield_prefill_to_decode = False
        max_prefill_requests = default_prefill_max_requests

        min_prefill_cost_s = self._estimate_prefill_cost(self.min_chunk_size)
        yield_guard_s = self._yield_guard_s(min_prefill_cost_s)
        ttft_slack_s = self._ttft_slack_s(ttft_pressure)
        yield_rhs_s = (
            self._active_decode_cost_s + min_prefill_cost_s + yield_guard_s
        )

        if objective == "tpot" and has_decode_work:
            if self._can_yield_to_decode(ttft_pressure):
                allow_prefill = False
                yield_prefill_to_decode = True

        if chunked_req is not None:
            allow_prefill = True

        self._last_objective = objective
        return SloAwarePrefillDecision(
            chunked_prefill_size=chunked_prefill_size,
            max_prefill_requests=max_prefill_requests,
            allow_prefill=allow_prefill,
            objective=objective,
            ttft_pressure=ttft_pressure,
            tpot_pressure=tpot_pressure,
            has_decode_work=has_decode_work,
            yield_prefill_to_decode=yield_prefill_to_decode,
            prefill_cost_per_token_s=self._active_prefill_cost_per_token_s,
            decode_cost_s=self._active_decode_cost_s,
            decode_context_len=pressure_state.decode_context_len,
            ttft_future_prefill_cost_s=pressure_state.ttft_future_prefill_cost_s,
            ttft_future_miss_tokens=pressure_state.ttft_future_miss_tokens,
            ttft_future_hit_tokens=pressure_state.ttft_future_hit_tokens,
            ttft_future_io_cost_s=pressure_state.ttft_future_io_cost_s,
            ttft_cache_hit_rate=pressure_state.ttft_cache_hit_rate,
            ttft_slack_s=ttft_slack_s,
            yield_rhs_s=yield_rhs_s,
            yield_guard_s=yield_guard_s,
            min_prefill_cost_s=min_prefill_cost_s,
            ttft_stat=self.ttft_stat,
            tpot_stat=self.tpot_stat,
        )

    def _normalize_pressure_stat(self, stat: str) -> str:
        if stat not in ("max", "mean", "p90"):
            raise ValueError(f"Unsupported SLO pressure stat: {stat}")
        return stat

    def set_startup_cost_profile(
        self,
        *,
        prefill_cost_ms: Sequence[tuple[int, float]],
        decode_cost_ms: Sequence[tuple[int, float]] = (),
        decode_cost_by_context_ms: Sequence[tuple[int, int, float]] = (),
    ) -> None:
        prefill_points = [
            (int(tokens), float(cost_ms) / 1000.0)
            for tokens, cost_ms in prefill_cost_ms
            if tokens > 0 and cost_ms > 0.0
        ]
        decode_points = [
            (int(batch_size), float(cost_ms) / 1000.0)
            for batch_size, cost_ms in decode_cost_ms
            if batch_size > 0 and cost_ms > 0.0
        ]
        self._prefill_cost_points_s = self._monotonic_cost_points(prefill_points)
        self._decode_cost_points_s = self._monotonic_cost_points(decode_points)
        self._decode_cost_table_s = self._build_decode_cost_table(
            decode_cost_by_context_ms
        )
        if self._prefill_cost_points_s:
            tokens, cost_s = self._prefill_cost_points_s[-1]
            self._prefill_cost_per_token_s = max(cost_s / max(tokens, 1), 1e-9)
        if self._decode_cost_table_s or self._decode_cost_points_s:
            self._decode_cost_s = self._decode_cost_for_batch(1, 0)

    def _choose_objective(
        self, ttft_pressure: float, tpot_pressure: float, has_decode_work: bool
    ) -> str:
        if not has_decode_work:
            return "ttft"

        ttft_pressure = self._quantize_pressure(ttft_pressure)
        tpot_pressure = self._quantize_pressure(tpot_pressure)
        margin = self.objective_margin

        if tpot_pressure >= 1.0 and ttft_pressure < 1.0:
            return "tpot"
        if ttft_pressure >= 1.0 and tpot_pressure < 1.0:
            return "ttft"
        if tpot_pressure > ttft_pressure + margin:
            return "tpot"
        return "ttft"

    def _quantize_pressure(self, pressure: float) -> float:
        return round(pressure, 2)

    def _select_chunk(
        self,
        base_chunk: int,
        objective: str,
        has_decode_work: bool,
    ) -> int:
        if not has_decode_work or objective == "ttft":
            return self._clamp_chunk(base_chunk, base_chunk)
        return self._clamp_chunk(self.min_chunk_size, base_chunk)

    def _clamp_chunk(self, chunk: int, base_chunk: int) -> int:
        return max(1, min(chunk, base_chunk, self.max_prefill_tokens))

    def _can_yield_to_decode(self, ttft_pressure: float) -> bool:
        return self._has_ttft_slack_for_decode_yield(ttft_pressure)

    def _has_ttft_slack_for_decode_yield(self, ttft_pressure: float) -> bool:
        if ttft_pressure >= 1.0:
            return False
        min_prefill_cost_s = self._estimate_prefill_cost(self.min_chunk_size)
        guard_s = self._yield_guard_s(min_prefill_cost_s)
        return (
            self._ttft_slack_s(ttft_pressure)
            > self._active_decode_cost_s + min_prefill_cost_s + guard_s
        )

    def _ttft_slack_s(self, ttft_pressure: float) -> float:
        return max((1.0 - ttft_pressure) * self.ttft_slo_s, 0.0)

    def _yield_guard_s(self, min_prefill_cost_s: float) -> float:
        return max(
            2.0 * self._active_decode_cost_s,
            0.2 * min_prefill_cost_s,
            self.yield_guard_ratio * self.ttft_slo_s,
        )

    def _estimate_prefill_cost(self, tokens: int) -> float:
        tokens = max(tokens, 0)
        if tokens == 0:
            return 0.0
        if self._prefill_cost_points_s:
            return self._profiled_cost(self._prefill_cost_points_s, tokens)
        return tokens * max(self._active_prefill_cost_per_token_s, 1e-9)

    def _decode_cost_for_batch(self, batch_size: int, context_len: int = 0) -> float:
        if batch_size <= 0:
            return self._decode_cost_s
        if self._decode_cost_table_s:
            return self._contextual_decode_cost(batch_size, context_len)
        if not self._decode_cost_points_s:
            return self._decode_cost_s
        return self._profiled_cost(self._decode_cost_points_s, batch_size)

    def _contextual_decode_cost(self, batch_size: int, context_len: int) -> float:
        table = self._decode_cost_table_s
        if not table:
            return self._decode_cost_s

        context_len = max(int(context_len), 0)
        if context_len <= table[0][0] or len(table) == 1:
            return self._profiled_cost(table[0][1], batch_size)

        for (left_ctx, left_points), (right_ctx, right_points) in zip(
            table, table[1:]
        ):
            if context_len <= right_ctx:
                left_cost_s = self._profiled_cost(left_points, batch_size)
                right_cost_s = max(
                    self._profiled_cost(right_points, batch_size), left_cost_s
                )
                ratio = (context_len - left_ctx) / max(right_ctx - left_ctx, 1)
                return left_cost_s + ratio * (right_cost_s - left_cost_s)

        prev_ctx, prev_points = table[-2] if len(table) > 1 else (0, [])
        last_ctx, last_points = table[-1]
        last_cost_s = self._profiled_cost(last_points, batch_size)
        if not prev_points or last_ctx <= prev_ctx:
            return last_cost_s
        prev_cost_s = self._profiled_cost(prev_points, batch_size)
        slope = max((last_cost_s - prev_cost_s) / max(last_ctx - prev_ctx, 1), 0.0)
        return last_cost_s + (context_len - last_ctx) * slope

    def _build_decode_cost_table(
        self, decode_cost_by_context_ms: Sequence[tuple[int, int, float]]
    ) -> list[tuple[int, list[tuple[int, float]]]]:
        grouped: dict[int, list[tuple[int, float]]] = {}
        for context_len, batch_size, cost_ms in decode_cost_by_context_ms:
            if context_len <= 0 or batch_size <= 0 or cost_ms <= 0.0:
                continue
            grouped.setdefault(int(context_len), []).append(
                (int(batch_size), float(cost_ms) / 1000.0)
            )
        return [
            (context_len, self._monotonic_cost_points(points))
            for context_len, points in sorted(grouped.items())
        ]

    def _profiled_cost(self, points: list[tuple[int, float]], size: int) -> float:
        if size <= 0:
            return 0.0
        if len(points) == 1:
            point_size, point_cost_s = points[0]
            return point_cost_s * size / max(point_size, 1)
        if size <= points[0][0]:
            return points[0][1] * size / max(points[0][0], 1)
        for left, right in zip(points, points[1:]):
            left_size, left_cost_s = left
            right_size, right_cost_s = right
            if size <= right_size:
                ratio = (size - left_size) / max(right_size - left_size, 1)
                return left_cost_s + ratio * (right_cost_s - left_cost_s)
        prev_size, prev_cost_s = points[-2]
        last_size, last_cost_s = points[-1]
        slope = (last_cost_s - prev_cost_s) / max(last_size - prev_size, 1)
        slope = max(slope, last_cost_s / max(last_size, 1), 1e-9)
        return last_cost_s + (size - last_size) * slope

    def _monotonic_cost_points(
        self, points: list[tuple[int, float]]
    ) -> list[tuple[int, float]]:
        points = sorted(points)
        ret = []
        max_cost = 0.0
        for size, cost_s in points:
            if ret and size == ret[-1][0]:
                max_cost = max(ret[-1][1], cost_s, max_cost)
                ret[-1] = (size, max_cost)
                continue
            max_cost = max(cost_s, max_cost)
            ret.append((size, max_cost))
        return ret

    def _decode_reqs(self, running_reqs: Iterable["Req"]) -> list["Req"]:
        return [
            req
            for req in running_reqs
            if not req.finished()
            and not req.is_retracted
            and len(req.output_ids) > 0
        ]

    def _decode_context_len(self, decode_reqs: Sequence["Req"]) -> int:
        if not decode_reqs:
            return 0
        return max(max(getattr(req, "seqlen", 0), 0) for req in decode_reqs)

    def _ttft_pressure(
        self, now: float, waiting_queue: Sequence["Req"], chunked_req: Optional["Req"]
    ) -> tuple[float, float, int, int, float, float]:
        if self.ttft_slo_s <= 0:
            return 0.0, 0.0, 0, 0, 0.0, 0.0
        pressures = []
        future_costs_s = []
        future_miss_tokens = []
        future_hit_tokens = []
        future_io_costs_s = []
        cache_hit_rates = []
        for req in self._prefill_candidates(waiting_queue, chunked_req):
            entry = (
                req.time_stats.wait_queue_entry_time
                or req.time_stats.scheduler_recv_time
            )
            if entry <= 0.0:
                continue
            future_cost_s, miss_tokens, hit_tokens, io_cost_s, cache_hit_rate = (
                self._estimate_future_prefill_cost(req)
            )
            future_costs_s.append(future_cost_s)
            future_miss_tokens.append(float(miss_tokens))
            future_hit_tokens.append(float(hit_tokens))
            future_io_costs_s.append(io_cost_s)
            cache_hit_rates.append(cache_hit_rate)
            pressures.append((now - entry + future_cost_s) / self.ttft_slo_s)
        return (
            self._aggregate_pressure(pressures, self.ttft_stat),
            self._aggregate_pressure(future_costs_s, self.ttft_stat),
            int(self._aggregate_pressure(future_miss_tokens, self.ttft_stat)),
            int(self._aggregate_pressure(future_hit_tokens, self.ttft_stat)),
            self._aggregate_pressure(future_io_costs_s, self.ttft_stat),
            self._aggregate_pressure(cache_hit_rates, self.ttft_stat),
        )

    def _estimate_future_prefill_cost(
        self, req: "Req"
    ) -> tuple[float, int, int, float, float]:
        future_miss_tokens, future_hit_tokens, cache_hit_rate = (
            self._estimate_future_prefill_tokens(req)
        )
        compute_cost_s = self._estimate_prefill_cost(future_miss_tokens)
        io_cost_s = self.cache_hit_io_cost_ratio * self._estimate_prefill_cost(
            future_hit_tokens
        )
        return (
            compute_cost_s + io_cost_s,
            future_miss_tokens,
            future_hit_tokens,
            io_cost_s,
            cache_hit_rate,
        )

    def _estimate_future_prefill_tokens(
        self, req: "Req"
    ) -> tuple[int, int, float]:
        total_tokens = self._total_prefill_tokens(req)
        if total_tokens <= 0:
            return 0, 0, 0.0

        processed_tokens = min(self._processed_prefill_tokens(req), total_tokens)
        remaining_tokens = max(total_tokens - processed_tokens, 0)
        if remaining_tokens <= 0:
            return 0, 0, 1.0

        known_cached_tokens = min(self._known_cached_tokens(req), total_tokens)
        request_hit_rate = known_cached_tokens / max(total_tokens, 1)
        cache_hit_rate = min(max(request_hit_rate, 0.0), 1.0)

        estimated_total_miss_tokens = math.ceil(total_tokens * (1.0 - cache_hit_rate))
        already_computed_miss_tokens = max(processed_tokens - known_cached_tokens, 0)
        future_miss_tokens = min(
            max(estimated_total_miss_tokens - already_computed_miss_tokens, 0),
            remaining_tokens,
        )
        future_hit_tokens = max(remaining_tokens - future_miss_tokens, 0)
        return future_miss_tokens, future_hit_tokens, cache_hit_rate

    def _total_prefill_tokens(self, req: "Req") -> int:
        full_ids = getattr(req, "full_untruncated_fill_ids", None)
        if full_ids is not None and len(full_ids) > 0:
            return len(full_ids)
        origin_ids = getattr(req, "origin_input_ids", None)
        if origin_ids is not None:
            return len(origin_ids)
        return max(getattr(req, "seqlen", 0), 0)

    def _processed_prefill_tokens(self, req: "Req") -> int:
        processed_tokens = max(getattr(req, "num_matched_prefix_tokens", 0), 0)
        prefix_indices = getattr(req, "prefix_indices", None)
        if prefix_indices is not None:
            processed_tokens = max(processed_tokens, len(prefix_indices))
        extend_range = getattr(req, "extend_range", None)
        if extend_range is not None:
            processed_tokens = max(processed_tokens, extend_range.end)
        processed_tokens = max(processed_tokens, getattr(req, "already_computed", 0))
        return processed_tokens

    def _known_cached_tokens(self, req: "Req") -> int:
        cached_tokens = max(getattr(req, "cached_tokens", 0), 0)
        detailed_cached_tokens = (
            max(getattr(req, "cached_tokens_device", 0), 0)
            + max(getattr(req, "cached_tokens_host", 0), 0)
            + max(getattr(req, "cached_tokens_storage", 0), 0)
        )
        cached_tokens = max(cached_tokens, detailed_cached_tokens)
        cached_tokens = max(
            cached_tokens, getattr(req, "num_matched_prefix_tokens", 0)
        )
        prefix_indices = getattr(req, "prefix_indices", None)
        if prefix_indices is not None:
            cached_tokens = max(cached_tokens, len(prefix_indices))
        host_hit_length = max(getattr(req, "host_hit_length", 0), 0)
        storage_hit_length = max(getattr(req, "storage_hit_length", 0), 0)
        return max(cached_tokens, host_hit_length, storage_hit_length)

    def _tpot_pressure(self, now: float, running_reqs: Iterable["Req"]) -> float:
        if self.tpot_slo_s <= 0:
            return 0.0
        pressures = []
        for req in running_reqs:
            if req.finished() or req.is_retracted or len(req.output_ids) == 0:
                continue
            start = req.time_stats.prefill_finished_time
            last = req.time_stats.last_decode_finish_time or now
            if start > 0.0:
                tpot_s = 0.0
                decode_anchor = req.time_stats.last_decode_finish_time or start
                if now > decode_anchor:
                    tpot_s = max(tpot_s, now - decode_anchor)
                if last > start and len(req.output_ids) > 1:
                    decode_tokens = max(len(req.output_ids) - 1, 1)
                    tpot_s = max(tpot_s, (last - start) / decode_tokens)
                pressures.append(tpot_s / self.tpot_slo_s)
        return self._aggregate_pressure(pressures, self.tpot_stat)

    def _aggregate_pressure(self, pressures: Sequence[float], stat: str) -> float:
        if len(pressures) == 0:
            return 0.0
        if stat == "mean":
            return sum(pressures) / len(pressures)
        if stat == "p90":
            sorted_pressures = sorted(pressures)
            index = max(math.ceil(0.90 * len(sorted_pressures)) - 1, 0)
            return sorted_pressures[min(index, len(sorted_pressures) - 1)]
        return max(pressures)

    def _prefill_candidates(
        self, waiting_queue: Sequence["Req"], chunked_req: Optional["Req"]
    ) -> Iterable["Req"]:
        if chunked_req is not None:
            yield chunked_req
        for req in waiting_queue:
            if len(req.output_ids) == 0:
                yield req
