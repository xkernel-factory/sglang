# SLO-Aware Prefill Scheduling 设计文档

本文档描述当前分支中的 **SLO-aware prefill scheduling**。该功能在每个 scheduler iteration 根据 TTFT/TPOT pressure 决定下一轮 forward 跑 prefill batch 还是 decode batch，并在需要 prefill 时选择固定的 chunk 档位。

当前实现只保留两个 prefill chunk 档位：

- `chunked_prefill_size`：无 decode work 或 TTFT 优先时使用。
- `slo_prefill_min_chunk_size`：TPOT 优先但不能安全让 decode 插队时使用。

## 启用参数

主要参数定义在 `python/sglang/srt/server_args.py`：

```bash
--enable-slo-aware-prefill
--slo-prefill-ttft-slo-ms <float>
--slo-prefill-tpot-slo-ms <float>
--slo-prefill-ttft-stat <max|mean|p90>
--slo-prefill-tpot-stat <max|mean|p90>
--slo-prefill-min-chunk-size <int>
--slo-prefill-yield-guard-ratio <float>
--slo-prefill-cost-profile-path <json>
--slo-prefill-cost-profile-output-path <json>
--enable-slo-prefill-startup-profiling
--disable-slo-prefill-startup-profiling
--slo-prefill-profile-prefill-token-sizes <int...>
--slo-prefill-profile-decode-context-len <int>
--slo-prefill-profile-decode-context-lens <int...>
--slo-prefill-profile-decode-batch-sizes <int...>
```

示例：

```bash
sglang serve \
  --model-path /path/to/model \
  --tp-size 4 \
  --chunked-prefill-size 32768 \
  --enable-slo-aware-prefill \
  --slo-prefill-ttft-slo-ms 15000 \
  --slo-prefill-ttft-stat p90 \
  --slo-prefill-tpot-slo-ms 60 \
  --slo-prefill-tpot-stat mean \
  --slo-prefill-min-chunk-size 4096
```

参数语义：

- `--chunked-prefill-size` 是无 decode work 或 TTFT 优先时使用的 prefill chunk。
- `--slo-prefill-min-chunk-size` 是 TPOT 优先但仍必须跑 prefill 时使用的 chunk。显式传入时使用用户传入值；未传入时默认等于 effective `chunked_prefill_size`。
- DP attention 开启时，SGLang 会先把 `--chunked-prefill-size` 除以 `dp_size` 得到本地有效值；SLO controller 直接使用该本地值，不会再把 `--slo-prefill-min-chunk-size` 除以 `dp_size`。
- `--slo-prefill-ttft-stat` / `--slo-prefill-tpot-stat` 控制 pressure 聚合口径，可选 `max`、`mean`、`p90`。
- `--slo-prefill-yield-guard-ratio` 是 TTFT slack 安全垫，默认 `0.05`。
- `--slo-prefill-cache-hit-io-cost-ratio` 是 cache hit token 的 HiCache IO 成本倍率，默认 `0.3`。
- `--slo-prefill-cost-profile-path` 加载离线生成的 Cp/Cd JSON 表；正式服务只做 CPU 侧表查找/插值，不额外跑 profiling forward。
- `--slo-prefill-cost-profile-output-path` 只在显式开启 startup profiling 时使用，把采样到的 Cp/Cd 表写出给后续正式服务复用。
- 启动 cost profiling 默认关闭；SLO 默认只使用 scheduler 公共状态和初始/默认成本估计，避免 synthetic forward 进入模型私有路径。需要离线生成 profile 表时可显式传入 `--enable-slo-prefill-startup-profiling`；失败或不支持时回退到初始值或默认成本估计。

## 调度流程

每轮 prefill admission 前，scheduler 计算并同步一个 `SloAwarePrefillPressureState`：

```text
(ttft_pressure, tpot_pressure, has_decode_work,
 prefill_cost, decode_cost, decode_context_len,
 ttft_future_prefill_cost, ttft_future_miss_tokens,
 ttft_future_hit_tokens, ttft_future_io_cost,
 ttft_cache_hit_rate)
```

然后按下面流程决策：

```text
objective = choose_objective(ttft_pressure, tpot_pressure, has_decode_work)

if objective == tpot and has_decode_work:
    if can_yield_to_decode(ttft_pressure):
        run decode
    else:
        run prefill with chunk = slo_prefill_min_chunk_size
else:
    run prefill with chunk = chunked_prefill_size
```

如果存在正在进行的 chunked prefill request，controller 不会阻塞它继续 prefill；scheduler 仍会优先处理 `yield_to_decode` 返回路径，避免不同 forward path 冲突。

## Pressure 计算

### TTFT Pressure

waiting queue 中尚未 prefill 的请求、以及正在 chunked prefill 的请求都会参与 TTFT pressure 统计：

```text
cache_hit_rate = request_known_hit_rate
future_miss_tokens = clamp(total_prompt_tokens * (1 - cache_hit_rate) - already_computed_miss_tokens, 0, remaining_tokens)
future_hit_tokens = remaining_tokens - future_miss_tokens
future_compute_cost = Cp(future_miss_tokens)
future_io_cost = cache_hit_io_cost_ratio * Cp(future_hit_tokens)
future_prefill_cost = future_compute_cost + future_io_cost
per_req_ttft_pressure = (prefill_wait_time + future_prefill_cost) / ttft_slo
ttft_pressure = aggregate(per_req_ttft_pressure, stat=max|mean|p90)
```

`future_miss_tokens` 表示预计还需要实际 prefill 计算的 token；`future_hit_tokens` 表示预计从 prefix cache / HiCache 命中的 token。cache hit token 不视为免费，而是按 `cache_hit_io_cost_ratio * Cp(hit_tokens)` 计入 IO 近似成本。

### TPOT Pressure

running decode 请求参与 TPOT pressure 统计：

```text
decode_gap = now - last_decode_finish_time
historical_avg_tpot = (last_decode_finish_time - prefill_finished_time) / decoded_tokens
per_req_tpot_pressure = max(decode_gap, historical_avg_tpot) / tpot_slo
tpot_pressure = aggregate(per_req_tpot_pressure, stat=max|mean|p90)
```

`decode_gap` 捕捉当前 decode 被 prefill 阻塞的实时风险；`historical_avg_tpot` 保留已观察到的慢 decode 信息。

## Objective 与 Decode Yield

`objective` 是中间目标，用于判断当前更应该保护 TTFT 还是 TPOT：

```text
if no active decode:
    objective = ttft
elif tpot_pressure >= 1 and ttft_pressure < 1:
    objective = tpot
elif ttft_pressure >= 1 and tpot_pressure < 1:
    objective = ttft
elif tpot_pressure > ttft_pressure + margin:
    objective = tpot
else:
    objective = ttft
```

`margin=0.10`，比较前 pressure 会 round 到两位小数，避免 TP ranks 因微小时间差选择不同 objective。

`objective=ttft` 会直接跑 prefill。`objective=tpot` 只表示优先考虑 decode，最终是否真的跑 decode 还要通过 TTFT slack 判断：

```text
can_yield_to_decode =
    ttft_pressure < 1
    and (1 - ttft_pressure) * ttft_slo
        > Cd(batch, kv_len_bucket) + Cp(min_chunk) + guard
```

```text
guard = max(
  2 * Cd(batch, kv_len_bucket),
  0.2 * Cp(min_chunk),
  yield_guard_ratio * ttft_slo,
)
```

- `can_yield_to_decode=True`：本轮跑 decode。
- `can_yield_to_decode=False`：本轮仍跑 prefill，但只使用 `slo_prefill_min_chunk_size`；`prefill_max_requests` 继承用户原始配置。

## Cost Profiling

SLO controller 使用两类成本估计。默认情况下不会为了建模成本额外构造 synthetic request，也不会在启动阶段额外调用模型 forward，因此 SLO 调度能力不绑定到任何具体模型族：

```text
Cp(tokens)                  # prefill token 数 -> prefill forward latency
Cd(context_len, batch_size)  # decode KV length bucket + batch size -> decode forward latency
```

生产推荐使用两阶段模式：

1. 离线生成 profile 表：

```bash
python3 benchmark/slo_prefill_cost_profiler.py \
  --output /tmp/slo_prefill_cost_profile.json \
  -- \
  --model-path /path/to/model \
  --tp-size 4 \
  --chunked-prefill-size 32768 \
  --slo-prefill-min-chunk-size 4096 \
  --slo-prefill-profile-prefill-token-sizes 4096 32768 \
  --slo-prefill-profile-decode-context-lens 4096 8192 16384 32768 \
  --slo-prefill-profile-decode-batch-sizes 1 2 4 8
```

该脚本会启动一次服务，自动追加 `--enable-slo-aware-prefill`、`--enable-slo-prefill-startup-profiling` 和 `--slo-prefill-cost-profile-output-path`，等待 JSON 写出后终止服务。

2. 正式服务加载 profile 表：

```bash
sglang serve \
  --model-path /path/to/model \
  --tp-size 4 \
  --chunked-prefill-size 32768 \
  --enable-slo-aware-prefill \
  --slo-prefill-ttft-slo-ms 15000 \
  --slo-prefill-tpot-slo-ms 60 \
  --slo-prefill-min-chunk-size 4096 \
  --slo-prefill-cost-profile-path /tmp/slo_prefill_cost_profile.json
```

正式服务只读取 JSON 并调用：

```text
controller.set_startup_cost_profile(
  prefill_cost_ms=[(tokens, latency_ms), ...],
  decode_cost_by_context_ms=[(context_len, batch_size, latency_ms), ...],
)
```

profile JSON schema：

```json
{
  "schema_version": 1,
  "metadata": {"model_path": "/path/to/model", "tp_size": 4},
  "prefill_cost_ms": [[4096, 12.3], [32768, 91.0]],
  "decode_cost_ms": [],
  "decode_cost_by_context_ms": [[4096, 1, 3.1], [4096, 4, 7.8]]
}
```

当前 startup profiler 的 Cp 默认只采样一个 `min_chunk` 点；生产离线 profiling 建议通过 `--slo-prefill-profile-prefill-token-sizes` 显式传入多个 token bucket，例如 `4096 32768`。Cd 按 `context_len + batch_size` 二维建模，每个 Cd 点先做 1 次 warmup，再采样 10 次 decode forward 取均值。长上下文 decode 场景建议显式传入多个 context bucket，例如：

```bash
--slo-prefill-profile-decode-context-lens 4096 8192 16384 32768
```

startup profiling 是显式 opt-in 的 best effort 实验能力：非 generation、disaggregation、pipeline parallelism 等场景会跳过；任一采样失败只丢弃该样本并回退初始值或默认成本估计。由于它使用 synthetic request 直接进入模型 forward，正式服务的官方模型兼容性不依赖该路径。

## TP / DP / HiCache 兼容性

- TP/DP attention 场景同步的是 pressure/cost 输入，而不是 Python decision 对象；同步后各 rank 基于相同输入本地计算相同 decision。
- DP attention 场景会覆盖 attention TP / CP 相关 group，避免不同 rank 进入不同 forward path。
- SLO controller 不重排 waiting queue，保留原有 schedule policy / prefix-cache locality 行为。
- 默认 SLO 路径不构造 synthetic request；加载离线 profile 表只读 JSON。显式开启 startup profiling 时才使用 synthetic request，并跳过 radix/cache insert，避免污染线上 prefix/HiCache 状态。

## 日志

启动成功会看到：

```text
SLO-aware prefill enabled: ... startup_profiling=False ...
```

加载离线 profile 表时会看到 `Loaded SLO prefill cost profile: ...`。显式开启并成功完成 startup profiling 时，还会看到：

```text
SLO prefill startup cost profile: Cp(ms)=[...], Cd_mean(ms)=[...], Cd_warmup=1, Cd_samples=10
```

运行时关键日志：

```text
SLO prefill decision: objective=..., allow=..., yield_to_decode=..., has_decode=..., chunk=..., prefill_max_requests=..., ttft_pressure=..., tpot_pressure=..., prefill_cost_ms_per_1k=..., decode_cost_ms=..., decode_context_len=..., ttft_future_cost_ms=..., ttft_future_miss_tokens=..., ttft_future_hit_tokens=..., ttft_future_io_cost_ms=..., cache_hit_io_cost_ratio=..., ttft_cache_hit_rate=..., ttft_slack_ms=..., yield_rhs_ms=..., yield_guard_ms=..., min_prefill_cost_ms=..., waiting=..., running=...
```

其中：

- `objective` 表示当前保护目标。
- `yield_to_decode=True` 表示本轮可让 decode 插队。
- `chunk` 表示本轮 prefill 被允许时使用的 chunk。
- `decode_cost_ms` / `decode_context_len` 表示当前 decode batch 的 Cd 估计输入。
- `ttft_future_*` 表示 TTFT pressure 中计入的未来 prefill 计算与 cache-hit IO 估计。
- `ttft_slack_ms` 与 `yield_rhs_ms` 是 decode yield 判断的左右两边。

## 验证

单测：

```bash
python3 test/registered/unit/managers/test_slo_aware_prefill.py
```

基础编译检查：

```bash
python3 -m py_compile \
  python/sglang/srt/managers/slo_aware_prefill.py \
  python/sglang/srt/managers/slo_prefill_cost_profile.py \
  python/sglang/srt/managers/scheduler.py \
  python/sglang/srt/server_args.py \
  benchmark/slo_prefill_cost_profiler.py
```
