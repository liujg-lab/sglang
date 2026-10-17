#!/usr/bin/env python3
"""SPECTRE RSTeller serving-style 评测客户端（开放到达 + 流式百分位）。

功能
    HTTP 客户端，打已启动的 SPECTRE Target（默认 :30000），不拉起服务。
    与 eval_spectre_rsteller.py（闭式 chunk）互补：本脚本用 bench_serving 的
    开放到达（Poisson / burst）、全程 stream 计时，以及客户端 TTFT / Decode / E2E。
    仍走 RSTeller 图文 caption、服务端投机 meta_info，结束后调用
    eval_rsteller_metrics.py 补 BLEU / ROUGE。
    不设置 ignore_eos（caption 遇 EOS 应停下）。

    闭式脚本的 Decode tok/s 用各 chunk 墙钟之和，本脚本的 Decode tok/s 只覆盖
    客户端输出阶段。两列不要直接横比。Output tok/s (wall) 才是整场墙钟吞吐。

到达模型
    --request-rate inf（默认）  立刻把全部请求发出（burst）。
    --request-rate R            Poisson：相邻请求间隔 ~ Exp(R)。
    --max-concurrency N         asyncio.Semaphore，只限制在飞请求数。

单条时间轴（全部使用客户端 time.perf_counter）
    t_send              t_first                 t_last           t_end
    发起 HTTP ────────── 首次 token 事件 ──────── 最后新增 token ── 响应结束
              TTFT                  Decode Time
    TTFT          t_first - t_send。含传输、预处理、prefill 和首 token 等待。
                  不叫 Prefill。token 事件看 completion_tokens 增长，不要求 text 非空。
    Decode Time   t_last - t_first。不含 TTFT，不含结束通知到 t_end 的等待。
                  含后续调度、网络，以及 SR 草稿、通信和核验等待。
    Client E2E    t_end - t_send。
    Server E2E    meta_info.e2e_latency，单独保存，不与客户端 TTFT 相减。
    first_text_s  首次非空 text 的等待，用来看反分词造成的可见输出延迟。

    设首个 token 事件累计 token 数为 K，最后一次增长的累计数为 N。
    Decode tokens = N - K。只有一个 token 事件时 Decode Time 为 0，速度留空。
    有后续 token 但时间差为 0 时，速度留空，不写无穷大。
    多 token chunk 的 ITL 按该次交付间隔均摊（itl_kind=delivery_interval_estimate），
    不表示 chunk 内部的真实生成时间。

jsonl 每条字段（在原始 sample 上追加）
    extract_perf 提供 model_output、output_ids、finish_reason、response_meta_info
    和投机统计。它写出的 prefill_s / decode_s / decode_tok_s 不用于本脚本汇总；
    时间字段按上面的定义重新计算：
      ttft_s / first_text_s / first_chunk_tokens
      decode_s / decode_tokens / decode_tok_s
      client_e2e_s / server_e2e_s
      token_events / itl / itl_kind / token_metrics_available
      global_index
    error：缺图、HTTP 失败、服务端 error，或流在 finish_reason 之前断开。
    completion_tokens 缺失、非整数或倒退时，token 计数和依赖它的速度留空，不猜 token 数。
    精度脚本会跳过带 error 的条。

CSV 汇总列（只统计无 error 的成功条；百分位为排序后线性插值）
    Test Time                     评测结束墙钟
    request_rate                  发起速率；inf 为 burst
    max_concurrency               在飞上限；空表示不限制
    traffic                       固定为 open
    Total Items                   实际跑的条数（含失败）
    Total Generated Tokens        成功且 completion_tokens 有值的条之和

    Output tok/s (wall)           sum(completion_tokens) / 整场墙钟
                                  墙钟 = 主跑 create_task 起到 gather 结束（不含 warmup）
                                  有限 request-rate 时分母含空档，会低于饱和吞吐
    Request/s                     成功条数 / 整场墙钟（完成 QPS，不是 --request-rate）
    Concurrency                   sum(client_e2e_s) / 整场墙钟（Little's Law）

    Mean/Median/P90/P99 E2E (ms)  各条 client_e2e_s ×1000
    Mean Server E2E (ms)          各条 server_e2e_s 的算术平均 ×1000；缺省不回退到客户端
    Mean/P99 TTFT (ms)            各条 ttft_s ×1000
    Mean/P99 Decode Time (ms)     各条 decode_s ×1000，含单 token 事件的 0
    Total Decode Tokens           可测请求的 decode_tokens 之和
    Total Decode Time (s)         同一批请求的 decode_s 之和
    Decode tok/s                  上面两组和的商，不对单条速度做平均
                                  可测 = 成功、decode_tokens 有值且 decode_s > 0
                                  无可测样本时这三列留空
    Decode Samples                进入 Decode tok/s 的请求数，无样本时为 0
                                  串行时该速度适合比较 AR/SR 的输出阶段。
                                  并发时它是请求级合并速度，不是服务器聚合吞吐。
    Mean/P99 ITL (ms)             首事件之后各次交付间隔估计摊平后的均值 / 99 分位
                                  长请求权重大；与按条等权的 TPOT 不同
    Mean TPOT (ms)                仅 K=1、有后续 token 且 decode_s>0：
                                  decode_s / (N-1) 的算术平均 ×1000
    Mean Post-first Delivery (ms) 仅 K>1、有后续 token 且 decode_s>0：
                                  decode_s / (N-K) 的算术平均 ×1000
                                  首事件之后的平均 token 交付间隔

    Avg Accept Length             spec_accept_length 的逐条等权均值（toks / verify）
                                  分子含 prefill 首 token，且不含 AR 回退步，有偏
    Avg Accept Rate               spec_accept_rate 的逐条等权均值，同样有偏
                                  分母是 verify × (num_draft_tokens-1)。树模式
                                  topk=2, ndt=8, steps=4 时上限约为 4/7
    Avg Draft Tokens per Verify   draft_tokens_per_verify 均值
    Total Verify                  spec_verify_ct 之和，含 0
    Accept Length (weighted)      Σ(c-1-r-f) / Σv。每次验证产出的 token，含 bonus
                                  r 是 total_retractions，缺失按 0，不另加未知标记
                                  无回退字段时 f=0，结果是上界，Fallback Known=no
                                  Σv=0 时留空
    Tokens per Target Step        Σ(c-1-r) / Σ(v+f)。每个 Target decode 步的 token 数
    Accepted Draft per Verify     Σa / Σv。不含 bonus，树模式可读到接近 steps
    Accept Rate (weighted)        Σa / Σspec_draft_token_num。树模式上限与旧列相同
    Total Fallback Steps          Σf。字段缺失的请求按 0 计入，看 Fallback Known
    Fallback Ratio                Σf / Σ(v+f)
    Spec Samples                  带 spec_verify_ct（含 0）的成功请求数
    Fallback Known                yes=每条投机样本都有 spec_fallback_ct；否则 no
    Avg BLEU / ROUGE-1/2/L        本脚本先留空，精度脚本补上

    列集合变化后，已有旧表头的 CSV 会拒绝追加，需换一个 --output-csv。

运行（仓库根目录，Target + Draft 已启动）
    python test_sglang_liujg/evaluation/eval_spectre_rsteller_serving.py \
      --port 30000 --max-items 100 --request-rate inf --max-concurrency 4
    python test_sglang_liujg/evaluation/eval_spectre_rsteller_serving.py \
      --port 30000 --max-items 100 --request-rate 8 --max-concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

from eval_spectre_rsteller import (
    DEFAULT_DATA,
    DEFAULT_IMG_DIR,
    DEFAULT_RESULT_DIR,
    _fmt,
    append_jsonl,
    call_metrics_script,
    create_run_paths,
    extract_perf,
    flush_cache,
    load_dataset,
    mean,
    prepare_request,
    server_snapshot,
    spec_condition_columns,
    warn_eval_conditions,
)

CSV_FIELDS = [
    "Test Time",
    "request_rate",
    "max_concurrency",
    "traffic",
    "Algorithm",
    "Spec Config",
    "Total Items",
    "Total Generated Tokens",
    "Output tok/s (wall)",
    "Request/s",
    "Concurrency",
    "Mean E2E (ms)",
    "Median E2E (ms)",
    "P90 E2E (ms)",
    "P99 E2E (ms)",
    "Mean Server E2E (ms)",
    "Mean TTFT (ms)",
    "P99 TTFT (ms)",
    "Mean Decode Time (ms)",
    "P99 Decode Time (ms)",
    "Total Decode Tokens",
    "Total Decode Time (s)",
    "Decode tok/s",
    "Decode Samples",
    "Mean ITL (ms)",
    "P99 ITL (ms)",
    "Mean TPOT (ms)",
    "Mean Post-first Delivery (ms)",
    "Avg Accept Length",
    "Avg Accept Rate",
    "Avg Draft Tokens per Verify",
    "Total Verify",
    "Accept Length (weighted)",
    "Tokens per Target Step",
    "Accepted Draft per Verify",
    "Accept Rate (weighted)",
    "Total Fallback Steps",
    "Fallback Ratio",
    "Spec Samples",
    "Fallback Known",
    "Avg BLEU",
    "Avg ROUGE-1",
    "Avg ROUGE-2",
    "Avg ROUGE-L",
]

# Columns shared with every client that uses summarize_serving.
METRIC_CSV_FIELDS = CSV_FIELDS[CSV_FIELDS.index("Output tok/s (wall)") :]

AIOHTTP_READ_BUFSIZE = 10 * 1024**2  # 10 MB, same as bench_serving
# Amortized gaps are delivery intervals, not intra-chunk generation times.
ITL_KIND = "delivery_interval_estimate"


def _remove_sse_prefix(chunk: str) -> str:
    if chunk.startswith("data: "):
        return chunk[6:]
    if chunk.startswith("data:"):
        return chunk[5:].lstrip()
    return chunk


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile (numpy.percentile default style)."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _ms(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 1000.0


def _csv_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    if type(value) is not int:
        return None
    return value


def write_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    if file_exists:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != CSV_FIELDS:
            raise SystemExit(
                f"CSV header in {csv_path} does not match the current columns. "
                "Refusing to append. Use a new --output-csv."
            )
    with csv_path.open("a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


async def get_request(
    items: list[tuple[int, dict[str, Any]]],
    request_rate: float,
) -> AsyncGenerator[tuple[int, dict[str, Any]], None]:
    """Open-loop arrivals: burst if rate is inf, else Poisson Exp(rate)."""
    for idx, item in items:
        yield idx, item
        if request_rate == float("inf"):
            continue
        await asyncio.sleep(random.expovariate(request_rate))


def parse_sse_payload(chunk: str) -> Any | None:
    """Return a parsed SSE object, '[DONE]', or None for a blank line."""
    body = _remove_sse_prefix(chunk.strip())
    if not body:
        return None
    if body == "[DONE]":
        return "[DONE]"
    return json.loads(body)


def _int_token_ids(value: Any) -> list[int] | None:
    if not isinstance(value, list) or not all(type(token) is int for token in value):
        return None
    return list(value)


def _chunk_has_content(payload: dict[str, Any]) -> bool:
    text = payload.get("text")
    if isinstance(text, str) and text:
        return True
    ids = payload.get("output_ids")
    return isinstance(ids, list) and len(ids) > 0


@dataclass
class StreamObservation:
    """Client-clock view of one streamed /generate response."""

    text: str
    output_ids: list[int]
    meta_info: dict[str, Any]
    token_events: list[dict[str, Any]]
    itl: list[float]
    ttft_s: float | None
    first_text_s: float | None
    first_chunk_tokens: int | None
    decode_s: float | None
    decode_tokens: int | None
    decode_tok_s: float | None
    client_e2e_s: float
    server_e2e_s: float | None
    completion_tokens: int | None
    token_metrics_available: bool
    error: str | None = None


@dataclass
class StreamAccumulator:
    """Fold SSE events into client TTFT / Decode metrics.

    Times are seconds since t_send on one perf_counter clock. Token events are
    completion_tokens increases. Duplicate text, metadata, a finish notice with
    no new tokens, and [DONE] do not move t_last.
    """

    last_meta: dict[str, Any] = field(default_factory=dict)
    output_text: str = ""
    output_ids: list[int] = field(default_factory=list)
    token_events: list[dict[str, Any]] = field(default_factory=list)
    itl: list[float] = field(default_factory=list)
    last_completion: int | None = None
    t_first: float | None = None
    t_last: float | None = None
    first_text_s: float | None = None
    first_chunk_tokens: int | None = None
    saw_finish: bool = False
    token_metrics_available: bool = True

    def add_event(self, t_rel_s: float, payload: Any) -> None:
        if payload == "[DONE]":
            return
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid streaming chunk: {payload!r}")
        if "error" in payload:
            raise RuntimeError(f"server error: {payload['error']}")
        meta = payload.get("meta_info")
        if isinstance(meta, dict):
            self.last_meta = meta
            if meta.get("finish_reason") is not None:
                self.saw_finish = True
        self._note_text(t_rel_s, payload)
        if not self.token_metrics_available:
            self._absorb_text_only(payload)
            return
        self._note_tokens(t_rel_s, payload)

    def finish(self, client_e2e_s: float) -> StreamObservation:
        timing = self._timing()
        error = None if self.saw_finish else "stream ended without finish_reason"
        return StreamObservation(
            text=self.output_text,
            output_ids=list(self.output_ids),
            meta_info=dict(self.last_meta),
            token_events=[dict(event) for event in self.token_events],
            itl=timing["itl"],
            ttft_s=timing["ttft_s"],
            first_text_s=self.first_text_s,
            first_chunk_tokens=timing["first_chunk_tokens"],
            decode_s=timing["decode_s"],
            decode_tokens=timing["decode_tokens"],
            decode_tok_s=timing["decode_tok_s"],
            client_e2e_s=client_e2e_s,
            server_e2e_s=_as_float(self.last_meta.get("e2e_latency")),
            completion_tokens=timing["completion_tokens"],
            token_metrics_available=self.token_metrics_available,
            error=error,
        )

    def _note_text(self, t_rel_s: float, payload: dict[str, Any]) -> None:
        text = payload.get("text")
        if isinstance(text, str) and text and self.first_text_s is None:
            self.first_text_s = t_rel_s

    def _note_tokens(self, t_rel_s: float, payload: dict[str, Any]) -> None:
        meta = payload.get("meta_info")
        if not isinstance(meta, dict) or "completion_tokens" not in meta:
            if _chunk_has_content(payload):
                self._invalidate()
                self._absorb_text_only(payload)
            return
        raw = meta.get("completion_tokens")
        count = _as_int(raw)
        if count is None or count < 0:
            self._invalidate()
            self._absorb_text_only(payload)
            return
        if self.last_completion is not None and count < self.last_completion:
            self._invalidate()
            self._absorb_text_only(payload)
            return
        delta = count if self.last_completion is None else count - self.last_completion
        if delta == 0:
            self.last_completion = count
            self._absorb_output(payload, count, delta)
            return
        if self.t_first is None:
            self.t_first = t_rel_s
            self.first_chunk_tokens = count
        elif self.t_last is None:
            self._invalidate()
            self._absorb_text_only(payload)
            return
        else:
            gap = t_rel_s - self.t_last
            self.itl.extend([gap / delta] * delta)
        self.t_last = t_rel_s
        self.token_events.append(
            {
                "t_rel_s": t_rel_s,
                "completion_tokens": count,
                "delta_tokens": delta,
            }
        )
        self.last_completion = count
        self._absorb_output(payload, count, delta)

    def _invalidate(self) -> None:
        self.token_metrics_available = False
        self.itl = []
        self.t_first = None
        self.t_last = None
        self.first_chunk_tokens = None

    def _absorb_text_only(self, payload: dict[str, Any]) -> None:
        text = payload.get("text")
        if not isinstance(text, str) or not text:
            return
        if not self.output_text or text.startswith(self.output_text):
            self.output_text = text

    def _absorb_output(self, payload: dict[str, Any], count: int, delta: int) -> None:
        ids = _int_token_ids(payload.get("output_ids"))
        text = payload.get("text")
        text_s = text if isinstance(text, str) and text else None
        if ids is None:
            if text_s and (not self.output_text or text_s.startswith(self.output_text)):
                self.output_text = text_s
            return
        if len(ids) == count:
            self.output_ids = ids
            if text_s:
                self.output_text = text_s
            return
        if delta > 0 and len(self.output_ids) + len(ids) == count:
            self.output_ids.extend(ids)
            if text_s:
                self.output_text += text_s
            return
        if delta == 0 and not ids:
            if text_s and (len(self.output_ids) == count or not self.output_text):
                self.output_text = text_s
            return
        self.output_ids = []
        if text_s:
            self.output_text = text_s

    def _timing(self) -> dict[str, Any]:
        empty = {
            "ttft_s": None,
            "first_chunk_tokens": None,
            "decode_s": None,
            "decode_tokens": None,
            "decode_tok_s": None,
            "completion_tokens": None,
            "itl": [],
        }
        if not self.token_metrics_available:
            return empty
        if (
            self.t_first is None
            or self.t_last is None
            or self.first_chunk_tokens is None
            or self.last_completion is None
        ):
            empty["completion_tokens"] = self.last_completion
            return empty
        decode_tokens = self.last_completion - self.first_chunk_tokens
        decode_s = 0.0 if len(self.token_events) <= 1 else self.t_last - self.t_first
        decode_tok_s = None
        if decode_s < 0:
            decode_s_out: float | None = None
        else:
            decode_s_out = decode_s
            if len(self.token_events) > 1 and decode_s > 0:
                decode_tok_s = decode_tokens / decode_s
        return {
            "ttft_s": self.t_first,
            "first_chunk_tokens": self.first_chunk_tokens,
            "decode_s": decode_s_out,
            "decode_tokens": decode_tokens,
            "decode_tok_s": decode_tok_s,
            "completion_tokens": self.last_completion,
            "itl": list(self.itl),
        }


def observe_stream(
    events: list[tuple[float, Any]], client_e2e_s: float
) -> StreamObservation:
    """Replay already-parsed SSE events. ``t_rel`` is seconds since t_send."""
    acc = StreamAccumulator()
    for t_rel, payload in events:
        acc.add_event(t_rel, payload)
    return acc.finish(client_e2e_s)


def _blank_stream_fields(rec: dict[str, Any]) -> None:
    rec["itl"] = []
    rec["itl_kind"] = ITL_KIND
    rec["token_events"] = []
    rec["ttft_s"] = None
    rec["first_text_s"] = None
    rec["first_chunk_tokens"] = None
    rec["decode_s"] = None
    rec["decode_tokens"] = None
    rec["decode_tok_s"] = None
    rec["client_e2e_s"] = None
    rec["server_e2e_s"] = None
    rec["token_metrics_available"] = False
    rec["prefill_s"] = None
    rec["e2e_tok_s"] = None


def apply_stream_observation(rec: dict[str, Any], obs: StreamObservation) -> dict[str, Any]:
    """Overwrite extract_perf's mixed-clock times with the client-clock view."""
    rec["model_output"] = obs.text
    rec["output_ids"] = obs.output_ids
    rec["prefill_s"] = None
    rec["e2e_tok_s"] = None
    rec["ttft_s"] = obs.ttft_s
    rec["first_text_s"] = obs.first_text_s
    rec["first_chunk_tokens"] = obs.first_chunk_tokens
    rec["decode_s"] = obs.decode_s
    rec["decode_tokens"] = obs.decode_tokens
    rec["decode_tok_s"] = obs.decode_tok_s
    rec["client_e2e_s"] = obs.client_e2e_s
    rec["server_e2e_s"] = obs.server_e2e_s
    rec["token_events"] = obs.token_events
    rec["itl"] = obs.itl
    rec["itl_kind"] = ITL_KIND
    rec["token_metrics_available"] = obs.token_metrics_available
    rec["completion_tokens"] = obs.completion_tokens
    ids = obs.output_ids
    rec["token_ids_complete"] = (
        obs.token_metrics_available
        and obs.completion_tokens is not None
        and len(ids) == obs.completion_tokens
        and all(type(token) is int for token in ids)
    )
    if obs.error:
        rec["error"] = obs.error
    return rec


async def async_post_generate_stream(
    session: Any,
    url: str,
    payload: dict[str, Any],
) -> StreamObservation:
    """Stream /generate and time token arrivals on perf_counter."""
    acc = StreamAccumulator()
    st = time.perf_counter()
    async with session.post(url=url, json=payload) as response:
        if response.status != 200:
            detail = await response.text()
            raise RuntimeError(
                f"HTTP {response.status} from {url}: {response.reason or ''}: {detail}"
            )
        async for chunk_bytes in response.content:
            chunk_bytes = chunk_bytes.strip()
            if not chunk_bytes:
                continue
            chunk = chunk_bytes.decode("utf-8", errors="replace")
            t_rel = time.perf_counter() - st
            payload_obj = parse_sse_payload(chunk)
            if payload_obj is None:
                continue
            acc.add_event(t_rel, payload_obj)
    return acc.finish(time.perf_counter() - st)


async def async_run_one(
    session: Any,
    generate_url: str,
    item: dict[str, Any],
    img_dir: Path,
    sampling: dict[str, Any],
) -> dict[str, Any]:
    prompt, extra = prepare_request(item, img_dir)
    if prompt is None:
        rec = dict(item)
        rec["error"] = extra
        rec["model_output"] = ""
        _blank_stream_fields(rec)
        return rec
    payload = {
        "text": prompt,
        "image_data": extra,
        "sampling_params": sampling,
        "stream": True,
    }
    t0 = time.perf_counter()
    try:
        obs = await async_post_generate_stream(session, generate_url, payload)
        result = {
            "text": obs.text,
            "output_ids": obs.output_ids,
            "meta_info": obs.meta_info,
        }
        rec = extract_perf(item, result, ttft_s=None, client_e2e_s=None)
        return apply_stream_observation(rec, obs)
    except Exception as e:  # noqa: BLE001 — keep eval going
        rec = dict(item)
        rec["error"] = str(e)
        rec["model_output"] = ""
        elapsed = time.perf_counter() - t0
        rec["e2e_latency"] = elapsed
        _blank_stream_fields(rec)
        rec["client_e2e_s"] = elapsed
        return rec


def _spec_values(ok: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for rec in ok:
        value = _as_float(rec.get(key))
        if value is not None:
            values.append(value)
    return values


def _weighted_spec(ok: list[dict[str, Any]]) -> dict[str, Any]:
    """Token-weighted speculative totals.

    ``c`` is completion tokens, ``v`` is verify steps, ``a`` is accepted draft
    tokens, ``f`` is AR fallback steps, and ``r`` is retractions. Each
    retraction re-prefills one token that is neither verified nor a fallback
    step, so the numerators are ``c-1-r-f`` and ``c-1-r``. A missing retraction
    field counts as 0 and does not mark the result unknown. A missing fallback
    field is treated as 0 and marks the accept length as an upper bound.
    Ratios with a zero denominator stay empty. Tree-mode accept rate still uses
    ``verify * (num_draft_tokens - 1)`` as its denominator, so its ceiling is
    about ``steps / (num_draft_tokens - 1)``.
    """
    sum_verify_tokens = 0.0
    sum_decode_tokens = 0.0
    sum_v = 0.0
    sum_f = 0.0
    sum_steps = 0.0
    sum_a = 0.0
    sum_a_v = 0.0
    sum_draft = 0.0
    sum_rate_a = 0.0
    spec_samples = 0
    missing_fallback = False
    saw_completion = False
    saw_accepted = False
    saw_draft = False

    for rec in ok:
        verify = _as_float(rec.get("spec_verify_ct"))
        if verify is None:
            continue
        spec_samples += 1
        raw_fallback = rec.get("spec_fallback_ct")
        fallback = _as_float(raw_fallback)
        if fallback is None:
            missing_fallback = True
            fallback = 0.0
        completion = _as_float(rec.get("completion_tokens"))
        if completion is not None:
            saw_completion = True
            retract = _as_float(rec.get("total_retractions")) or 0.0
            prefill_tokens = 1.0 + retract
            sum_verify_tokens += completion - prefill_tokens - fallback
            sum_decode_tokens += completion - prefill_tokens
            sum_v += verify
            sum_f += fallback
            sum_steps += verify + fallback
        accepted = _as_float(rec.get("spec_accept_token_num"))
        if accepted is not None:
            saw_accepted = True
            sum_a += accepted
            sum_a_v += verify
        draft = _as_float(rec.get("spec_draft_token_num"))
        if accepted is not None and draft is not None:
            saw_draft = True
            sum_rate_a += accepted
            sum_draft += draft

    if spec_samples == 0:
        fallback_known = None
    elif missing_fallback:
        fallback_known = "no"
    else:
        fallback_known = "yes"

    def _ratio(numerator: float, denominator: float, ready: bool) -> float | None:
        if not ready or denominator <= 0:
            return None
        return numerator / denominator

    return {
        "accept_length_weighted": _ratio(sum_verify_tokens, sum_v, saw_completion),
        "tokens_per_target_step": _ratio(sum_decode_tokens, sum_steps, saw_completion),
        "accepted_draft_per_verify": _ratio(sum_a, sum_a_v, saw_accepted),
        "accept_rate_weighted": _ratio(sum_rate_a, sum_draft, saw_draft),
        "total_fallback_steps": sum_f if spec_samples and saw_completion else None,
        "fallback_ratio": _ratio(sum_f, sum_steps, saw_completion),
        "spec_samples": spec_samples,
        "fallback_known": fallback_known,
    }


def summarize_serving(records: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    ok = [r for r in records if not r.get("error")]
    toks = [
        value
        for value in (_as_float(r.get("completion_tokens")) for r in ok)
        if value is not None
    ]
    client_e2es = [
        value
        for value in (_as_float(r.get("client_e2e_s")) for r in ok)
        if value is not None
    ]
    server_e2es = [
        value
        for value in (_as_float(r.get("server_e2e_s")) for r in ok)
        if value is not None
    ]
    ttfts = [
        value for value in (_as_float(r.get("ttft_s")) for r in ok) if value is not None
    ]
    decode_times = [
        value
        for value in (_as_float(r.get("decode_s")) for r in ok)
        if value is not None
    ]
    itls: list[float] = []
    tpots: list[float] = []
    post_first: list[float] = []
    measurable: list[tuple[float, float]] = []
    for rec in ok:
        chunk_itl = rec.get("itl") or []
        if isinstance(chunk_itl, list):
            itls.extend(
                value for value in (_as_float(x) for x in chunk_itl) if value is not None
            )
        decode_tokens = _as_float(rec.get("decode_tokens"))
        decode_s = _as_float(rec.get("decode_s"))
        if (
            decode_tokens is not None
            and decode_s is not None
            and decode_s > 0
            and decode_tokens >= 0
        ):
            measurable.append((decode_tokens, decode_s))
        first_chunk = _as_int(rec.get("first_chunk_tokens"))
        output_len = _as_int(rec.get("completion_tokens"))
        if (
            first_chunk == 1
            and output_len is not None
            and output_len > 1
            and decode_s is not None
            and decode_s > 0
            and decode_tokens == output_len - 1
        ):
            tpots.append(decode_s / (output_len - 1))
        elif (
            first_chunk is not None
            and first_chunk > 1
            and decode_tokens is not None
            and decode_tokens > 0
            and decode_s is not None
            and decode_s > 0
        ):
            post_first.append(decode_s / decode_tokens)
    alens = _spec_values(ok, "spec_accept_length")
    rates = _spec_values(ok, "spec_accept_rate")
    drafts = _spec_values(ok, "draft_tokens_per_verify")
    verifies = _spec_values(ok, "spec_verify_ct")
    weighted = _weighted_spec(ok)
    total_toks = sum(toks)
    n_ok = len(ok)
    total_decode_tokens = sum(tokens for tokens, _ in measurable)
    total_decode_time = sum(seconds for _, seconds in measurable)
    decode_tok_s = (
        total_decode_tokens / total_decode_time
        if measurable and total_decode_time > 0
        else None
    )
    return {
        "n_ok": n_ok,
        "n_total": len(records),
        "total_tokens": total_toks,
        "output_tok_s": (total_toks / wall_s) if wall_s > 0 and total_toks else None,
        "request_s": (n_ok / wall_s) if wall_s > 0 and n_ok else None,
        "concurrency": (sum(client_e2es) / wall_s) if wall_s > 0 and client_e2es else None,
        "mean_e2e_ms": _ms(mean(client_e2es)),
        "median_e2e_ms": _ms(statistics.median(client_e2es) if client_e2es else None),
        "p90_e2e_ms": _ms(percentile(client_e2es, 90)),
        "p99_e2e_ms": _ms(percentile(client_e2es, 99)),
        "mean_server_e2e_ms": _ms(mean(server_e2es)),
        "mean_ttft_ms": _ms(mean(ttfts)),
        "p99_ttft_ms": _ms(percentile(ttfts, 99)),
        "mean_decode_ms": _ms(mean(decode_times)),
        "p99_decode_ms": _ms(percentile(decode_times, 99)),
        "total_decode_tokens": total_decode_tokens if measurable else None,
        "total_decode_time_s": total_decode_time if measurable else None,
        "decode_tok_s": decode_tok_s,
        "decode_samples": len(measurable),
        "mean_itl_ms": _ms(mean(itls)),
        "p99_itl_ms": _ms(percentile(itls, 99)),
        "mean_tpot_ms": _ms(mean(tpots)),
        "mean_post_first_ms": _ms(mean(post_first)),
        "avg_alen": mean(alens),
        "avg_accept": mean(rates),
        "avg_draft_per_verify": mean(drafts),
        "total_verify": sum(verifies) if verifies else 0.0,
        **weighted,
    }


def metric_csv_values(stats: dict[str, Any]) -> dict[str, str]:
    """CSV cells for the client-clock summary. Callers add their traffic columns."""
    return {
        "Total Items": stats["n_total"],
        "Total Generated Tokens": f"{stats['total_tokens']:.0f}",
        "Output tok/s (wall)": _csv_num(stats["output_tok_s"]),
        "Request/s": _csv_num(stats["request_s"]),
        "Concurrency": _csv_num(stats["concurrency"]),
        "Mean E2E (ms)": _csv_num(stats["mean_e2e_ms"]),
        "Median E2E (ms)": _csv_num(stats["median_e2e_ms"]),
        "P90 E2E (ms)": _csv_num(stats["p90_e2e_ms"]),
        "P99 E2E (ms)": _csv_num(stats["p99_e2e_ms"]),
        "Mean Server E2E (ms)": _csv_num(stats["mean_server_e2e_ms"]),
        "Mean TTFT (ms)": _csv_num(stats["mean_ttft_ms"]),
        "P99 TTFT (ms)": _csv_num(stats["p99_ttft_ms"]),
        "Mean Decode Time (ms)": _csv_num(stats["mean_decode_ms"]),
        "P99 Decode Time (ms)": _csv_num(stats["p99_decode_ms"]),
        "Total Decode Tokens": (
            f"{stats['total_decode_tokens']:.0f}" if stats["decode_samples"] else ""
        ),
        "Total Decode Time (s)": (
            _csv_num(stats["total_decode_time_s"]) if stats["decode_samples"] else ""
        ),
        "Decode tok/s": _csv_num(stats["decode_tok_s"]),
        "Decode Samples": str(stats["decode_samples"]),
        "Mean ITL (ms)": _csv_num(stats["mean_itl_ms"]),
        "P99 ITL (ms)": _csv_num(stats["p99_itl_ms"]),
        "Mean TPOT (ms)": _csv_num(stats["mean_tpot_ms"]),
        "Mean Post-first Delivery (ms)": _csv_num(stats["mean_post_first_ms"]),
        "Avg Accept Length": _csv_num(stats["avg_alen"], 4),
        "Avg Accept Rate": _csv_num(stats["avg_accept"], 4),
        "Avg Draft Tokens per Verify": _csv_num(stats["avg_draft_per_verify"], 4),
        "Total Verify": f"{stats['total_verify']:.0f}",
        "Accept Length (weighted)": _csv_num(stats["accept_length_weighted"], 4),
        "Tokens per Target Step": _csv_num(stats["tokens_per_target_step"], 4),
        "Accepted Draft per Verify": _csv_num(stats["accepted_draft_per_verify"], 4),
        "Accept Rate (weighted)": _csv_num(stats["accept_rate_weighted"], 4),
        "Total Fallback Steps": (
            f"{stats['total_fallback_steps']:.0f}"
            if stats["total_fallback_steps"] is not None
            else ""
        ),
        "Fallback Ratio": _csv_num(stats["fallback_ratio"], 4),
        "Spec Samples": str(stats["spec_samples"]),
        "Fallback Known": stats["fallback_known"] or "",
        "Avg BLEU": "",
        "Avg ROUGE-1": "",
        "Avg ROUGE-2": "",
        "Avg ROUGE-L": "",
    }


def format_serving_summary(stats: dict[str, Any], wall_s: float) -> str:
    """One-line client-clock summary shared by the open-loop scripts."""
    return (
        f"summary items={stats['n_ok']}/{stats['n_total']}  "
        f"toks={stats['total_tokens']:.0f}  wall={wall_s:.3f}s  "
        f"decode_tok/s={_fmt(stats['decode_tok_s'])}  "
        f"decode_samples={stats['decode_samples']}  "
        f"output_tok/s={_fmt(stats['output_tok_s'])}  "
        f"req/s={_fmt(stats['request_s'])}  "
        f"concurrency={_fmt(stats['concurrency'])}  "
        f"ttft_ms={_fmt(stats['mean_ttft_ms'], 2)}  "
        f"decode_ms={_fmt(stats['mean_decode_ms'], 2)}  "
        f"p99_e2e_ms={_fmt(stats['p99_e2e_ms'], 2)}  "
        f"p99_ttft_ms={_fmt(stats['p99_ttft_ms'], 2)}  "
        f"mean_itl_ms={_fmt(stats['mean_itl_ms'], 2)}  "
        f"tpot_ms={_fmt(stats['mean_tpot_ms'], 2)}  "
        f"post_first_ms={_fmt(stats['mean_post_first_ms'], 2)}  "
        f"accept={_fmt(stats['avg_accept'])}  "
        f"alen={_fmt(stats['avg_alen'])}  "
        f"alen_w={_fmt(stats['accept_length_weighted'])}  "
        f"tok/step={_fmt(stats['tokens_per_target_step'])}  "
        f"fallback={_fmt(stats['fallback_ratio'])}  "
        f"fallback_known={stats['fallback_known'] or ''}"
    )


def print_serving_item(idx: int, rec: dict[str, Any]) -> None:
    label = rec.get("image") or rec.get("id") or str(idx)
    text = (rec.get("model_output") or "").replace("\n", " ").strip()
    if len(text) > 80:
        text = text[:77] + "..."
    err = rec.get("error")
    if err:
        print(f"  [{idx}] {label}  ERROR {err}")
        return
    print(
        f"  [{idx}] toks={_fmt(rec.get('completion_tokens'), 0):>4s}  "
        f"ttft={_fmt(rec.get('ttft_s')):>6s}s  "
        f"decode={_fmt(rec.get('decode_s')):>6s}s  "
        f"client_e2e={_fmt(rec.get('client_e2e_s')):>6s}s  "
        f"accept={_fmt(rec.get('spec_accept_rate')):>6s}  "
        f"alen={_fmt(rec.get('spec_accept_length')):>6s}  "
        f"verify={_fmt(rec.get('spec_verify_ct'), 0):>4s}  "
        f"text={text!r}"
    )


async def run_benchmark(args: argparse.Namespace) -> int:
    import aiohttp

    generate_url = f"http://{args.host}:{args.port}/generate"
    base_url = f"http://{args.host}:{args.port}"
    sampling = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    dataset = load_dataset(args.data_file, args.max_items)
    indexed = list(enumerate(dataset))
    conc = args.max_concurrency if args.max_concurrency else "not set"
    rate_s = "inf" if args.request_rate == float("inf") else str(args.request_rate)
    print(f"loaded {len(dataset)} items from {args.data_file}")
    print(
        f"target={generate_url}  traffic=open  request_rate={rate_s}  "
        f"max_concurrency={conc}  max_new_tokens={args.max_new_tokens}"
    )
    snapshot = server_snapshot(base_url, args.timeout)
    warn_eval_conditions(snapshot, args.max_concurrency, bool(args.flush_cache))
    conditions = spec_condition_columns(snapshot)
    run_id, args.output_jsonl, args.output_csv, manifest_path = create_run_paths(
        args.output_jsonl, args.output_csv
    )
    print(f"run_id={run_id}  manifest={manifest_path}")

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(
        timeout=timeout, read_bufsize=AIOHTTP_READ_BUFSIZE
    ) as session:
        if not args.no_warmup and dataset:
            print("=== warmup (first item, discarded) ===")
            warmup = await async_run_one(
                session, generate_url, dataset[0], args.img_dir, sampling
            )
            if warmup.get("error"):
                raise RuntimeError(f"Warmup failed: {warmup['error']}")
            print("warmup ok")
            if args.flush_cache:
                flush_cache(base_url, args.timeout)

        semaphore = (
            asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None
        )

        async def limited(idx: int, item: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            if semaphore is None:
                rec = await async_run_one(
                    session, generate_url, item, args.img_dir, sampling
                )
            else:
                async with semaphore:
                    rec = await async_run_one(
                        session, generate_url, item, args.img_dir, sampling
                    )
            return idx, rec

        print("=== main run ===")
        t0 = time.perf_counter()
        tasks: list[asyncio.Task[tuple[int, dict[str, Any]]]] = []
        async for idx, item in get_request(indexed, args.request_rate):
            tasks.append(asyncio.create_task(limited(idx, item)))
        paired = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

    paired_sorted = sorted(paired, key=lambda x: x[0])
    records: list[dict[str, Any]] = []
    for idx, rec in paired_sorted:
        rec["global_index"] = idx
        print_serving_item(idx, rec)
        append_jsonl(args.output_jsonl, rec)
        records.append(rec)

    stats = summarize_serving(records, wall)
    test_time = time.strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "Test Time": test_time,
        "request_rate": rate_s,
        "max_concurrency": args.max_concurrency if args.max_concurrency else "",
        "traffic": "open",
        **conditions,
    }
    row.update(metric_csv_values(stats))
    write_csv_row(args.output_csv, row)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "server": snapshot,
                "results_jsonl": str(args.output_jsonl),
                "summary_csv": str(args.output_csv),
                "summary": stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(format_serving_summary(stats, wall))
    print(f"wrote {args.output_jsonl}")
    print(f"wrote {args.output_csv}")

    if not args.skip_metrics:
        call_metrics_script(args.output_jsonl, args.output_csv)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SPECTRE RSTeller serving-style evaluation (open-loop)"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Arrivals per second. inf sends all requests immediately (burst). "
        "Finite values use a Poisson process. Default: inf.",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Cap on in-flight requests (asyncio.Semaphore). Default: unlimited.",
    )
    p.add_argument("--max-items", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    p.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_RESULT_DIR / "RSTeller_spectre_serving_results.jsonl",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_RESULT_DIR / "RSTeller_spectre_serving_statistics.csv",
    )
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument(
        "--flush-cache",
        action="store_true",
        help="POST /flush_cache after warmup, once before the main run",
    )
    p.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Do not call eval_rsteller_metrics.py after generation",
    )
    p.add_argument("--seed", type=int, default=1, help="RNG seed for Poisson arrivals")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_items < 1:
        raise SystemExit("--max-items must be >= 1")
    if args.max_concurrency is not None and args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be >= 1")
    if args.request_rate <= 0:
        raise SystemExit("--request-rate must be > 0")
    random.seed(args.seed)
    return asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    raise SystemExit(main())
