"""Local error of slot 4, layers 30-45. Does not change the forward."""

from __future__ import annotations

import json
import logging
import os

import torch
import torch.nn.functional as F

PREFIX = 4163
ROW_SPEC = 4
POS = 4167

_bag: dict = {}
_mods: dict = {}
_current: int | None = None
_mode = ""
_row = 0
_token_pos: int | None = None
_armed = False
_missed = False
_done = False
_pos_cos = None
_pos_sin = None
_rotary = None
_positions: list = []


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _log(line: str) -> None:
    logging.warning("FORKSUB %s", line)


def _capturing() -> bool:
    owners = [getattr(torch, "npu", None)]
    try:
        import torch_npu

        owners.append(getattr(torch_npu, "npu", None))
    except Exception:
        pass
    for owner in owners:
        check = getattr(owner, "is_current_stream_capturing", None)
        if check is None:
            continue
        try:
            return bool(check())
        except Exception:
            return False
    return False


def _cpu0(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else None
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            return None
        if value.numel() == 0:
            return None
        return int(value.reshape(-1)[0])
    return None


def _copy(tensor: torch.Tensor, row: int) -> torch.Tensor:
    src = tensor[row : row + 1]
    buf = torch.empty(src.shape, dtype=src.dtype, device=src.device)
    buf.copy_(src)
    return buf


def _keep(name: str, tensor: torch.Tensor) -> None:
    if _current is None or tensor is None or not torch.is_tensor(tensor):
        return
    try:
        _bag.setdefault(_current, {})[name] = _copy(tensor, _row)
    except Exception as exc:
        _log(f"ERR copy {name} L{_current} {exc}")


def begin_layer(decoder, hidden: torch.Tensor, forward_batch) -> None:
    global _current, _mode, _row, _token_pos, _armed, _missed
    _current = None
    if _capturing() or hidden is None or hidden.shape[0] == 0:
        return
    layer_id = int(decoder.self_attn.attn.layer_id)
    verify = bool(forward_batch.forward_mode.is_target_verify())
    if not verify and layer_id == 0 and hidden.shape[0] == 1:
        pos = getattr(forward_batch, "positions", None)
        if torch.is_tensor(pos) and pos.numel() > 0:
            _token_pos = int(pos.reshape(-1)[0])
        else:
            _token_pos = None
    if layer_id < 30 or layer_id > 45:
        return
    if verify:
        if hidden.shape[0] != 9:
            return
        seq = _cpu0(getattr(forward_batch, "seq_lens_cpu", None))
        if seq != PREFIX:
            if layer_id == 30 and not _missed and (seq is None or seq > PREFIX):
                _missed = True
                _log(f"miss seq={seq} rank={_rank()}")
            return
        _mode = "spec"
        _row = ROW_SPEC
    else:
        if hidden.shape[0] != 1 or _token_pos != POS:
            return
        _mode = "decode"
        _row = 0
    _current = layer_id
    _mods[layer_id] = decoder
    if not _armed:
        _armed = True
        _log(f"arm mode={_mode} rank={_rank()}")


def end_layer() -> None:
    global _current
    _current = None


def note_input_norm(hidden: torch.Tensor, residual: torch.Tensor) -> None:
    _keep("norm_out", hidden)
    _keep("norm_in", residual)


def note_qkv(qkv: torch.Tensor) -> None:
    _keep("qkv", qkv)


def note_qk(attn, q: torch.Tensor, k: torch.Tensor) -> None:
    global _pos_cos, _pos_sin, _rotary
    _keep("q", q)
    _keep("k", k)
    if _pos_cos is not None or _current is None:
        return
    rotary = attn.rotary_emb
    cos = getattr(rotary, "position_cos", None)
    sin = getattr(rotary, "position_sin", None)
    if cos is None or sin is None or cos.shape[0] <= _row:
        return
    try:
        _pos_cos = _copy(cos, _row)
        _pos_sin = _copy(sin, _row)
        _rotary = rotary
    except Exception as exc:
        _log(f"ERR copy rope {exc}")


def note_attn(attn_out: torch.Tensor) -> None:
    _keep("attn", attn_out)


def note_oproj(out: torch.Tensor) -> None:
    _keep("o_proj", out)


def note_post_norm(hidden: torch.Tensor, residual: torch.Tensor) -> None:
    _keep("post_out", hidden)
    _keep("post_in", residual)


def note_gate(gate_up: torch.Tensor) -> None:
    _keep("gate_up", gate_up)


def note_mlp(hidden: torch.Tensor) -> None:
    _keep("mlp", hidden)


def note_pair(hidden: torch.Tensor, residual: torch.Tensor) -> None:
    _keep("mlp_out", hidden)
    _keep("resid", residual)


def _hops(parents: list[int]) -> list[int]:
    hops: list[int] = []
    for i, parent in enumerate(parents):
        if parent == -1:
            hops.append(0)
        else:
            hops.append(hops[parent] + 1)
    return hops


def _log_m4(spec_info, positions: list[int]) -> None:
    if _rank() != 0:
        return
    rows = getattr(spec_info, "_fork_parents", None)
    if not rows:
        _log("m4 parents absent")
        return
    parents = [int(item) for item in rows[0]]
    hops = _hops(parents)
    depth = [0] + [item + 1 for item in hops]
    bad = []
    parts = []
    seen: dict[int, int] = {}
    for slot, hop in enumerate(depth):
        pos = positions[slot] if slot < len(positions) else None
        parent = -1 if slot == 0 else parents[slot - 1]
        expect = PREFIX + hop
        flag = ""
        if pos != expect:
            flag = " BAD"
            bad.append(slot)
        if hop in seen and seen[hop] != pos:
            flag = " BAD"
            bad.append(slot)
        seen[hop] = pos
        parts.append(f"{slot}:p={parent}:h={hop}:pos={pos}:exp={expect}{flag}")
    status = "fail" if bad else "pass"
    _log(
        f"m4 {status} parents={parents} hops={hops} positions={positions[:9]} "
        + " ".join(parts)
    )


def _brief(linear) -> str:
    method = getattr(linear, "quant_method", None)
    name = type(method).__name__ if method is not None else "none"
    bits = [name]
    for owner in (linear, method):
        if owner is None:
            continue
        for key in (
            "weight",
            "weight_scale",
            "input_scale",
            "smooth_scale",
            "deq_scale",
            "quant_bias",
            "weight_offset",
        ):
            val = getattr(owner, key, None)
            if torch.is_tensor(val):
                bits.append(f"{key}:{tuple(val.shape)}:{val.dtype}")
    for key in ("quant_type", "act_quant_type", "weight_quant_type", "dynamic"):
        if method is not None and hasattr(method, key):
            bits.append(f"{key}={getattr(method, key)}")
    return " ".join(bits)


def _log_m3() -> None:
    if _rank() != 0 or 36 not in _mods:
        return
    from sglang.srt.server_args import get_global_server_args

    args = get_global_server_args()
    quant = getattr(args, "quantization", None) or "none"
    path = getattr(args, "model_path", "") or ""
    cfg = None
    cfg_path = os.path.join(path, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as handle:
            cfg = json.load(handle).get("quantization_config")
    files = []
    if os.path.isdir(path):
        files = sorted(name for name in os.listdir(path) if "quant" in name.lower())
    dec = _mods[36]
    parts = [
        f"qkv={_brief(dec.self_attn.qkv_proj)}",
        f"o={_brief(dec.self_attn.o_proj)}",
        f"gate={_brief(dec.mlp.gate_up_proj)}",
        f"down={_brief(dec.mlp.down_proj)}",
    ]
    _log(f"m3 quant={quant} config={cfg} files={files} " + " ".join(parts))


def _weight(linear):
    weight = getattr(linear, "weight", None)
    if not torch.is_tensor(weight):
        return None
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    return weight


def _gemm(x: torch.Tensor, linear):
    weight = _weight(linear)
    if weight is None or x is None:
        return None
    return F.linear(x.float(), weight.float())


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(var + eps) * weight.float()


def _rel(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    a = got.float().reshape(-1)
    b = ref.float().reshape(-1)
    rel = float((a - b).norm() / b.norm().clamp_min(1e-6))
    cos = float(F.cosine_similarity(a, b, dim=0))
    return rel, cos


def _ulp(got: torch.Tensor, ref: torch.Tensor) -> tuple[int, float]:
    abs_v = float((got.float() - ref.float()).abs().max())
    try:
        a = got.detach().to(dtype=torch.bfloat16).contiguous().cpu().view(torch.int16)
        b = ref.detach().to(dtype=torch.bfloat16).contiguous().cpu().view(torch.int16)
        ulp = int((a.to(torch.int32) - b.to(torch.int32)).abs().max())
    except Exception:
        ulp = -1
    return ulp, abs_v


def _call(linear, x: torch.Tensor) -> torch.Tensor:
    out = linear(x)
    if isinstance(out, tuple):
        out = out[0]
    return out


def _rotate(x: torch.Tensor, neox: bool) -> torch.Tensor:
    if neox:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _cache_cos_sin(rotary, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    row = rotary.cos_sin_cache[index]
    last = int(row.numel())
    pair = row.reshape(2, last // 2)
    return pair[0].repeat(2), pair[1].repeat(2)


def _rope_ref(attn, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    if _rotary is None:
        return None
    q_size = int(attn.q_size)
    kv_size = int(attn.kv_size)
    head = int(attn.head_dim)
    q, k, _v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = _rms(q.view(1, -1, head), attn.q_norm.weight, float(attn.q_norm.variance_epsilon))
    k = _rms(k.view(1, -1, head), attn.k_norm.weight, float(attn.k_norm.variance_epsilon))
    cos, sin = _cache_cos_sin(_rotary, POS)
    cos = cos.float().view(1, 1, -1)
    sin = sin.float().view(1, 1, -1)
    neox = bool(getattr(_rotary, "is_neox_style", True))
    q = q * cos + _rotate(q, neox) * sin
    k = k * cos + _rotate(k, neox) * sin
    return q.reshape(1, -1), k.reshape(1, -1)


def _log_rope_index(positions: list[int]) -> None:
    if _rotary is None or _pos_cos is None or _rank() != 0:
        return
    cos, _sin = _cache_cos_sin(_rotary, POS)
    got = _pos_cos.reshape(-1).float()
    ref = cos.reshape(-1).float()
    n = min(got.numel(), ref.numel())
    abs_v = float((got[:n] - ref[:n]).abs().max()) if n else -1.0
    pos4 = positions[4] if len(positions) > 4 else None
    _log(f"rope_index pos4={pos4} cache={POS} abs={abs_v:.3e} match={int(abs_v == 0)}")


def _silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    d = gate_up.shape[-1] // 2
    return F.silu(gate_up[..., :d].float()) * gate_up[..., d:].float()


def _report(stage: str, layer: int, got, ref) -> None:
    if got is None or ref is None:
        _log(f"m1 mode={_mode} L{layer} {stage} skip rank={_rank()}")
        return
    rel, cos = _rel(got, ref)
    _log(f"m1 mode={_mode} L{layer} {stage} rel={rel:.6g} cos={cos:.6f} rank={_rank()}")


def _report_m2(stage: str, layer: int, got, ref) -> None:
    if got is None or ref is None:
        _log(f"m2 mode={_mode} L{layer} {stage} skip rank={_rank()}")
        return
    ulp, abs_v = _ulp(got, ref)
    _log(
        f"m2 mode={_mode} L{layer} {stage} ulp={ulp} abs={abs_v:.3e} rank={_rank()}"
    )


def _flush() -> None:
    global _done
    if _done or not _bag:
        return
    _log_m3()
    _log("m1 attn cite k-prime rank=%s" % _rank())
    if _mode == "spec":
        _log_rope_index(_positions)
    from sglang.srt.distributed import tensor_model_parallel_all_reduce

    for layer in range(30, 46):
        dec = _mods.get(layer)
        if dec is None:
            continue
        item = _bag.get(layer, {})
        attn = dec.self_attn
        rank = _rank()
        try:
            if "norm_in" in item and "norm_out" in item and rank == 0:
                _report(
                    "input_norm",
                    layer,
                    _rms(
                        item["norm_in"],
                        dec.input_layernorm.weight,
                        float(dec.input_layernorm.variance_epsilon),
                    ),
                    item["norm_out"],
                )
            if "norm_out" in item and "qkv" in item:
                _report("qkv", layer, _gemm(item["norm_out"], attn.qkv_proj), item["qkv"])
                _report_m2(
                    "qkv", layer, _call(attn.qkv_proj, item["norm_out"]), item["qkv"]
                )
            if "qkv" in item and "q" in item and rank == 0:
                ref = _rope_ref(attn, item["qkv"])
                if ref is None:
                    _log(f"m1 mode={_mode} L{layer} rope skip rank={rank}")
                else:
                    q_rel, q_cos = _rel(ref[0], item["q"])
                    k_rel, k_cos = _rel(ref[1], item["k"])
                    _log(
                        f"m1 mode={_mode} L{layer} rope q_rel={q_rel:.6g} q_cos={q_cos:.6f} "
                        f"k_rel={k_rel:.6g} k_cos={k_cos:.6f} rank={rank}"
                    )
            if "attn" in item and "o_proj" in item:
                _report("o_proj", layer, _gemm(item["attn"], attn.o_proj), item["o_proj"])
                _report_m2(
                    "o_proj", layer, _call(attn.o_proj, item["attn"]), item["o_proj"]
                )
            if "post_in" in item and "post_out" in item and rank == 0:
                _report(
                    "post_norm",
                    layer,
                    _rms(
                        item["post_in"],
                        dec.post_attention_layernorm.weight,
                        float(dec.post_attention_layernorm.variance_epsilon),
                    ),
                    item["post_out"],
                )
            if "post_out" in item and "gate_up" in item:
                _report(
                    "gate_up",
                    layer,
                    _gemm(item["post_out"], dec.mlp.gate_up_proj),
                    item["gate_up"],
                )
                _report_m2(
                    "gate_up",
                    layer,
                    _call(dec.mlp.gate_up_proj, item["post_out"]),
                    item["gate_up"],
                )
        except Exception as exc:
            _log(f"ERR m1 L{layer} {exc} rank={rank}")
        hidden = int(dec.hidden_size)
        device = dec.input_layernorm.weight.device
        dtype = dec.input_layernorm.weight.dtype
        partial = torch.zeros(1, hidden, dtype=torch.float32, device=device)
        comparable = False
        if "post_out" in item and _weight(dec.mlp.down_proj) is not None:
            gu = _gemm(item["post_out"], dec.mlp.gate_up_proj)
            if gu is not None:
                partial = F.linear(_silu_mul(gu), _weight(dec.mlp.down_proj).float())
                comparable = True
        full = tensor_model_parallel_all_reduce(partial)
        if rank == 0:
            if comparable and "mlp" in item:
                _report("mlp", layer, full, item["mlp"])
            else:
                _log(f"m1 mode={_mode} L{layer} mlp skip rank={rank}")
        x = item.get("post_out")
        if x is None:
            x = torch.zeros(1, hidden, dtype=dtype, device=device)
        replay = dec.mlp(x)
        if "post_out" in item and "mlp" in item:
            _report_m2("mlp", layer, replay, item["mlp"])
        elif rank == 0:
            _log(f"m2 mode={_mode} L{layer} mlp skip rank={rank}")
        nxt = _bag.get(layer + 1, {})
        if rank == 0 and "mlp_out" in item and "resid" in item:
            added = item["mlp_out"].float() + item["resid"].float()
            target = nxt.get("norm_in", item["mlp_out"] + item["resid"])
            _report("resid", layer, added, target)
    _done = True
    _bag.clear()
    _log(f"done mode={_mode} rank={_rank()}")


def flush_spec(spec_info) -> None:
    global _positions
    if _done or not _bag or spec_info is None:
        return
    pos = getattr(spec_info, "positions", None)
    if pos is None:
        return
    values = [int(item) for item in pos.detach().reshape(-1)[:9].cpu().tolist()]
    if len(values) < 9 or values[0] != PREFIX or values[4] != POS:
        _bag.clear()
        _log(f"skip positions={values} rank={_rank()}")
        return
    _positions = values
    _log_m4(spec_info, values)
    _flush()


def flush_decode(forward_batch) -> None:
    if forward_batch is not None and forward_batch.forward_mode.is_target_verify():
        return
    _flush()
