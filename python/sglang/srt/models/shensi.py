from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.configs.shensi import ShensiConfig
from sglang.srt.distributed import get_pp_group, tensor_model_parallel_all_reduce
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import LayerNorm, RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_runner_backend,
    should_skip_post_experts_all_reduce,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _load_fused_expert_tensor,
)
from sglang.srt.models.deepseek_v2 import _is_cuda, _is_hip, _is_npu
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
)
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - non-triton platforms fall back to torch
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _attn_res_prenorm_kernel(
        prefix_ptr,
        arg1_ptr,
        x_ptr,
        rs_ptr,
        eps,
        HAS_ARG1: tl.constexpr,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        sq = 0.0
        for h0 in tl.static_range(0, H, BLOCK_H):
            o = h0 + offs
            m = o < H
            p = tl.load(prefix_ptr + row * H + o, mask=m, other=0.0).to(tl.float32)
            if HAS_ARG1:
                a = tl.load(arg1_ptr + row * H + o, mask=m, other=0.0).to(tl.float32)
                x = p + a
            else:
                x = p
            tl.store(x_ptr + row * H + o, x, mask=m)
            sq += tl.sum(x * x)
        tl.store(rs_ptr + row, 1.0 / tl.sqrt(sq / H + eps))

    @triton.jit
    def _attn_res_delta_kernel(
        proj_ptr,
        rs_ptr,
        prefix_ptr,
        arg1_ptr,
        bias_ptr,
        out_ptr,
        HAS_ARG1: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        rs = tl.load(rs_ptr + row)
        offs = tl.arange(0, BLOCK_H)
        s_k2 = 0.0
        s_r = 0.0
        for h0 in tl.static_range(0, H, BLOCK_H):
            o = h0 + offs
            m = o < H
            g0 = tl.load(proj_ptr + row * 5 * H + o, mask=m, other=0.0)
            g1 = tl.load(proj_ptr + row * 5 * H + H + o, mask=m, other=0.0)
            kv = tl.load(proj_ptr + row * 5 * H + 3 * H + o, mask=m, other=0.0)
            if HAS_BIAS:
                b0 = tl.load(bias_ptr + o, mask=m, other=0.0)
                b1 = tl.load(bias_ptr + H + o, mask=m, other=0.0)
            else:
                b0 = 0.0
                b1 = 0.0
            decay = tl.sigmoid(g0 * rs + b0)
            erase = tl.sigmoid(g1 * rs + b1)
            pre = tl.load(prefix_ptr + row * H + o, mask=m, other=0.0).to(tl.float32)
            k = kv * rs
            s_k2 += tl.sum(k * k)
            s_r += tl.sum(k * erase * decay * pre)
        inv = 1.0 / tl.maximum(tl.sqrt(s_k2), 1e-12)
        r = s_r * inv
        for h0 in tl.static_range(0, H, BLOCK_H):
            o = h0 + offs
            m = o < H
            g0 = tl.load(proj_ptr + row * 5 * H + o, mask=m, other=0.0)
            g1 = tl.load(proj_ptr + row * 5 * H + H + o, mask=m, other=0.0)
            g2 = tl.load(proj_ptr + row * 5 * H + 2 * H + o, mask=m, other=0.0)
            kv = tl.load(proj_ptr + row * 5 * H + 3 * H + o, mask=m, other=0.0)
            if HAS_BIAS:
                b0 = tl.load(bias_ptr + o, mask=m, other=0.0)
                b2 = tl.load(bias_ptr + 2 * H + o, mask=m, other=0.0)
            else:
                b0 = 0.0
                b2 = 0.0
            decay = tl.sigmoid(g0 * rs + b0)
            write = tl.sigmoid(g2 * rs + b2)
            pre = tl.load(prefix_ptr + row * H + o, mask=m, other=0.0).to(tl.float32)
            if HAS_ARG1:
                d = tl.load(arg1_ptr + row * H + o, mask=m, other=0.0).to(tl.float32)
            else:
                d = 0.0
            khat = kv * rs * inv
            upd = decay * pre - khat * r + write * d
            tl.store(out_ptr + row * H + o, upd, mask=m)

    @triton.jit
    def _attn_res_score_kernel(
        bank_ptr,
        proj_ptr,
        rs_ptr,
        updated_ptr,
        logits_ptr,
        hc,
        stride_t,
        stride_hc,
        stride_r,
        eps,
        R: tl.constexpr,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = row % hc
        base = (row // hc) * stride_t + h * stride_hc
        rs = tl.load(rs_ptr + row)
        offs = tl.arange(0, BLOCK_H)
        for r in tl.static_range(R):
            dot = 0.0
            sq = 0.0
            for h0 in tl.static_range(0, H, BLOCK_H):
                o = h0 + offs
                m = o < H
                b = tl.load(bank_ptr + base + r * stride_r + o, mask=m, other=0.0).to(
                    tl.float32
                )
                qv = tl.load(proj_ptr + row * 5 * H + 4 * H + o, mask=m, other=0.0) * rs
                dot += tl.sum(b * qv)
                sq += tl.sum(b * b)
            tl.store(logits_ptr + row * (R + 1) + r, dot / tl.sqrt(sq / H + eps))
        dot = 0.0
        sq = 0.0
        for h0 in tl.static_range(0, H, BLOCK_H):
            o = h0 + offs
            m = o < H
            u = tl.load(updated_ptr + row * H + o, mask=m, other=0.0)
            qv = tl.load(proj_ptr + row * 5 * H + 4 * H + o, mask=m, other=0.0) * rs
            dot += tl.sum(u * qv)
            sq += tl.sum(u * u)
        tl.store(logits_ptr + row * (R + 1) + R, dot / tl.sqrt(sq / H + eps))

    @triton.jit
    def _attn_res_mix_kernel(
        bank_ptr,
        updated_ptr,
        logits_ptr,
        out_ptr,
        hc,
        stride_t,
        stride_hc,
        stride_r,
        R: tl.constexpr,
        BLOCK_R: tl.constexpr,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = row % hc
        base = (row // hc) * stride_t + h * stride_hc
        r_offs = tl.arange(0, BLOCK_R)
        logits = tl.load(
            logits_ptr + row * (R + 1) + r_offs,
            mask=r_offs < R + 1,
            other=-float("inf"),
        )
        probs = tl.exp(logits - tl.max(logits, 0))
        probs = probs / tl.sum(probs, 0)
        o = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = o < H
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for r in tl.static_range(R):
            p = tl.sum(tl.where(r_offs == r, probs, 0.0))
            b = tl.load(bank_ptr + base + r * stride_r + o, mask=mask, other=0.0).to(
                tl.float32
            )
            acc += p * b
        p_u = tl.sum(tl.where(r_offs == R, probs, 0.0))
        u = tl.load(updated_ptr + row * H + o, mask=mask, other=0.0)
        acc += p_u * u
        acc += u
        tl.store(out_ptr + row * H + o, acc, mask=mask)

    @triton.jit
    def _hc_prenorm_kernel(
        streams_ptr,
        x_ptr,
        rs_ptr,
        ch,
        eps,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        sq = 0.0
        for h0 in range(0, ch, BLOCK):
            o = h0 + tl.arange(0, BLOCK)
            m = o < ch
            v = tl.load(streams_ptr + row * ch + o, mask=m, other=0.0).to(tl.float32)
            tl.store(x_ptr + row * ch + o, v, mask=m)
            sq += tl.sum(v * v)
        tl.store(rs_ptr + row, 1.0 / tl.sqrt(sq / ch + eps))

    @triton.jit
    def _hc_pre_finish_kernel(
        logits_ptr,
        scale_ptr,
        base_ptr,
        streams_ptr,
        out_ptr,
        c,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        hb = tl.program_id(1)
        h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        m = h < H
        scale = tl.load(scale_ptr)
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for ci in range(c):
            lg = tl.load(logits_ptr + row * c + ci)
            p = tl.sigmoid(lg * scale + tl.load(base_ptr + ci))
            v = tl.load(streams_ptr + row * c * H + ci * H + h, mask=m, other=0.0).to(
                tl.float32
            )
            acc += p * v
        tl.store(out_ptr + row * H + h, acc.to(out_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _hc_route_kernel(
        logits_ptr,
        scale_ptr,
        base_ptr,
        p_ptr,
        idx_ptr,
        hc: tl.constexpr,
        fixed: tl.constexpr,
        routed: tl.constexpr,
        active: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        m = offs < hc
        lg = tl.load(logits_ptr + row * hc + offs, mask=m, other=0.0)
        s = tl.sigmoid(
            lg * tl.load(scale_ptr) + tl.load(base_ptr + offs, mask=m, other=0.0)
        )
        s = tl.where(offs < fixed, -float("inf"), s)
        for k in range(fixed):
            tl.store(p_ptr + row * active + k, 1.0)
            tl.store(idx_ptr + row * active + k, k)
        taken = tl.zeros([BLOCK_H], dtype=tl.float32)
        for i in range(routed):
            cand = tl.where(m & (taken == 0.0), s, -float("inf"))
            best = tl.max(cand, 0)
            bi = tl.argmax(cand, 0)
            tl.store(p_ptr + row * active + fixed + i, best)
            tl.store(idx_ptr + row * active + fixed + i, bi.to(tl.int32))
            taken += tl.where(offs == bi, 1.0, 0.0)


def _hc_prenorm(streams: torch.Tensor, eps: float):
    rows, c, h = streams.shape
    if rows == 0:
        return None, None
    x = torch.empty((rows, c * h), dtype=torch.float32, device=streams.device)
    rs = torch.empty((rows,), dtype=torch.float32, device=streams.device)
    _hc_prenorm_kernel[(rows,)](streams, x, rs, c * h, eps, BLOCK=1024, num_warps=4)
    return x, rs


def _hc_pre_finish(logits, scale, base, streams, out_dtype):
    rows, c = logits.shape
    h = streams.shape[2]
    if rows == 0:
        return torch.zeros((rows, h), dtype=out_dtype, device=streams.device)
    out = torch.empty((rows, h), dtype=out_dtype, device=streams.device)
    _hc_pre_finish_kernel[(rows, triton.cdiv(h, 1024))](
        logits, scale, base, streams, out, c, H=h, BLOCK_H=1024, num_warps=4
    )
    return out


def _hc_route(logits, scale, base, fixed, routed, active):
    rows, hc = logits.shape
    p = torch.empty((rows, active), dtype=torch.float32, device=logits.device)
    idx = torch.empty((rows, active), dtype=torch.int32, device=logits.device)
    if rows == 0:
        return p, idx
    _hc_route_kernel[(rows,)](
        logits,
        scale,
        base,
        p,
        idx,
        hc=hc,
        fixed=fixed,
        routed=routed,
        active=active,
        BLOCK_H=triton.next_power_of_2(hc),
        num_warps=1,
    )
    return p, idx


def _attn_res_prenorm(
    prefix_sum: torch.Tensor, arg1: Optional[torch.Tensor], eps: float
):
    T, hc, H = prefix_sum.shape
    rows = T * hc
    if rows == 0:
        return None, None
    x = torch.empty((rows, H), dtype=torch.float32, device=prefix_sum.device)
    rs = torch.empty((rows,), dtype=torch.float32, device=prefix_sum.device)
    has_arg1 = arg1 is not None
    _attn_res_prenorm_kernel[(rows,)](
        prefix_sum,
        arg1 if has_arg1 else prefix_sum,
        x,
        rs,
        eps,
        HAS_ARG1=has_arg1,
        H=H,
        BLOCK_H=1024,
        num_warps=4,
    )
    return x, rs


def _attn_res_delta(proj, rs, prefix_sum, arg1, bias):
    T, hc, H = prefix_sum.shape
    rows = T * hc
    if rows == 0:
        return torch.zeros_like(prefix_sum, dtype=torch.float32)
    out = torch.empty((rows, H), dtype=torch.float32, device=proj.device)
    has_arg1 = arg1 is not None
    has_bias = bias is not None
    _attn_res_delta_kernel[(rows,)](
        proj,
        rs,
        prefix_sum,
        arg1 if has_arg1 else prefix_sum,
        bias if has_bias else proj,
        out,
        HAS_ARG1=has_arg1,
        HAS_BIAS=has_bias,
        H=H,
        BLOCK_H=1024,
        num_warps=4,
    )
    return out.view(T, hc, H)


def _attn_res_route(bank, proj, rs, updated, eps):
    T, hc, R, H = bank.shape
    rows = T * hc
    if rows == 0:
        return torch.zeros_like(updated)
    logits = torch.empty((rows, R + 1), dtype=torch.float32, device=bank.device)
    out = torch.empty((rows, H), dtype=torch.float32, device=bank.device)
    stride_t, stride_hc, stride_r, _ = bank.stride()
    _attn_res_score_kernel[(rows,)](
        bank,
        proj,
        rs,
        updated,
        logits,
        hc,
        stride_t,
        stride_hc,
        stride_r,
        eps,
        R=R,
        H=H,
        BLOCK_H=1024,
        num_warps=4,
    )
    _attn_res_mix_kernel[(rows, triton.cdiv(H, 1024))](
        bank,
        updated,
        logits,
        out,
        hc,
        stride_t,
        stride_hc,
        stride_r,
        R=R,
        BLOCK_R=triton.next_power_of_2(R + 1),
        H=H,
        BLOCK_H=1024,
        num_warps=4,
    )
    return out.view(T, hc, H)


class ShensiUnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(
            x.dtype
        )


class ShensiHyperConnection(nn.Module):
    def __init__(self, config: ShensiConfig, is_mlp: bool = False):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.active_streams = config.hc_active_streams
        self.fixed_streams = config.hc_fixed_streams
        self.routed_streams = self.active_streams - self.fixed_streams
        self.is_mlp = is_mlp

        self.input_norm = ShensiUnweightedRMSNorm(config.rms_norm_eps)
        self.pre_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_mult * config.hidden_size)
        )
        self.pre_base = nn.Parameter(torch.empty(self.hc_mult))
        self.pre_scale = nn.Parameter(torch.empty(1))

        self.route_norm = LayerNorm(self.hc_mult * config.hidden_size, eps=1e-5)
        self.route_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_mult * config.hidden_size)
        )
        self.route_base = nn.Parameter(torch.empty(self.hc_mult))
        self.route_scale = nn.Parameter(torch.empty(1))

        self.kr = (len(config.hc_conv_kernels) + 1) if is_mlp else 1
        if is_mlp:
            self.temporal_convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        config.hidden_size,
                        config.hidden_size,
                        ks,
                        padding=ks - 1,
                        groups=config.hidden_size,
                        bias=False,
                    )
                    for ks in config.hc_conv_kernels
                ]
            )
        self.post_fn = nn.Parameter(
            torch.empty(
                self.active_streams * self.kr, self.active_streams * config.hidden_size
            )
        )
        self.post_base = nn.Parameter(torch.empty(self.active_streams * self.kr))
        self.post_scale = nn.Parameter(torch.empty(1))
        self._fp32_cache = {}

    def _apply(self, fn, recurse=True):
        self._fp32_cache = {}
        return super()._apply(fn, recurse)

    def _fp32_weight(self, name: str) -> torch.Tensor:
        w = getattr(self, name)
        cached = self._fp32_cache.get(name)
        if cached is None or cached.dtype != torch.float32:
            cached = w.detach().float()
            self._fp32_cache[name] = cached
        return cached

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        # hidden_streams: [T, hc_mult, hidden]
        if _HAS_TRITON and hidden_streams.is_contiguous():
            x, rs = _hc_prenorm(hidden_streams, self.input_norm.eps)
            logits = F.linear(x, self._fp32_weight("pre_fn")) * rs.unsqueeze(1)
            return _hc_pre_finish(
                logits,
                self.pre_scale,
                self.pre_base,
                hidden_streams,
                hidden_streams.dtype,
            )
        flat = self.input_norm(hidden_streams.flatten(start_dim=1).float())
        pre = torch.sigmoid(
            F.linear(flat, self.pre_fn.float()) * self.pre_scale.float()
            + self.pre_base.float()
        )
        collapsed = (
            (pre.unsqueeze(-1) * hidden_streams).sum(dim=1).to(hidden_streams.dtype)
        )
        return collapsed

    def write_back(
        self, hidden_streams: torch.Tensor, sublayer_output: torch.Tensor
    ) -> torch.Tensor:
        if hidden_streams.shape[0] == 0:
            return hidden_streams

        T, hc, H = hidden_streams.shape
        flat = self.route_norm(
            hidden_streams.flatten(start_dim=1).to(self.route_norm.weight.dtype)
        ).float()
        route_logits = F.linear(flat, self._fp32_weight("route_fn"))
        if _HAS_TRITON and hidden_streams.is_contiguous():
            p, active_idx = _hc_route(
                route_logits,
                self.route_scale,
                self.route_base,
                self.fixed_streams,
                self.routed_streams,
                self.active_streams,
            )
            active_idx = active_idx.long()
        else:
            route_scores = torch.sigmoid(
                route_logits * self.route_scale.float() + self.route_base.float()
            )
            fixed_mask = (
                torch.arange(hc, device=route_scores.device) < self.fixed_streams
            )
            route_scores = route_scores.masked_fill(
                fixed_mask.view(1, -1), -float("inf")
            )
            fixed_idx = torch.arange(self.fixed_streams, device=hidden_streams.device)
            fixed_idx = fixed_idx.view(1, -1).expand(T, -1)
            routed_idx = route_scores.topk(self.routed_streams, dim=-1).indices
            active_idx = torch.cat([fixed_idx, routed_idx], dim=-1)
            p = torch.cat(
                [
                    torch.ones_like(fixed_idx, dtype=route_scores.dtype),
                    route_scores.gather(-1, routed_idx),
                ],
                dim=-1,
            )

        if self.is_mlp:
            x = (
                sublayer_output.transpose(0, 1)
                .unsqueeze(0)
                .to(self.temporal_convs[0].weight.dtype)
            )
            conv_outs = [conv(x)[..., :T] for conv in self.temporal_convs]
            ortho = []
            prevs = [x]
            for g in conv_outs:
                v = g
                for prev in prevs:
                    denom = (
                        (prev * prev)
                        .sum(dim=1, keepdim=True)
                        .clamp_min(self.input_norm.eps)
                    )
                    v = v - ((prev * v).sum(dim=1, keepdim=True) / denom) * prev
                ortho.append(v)
                prevs.append(v)
            out_aug = (
                torch.cat([x] + ortho, dim=1)
                .squeeze(0)
                .transpose(0, 1)
                .reshape(T, self.kr, H)
                .float()
            )
        else:
            out_aug = sublayer_output.float().unsqueeze(-2)

        active_streams = hidden_streams.gather(
            1, active_idx.unsqueeze(-1).expand(-1, -1, H)
        )
        post = 2 * torch.sigmoid(
            F.linear(
                self.input_norm(active_streams.flatten(start_dim=1).float()),
                self._fp32_weight("post_fn"),
            ).view(T, self.active_streams, self.kr)
            * self.post_scale.float()
            + self.post_base.float().view(self.active_streams, self.kr)
        )

        delta = torch.einsum("tkr,trh->tkh", post, out_aug) * p.unsqueeze(-1)
        updated = delta.to(hidden_streams.dtype)
        return hidden_streams.scatter(
            1, active_idx.unsqueeze(-1).expand(-1, -1, H), updated
        )


class ShensiAttentionResidual(nn.Module):
    def __init__(
        self,
        config: ShensiConfig,
        has_router: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.norm = ShensiUnweightedRMSNorm(config.rms_norm_eps)
        self.gate_proj = ReplicatedLinear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=True,
            params_dtype=torch.float32,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.q_proj = (
            nn.Parameter(
                torch.empty(config.hidden_size, config.hidden_size, dtype=torch.float32)
            )
            if has_router
            else None
        )
        self.k_proj = nn.Parameter(
            torch.empty(config.hidden_size, config.hidden_size, dtype=torch.float32)
        )
        self._fused_proj = None

    def _apply(self, fn, recurse=True):
        self._fused_proj = None
        return super()._apply(fn, recurse)

    def _proj_weight(self) -> torch.Tensor:
        if self._fused_proj is None:
            parts = [self.gate_proj.weight, self.k_proj]
            if self.q_proj is not None:
                parts.append(self.q_proj)
            fused = torch.cat(
                [p.detach().reshape(p.shape[0], -1).float() for p in parts], dim=0
            ).contiguous()
            n = self.gate_proj.weight.shape[0]
            with torch.no_grad():
                self.gate_proj.weight.data = fused[:n]
                self.k_proj.data = fused[n : n + self.k_proj.shape[0]]
                if self.q_proj is not None:
                    self.q_proj.data = fused[n + self.k_proj.shape[0] :]
            self._fused_proj = fused
        return self._fused_proj

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        residual: torch.Tensor,
        prefix_sum: torch.Tensor,
        output_norm_weight: Optional[torch.Tensor],
        num_blocks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            _HAS_TRITON
            and prefix_sum.is_contiguous()
            and (hidden_states is None or hidden_states.is_contiguous())
        ):
            x, rs = _attn_res_prenorm(prefix_sum, hidden_states, self.norm.eps)
            proj = F.linear(x, self._proj_weight())
            updated = _attn_res_delta(
                proj, rs, prefix_sum, hidden_states, self.gate_proj.bias
            )
            if num_blocks > 0:
                output = _attn_res_route(
                    residual[..., :num_blocks, :],
                    proj,
                    rs,
                    updated,
                    self.norm.eps,
                )
            else:
                output = updated
        else:
            delta = hidden_states
            updated = prefix_sum.float() + (delta.float() if delta is not None else 0.0)
            state = self.norm(updated)
            h = state.shape[-1]
            proj = F.linear(state, self._proj_weight())
            gate = proj[..., : 3 * h]
            if self.gate_proj.bias is not None:
                gate = gate + self.gate_proj.bias.float()
            decay, erase, write = torch.sigmoid(
                gate.reshape(*state.shape[:-1], 3, -1)
            ).unbind(-2)
            forgotten = decay * prefix_sum.float()
            khat = F.normalize(proj[..., 3 * h : 4 * h], dim=-1)
            r = (khat * erase * forgotten).sum(dim=-1, keepdim=True)
            updated = (
                forgotten
                - khat * r
                + write * (delta.float() if delta is not None else 0.0)
            )
            if num_blocks > 0:
                query = proj[..., 4 * h : 5 * h]
                bank = residual[..., :num_blocks, :].float()
                updated_logits = (updated * query).sum(
                    dim=-1, keepdim=True
                ) * torch.rsqrt(
                    updated.square().mean(dim=-1, keepdim=True) + self.norm.eps
                )
                bank_logits = torch.matmul(bank, query.unsqueeze(-1)).squeeze(
                    -1
                ) * torch.rsqrt(bank.square().mean(dim=-1) + self.norm.eps)
                scores = F.softmax(
                    torch.cat([bank_logits, updated_logits], dim=-1), dim=-1
                )
                routed = torch.matmul(
                    scores[..., :num_blocks].unsqueeze(-2), bank
                ).squeeze(-2)
                routed = routed + scores[..., num_blocks:] * updated
            else:
                routed = torch.zeros_like(updated)
            output = updated + routed
        if output_norm_weight is not None:
            output = (
                output
                * torch.rsqrt(
                    output.square().mean(dim=-1, keepdim=True) + self.norm.eps
                )
                * output_norm_weight.float()
            )
        return output.to(prefix_sum.dtype), updated.to(prefix_sum.dtype), residual


class ShensiHashMLP(nn.Module):
    def __init__(
        self,
        config: ShensiConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.limit = config.swiglu_limit
        intermediate_size = config.routed_expert_hidden_size
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            intermediate_size,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            intermediate_size,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=add_prefix("up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.deepemb = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("deepemb", prefix),
        )

    def forward(
        self, hidden_states: torch.Tensor, input_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        out, _ = self.down_proj(F.silu(gate) * up)
        return out * self.deepemb(input_ids)


def shensi_route(
    gating_output: torch.Tensor, topk: int, renormalize: bool, scoring_func: str
):
    """Shensi routing: score -> top-k -> (renormalize). The routed scale is
    applied by the MoE runner, matching the HF reference math."""
    if scoring_func == "sqrtsoftplus":
        scores = F.softplus(gating_output).sqrt()
    elif scoring_func == "softmax":
        scores = F.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = torch.sigmoid(gating_output)
    else:
        raise ValueError(f"Unsupported scoring_func: {scoring_func}")
    topk_ids = torch.topk(scores, topk, dim=-1, sorted=False).indices
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_weights, topk_ids


class ShensiRouter(nn.Module):
    def __init__(
        self,
        config: ShensiConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        apply_routed_scaling_factor_on_output: bool,
    ):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=False,
            scoring_func=config.scoring_func,
            correction_bias=None,
            quant_config=quant_config,
            routed_scaling_factor=config.routed_scaling_factor,
            # The routed scale is applied by the MoE runner (not the topk), so the
            # custom-routing branch below keeps its `not apply_...` invariant.
            apply_routed_scaling_factor_on_output=False,
            custom_routing_function=lambda hidden_states, gating_output, topk, renormalize: (
                shensi_route(gating_output, topk, renormalize, config.scoring_func)
            ),
            is_fp4_experts=getattr(quant_config, "is_fp4_experts", False),
            output_format=(
                TopKOutputFormat.STANDARD
                if (quant_config is None)
                and (not get_moe_runner_backend().is_flashinfer_trtllm())
                else None
            ),
        )

    def forward(self, hidden_states: torch.Tensor, expert_location_dispatch_info=None):
        router_logits = F.linear(hidden_states, self.weight)
        return self.topk(
            hidden_states,
            router_logits,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )


class ShensiSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: ShensiConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        is_block_write_layer: bool,
    ):
        super().__init__()
        self.is_block_write_layer = is_block_write_layer
        self.layer_id = layer_idx
        self.tp_size = get_parallel().tp_size
        expert_hidden = config.routed_expert_hidden_size
        self.experts = (
            get_moe_impl_class(quant_config)(
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=expert_hidden,
                intermediate_size=config.moe_intermediate_size,
                layer_id=layer_idx,
                quant_config=quant_config,
                routed_scaling_factor=config.routed_scaling_factor,
                prefix=add_prefix("experts", prefix),
            )
            if is_block_write_layer
            else None
        )
        self.gate = (
            ShensiRouter(
                config,
                layer_idx,
                quant_config,
                self.experts.should_fuse_routed_scaling_factor_in_topk,
            )
            if is_block_write_layer
            else None
        )
        self.routed_expert_down_proj = ReplicatedLinear(
            config.hidden_size,
            expert_hidden,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=add_prefix("routed_expert_down_proj", prefix),
        )
        self.routed_expert_norm = RMSNorm(expert_hidden, eps=config.rms_norm_eps)
        self.routed_expert_up_proj = ReplicatedLinear(
            expert_hidden,
            config.hidden_size,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=add_prefix("routed_expert_up_proj", prefix),
        )

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name in ("w13_weight", "w2_weight")
        ]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        down, _ = self.routed_expert_down_proj(hidden_states)
        dispatch_info = (
            ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
            if get_exec().moe.enable_eplb
            else None
        )
        topk_output = self.gate(hidden_states, dispatch_info)
        routed = self.experts(down, topk_output)
        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True
        ):
            routed = tensor_model_parallel_all_reduce(routed)
        routed = self.routed_expert_norm(routed)
        out, _ = self.routed_expert_up_proj(routed)
        return out


class ShensiDecoderLayer(DeepseekV4DecoderLayer):
    def __init__(
        self,
        config: ShensiConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        self.self_attn = self._build_self_attn(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_streams=alt_streams,
            compress_ratio_override=(
                compress_ratio_override
                if compress_ratio_override is not None
                else config.compress_ratios[layer_id]
            ),
        )

        self.is_hash = config.mlp_layer_types[layer_id] == "hash_moe"
        if self.is_hash:
            self.mlp = ShensiHashMLP(config, quant_config, add_prefix("mlp", prefix))
        else:
            layer_types = config.attn_res_block_layer_types()
            self.mlp = ShensiSparseMoeBlock(
                config,
                layer_id,
                quant_config,
                add_prefix("mlp", prefix),
                is_block_write_layer=layer_types[layer_id] == "block_write_layer",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_hc = ShensiHyperConnection(config, is_mlp=False)
        self.ffn_hc = ShensiHyperConnection(config, is_mlp=True)

        layer_types = config.attn_res_block_layer_types()
        self.is_block_write_layer = layer_types[layer_id] == "block_write_layer"
        self.prev_valid_blocks = sum(
            1 for role in layer_types[:layer_id] if role == "block_write_layer"
        )
        self.block_write_idx = self.prev_valid_blocks
        self.self_attention_attn_res = ShensiAttentionResidual(
            config,
            self.prev_valid_blocks > 0,
            add_prefix("self_attention_attn_res", prefix),
        )
        self.mlp_attn_res = ShensiAttentionResidual(
            config,
            self.prev_valid_blocks + self.is_block_write_layer > 0,
            add_prefix("mlp_attn_res", prefix),
        )

    def refresh_mhc_norm_weight_cache(self) -> None:
        pass

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        residual: torch.Tensor,
        prefix_sum: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = hidden_states - prefix_sum if hidden_states is not None else None

        hidden_states, prefix_sum, residual = self.self_attention_attn_res(
            delta,
            residual,
            prefix_sum,
            output_norm_weight=self.input_layernorm.weight,
            num_blocks=self.prev_valid_blocks,
        )
        if self.is_block_write_layer:
            written = (hidden_states if hidden_states is not None else prefix_sum).to(
                residual.dtype
            )
            residual = torch.cat(
                [
                    residual[..., : self.block_write_idx, :],
                    written.unsqueeze(-2),
                    residual[..., self.block_write_idx + 1 :, :],
                ],
                dim=-2,
            )
            prefix_sum = None

        collapsed = self.attn_hc(hidden_states)
        attn_output = self.self_attn(
            x=collapsed, positions=positions, forward_batch=forward_batch
        )
        hidden_states = self.attn_hc.write_back(hidden_states, attn_output)

        if prefix_sum is None:
            prefix_sum = hidden_states
        else:
            prefix_sum = prefix_sum + hidden_states

        hidden_states, prefix_sum, residual = self.mlp_attn_res(
            prefix_sum,
            residual,
            prefix_sum,
            output_norm_weight=self.post_attention_layernorm.weight,
            num_blocks=self.prev_valid_blocks + self.is_block_write_layer,
        )

        collapsed = self.ffn_hc(hidden_states)
        if self.is_hash:
            mlp_output = self.mlp(collapsed, input_ids)
        else:
            mlp_output = self.mlp(collapsed)
        hidden_states = self.ffn_hc.write_back(hidden_states, mlp_output)

        prefix_sum = prefix_sum + hidden_states
        return hidden_states, prefix_sum, residual


class ShensiHyperHead(nn.Module):
    def __init__(self, config: ShensiConfig):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.input_norm = ShensiUnweightedRMSNorm(config.rms_norm_eps)
        self.hc_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_mult * config.hidden_size)
        )
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))
        self._fp32_cache = {}

    def _apply(self, fn, recurse=True):
        self._fp32_cache = {}
        return super()._apply(fn, recurse)

    def _fp32_weight(self, name: str) -> torch.Tensor:
        w = getattr(self, name)
        cached = self._fp32_cache.get(name)
        if cached is None or cached.dtype != torch.float32:
            cached = w.detach().float()
            self._fp32_cache[name] = cached
        return cached

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_TRITON and x.is_contiguous():
            x32, rs = _hc_prenorm(x, self.input_norm.eps)
            logits = F.linear(x32, self._fp32_weight("hc_fn")) * rs.unsqueeze(1)
            return _hc_pre_finish(logits, self.hc_scale, self.hc_base, x, x.dtype)
        flat = self.input_norm(x.flatten(start_dim=1).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float())
        return (pre.unsqueeze(-1) * x).sum(dim=1).to(x.dtype)


class ShensiModel(DeepseekV4Model):
    def __init__(
        self,
        config: ShensiConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.pp_group = get_pp_group()
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult

        use_stream_pool = (
            _is_cuda
            or (
                _is_hip
                and (
                    envs.SGLANG_ROCM_USE_MULTI_STREAM.get()
                    or envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
                )
            )
            or (_is_npu and envs.SGLANG_NPU_USE_MULTI_STREAM.get())
        )
        device_module = torch.get_device_module()
        num_alt_streams = 5 if (_is_cuda or _is_npu) else 2
        self.alt_streams = (
            [device_module.Stream() for _ in range(num_alt_streams)]
            if use_stream_pool
            else None
        )

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: ShensiDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_streams=self.alt_streams,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.num_attn_res_blocks = config.attn_res_block_layer_types().count(
            "block_write_layer"
        )
        if self.pp_group.is_last_rank:
            self.output_attn_res = ShensiAttentionResidual(
                config, prefix=add_prefix("output_attn_res", prefix)
            )
            self.hc_head = ShensiHyperHead(config)
        else:
            self.output_attn_res = None
            self.hc_head = None

        self.tie_moe_groups()

    # `DeepseekV4Model` defines an `hc_head` *method*; this property keeps the
    # `hc_head.*` checkpoint keys while routing attribute access to the module.
    @property
    def hc_head(self):
        return self._modules.get("hc_head")

    @hc_head.setter
    def hc_head(self, value):
        self._modules["hc_head"] = value

    def tie_moe_groups(self) -> None:
        if self.pp_group.world_size > 1:
            return
        layer_types = self.config.attn_res_block_layer_types()
        write_layers = [
            i for i, role in enumerate(layer_types) if role == "block_write_layer"
        ]
        shared = {}
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx]
            if not isinstance(layer, ShensiDecoderLayer) or layer.is_hash:
                continue
            block_id = max(w for w in write_layers if w <= layer_idx)
            group = shared.setdefault(block_id, [layer.mlp, [layer_idx]])
            if group[0] is not layer.mlp:
                group[1].append(layer_idx)
                layer.mlp.gate = group[0].gate
                layer.mlp.experts = group[0].experts

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            if hidden_states.ndim == 2:
                hidden_states = hidden_states.view(-1, self.hc_mult, self.hidden_size)

        num_tokens = hidden_states.shape[0]
        residual = hidden_states.new_zeros(
            num_tokens, self.hc_mult, self.num_attn_res_blocks, self.hidden_size
        )
        prefix_sum = hidden_states
        hidden_states = None

        for i in range(self.start_layer, self.end_layer):
            hidden_states, prefix_sum, residual = self.layers[i](
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                prefix_sum=prefix_sum,
                forward_batch=forward_batch,
                input_ids=input_ids,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors({"hidden_states": hidden_states.flatten(1)})

        hidden_states, _, _ = self.output_attn_res(
            hidden_states,
            residual,
            prefix_sum,
            output_norm_weight=None,
            num_blocks=self.num_attn_res_blocks,
        )

        pre_hc_head = hidden_states.flatten(1)
        hidden_states = self.norm(self.hc_head(hidden_states))
        return hidden_states, pre_hc_head

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens


def _remap_shensi_attention_names(name: str) -> str:
    if ".self_attn." not in name:
        return name
    name = name.replace(".indexer.q_b_proj.", ".indexer.wq_b.")
    name = name.replace(".indexer.scorer.weights_proj.", ".indexer.weights_proj.")
    name = name.replace(".indexer.kv_proj.", ".indexer.compressor.wkv.")
    name = name.replace(".indexer.gate_proj.", ".indexer.compressor.wgate.")
    name = name.replace(".indexer.position_bias", ".indexer.compressor.ape")
    name = name.replace(".indexer.kv_norm.", ".indexer.compressor.norm.")
    name = name.replace(".compressor.kv_proj.", ".compressor.wkv.")
    name = name.replace(".compressor.gate_proj.", ".compressor.wgate.")
    name = name.replace(".compressor.position_bias", ".compressor.ape")
    name = name.replace(".compressor.kv_norm.", ".compressor.norm.")
    name = name.replace(".self_attn.q_a_proj.", ".self_attn.wq_a.")
    name = name.replace(".self_attn.q_a_norm.", ".self_attn.q_norm.")
    name = name.replace(".self_attn.q_b_proj.", ".self_attn.wq_b.")
    name = name.replace(".self_attn.kv_proj.", ".self_attn.wkv.")
    name = name.replace(".self_attn.o_a_proj.", ".self_attn.wo_a.")
    name = name.replace(".self_attn.o_b_proj.", ".self_attn.wo_b.")
    if name.endswith(".self_attn.sinks"):
        name = name[: -len("sinks")] + "attn_sink"
    return name


class ShensiForCausalLM(DeepseekV4ForCausalLM):
    def __init__(
        self,
        config: ShensiConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.num_fused_shared_experts = 0
        self.model = ShensiModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.pp_group = get_pp_group()
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        get_attn_tp_context().init_context(config.q_lora_rank, is_dsa=True)
        self.start_layer = self.model.start_layer
        self.end_layer = self.model.end_layer

    @property
    def routed_experts_weights_of_layer(self):
        return {
            layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
            for layer_id in range(self.model.start_layer, self.model.end_layer)
            if not self.model.layers[layer_id].is_hash
        }

    def _prewarm_mhc_kernels(self) -> None:
        pass

    def post_load_weights(self, is_nextn=False, weight_names=None):
        super().post_load_weights(is_nextn=is_nextn, weight_names=weight_names)
        # Merge the AttnRes projections before any forward so the fused fp32
        # weights are never (re)built inside a CUDA-graph capture.
        for layer_id in range(self.model.start_layer, self.model.end_layer):
            layer = self.model.layers[layer_id]
            layer.self_attention_attn_res._proj_weight()
            layer.mlp_attn_res._proj_weight()
        if self.model.output_attn_res is not None:
            self.model.output_attn_res._proj_weight()

    def _moe_block_owner_map(self) -> dict:
        layer_types = self.config.attn_res_block_layer_types()
        write_layers = [
            i for i, role in enumerate(layer_types) if role == "block_write_layer"
        ]
        return {
            i: max(w for w in write_layers if w <= i) for i in range(len(layer_types))
        }

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn: bool = False
    ):
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        block_owner = self._moe_block_owner_map()
        compressor_cache = {}

        def auto_weight_loader(param):
            return getattr(param, "weight_loader", default_weight_loader)

        def remap_tied_moe(name: str) -> str:
            match = re.search(r"model\.layers\.(\d+)\.mlp\.(gate|experts)\.", name)
            if match is None:
                return name
            layer_idx = int(match.group(1))
            owner = block_owner.get(layer_idx)
            if owner is None or owner == layer_idx:
                return name
            return name.replace(
                f"model.layers.{layer_idx}.mlp.", f"model.layers.{owner}.mlp.", 1
            )

        for name, loaded_weight in weights:
            name = _remap_shensi_attention_names(name)
            name = remap_tied_moe(name)

            layer_id = get_layer_id(name)
            if layer_id is not None and not (
                self.start_layer <= layer_id < self.end_layer
            ):
                continue

            if _load_fused_expert_tensor(name, loaded_weight, params_dict):
                loaded_params.add(name)
                continue

            if ".compressor.wkv." in name or ".compressor.wgate." in name:
                key = name.rsplit(".", 2)[0]
                is_kv = name.endswith(".wkv.weight")
                cached = compressor_cache.get(key)
                if cached is None:
                    compressor_cache[key] = (is_kv, loaded_weight)
                    continue
                cached_is_kv, cached_weight = cached
                kv = loaded_weight if is_kv else cached_weight
                wgate = cached_weight if is_kv else loaded_weight
                target = key + ".wkv_gate.weight"
                param = params_dict.get(target)
                if param is None:
                    logger.warning(f"{target} not found in params_dict.")
                    compressor_cache.pop(key)
                    continue
                auto_weight_loader(param)(param, torch.cat([kv, wgate], dim=0))
                loaded_params.add(target)
                compressor_cache.pop(key)
                continue

            if name not in params_dict:
                logger.warning(f"{name} not found in params_dict.")
                continue
            param = params_dict[name]
            auto_weight_loader(param)(param, loaded_weight)
            loaded_params.add(name)

        assert not compressor_cache, compressor_cache.keys()

        unloaded = params_dict.keys() - loaded_params
        skipped = ["attn_mqa.k_scale", "attn_mqa.v_scale", "blockscale_swizzled"]
        if not self.pp_group.is_first_rank:
            skipped.append("embed_tokens")
        if not self.pp_group.is_last_rank:
            skipped.extend(["output_attn_res", "hc_head", "model.norm", "lm_head"])
        unloaded = {p for p in unloaded if all(s not in p for s in skipped)}
        if unloaded:
            logger.warning(
                f"Some weights are not initialized from checkpoints: {unloaded}"
            )

        self.post_load_weights(is_nextn=is_nextn)


EntryClass = [ShensiForCausalLM]
