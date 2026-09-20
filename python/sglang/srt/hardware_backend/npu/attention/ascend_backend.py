from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import torch_npu
from sgl_kernel_npu.attention.sinks_attention import (
    attention_sinks_prefill_triton,
    attention_sinks_triton,
)

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.hardware_backend.npu.attention.ascend_torch_native_backend import (
    AscendTorchNativeAttnBackend,
)
from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
    is_fia_nz,
    is_mla_preprocess_enabled,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.speculative.tree_shared_prefix import (
    SHARED_PREFIX_IMPL,
    SharedPrefixMetadata,
    cache_view,
    cpu_prefix_lengths,
    fill_shared_draft_,
    fill_shared_verify_,
    shared_prefix_attention,
    shared_prefix_layer_supported,
)
from sglang.srt.speculative.standalone_remote.drafter.sr_tree_paged_layout import (
    IMPL_PAGED_ATB,
    IMPL_PAGED_FIA,
    SRTreePagedMetadata,
    build_step_context_lens,
    context_lens_list,
    kv_buckets_to_page_buckets,
    make_dummy_block_tables,
    max_query_pages_for_tree,
    prepare_tree_paged_view,
    quantize_page_width,
    read_sr_tree_paged_env,
    resolve_eager_page_buckets,
    select_page_bucket,
)
from sglang.srt.speculative.standalone_remote.verifier.sr_target_tree_fia import (
    IMPL_TREE_PAGED_FIA,
    SRTargetTreeFiaMetadata,
    fill_target_tree_fia_metadata_,
    maybe_select_target_tree_fia,
    pages_for_s_cap,
    prime_target_tree_fia_capture_,
    read_sr_target_tree_fia_env,
    target_tree_fia_blocked_extra_combos,
)
from sglang.srt.speculative.standalone_remote.sr_tail_attention import (
    SRTailAttentionMetadata,
    build_tail_attention_metadata,
)
from sglang.srt.speculative.spec_utils import (
    NpuGraphPreparationError,
    build_tree_draft_block_tables,
    build_tree_draft_kv_slots,
    expand_seq_lens_for_spec_topk,
    fill_tree_draft_metadata_,
    normalize_tree_draft_kv_lens,
)
from sglang.srt.speculative.tree_attn_fallback import (
    TREE_GRAPH_KV_BUCKETS_ENV,
    TREE_GRAPH_MAX_KV_ENV,
    build_tree_verify_kv_slots,
    fill_tree_verify_kv_slots_,
    flatten_paged_kv,
    gather_kv_into,
    log_tree_draft_slot_gather_once,
    log_tree_verify_fallback_once,
    log_tree_verify_kv_slot_layout_once,
    parse_tree_graph_kv_buckets,
    parse_tree_graph_max_kv,
    select_tree_kv_bucket,
    tree_attn_chunk_width,
    tree_compact_fia_layout_supported,
    tree_draft_attention,
    tree_fia_actual_seq_lengths_kv,
    tree_graph_slot_max_kv,
    tree_slot_graph_can_run_batch,
    tree_slot_graph_needed_kv,
    tree_verify_attention,
    use_tree_verify_fallback,
    verify_tree_topk_from_server_args,
    zero_gathered_kv_padding,
)
from sglang.srt.speculative.tree_attn_mask import (
    custom_mask_to_ascend_masked,
    full_mask_numel,
    inplace_update_graph_tree_attn_mask,
    resolve_tree_verify_mask_seq_lens,
)
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

import logging

import numpy as np


def _reshape_kv_for_fia_nz(
    tensor: torch.Tensor, num_heads: int, head_dim: int, page_size: int
) -> torch.Tensor:
    """Reshapes a tensor for FIA NZ format."""
    return tensor.view(-1, 1, num_heads * head_dim // 16, page_size, 16)


logger = logging.getLogger(__name__)


class AttnGraphRole:
    DECODE = "decode"
    TREE_DRAFT = "tree_draft"
    TARGET_VERIFY = "target_verify"


@dataclass
class ForwardMetadata:

    tree_shared: Optional[SharedPrefixMetadata] = None
    sr_tail: Optional[SRTailAttentionMetadata] = None
    sr_tree_paged: Optional[SRTreePagedMetadata] = None
    sr_target_tree_fia: Optional[SRTargetTreeFiaMetadata] = None

    # calculated map for kv positions [bs * maxseqlen]
    block_tables: Optional[torch.Tensor] = None

    # mapped block_tables for swa
    block_tables_swa: Optional[torch.Tensor] = None

    # seq len inputs
    extend_seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_int: Optional[torch.Tensor] = None
    seq_lens_cpu_list: Optional[List[int]] = None
    seq_lens_list_cumsum: Optional[List[int]] = None
    seq_lens: Optional[torch.Tensor] = None
    actual_seq_lengths_q: Optional[torch.Tensor] = None
    actual_seq_lengths_kv: Optional[torch.Tensor] = None

    # prefix cache
    prefix_lens: Optional[torch.Tensor] = None
    flatten_prefix_block_tables: Optional[torch.Tensor] = None

    # TARGET_VERIFY tree attention (True = masked, Ascend polarity)
    tree_attn_mask: Optional[torch.Tensor] = None
    tree_kv_lens: Optional[List[int]] = None

    # Tree-draft token-level slots (one row per branch)
    tree_draft_kv_slots: Optional[torch.Tensor] = None
    tree_draft_kv_lens_t: Optional[torch.Tensor] = None
    tree_draft_kv_slots_swa: Optional[torch.Tensor] = None

    # TARGET_VERIFY token-level slots (one row per draft query)
    tree_verify_kv_slots: Optional[torch.Tensor] = None
    tree_verify_kv_lens_t: Optional[torch.Tensor] = None

    # CPU FIA KV lengths for compact tree attention (zeros already mapped to 1)
    tree_fia_kv_lens_cpu: Optional[List[int]] = None


class AscendAttnMaskBuilder:
    def __init__(self, model_runner: ModelRunner, device, use_fia, use_mla):
        """
        Initialize the AscendAttnMaskBuilder class.

        :param model_runner: ModelRunner instance for model execution.
        :param device: Device to run the model on (e.g., 'cuda', 'npu').
        :param use_fia: Boolean flag to indicate if environment variable ASCEND_USE_FIA is set to 1.
        """
        self.use_fia = use_fia
        self.model_runner = model_runner
        self.device = device

        # Initialize mask
        mask_len = 128
        self.mask = self.generate_attn_mask(mask_len, "norm", model_runner.dtype).to(
            self.device
        )

        # Initialize FIA mask
        fia_mask_len = 2048
        self.fia_mask = self.generate_mask_flag(fia_mask_len).to(self.device)

        # Initialize MTP mask
        mtp_mask_len = 2048
        self.mtp_mask = self.generate_mask_flag(mtp_mask_len).to(self.device)

        # Initialize mixed chunk mask cache
        mixed_mask_len = 2048
        self.mixed_chunk_attn_mask = self.get_splitfuse_attn_mask(mixed_mask_len)

        if use_mla:
            # Initialize RingMla mask
            ringmla_mask_len = 512
            self.ringmla_mask = self.generate_attn_mask(
                ringmla_mask_len, "norm", torch.bfloat16
            ).to(self.device)

    @staticmethod
    def generate_mask_flag(max_seq_len):
        """
        Generate a mask flag for attention masks.

        :param max_seq_len: Maximum sequence length for the mask.
        :return: A boolean tensor representing the mask flag.
        """
        # Construct lower triangle matrix.
        mask_flag = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool).tril_()
        # Create upper triangle matrix used to mark mask positions.
        mask_flag = ~mask_flag
        return mask_flag

    @staticmethod
    def generate_attn_mask(max_seq_len, mode, dtype=torch.float16):
        """
        Generate an attention mask.

        :param max_seq_len: Maximum sequence length for the mask.
        :param mode: Mode of the mask ('mix' or 'norm').
        :param dtype: Data type of the mask tensor.
        :return: A tensor representing the attention mask.
        """
        mask_flag = AscendAttnMaskBuilder.generate_mask_flag(max_seq_len)
        if mode == "mix":
            mask_value = (
                float("-inf") if dtype in [torch.float16, torch.bfloat16] else 1
            )
        else:
            mask_value = torch.finfo(torch.float32).min if dtype == torch.float16 else 1
        attn_mask = (
            torch.zeros(size=(max_seq_len, max_seq_len))
            .masked_fill_(mask_flag, mask_value)
            .to(dtype)
        )
        return attn_mask

    @staticmethod
    def get_attention_mask_id(seq_lens, extend_lens):
        """
        Generate attention mask IDs based on sequence lengths and extended lengths.

        :param seq_lens: Sequence lengths.
        :param extend_lens: Extended lengths.
        :return: A tensor containing the attention mask IDs.
        """
        starts = seq_lens - extend_lens
        ends = seq_lens

        # Use torch.stack to stack the start and end indices together
        ranges = torch.stack((starts, ends), dim=-1)

        # Use list comprehension to generate tensors for each range and concatenate them
        attn_mask_id = torch.cat([torch.arange(start, end) for start, end in ranges])
        return attn_mask_id

    def update_attn_cache(
        self,
        seqlen: int,
        mask_cache: torch.Tensor,
        seq_len_cached: int,
        dtype: torch.dtype,
        mode,
    ):
        """
        Update the attention mask cache.

        :param seqlen: Maximum sequence length.
        :param mask_cache: Current attention mask cache.
        :param seq_len_cached: Cached sequence length.
        :param dtype: Data type of the mask tensor.
        :param mode: Mode of the mask ('mix' or 'norm').
        :return: Updated mask cache and sequence length cache.
        """
        if seqlen > seq_len_cached:
            seq_len_cached = seqlen
            mask_cache = self.generate_attn_mask(seqlen, mode, dtype)
        if mask_cache.dtype != dtype:
            mask_cache = mask_cache.to(dtype)
        return mask_cache, seq_len_cached

    def get_splitfuse_attn_mask(
        self,
        seq_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate a splitfuse attention mask.

        :param seq_lens: Sequence lengths.
        :return: A tensor representing the splitfuse attention mask.
        """
        attn_mask = (
            torch.triu(torch.ones(seq_lens, seq_lens), diagonal=1)
            .to(torch.int8)
            .to(self.device)
        )
        return attn_mask


class AscendAttnBackend(AttentionBackend):

    def __init__(
        self,
        model_runner: ModelRunner,
        speculative_step_id: int = 0,
        draft_topk: int = 1,
        draft_num_steps: int = 0,
        graph_roles=None,
    ):
        super().__init__()
        self.forward_metadata = None
        self.model_runner = model_runner
        self.device = model_runner.device
        self.speculative_step_id = speculative_step_id
        self.draft_topk = max(int(draft_topk), 1)
        self.verify_tree_topk = verify_tree_topk_from_server_args(
            model_runner.server_args
        )
        self.draft_num_steps = max(int(draft_num_steps), 0)
        if graph_roles is None:
            self.graph_roles = None
        else:
            self.graph_roles = frozenset(graph_roles)
        self._central_tree_draft_fill = False
        self._sr_tree_paged_requested = False
        self._sr_tree_paged_meta = None
        self._sr_tree_paged_prep_count = 0
        self._sr_tree_paged_copy_count = 0
        self.cuda_graph_verify_workspace = None
        self.speculative_step_offset_npu = torch.tensor(
            speculative_step_id + 1, device="npu"
        )
        self.page_size = model_runner.page_size
        self.model_dtype = model_runner.model_config.dtype
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        if self.use_mla:
            self.kv_lora_rank = model_runner.model_config.kv_lora_rank
            self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
            if (
                "MiniCPM3ForCausalLM"
                in model_runner.model_config.hf_config.architectures
            ):
                self.qk_nope_head_dim = (
                    model_runner.model_config.hf_config.qk_nope_head_dim
                )
            else:
                self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
            self.q_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        else:
            self.use_alibi = getattr(model_runner.model_config, "use_alibi", False)
            if (
                "Gemma2ForSequenceClassification"
                in model_runner.model_config.hf_config.architectures
            ):
                self.use_native_sdpa = True
        self.native_attn = AscendTorchNativeAttnBackend()
        self.graph_metadata = {}
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.graph_mode = False
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.cuda_graph_custom_mask = None
        self.cuda_graph_tree_attn_mask = None
        self.cuda_graph_verify_positions = None
        self.cuda_graph_kv_slots = None
        self.cuda_graph_kv_lens = None
        self.cuda_graph_kv_slots_swa = None
        self.cuda_graph_verify_workspace = None
        self._cuda_graph_draft_slot_views = {}
        self._cuda_graph_verify_slot_views = {}
        self.tree_graph_max_kv = parse_tree_graph_max_kv(
            os.environ.get(TREE_GRAPH_MAX_KV_ENV)
        )
        self.tree_kv_buckets = []
        self._active_tree_s_cap = None
        self._replay_tree_s_cap = None
        self._tree_kv_scratch = {}
        self._tree_scratch_max_rows = 0
        self._tree_scratch_max_cols = 0
        self.tree_fia_kv_lens_cpu = None
        self.tree_target_replay_count = 0
        self.tree_draft_replay_count = 0
        self.tree_capacity_fallback_count = 0
        self.tree_layout_fallback_count = 0
        self._logged_tree_fallback = set()
        self.ascend_attn_mask_builder = AscendAttnMaskBuilder(
            model_runner, self.device, self.use_fia, self.use_mla
        )
        self.mask, self.fia_mask, self.mtp_mask, self.mix_mask = (
            self.ascend_attn_mask_builder.mask,
            self.ascend_attn_mask_builder.fia_mask,
            self.ascend_attn_mask_builder.mtp_mask,
            self.ascend_attn_mask_builder.mixed_chunk_attn_mask,
        )
        if self.use_mla:
            self.ringmla_mask = self.ascend_attn_mask_builder.ringmla_mask
        self.is_hybrid_swa = model_runner.is_hybrid_swa
        if self.is_hybrid_swa:
            self.full_to_swa_index_mapping = (
                model_runner.token_to_kv_pool.full_to_swa_index_mapping
            )

        # head num padding
        self.padding_size_list = [1, 2, 4, 8, 16, 32, 64, 128]
        self.q_head_num_padding = None
        if hasattr(model_runner.model_config, "num_attention_heads") and self.use_mla:
            self.tp_q_head_num = (
                model_runner.model_config.num_attention_heads // get_attention_tp_size()
            )
            for num in self.padding_size_list:
                if num >= self.tp_q_head_num:
                    self.q_head_num_padding = num
                    break

        # dllm model config
        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm_model = False
        if self.dllm_config is not None:
            self.is_dllm_model = True
            self.dllm_block_size = self.dllm_config.block_size

        self._shared_graph_metadata = {}
        self._target_fia_graph_metadata = {}
        self._target_fia_dummy_page = 0
        self._init_tree_shared_prefix()

    def _paged_impl_selected(self) -> bool:
        return getattr(self, "tree_attention_impl", None) in (
            IMPL_PAGED_ATB,
            IMPL_PAGED_FIA,
        )

    def _init_tree_shared_prefix(self):
        """Select once, before either runner starts capturing graphs."""
        from sglang.srt.layers.radix_attention import RadixAttention

        self._sr_tree_paged_requested = read_sr_tree_paged_env()
        self.tree_attention_impl = (
            "compact_fia" if self._use_tree_compact_fia() else "chunked"
        )
        reason = None
        args = self.model_runner.server_args
        if (
            args.speculative_algorithm != "STANDALONE_REMOTE"
            or self.verify_tree_topk <= 1
        ):
            return
        if self.use_mla or self.is_hybrid_swa or self.is_dllm_model or self.use_alibi:
            reason = "specialized attention architecture"
        else:
            modules = list(self.model_runner.model.modules())
            layers = [m for m in modules if isinstance(m, RadixAttention)]
            if not layers or not all(shared_prefix_layer_supported(m) for m in layers):
                reason = "unsupported attention layer"
            elif any(getattr(m, "sinks", None) is not None for m in modules):
                reason = "attention sinks"
            else:
                pool = self.model_runner.token_to_kv_pool
                # Public accessors may wait for layer-wise cache transfers;
                # capability inspection must not wait for request data.
                get_key = getattr(pool, "_get_key_buffer", pool.get_key_buffer)
                get_value = getattr(pool, "_get_value_buffer", pool.get_value_buffer)
                try:
                    for layer in layers:
                        for cache in (
                            get_key(layer.layer_id),
                            get_value(layer.layer_id),
                        ):
                            if (
                                cache.dtype not in (torch.float16, torch.bfloat16)
                                or cache.dtype != self.model_dtype
                            ):
                                raise ValueError("quantized or mixed KV dtype")
                            if torch_npu.get_npu_format(cache) == 29:
                                raise ValueError("NZ KV layout")
                            cache_view(cache, layer.tp_k_head_num, layer.qk_head_dim)
                except (ValueError, RuntimeError) as exc:
                    reason = str(exc)
        # Capture layer/cache capability before Draft/Target policy rewrites reason.
        paged_capable = reason is None
        if reason is None:
            # Remote Draft is a standalone server, not a local draft worker.
            # Three-query, multi-step expansion regresses with the FP32 torch
            # path. Keep paged Draft on fused FIA; Target still shares prefix KV.
            if (
                getattr(args, "standalone_remote_role", None) == "draft"
                and self.page_size > 1
            ):
                reason = "paged SR Draft latency policy: prefer compact-FIA"
            else:
                self.tree_attention_impl = SHARED_PREFIX_IMPL
        if (
            self._sr_tree_paged_requested
            and paged_capable
            and self.tree_attention_impl == "compact_fia"
            and getattr(args, "speculative_algorithm", None) == "STANDALONE_REMOTE"
            and getattr(args, "standalone_remote_role", None) == "draft"
            and self.draft_topk > 1
            and int(self.page_size) > 1
            and not self.use_mla
            and not self.is_hybrid_swa
            and not self.use_alibi
            and not self.is_dllm_model
        ):
            self.tree_attention_impl = (
                "paged_fia" if getattr(self, "use_fia", False) else "paged_atb"
            )
            reason = None
        extra = target_tree_fia_blocked_extra_combos(args)
        self.tree_attention_impl, reason = maybe_select_target_tree_fia(
            self.tree_attention_impl,
            reason,
            requested=read_sr_target_tree_fia_env(),
            capable=paged_capable,
            extra_reason=extra,
            page_size=self.page_size,
            role=getattr(args, "standalone_remote_role", None),
            verify_topk=self.verify_tree_topk,
        )
        logger.info(
            "NPU SR tree attention implementation=%s fallback_reason=%s",
            self.tree_attention_impl,
            reason,
        )

    def _use_tree_shared_prefix(self):
        return getattr(self, "tree_attention_impl", None) == SHARED_PREFIX_IMPL

    def _use_target_tree_paged_fia(self):
        return getattr(self, "tree_attention_impl", None) == IMPL_TREE_PAGED_FIA

    def _target_fia_metadata(self, bs, queries, pages, *, graph):
        key = (int(bs), int(queries), int(pages))
        cache = getattr(self, "_target_fia_graph_metadata", None)
        if cache is None:
            cache = {}
            self._target_fia_graph_metadata = cache
        if graph and key in cache:
            return cache[key]
        md = SRTargetTreeFiaMetadata.allocate(
            bs, queries, pages, self.page_size, self.device
        )
        if graph:
            cache[key] = md
        return md

    def _prepare_target_tree_fia_eager(self, batch) -> bool:
        """Allocate page-aligned metadata for the real batch. Always succeeds."""
        lengths, raw_bs = self._tree_verify_mask_layout(
            batch.spec_info, batch.seq_lens_cpu
        )
        queries = int(batch.spec_info.draft_token_num)
        prefixes = cpu_prefix_lengths(lengths, raw_bs)
        needed = max((p + queries for p in prefixes), default=queries)
        pages = pages_for_s_cap(needed, self.page_size)
        md = self._target_fia_metadata(raw_bs, queries, pages, graph=False)
        fill_target_tree_fia_metadata_(
            md,
            self.req_to_token,
            batch.req_pool_indices,
            batch.spec_info.custom_mask,
            prefixes,
            queries,
            raw_bs,
            dummy_page=int(getattr(self, "_target_fia_dummy_page", 0)),
        )
        self.forward_metadata.sr_target_tree_fia = md
        return True

    def _run_sr_target_tree_fia(self, q, k_cache, v_cache, layer):
        md = getattr(self.forward_metadata, "sr_target_tree_fia", None)
        if md is None:
            raise RuntimeError("target tree FIA metadata missing")
        bs = int(md.block_tables.shape[0])
        queries = int(md.blocked_mask.shape[2])
        hq = layer.tp_q_head_num
        hkv = layer.tp_k_head_num
        query = q.reshape(bs, queries, hq, layer.qk_head_dim)
        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query,
            k_cache.view(-1, self.page_size, hkv * layer.qk_head_dim),
            v_cache.view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim),
            input_layout="BSND",
            num_heads=hq,
            num_key_value_heads=hkv,
            scale=layer.scaling,
            block_table=md.block_tables,
            block_size=self.page_size,
            atten_mask=md.blocked_mask,
            actual_seq_lengths=md.q_lens_cpu,
            actual_seq_lengths_kv=md.kv_lens_cpu,
            sparse_mode=0,
        )
        return output.reshape(bs * queries, hq * layer.v_head_dim)

    def _shared_metadata(self, bs, queries, width, *, draft, graph):
        key = (bs, queries, width, draft)
        if graph and key in self._shared_graph_metadata:
            return self._shared_graph_metadata[key]
        nodes = self.draft_topk * self.draft_num_steps if draft else queries
        path = self.draft_num_steps if draft else queries
        md = SharedPrefixMetadata.allocate(bs, queries, width, nodes, path, self.device)
        if graph:
            self._shared_graph_metadata[key] = md
        return md

    def _prepare_shared_eager(self, batch):
        draft = self._is_tree_draft(batch)
        if draft:
            lengths = cpu_prefix_lengths(batch.seq_lens_cpu, batch.batch_size)
            queries = self.draft_topk
        else:
            lengths, raw_bs = self._tree_verify_mask_layout(
                batch.spec_info, batch.seq_lens_cpu
            )
            lengths = cpu_prefix_lengths(lengths, raw_bs)
            queries = int(batch.spec_info.draft_token_num)
        extra = self.draft_num_steps if draft else queries
        if self.tree_kv_buckets and max(lengths, default=0) + extra > max(
            self.tree_kv_buckets
        ):
            self._log_tree_fallback_once(
                "capacity",
                "shared-prefix batch exceeds captured buckets; using eager compact-FIA",
            )
            return False
        md = self._shared_metadata(
            len(lengths), queries, max(lengths, default=0), draft=draft, graph=False
        )
        self.forward_metadata.tree_shared = md
        self._fill_shared_metadata(
            md, batch.req_pool_indices, lengths, draft=draft, batch=batch
        )
        return True

    def _fill_shared_metadata(self, md, pool, lengths, *, draft, batch=None):
        if draft:
            fill_shared_draft_(
                md,
                self.req_to_token,
                pool,
                lengths,
                page_size=self.page_size,
                topk=self.draft_topk,
                steps=self.draft_num_steps,
                step=self.speculative_step_id,
            )
        else:
            fill_shared_verify_(
                md,
                self.req_to_token,
                pool,
                lengths,
                batch.out_cache_loc,
                batch.spec_info.custom_mask,
                int(batch.spec_info.draft_token_num),
            )

    def _run_tree_shared_prefix_attention(self, q, k_cache, v_cache, layer):
        md = self.forward_metadata.tree_shared
        if md is None or not shared_prefix_layer_supported(layer):
            raise RuntimeError(
                "shared-prefix attention capability changed after graph selection"
            )
        bs, queries = md.path_lens.shape
        return shared_prefix_attention(
            q.view(bs, queries, layer.tp_q_head_num, layer.qk_head_dim),
            k_cache,
            v_cache,
            md,
            scale=layer.scaling,
            kv_heads=layer.tp_k_head_num,
        )

    def _graph_row_capacity(self, max_bs: int, max_num_tokens: int) -> int:
        roles = getattr(self, "graph_roles", None)
        if not roles:
            draft = max(int(self.speculative_num_draft_tokens or 1), 1)
            return max(
                int(max_num_tokens),
                int(max_bs) * max(draft, int(self.draft_topk), 1),
            )
        caps = []
        if AttnGraphRole.TREE_DRAFT in roles:
            caps.append(int(max_bs) * max(int(self.draft_topk), 1))
        if AttnGraphRole.TARGET_VERIFY in roles:
            caps.append(
                int(max_bs) * max(int(self.speculative_num_draft_tokens or 1), 1)
            )
        if AttnGraphRole.DECODE in roles:
            caps.append(int(max_num_tokens))
        return max(caps) if caps else int(max_num_tokens)

    def get_verify_buffers_to_fill_after_draft(self):
        """Tree mask and position buffers filled after draft for TARGET_VERIFY."""
        return [self.cuda_graph_custom_mask, self.cuda_graph_verify_positions]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        custom_mask = getattr(spec_info, "custom_mask", None)
        if custom_mask is None or self.cuda_graph_custom_mask is None:
            return
        n = custom_mask.numel()
        if n > self.cuda_graph_custom_mask.numel():
            raise RuntimeError(
                f"Ascend tree custom_mask ({n}) exceeds graph buffer "
                f"({self.cuda_graph_custom_mask.numel()})"
            )
        self.cuda_graph_custom_mask[:n].copy_(custom_mask.reshape(-1))
        spec_info.custom_mask = self.cuda_graph_custom_mask[:n]
        positions = getattr(spec_info, "positions", None)
        if positions is not None and self.cuda_graph_verify_positions is not None:
            pn = positions.numel()
            self.cuda_graph_verify_positions[:pn].copy_(positions.reshape(-1))
            spec_info.positions = self.cuda_graph_verify_positions[:pn]

    def _slot_gather_tree_attn(self) -> bool:
        """True when tree attention reads token slots instead of an FIA mask.

        Target verify keys off ``speculative_eagle_topk``; tree draft needs a
        paged KV cache on top of its own topk.
        """
        return int(self.verify_tree_topk) > 1 or (
            int(self.draft_topk) > 1 and int(self.page_size) > 1
        )

    def _tree_verify_mask_layout(self, spec_info, fallback_seq_lens):
        """Producer-side seq_lens / raw_bs that laid out ``custom_mask``.

        Prefers ``spec_info.seq_lens_cpu`` (recorded when the mask was built)
        over graph-padded ``buffers.seq_lens[:bs]``.
        """
        seq_list, raw_bs, _source = resolve_tree_verify_mask_seq_lens(
            spec_info, fallback_seq_lens
        )
        if isinstance(fallback_seq_lens, torch.Tensor):
            seq_t = torch.as_tensor(
                seq_list,
                dtype=fallback_seq_lens.dtype,
                device=fallback_seq_lens.device,
            )
        else:
            seq_t = torch.tensor(seq_list, dtype=torch.int32)
        return seq_t, int(raw_bs)

    def _fill_tree_verify_mask(self, forward_batch: ForwardBatch) -> None:
        if self._slot_gather_tree_attn():
            # Slot-gather carries visibility in tree_verify_kv_slots/_lens, so
            # the dense mask has no reader and would be context_len wide.
            return
        spec_info = getattr(forward_batch, "spec_info", None)
        custom_mask = getattr(spec_info, "custom_mask", None)
        if spec_info is None or custom_mask is None or custom_mask.numel() == 0:
            return
        num_draft = int(
            getattr(spec_info, "draft_token_num", None)
            or self.speculative_num_draft_tokens
            or 1
        )
        seq_lens, _raw_bs = self._tree_verify_mask_layout(
            spec_info, forward_batch.seq_lens
        )
        if seq_lens is None or int(seq_lens.numel()) == 0:
            return
        tree_mask = custom_mask_to_ascend_masked(
            custom_mask,
            seq_lens,
            num_draft,
            device=self.device,
        )
        if self.graph_mode and self.cuda_graph_tree_attn_mask is not None:
            tree_mask = inplace_update_graph_tree_attn_mask(
                self.cuda_graph_tree_attn_mask, tree_mask
            )
        self.forward_metadata.tree_attn_mask = tree_mask
        self.forward_metadata.tree_kv_lens = [
            int(s) + num_draft for s in seq_lens.tolist()
        ]

    def _is_tree_draft(self, forward_batch: ForwardBatch) -> bool:
        return (
            self.draft_topk > 1
            and forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.spec_info is not None
        )

    def _use_tree_draft_slot_gather(self, forward_batch: ForwardBatch) -> bool:
        return (
            self._is_tree_draft(forward_batch)
            and (self.page_size > 1 or self._use_tree_shared_prefix())
            and not self._paged_impl_selected()
        )

    def _fill_tree_draft_kv_slots(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        dest = getattr(self.forward_metadata, "tree_draft_kv_slots", None)
        dest_lens = getattr(self.forward_metadata, "tree_draft_kv_lens_t", None)
        max_kv = int(dest.shape[1]) if dest is not None else None
        slots, lens = build_tree_draft_kv_slots(
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            page_size=self.page_size,
            topk=self.draft_topk,
            step_id=self.speculative_step_id,
            num_steps=self.draft_num_steps,
            max_kv=max_kv,
        )
        dest = getattr(self.forward_metadata, "tree_draft_kv_slots", None)
        dest_lens = getattr(self.forward_metadata, "tree_draft_kv_lens_t", None)
        if dest is not None and dest_lens is not None:
            self._copy_into_graph_slot_buffers(dest, dest_lens, slots, lens)
            self._store_tree_fia_kv_lens_cpu(dest_lens, int(dest_lens.shape[0]))
        else:
            self.forward_metadata.tree_draft_kv_slots = slots
            self.forward_metadata.tree_draft_kv_lens_t = lens
            self._store_tree_fia_kv_lens_cpu(lens, int(lens.shape[0]))
        if self.is_hybrid_swa:
            swa_max_kv = max_kv
            swa_dest = getattr(self.forward_metadata, "tree_draft_kv_slots_swa", None)
            if swa_dest is not None:
                swa_max_kv = int(swa_dest.shape[1])
            swa_slots, swa_lens = build_tree_draft_kv_slots(
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                page_size=self.page_size,
                topk=self.draft_topk,
                step_id=self.speculative_step_id,
                num_steps=self.draft_num_steps,
                index_mapping=self.full_to_swa_index_mapping,
                max_kv=swa_max_kv,
            )
            if swa_dest is not None and dest_lens is not None:
                self._copy_into_graph_slot_buffers(
                    swa_dest, dest_lens, swa_slots, swa_lens
                )
            else:
                self.forward_metadata.tree_draft_kv_slots_swa = swa_slots

    def _slot_gather_kv_pool_size(self) -> int:
        """Token slots the KV pool physically holds.

        ``req_to_token.shape[1]`` is ``context_len`` wide, so it is a mapping
        table bound, not a capacity bound, and must not be used here.
        """
        pool = getattr(self.model_runner, "token_to_kv_pool", None)
        size = int(getattr(pool, "size", 0) or 0) if pool is not None else 0
        if size <= 0:
            size = int(getattr(self.model_runner, "max_total_num_tokens", 0) or 0)
        return max(size, 0)

    def _slot_gather_graph_max_kv(self) -> int:
        extra = max(
            int(self.draft_num_steps),
            int(self.speculative_num_draft_tokens or 1),
            1,
        )
        pool = self._slot_gather_kv_pool_size()
        cap = int(self.max_context_len)
        if pool > 0:
            cap = min(cap, pool)
        return max(cap, 1) + extra

    def _slot_gather_graph_slot_max_kv(self) -> int:
        """kv_slots width: original graph bound capped by the frozen S_cap."""
        return tree_graph_slot_max_kv(
            self._slot_gather_graph_max_kv(), self.tree_graph_max_kv
        )

    def tree_slot_graph_width(self) -> Optional[int]:
        """Captured slot-table columns, or None before graph buffers exist."""
        if self._use_tree_shared_prefix() or self._use_target_tree_paged_fia():
            return (
                self._replay_tree_s_cap
                or self._active_tree_s_cap
                or max(self.tree_kv_buckets, default=self.tree_graph_max_kv)
            )
        slots = self.cuda_graph_kv_slots
        if slots is None:
            return None
        s_cap = self._active_tree_s_cap or self._replay_tree_s_cap
        if s_cap is not None:
            return min(int(s_cap), int(slots.shape[1]))
        return int(slots.shape[1])

    def _tree_slot_bind_cols(self) -> int:
        slots = self.cuda_graph_kv_slots
        if slots is None:
            return 0
        width = self.tree_slot_graph_width()
        if width is None:
            return int(slots.shape[1])
        return int(width)

    def _log_tree_fallback_once(self, kind: str, reason: str) -> None:
        key = f"{kind}:{reason}"
        if key in self._logged_tree_fallback:
            return
        self._logged_tree_fallback.add(key)
        logger.info("tree attention %s fallback: %s", kind, reason)

    def _use_tree_compact_fia(self, q_rope: Optional[torch.Tensor] = None) -> bool:
        return tree_compact_fia_layout_supported(
            use_mla=self.use_mla, has_rope_split=q_rope is not None
        )

    def tree_slot_graph_can_run(self, forward_batch: ForwardBatch) -> bool:
        """True when this tree batch's needed KV fits a captured bucket.

        Uses captured buckets when they exist, otherwise the slot-table width.
        Verify reads mask-producer seq_lens; draft uses seq+num_steps as a
        conservative bound. Per-batch: overflow does not disable later graph hits.
        """
        fm = getattr(forward_batch, "forward_mode", None)
        is_verify = bool(fm is not None and fm.is_target_verify())
        is_draft = bool(
            fm is not None
            and fm.is_decode_or_idle()
            and int(self.draft_topk) > 1
            and (int(self.page_size) > 1 or self._use_tree_shared_prefix())
        )
        seq = getattr(forward_batch, "seq_lens_cpu", None)
        if seq is None:
            seq = getattr(forward_batch, "seq_lens", None)
        spec_info = getattr(forward_batch, "spec_info", None)
        fallback = seq
        if is_verify and spec_info is not None:
            producer_seq_lens = getattr(spec_info, "seq_lens_cpu", None)
            fallback = seq if producer_seq_lens is None else producer_seq_lens
        buckets = list(self.tree_kv_buckets) if self.tree_kv_buckets else None
        if self._paged_impl_selected() and is_draft:
            prefixes = []
            if seq is not None:
                prefixes = [
                    int(x)
                    for x in (
                        seq.reshape(-1).tolist()
                        if isinstance(seq, torch.Tensor)
                        else seq
                    )
                ][: int(getattr(forward_batch, "batch_size", len(seq)))]
            needed_pages = max(
                max_query_pages_for_tree(
                    prefixes, self.draft_num_steps, self.page_size
                )
                or [1]
            )
            page_buckets = (
                kv_buckets_to_page_buckets(buckets, self.page_size)
                if buckets
                else None
            )
            if not page_buckets:
                max_pages = getattr(self, "_paged_graph_max_pages", None)
                if max_pages is not None:
                    page_buckets = [max(int(max_pages), 1)]
            if not page_buckets:
                self._replay_tree_s_cap = None
                return False
            chosen = select_page_bucket(needed_pages, page_buckets)
            if chosen is None:
                self.tree_capacity_fallback_count += 1
                self._log_tree_fallback_once(
                    "capacity",
                    f"needed_pages={needed_pages} page_buckets={page_buckets}",
                )
                self._replay_tree_s_cap = None
                return False
            self._replay_tree_s_cap = int(chosen)
            return True
        ok = tree_slot_graph_can_run_batch(
            slot_width=self.tree_slot_graph_width(),
            slot_gather_enabled=self._slot_gather_tree_attn(),
            is_target_verify=is_verify,
            is_tree_draft=is_draft,
            spec_info=spec_info,
            fallback_seq_lens=fallback if fallback is not None else [],
            draft_token_num_fallback=int(self.speculative_num_draft_tokens or 1),
            draft_num_steps=self.draft_num_steps,
            seq_lens=seq if seq is not None else [],
            buckets=buckets,
        )
        if not ok:
            self.tree_capacity_fallback_count += 1
            needed = tree_slot_graph_needed_kv(
                is_target_verify=is_verify,
                is_tree_draft=is_draft,
                spec_info=spec_info,
                fallback_seq_lens=fallback if fallback is not None else [],
                draft_token_num_fallback=int(self.speculative_num_draft_tokens or 1),
                draft_num_steps=self.draft_num_steps,
                seq_lens=seq if seq is not None else [],
            )
            self._log_tree_fallback_once(
                "capacity",
                f"needed_kv={needed} buckets={buckets}",
            )
            self._replay_tree_s_cap = None
            return False
        if buckets and (is_verify or is_draft):
            needed = tree_slot_graph_needed_kv(
                is_target_verify=is_verify,
                is_tree_draft=is_draft,
                spec_info=spec_info,
                fallback_seq_lens=fallback if fallback is not None else [],
                draft_token_num_fallback=int(self.speculative_num_draft_tokens or 1),
                draft_num_steps=self.draft_num_steps,
                seq_lens=seq if seq is not None else [],
            )
            self._replay_tree_s_cap = select_tree_kv_bucket(needed or 0, buckets)
            logger.debug(
                "tree attention bucket hit kind=%s bucket=%s needed=%s",
                "verify" if is_verify else "draft",
                self._replay_tree_s_cap,
                needed,
            )
        return True

    def _slot_gather_kv_bound(self, kv_lens) -> Optional[int]:
        """Columns worth visiting in the slot-gather loop, or None for all of them.

        Outside a device graph the all-padding tail chunks can be skipped.
        Inside one the trip count must stay capture-time constant, so the full
        slot-table width has to drive the loop.
        """
        if self.graph_mode or not isinstance(kv_lens, torch.Tensor):
            return None
        if kv_lens.numel() == 0:
            return None
        return int(kv_lens.max().item())

    def _copy_into_graph_slot_buffers(
        self,
        dest_slots: torch.Tensor,
        dest_lens: torch.Tensor,
        slots: torch.Tensor,
        lens: torch.Tensor,
    ) -> None:
        need = int(lens.reshape(-1).max().item()) if lens.numel() else 0
        if need > dest_slots.shape[1]:
            raise NpuGraphPreparationError(
                f"tree slot-gather kv_len {need} exceeds graph max_kv "
                f"{int(dest_slots.shape[1])}",
                scope="graph",
            )
        if slots.shape[0] > dest_slots.shape[0]:
            raise NpuGraphPreparationError(
                f"tree slot-gather rows {slots.shape[0]} exceed graph buffer "
                f"{int(dest_slots.shape[0])}",
                scope="graph",
            )
        dest_slots.fill_(0)
        dest_lens.fill_(0)
        n_rows = slots.shape[0]
        n_cols = min(int(slots.shape[1]), int(dest_slots.shape[1]))
        if n_rows and n_cols:
            dest_slots[:n_rows, :n_cols].copy_(slots[:, :n_cols])
        if n_rows:
            dest_lens[:n_rows].copy_(lens.reshape(-1)[:n_rows].to(dtype=dest_lens.dtype))

    def _bind_graph_draft_slot_views(self, metadata: ForwardMetadata, bs: int, rows: int) -> None:
        if self.cuda_graph_kv_slots is None:
            return
        rows = min(int(rows), int(self.cuda_graph_kv_slots.shape[0]))
        cols = self._tree_slot_bind_cols()
        slots = self.cuda_graph_kv_slots[:rows, :cols]
        lens = self.cuda_graph_kv_lens[:rows]
        slots.fill_(0)
        lens.fill_(0)
        swa = None
        if self.is_hybrid_swa and self.cuda_graph_kv_slots_swa is not None:
            swa = self.cuda_graph_kv_slots_swa[:rows, :cols]
            swa.fill_(0)
        metadata.tree_draft_kv_slots = slots
        metadata.tree_draft_kv_lens_t = lens
        metadata.tree_draft_kv_slots_swa = swa
        self._cuda_graph_draft_slot_views[int(bs)] = (slots, lens, swa)

    def _bind_graph_verify_slot_views(self, metadata: ForwardMetadata, bs: int, rows: int) -> None:
        if self.cuda_graph_kv_slots is None:
            return
        rows = min(int(rows), int(self.cuda_graph_kv_slots.shape[0]))
        cols = self._tree_slot_bind_cols()
        slots = self.cuda_graph_kv_slots[:rows, :cols]
        lens = self.cuda_graph_kv_lens[:rows]
        slots.fill_(0)
        lens.fill_(0)
        metadata.tree_verify_kv_slots = slots
        metadata.tree_verify_kv_lens_t = lens
        self._cuda_graph_verify_slot_views[int(bs)] = (slots, lens)

    def _restore_graph_draft_slot_views(self, metadata: ForwardMetadata, bs: int) -> None:
        views = self._cuda_graph_draft_slot_views.get(int(bs))
        if views is None or self.cuda_graph_kv_slots is None:
            return
        rows = int(views[0].shape[0])
        cols = self._tree_slot_bind_cols()
        slots = self.cuda_graph_kv_slots[:rows, :cols]
        lens = self.cuda_graph_kv_lens[:rows]
        swa = None
        if self.is_hybrid_swa and self.cuda_graph_kv_slots_swa is not None:
            swa = self.cuda_graph_kv_slots_swa[:rows, :cols]
        metadata.tree_draft_kv_slots = slots
        metadata.tree_draft_kv_lens_t = lens
        metadata.tree_draft_kv_slots_swa = swa

    def _restore_graph_verify_slot_views(self, metadata: ForwardMetadata, bs: int) -> None:
        views = self._cuda_graph_verify_slot_views.get(int(bs))
        if views is None or self.cuda_graph_kv_slots is None:
            return
        rows = int(views[0].shape[0])
        cols = self._tree_slot_bind_cols()
        slots = self.cuda_graph_kv_slots[:rows, :cols]
        lens = self.cuda_graph_kv_lens[:rows]
        metadata.tree_verify_kv_slots = slots
        metadata.tree_verify_kv_lens_t = lens

    def _sync_active_tree_s_cap(self) -> None:
        if self._use_tree_shared_prefix() or self._use_target_tree_paged_fia():
            self._active_tree_s_cap = (
                getattr(self, "_shared_capture_width", None) or self._replay_tree_s_cap
            )
            return
        runner = getattr(self.model_runner, "graph_runner", None)
        extra = getattr(runner, "_active_capture_extra", None)
        if extra is None:
            extra = getattr(self, "_replay_tree_s_cap", None)
        self._active_tree_s_cap = extra

    def _store_tree_fia_kv_lens_cpu(self, lens, capture_rows: int) -> None:
        self.tree_fia_kv_lens_cpu = tree_fia_actual_seq_lengths_kv(lens, capture_rows)
        if self.forward_metadata is not None:
            self.forward_metadata.tree_fia_kv_lens_cpu = self.tree_fia_kv_lens_cpu

    def _ensure_tree_kv_scratch(
        self,
        rows: int,
        s_cap: int,
        n_kv: int,
        dk: int,
        dv: int,
        dtype,
        device,
    ):
        alloc_r = max(int(rows), int(self._tree_scratch_max_rows or 0), 1)
        key = (int(n_kv), int(dk), int(dv), dtype, str(device), int(s_cap))
        pair = self._tree_kv_scratch.get(key)
        if pair is None or pair[0].shape[0] < alloc_r:
            if self.graph_mode and pair is not None:
                raise NpuGraphPreparationError(
                    "tree compact FIA scratch grew after graph capture",
                    scope="graph",
                )
            k_buf = torch.zeros(
                (alloc_r, int(s_cap), int(n_kv), int(dk)),
                dtype=dtype,
                device=device,
            )
            v_buf = torch.zeros(
                (alloc_r, int(s_cap), int(n_kv), int(dv)),
                dtype=dtype,
                device=device,
            )
            self._tree_kv_scratch[key] = (k_buf, v_buf)
            pair = (k_buf, v_buf)
        return pair[0][:rows], pair[1][:rows]

    def _run_tree_compact_fia(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        kv_slots: torch.Tensor,
        kv_lens,
        scale: float,
        n_q_heads: int,
        n_kv_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
    ) -> torch.Tensor:
        query = q.reshape(-1, n_q_heads, qk_head_dim)
        rows = int(query.shape[0])
        if rows == 0:
            return query.new_zeros(0, n_q_heads * v_head_dim)
        kv_slots = kv_slots.reshape(rows, -1).to(dtype=torch.int64, device=query.device)
        s_cap = int(kv_slots.shape[1])
        n_kv_heads = max(int(n_kv_heads), 1)
        flat_k = flatten_paged_kv(k_cache, n_kv_heads, qk_head_dim)
        flat_v = flatten_paged_kv(v_cache, n_kv_heads, v_head_dim)
        k_s, v_s = self._ensure_tree_kv_scratch(
            rows,
            s_cap,
            n_kv_heads,
            qk_head_dim,
            v_head_dim,
            query.dtype,
            query.device,
        )
        gather_kv_into(flat_k, flat_v, kv_slots.clamp(min=0), k_s, v_s)
        kv_len_t = kv_lens if isinstance(kv_lens, torch.Tensor) else torch.as_tensor(
            list(kv_lens), dtype=torch.int32, device=query.device
        )
        zero_gathered_kv_padding(k_s, v_s, kv_len_t)
        q_fia = query.reshape(rows, 1, n_q_heads, qk_head_dim)
        if not q_fia.is_contiguous():
            q_fia = q_fia.contiguous()
        actual = getattr(self.forward_metadata, "tree_fia_kv_lens_cpu", None)
        if actual is None:
            actual = self.tree_fia_kv_lens_cpu
        if actual is None or len(actual) != rows:
            if self.graph_mode:
                # Capture/replay must not D2H; replay fill updates CPU lens + graph.update.
                actual = [1] * rows
            else:
                actual = tree_fia_actual_seq_lengths_kv(kv_len_t, rows)
        fia_kwargs = dict(
            num_heads=n_q_heads,
            num_key_value_heads=n_kv_heads,
            input_layout="BSND",
            atten_mask=None,
            sparse_mode=0,
            actual_seq_lengths_kv=actual,
            scale=float(scale),
        )
        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            q_fia,
            k_s,
            v_s,
            **fia_kwargs,
        )
        attn_output = torch.empty(
            (rows, 1, n_q_heads, v_head_dim),
            dtype=query.dtype,
            device=query.device,
        )
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        torch_npu.npu_fused_infer_attention_score.out(
            q_fia,
            k_s,
            v_s,
            **fia_kwargs,
            workspace=workspace,
            out=[attn_output, softmax_lse],
        )
        empty = kv_len_t.reshape(-1).to(device=attn_output.device) <= 0
        if int(empty.numel()) == rows:
            attn_output = attn_output.masked_fill(empty.view(rows, 1, 1, 1), 0)
        return attn_output.reshape(rows, n_q_heads * v_head_dim)

    def _graph_out_cache_loc(self, num_tokens: int):
        runner = getattr(self.model_runner, "graph_runner", None)
        buffers = getattr(runner, "buffers", None)
        loc = getattr(buffers, "out_cache_loc", None) if buffers is not None else None
        if loc is None:
            return None
        return loc[: int(num_tokens)]

    def _fill_tree_verify_kv_slots(self, forward_batch: ForwardBatch) -> None:
        spec_info = getattr(forward_batch, "spec_info", None)
        custom_mask = getattr(spec_info, "custom_mask", None)
        if not use_tree_verify_fallback(
            forward_batch.forward_mode.is_target_verify(),
            self.verify_tree_topk,
            custom_mask,
        ):
            return
        num_draft = int(
            getattr(spec_info, "draft_token_num", None)
            or self.speculative_num_draft_tokens
            or 1
        )
        seq_lens, raw_bs = self._tree_verify_mask_layout(
            spec_info, forward_batch.seq_lens
        )
        if isinstance(seq_lens, torch.Tensor) and seq_lens.device.type != "cpu":
            seq_list, raw_bs, _ = resolve_tree_verify_mask_seq_lens(
                spec_info, forward_batch.seq_lens
            )
        else:
            seq_list = seq_lens
        max_kv = None
        dest = getattr(self.forward_metadata, "tree_verify_kv_slots", None)
        dest_lens = getattr(self.forward_metadata, "tree_verify_kv_lens_t", None)
        if dest is not None and dest_lens is not None:
            max_kv = int(dest.shape[1])
        expected = full_mask_numel(seq_list, num_draft)
        graph_rows = int(dest.shape[0]) if dest is not None else raw_bs * num_draft
        padded_bs = 0
        fb_seq = getattr(forward_batch, "seq_lens", None)
        if isinstance(fb_seq, torch.Tensor):
            padded_bs = int(fb_seq.numel())
        elif fb_seq is not None:
            padded_bs = len(fb_seq)
        log_tree_verify_kv_slot_layout_once(
            mask_numel=int(custom_mask.numel()),
            expected_numel=int(expected),
            bs=padded_bs,
            raw_bs=raw_bs,
            num_draft=num_draft,
            seq_lens_sum=int(sum(int(x) for x in seq_list)),
            graph_rows=graph_rows,
            max_kv=int(max_kv) if max_kv is not None else 0,
        )
        loc = forward_batch.out_cache_loc
        need_loc = int(raw_bs) * num_draft
        if loc is not None and int(loc.numel()) > need_loc:
            loc = loc[:need_loc]
        req_pool = forward_batch.req_pool_indices
        if req_pool is not None and int(req_pool.numel()) > raw_bs:
            req_pool = req_pool[:raw_bs]
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        try:
            if dest is not None and dest_lens is not None:
                workspace = self.cuda_graph_verify_workspace
                if workspace is None:
                    workspace = dest.new_zeros((dest.shape[0], dest.shape[1] + 1))
                fill_tree_verify_kv_slots_(
                    custom_mask,
                    seq_list,
                    req_to_token,
                    req_pool,
                    loc,
                    num_draft,
                    dest,
                    dest_lens,
                    workspace,
                    max_kv=max_kv,
                    rows_limit=need_loc,
                )
                self._store_tree_fia_kv_lens_cpu(dest_lens, int(dest_lens.shape[0]))
                return
            slots, lens = build_tree_verify_kv_slots(
                custom_mask,
                seq_list,
                req_to_token,
                req_pool,
                loc,
                num_draft,
                max_kv=max_kv,
                rows_limit=need_loc,
            )
        except (RuntimeError, ValueError) as e:
            if dest is not None:
                raise NpuGraphPreparationError(str(e), scope="graph") from e
            raise
        self.forward_metadata.tree_verify_kv_slots = slots
        self.forward_metadata.tree_verify_kv_lens_t = lens
        self._store_tree_fia_kv_lens_cpu(lens, int(lens.shape[0]))

    def _run_tree_verify_slot_gather(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        custom_mask,
        *,
        qk_head_dim: int,
        v_head_dim: int,
        q_rope: Optional[torch.Tensor] = None,
        k_rope_cache: Optional[torch.Tensor] = None,
        rope_head_dim: Optional[int] = None,
    ) -> torch.Tensor:
        if (
            self._use_tree_shared_prefix()
            and self.forward_metadata.tree_shared is not None
        ):
            return self._run_tree_shared_prefix_attention(q, k_cache, v_cache, layer)
        log_tree_verify_fallback_once(self.verify_tree_topk)
        num_draft = int(
            getattr(forward_batch.spec_info, "draft_token_num", None)
            or self.speculative_num_draft_tokens
            or 1
        )
        if self._use_tree_compact_fia(q_rope):
            return self._run_tree_compact_fia(
                q,
                k_cache,
                v_cache,
                kv_slots=self.forward_metadata.tree_verify_kv_slots,
                kv_lens=self.forward_metadata.tree_verify_kv_lens_t,
                scale=layer.scaling,
                n_q_heads=layer.tp_q_head_num,
                n_kv_heads=layer.tp_k_head_num,
                qk_head_dim=qk_head_dim,
                v_head_dim=v_head_dim,
            )
        self.tree_layout_fallback_count += 1
        self._log_tree_fallback_once(
            "layout",
            f"verify use_mla={self.use_mla} has_rope={q_rope is not None}",
        )
        return tree_verify_attention(
            q,
            k_cache,
            v_cache,
            custom_mask=custom_mask,
            seq_lens=forward_batch.seq_lens,
            req_to_token=forward_batch.req_to_token_pool.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices,
            out_cache_loc=forward_batch.out_cache_loc,
            num_draft=num_draft,
            scale=layer.scaling,
            n_q_heads=layer.tp_q_head_num,
            n_kv_heads=layer.tp_k_head_num,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            q_rope=q_rope,
            k_rope_cache=k_rope_cache,
            rope_head_dim=rope_head_dim,
            kv_slots=self.forward_metadata.tree_verify_kv_slots,
            kv_lens=self.forward_metadata.tree_verify_kv_lens_t,
            kv_bound=self._slot_gather_kv_bound(
                self.forward_metadata.tree_verify_kv_lens_t
            ),
        )

    def _run_tree_draft_slot_gather(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer: RadixAttention,
        *,
        qk_head_dim: int,
        v_head_dim: int,
        q_rope: Optional[torch.Tensor] = None,
        k_rope_cache: Optional[torch.Tensor] = None,
        rope_head_dim: Optional[int] = None,
    ) -> torch.Tensor:
        if (
            self._use_tree_shared_prefix()
            and self.forward_metadata.tree_shared is not None
        ):
            return self._run_tree_shared_prefix_attention(q, k_cache, v_cache, layer)
        slots = self.forward_metadata.tree_draft_kv_slots
        if slots is None:
            raise RuntimeError(
                "tree draft slot-gather requires tree_draft_kv_slots; "
                "init_forward_metadata did not fill them"
            )
        if (
            self.is_hybrid_swa
            and getattr(layer, "sliding_window_size", -1) not in (-1, None)
            and int(layer.sliding_window_size) != -1
        ):
            swa = self.forward_metadata.tree_draft_kv_slots_swa
            if swa is None:
                raise RuntimeError(
                    "tree draft SWA slot-gather requires tree_draft_kv_slots_swa"
                )
            slots = swa
        log_tree_draft_slot_gather_once(self.draft_topk, self.page_size)
        if self._use_tree_compact_fia(q_rope):
            return self._run_tree_compact_fia(
                q,
                k_cache,
                v_cache,
                kv_slots=slots,
                kv_lens=self.forward_metadata.tree_draft_kv_lens_t,
                scale=layer.scaling,
                n_q_heads=layer.tp_q_head_num,
                n_kv_heads=layer.tp_k_head_num,
                qk_head_dim=qk_head_dim,
                v_head_dim=v_head_dim,
            )
        self.tree_layout_fallback_count += 1
        self._log_tree_fallback_once(
            "layout",
            f"draft use_mla={self.use_mla} has_rope={q_rope is not None}",
        )
        return tree_draft_attention(
            q,
            k_cache,
            v_cache,
            kv_slots=slots,
            kv_lens=self.forward_metadata.tree_draft_kv_lens_t,
            scale=layer.scaling,
            n_q_heads=layer.tp_q_head_num,
            n_kv_heads=layer.tp_k_head_num,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            q_rope=q_rope,
            k_rope_cache=k_rope_cache,
            rope_head_dim=rope_head_dim,
            kv_bound=self._slot_gather_kv_bound(
                self.forward_metadata.tree_draft_kv_lens_t
            ),
        )

    def _tree_draft_kv_lens(self, seq_lens, num_q: int):
        if self.draft_topk > 1:
            return normalize_tree_draft_kv_lens(seq_lens, num_q, self.draft_topk)
        return expand_seq_lens_for_spec_topk(seq_lens, num_q)

    def _tree_draft_table_rows(self, num_seqs: int, num_tokens: Optional[int] = None) -> int:
        if self.draft_topk > 1:
            if num_tokens is not None:
                return max(int(num_seqs), int(num_tokens))
            return int(num_seqs) * self.draft_topk
        return int(num_seqs)

    def _build_tree_draft_block_tables(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        max_pages: Optional[int] = None,
        index_mapping: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return build_tree_draft_block_tables(
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            page_size=self.page_size,
            topk=self.draft_topk,
            step_id=self.speculative_step_id,
            num_steps=self.draft_num_steps,
            max_pages=max_pages,
            index_mapping=index_mapping,
        )

    def _copy_tree_draft_block_tables(
        self,
        dest: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        index_mapping: Optional[torch.Tensor] = None,
    ) -> None:
        tables = self._build_tree_draft_block_tables(
            req_pool_indices,
            seq_lens,
            max_pages=dest.shape[1],
            index_mapping=index_mapping,
        )
        if tables.shape[0] > dest.shape[0]:
            raise RuntimeError(
                "tree draft block_tables need "
                f"{tables.shape[0]} rows, graph buffer has {dest.shape[0]}"
            )
        n_rows = tables.shape[0]
        dest[:n_rows].copy_(tables)
        dest[n_rows:].fill_(0)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        self.forward_metadata = ForwardMetadata()
        if (
            self._use_target_tree_paged_fia()
            and forward_batch.forward_mode.is_target_verify()
            and self.verify_tree_topk > 1
        ):
            self._prepare_target_tree_fia_eager(forward_batch)
            self.graph_mode = False
            return
        if self._use_tree_shared_prefix() and (
            self._is_tree_draft(forward_batch)
            or (
                forward_batch.forward_mode.is_target_verify()
                and self.verify_tree_topk > 1
            )
        ):
            if self._prepare_shared_eager(forward_batch):
                self.graph_mode = False
                return
        is_sr_tail = getattr(forward_batch, "is_sr_tail_extend", False)
        self.sr_tail_attention_paths = set()
        seq_lens_max = (
            max(forward_batch.seq_lens_cpu.tolist(), default=0)
            if is_sr_tail
            else forward_batch.seq_lens.max()
        )
        if forward_batch.forward_mode.is_target_verify():
            seq_lens_max += self.speculative_num_draft_tokens
        elif (
            forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.spec_info is not None
        ):
            seq_lens_max += self.speculative_step_id + 1
        use_tree_draft_tables = (
            self.draft_topk > 1
            and forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.spec_info is not None
            and not self._paged_impl_selected()
        )
        if self._paged_impl_selected() and self._sr_tree_paged_meta is not None:
            self.forward_metadata.sr_tree_paged = self._sr_tree_paged_meta
            self.forward_metadata.block_tables = self._sr_tree_paged_meta.block_tables
        elif use_tree_draft_tables:
            self.forward_metadata.block_tables = self._build_tree_draft_block_tables(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
            )
        elif is_sr_tail:
            self.forward_metadata.block_tables = (
                forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, :seq_lens_max:self.page_size
                ] // self.page_size
            ).to(torch.int32).contiguous()
            self.forward_metadata.sr_tail = build_tail_attention_metadata(
                forward_batch.extend_prefix_lens_cpu,
                forward_batch.extend_seq_lens_cpu,
                self.forward_metadata.block_tables,
            )
        else:
            self.forward_metadata.block_tables = (
                forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, :seq_lens_max
                ][:, :: self.page_size]
                // self.page_size
            )
        if self.is_hybrid_swa:
            if use_tree_draft_tables:
                self.forward_metadata.block_tables_swa = (
                    self._build_tree_draft_block_tables(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        index_mapping=self.full_to_swa_index_mapping,
                    )
                    .to(torch.int32)
                    .contiguous()
                )
            else:
                self.forward_metadata.block_tables_swa = (
                    (
                        self.full_to_swa_index_mapping[
                            forward_batch.req_to_token_pool.req_to_token[
                                forward_batch.req_pool_indices, :seq_lens_max
                            ]
                        ][:, :: self.page_size]
                        // self.page_size
                    )
                    .to(torch.int32)
                    .contiguous()
                )
        if self._use_tree_draft_slot_gather(forward_batch) and not (
            self._paged_impl_selected() and self._sr_tree_paged_meta is not None
        ):
            self._fill_tree_draft_kv_slots(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
            )
        if forward_batch.extend_seq_lens is not None:
            self.forward_metadata.extend_seq_lens = forward_batch.extend_seq_lens
            self.forward_metadata.extend_seq_lens_cpu_int = (
                torch.tensor(forward_batch.extend_seq_lens_cpu, dtype=torch.int32)
                if is_sr_tail else forward_batch.extend_seq_lens.cpu().int()
            )
        if forward_batch.seq_lens is not None:
            self.forward_metadata.seq_lens = forward_batch.seq_lens.int()
        else:
            self.forward_metadata.seq_lens = forward_batch.seq_lens_cpu.to(
                self.device
            ).int()

        self.forward_metadata.seq_lens_cpu_int = forward_batch.seq_lens_cpu.int()
        if (
            not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_draft_extend()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            seq_lens_list_cumsum = np.cumsum(forward_batch.extend_seq_lens_cpu)
            self.forward_metadata.seq_lens_list_cumsum = seq_lens_list_cumsum

        if forward_batch.forward_mode.is_target_verify():
            self.forward_metadata.seq_lens_cpu_int += self.speculative_num_draft_tokens
            self._fill_tree_verify_mask(forward_batch)
            self._fill_tree_verify_kv_slots(forward_batch)
        elif (
            forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.spec_info is not None
        ):
            self.forward_metadata.seq_lens_cpu_int += self.speculative_step_id + 1

        if (
            self.use_mla
            and forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
            and not forward_batch.forward_mode.is_target_verify()
            and sum(forward_batch.extend_prefix_lens_cpu) > 0
        ):
            self.forward_metadata.prefix_lens = forward_batch.extend_prefix_lens.to(
                "cpu"
            )
            seq_prefix_lens = self.forward_metadata.prefix_lens.tolist()
            self.forward_metadata.flatten_prefix_block_tables = torch.empty(
                0, dtype=torch.int32
            ).to(self.device)
            for req_idx, seq_len in zip(
                forward_batch.req_pool_indices.tolist(), seq_prefix_lens
            ):
                req_indices = forward_batch.req_to_token_pool.req_to_token[req_idx]
                req_prefix_block_tables = (
                    req_indices[:seq_len][:: self.page_size] // self.page_size
                )
                self.forward_metadata.flatten_prefix_block_tables = torch.cat(
                    (
                        self.forward_metadata.flatten_prefix_block_tables,
                        torch.flatten(req_prefix_block_tables),
                    )
                )

        self.graph_mode = False

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        table_bs = (
            max(int(max_bs), int(max_num_tokens))
            if self.draft_topk > 1
            else int(max_bs)
        )
        self.graph_metadata = {
            "block_tables": torch.empty(
                (
                    table_bs,
                    (self.max_context_len + self.page_size - 1) // self.page_size,
                ),
                dtype=torch.int32,
                device=self.device,
            ),
        }
        max_q = self._graph_row_capacity(max_bs, max_num_tokens)
        mask_max_kv = self._slot_gather_graph_max_kv()
        slot_max_kv = self._slot_gather_graph_slot_max_kv()
        self.tree_kv_buckets = parse_tree_graph_kv_buckets(
            os.environ.get(TREE_GRAPH_KV_BUCKETS_ENV),
            orig_max_kv=slot_max_kv,
        )
        if self.tree_kv_buckets:
            slot_max_kv = max(self.tree_kv_buckets)
        if self._paged_impl_selected() or self._use_target_tree_paged_fia():
            page = max(int(self.page_size), 1)
            max_pages = max((int(slot_max_kv) + page - 1) // page, 1)
            if self._paged_impl_selected():
                # One contiguous buffer per (rows, pages) in both maps.
                # Sharing a max-shaped tensor and slicing dest[:rows, :pages]
                # yields a strided view that ATB paged attention does not
                # accept (ACL 507011). active contents depend only on
                # rows/raw_bs, not pages, so sharing one active across page
                # buckets used to be semantically safe; the maps still use
                # the same (rows, pages) key so an operator that later
                # reads active does not have to re-derive that invariant.
                self.cuda_graph_paged_tables = {}
                self.cuda_graph_paged_actives = {}
                self._paged_graph_max_pages = max_pages
            self._tree_scratch_max_rows = 0
            self._tree_scratch_max_cols = 0
        else:
            self._tree_scratch_max_rows = max_q
            self._tree_scratch_max_cols = slot_max_kv
        self.cuda_graph_custom_mask = torch.empty(
            (max_q * mask_max_kv,), dtype=torch.bool, device=self.device
        )
        if self._slot_gather_tree_attn():
            self.cuda_graph_tree_attn_mask = None
        else:
            self.cuda_graph_tree_attn_mask = torch.ones(
                (max_q, mask_max_kv), dtype=torch.bool, device=self.device
            )
        self.cuda_graph_verify_positions = torch.empty(
            (max_q,), dtype=torch.int64, device=self.device
        )
        skip_slots = self._use_tree_shared_prefix() or self._use_target_tree_paged_fia()
        self.cuda_graph_kv_slots = (
            None
            if skip_slots
            else torch.zeros(
                (max_q, slot_max_kv), dtype=torch.int64, device=self.device
            )
        )
        self.cuda_graph_kv_lens = (
            None
            if skip_slots
            else torch.zeros((max_q,), dtype=torch.int32, device=self.device)
        )
        self.cuda_graph_verify_workspace = (
            None
            if skip_slots
            else torch.zeros(
                (max_q, slot_max_kv + 1), dtype=torch.int64, device=self.device
            )
        )
        logger.info(
            "tree slot-gather graph buffers: max_q=%s slot_max_kv=%s "
            "mask_max_kv=%s kv_pool=%s context_len=%s s_cap=%s buckets=%s "
            "slots=%.1f MiB chunk_width=%s",
            max_q,
            slot_max_kv,
            mask_max_kv,
            self._slot_gather_kv_pool_size(),
            int(self.max_context_len),
            int(self.tree_graph_max_kv),
            self.tree_kv_buckets,
            max_q * slot_max_kv * 8 / (1024**2),
            tree_attn_chunk_width(slot_max_kv),
        )
        if self.is_hybrid_swa:
            self.cuda_graph_kv_slots_swa = torch.zeros(
                (max_q, slot_max_kv), dtype=torch.int64, device=self.device
            )
            self.graph_metadata["block_tables_swa"] = torch.empty(
                (
                    table_bs,
                    (self.max_context_len + self.page_size - 1) // self.page_size,
                ),
                dtype=torch.int32,
                device=self.device,
            )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        self._sync_active_tree_s_cap()
        metadata = ForwardMetadata()

        draft = (
            self.draft_topk > 1
            and spec_info is not None
            and not forward_mode.is_target_verify()
        )
        if self._use_target_tree_paged_fia() and (
            forward_mode.is_target_verify() and self.verify_tree_topk > 1
        ):
            queries = num_tokens // bs
            s_cap = int(
                getattr(self, "_shared_capture_width", None)
                or self._active_tree_s_cap
                or max(self.tree_kv_buckets, default=self.tree_graph_max_kv)
            )
            pages = pages_for_s_cap(s_cap, self.page_size)
            metadata.sr_target_tree_fia = self._target_fia_metadata(
                bs, queries, pages, graph=True
            )
            prime_target_tree_fia_capture_(
                metadata.sr_target_tree_fia,
                dummy_page=int(getattr(self, "_target_fia_dummy_page", 0)),
            )
            self.graph_metadata[bs] = metadata
            self.forward_metadata = metadata
            self.graph_mode = True
            return
        if self._use_tree_shared_prefix() and (
            draft or (forward_mode.is_target_verify() and self.verify_tree_topk > 1)
        ):
            queries = self.draft_topk if draft else num_tokens // bs
            width = int(self._active_tree_s_cap or max(self.tree_kv_buckets))
            metadata.tree_shared = self._shared_metadata(
                bs, queries, width, draft=draft, graph=True
            )
            metadata.tree_shared.clear()
            self.graph_metadata[bs] = metadata
            self.forward_metadata = metadata
            self.graph_mode = True
            return

        table_rows = self._tree_draft_table_rows(bs, num_tokens)
        metadata.block_tables = self.graph_metadata["block_tables"][:table_rows, :]
        if self.draft_topk > 1:
            metadata.block_tables.fill_(0)
        if self.is_dllm_model:
            max_len = int(seq_lens[:bs].max().item())
            max_seq_pages = (max_len + self.page_size - 1) // self.page_size
            metadata.block_tables[:bs, :max_seq_pages].copy_(
                (
                    self.req_to_token[req_pool_indices[:bs], :max_len][
                        :, :: self.page_size
                    ]
                    // self.page_size
                ).to(torch.int32)
            )
            metadata.block_tables[:bs, max_seq_pages:].fill_(0)
            metadata.block_tables[bs:, :].fill_(0)

        if self.is_hybrid_swa:
            metadata.block_tables_swa = self.graph_metadata["block_tables_swa"][
                :table_rows, :
            ]
            if self.draft_topk > 1:
                metadata.block_tables_swa.fill_(0)
        metadata.seq_lens_cpu_list = seq_lens.cpu().int().tolist()
        if num_tokens > bs:
            metadata.seq_lens_cpu_list = self._tree_draft_kv_lens(
                metadata.seq_lens_cpu_list, num_tokens
            )
        metadata.seq_lens = seq_lens
        if (
            forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
            or forward_mode.is_draft_extend()
        ):
            metadata.actual_seq_lengths_q = torch.arange(
                self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens
                + bs * self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=seq_lens.device,
            )
        else:
            metadata.actual_seq_lengths_q = torch.tensor(
                [1 + i * 1 for i in range(bs)],
                dtype=torch.int32,
                device=seq_lens.device,
            )
        if forward_mode.is_dllm_extend():
            extend_seq_lens_cpu_int = torch.tensor(
                [self.dllm_block_size for i in range(bs)],
                dtype=torch.int32,
                device=seq_lens.device,
            )
            metadata.seq_lens_list_cumsum = (
                torch.cumsum(extend_seq_lens_cpu_int, dim=0).int().tolist()
            )

        if self.draft_topk > 1 and spec_info is not None and not forward_mode.is_target_verify():
            if self._paged_impl_selected() and self._sr_tree_paged_meta is not None:
                metadata.sr_tree_paged = self._sr_tree_paged_meta
                metadata.block_tables = self._sr_tree_paged_meta.block_tables
            else:
                self._bind_graph_draft_slot_views(
                    metadata, bs, self._tree_draft_table_rows(bs, num_tokens)
                )
                dest_lens = getattr(metadata, "tree_draft_kv_lens_t", None)
                if dest_lens is not None:
                    self.forward_metadata = metadata
                    n_rows = int(dest_lens.shape[0])
                    self._store_tree_fia_kv_lens_cpu([0] * n_rows, n_rows)
        elif forward_mode.is_target_verify() and self.verify_tree_topk > 1:
            self._bind_graph_verify_slot_views(metadata, bs, num_tokens)
            dest_lens = getattr(metadata, "tree_verify_kv_lens_t", None)
            if dest_lens is not None:
                self.forward_metadata = metadata
                n_rows = int(dest_lens.shape[0])
                self._store_tree_fia_kv_lens_cpu([0] * n_rows, n_rows)

        if (
            self.q_head_num_padding is not None
            and self.q_head_num_padding > self.tp_q_head_num
        ):
            # In the MLA architecture, the FIA kernel requires the head count to be a power of 2.
            # Therefore, we pad the head dimension accordingly and initialize an empty tensor for padding.
            metadata.nope_padding = torch.empty(
                [
                    bs,
                    1,
                    self.q_head_num_padding - self.tp_q_head_num,
                    self.kv_lora_rank,
                ],
                dtype=(
                    self.model_dtype if self.model_dtype is not None else torch.bfloat16
                ),
                device=seq_lens.device,
            )
            metadata.rope_padding = torch.empty(
                [
                    bs,
                    1,
                    self.q_head_num_padding - self.tp_q_head_num,
                    self.qk_rope_head_dim,
                ],
                dtype=(
                    self.model_dtype if self.model_dtype is not None else torch.bfloat16
                ),
                device=seq_lens.device,
            )

        self.graph_metadata[bs] = metadata
        self.forward_metadata = metadata

        self.graph_mode = True

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        self._sync_active_tree_s_cap()
        metadata = self.graph_metadata[bs]
        draft = (
            self.draft_topk > 1
            and spec_info is not None
            and forward_mode.is_decode_or_idle()
        )
        if self._use_target_tree_paged_fia() and (
            forward_mode.is_target_verify() and self.verify_tree_topk > 1
        ):
            queries = int(spec_info.draft_token_num)
            s_cap = int(self._replay_tree_s_cap or self._active_tree_s_cap)
            pages = pages_for_s_cap(s_cap, self.page_size)
            key = (int(bs), int(queries), int(pages))
            cache = getattr(self, "_target_fia_graph_metadata", {})
            if key not in cache:
                raise NpuGraphPreparationError(
                    "missing captured target tree FIA buffers", scope="graph"
                )
            md = cache[key]
            lengths, raw_bs = self._tree_verify_mask_layout(spec_info, seq_lens_cpu)
            prefixes = cpu_prefix_lengths(lengths, raw_bs)
            fill_target_tree_fia_metadata_(
                md,
                self.req_to_token,
                req_pool_indices,
                spec_info.custom_mask,
                prefixes,
                queries,
                raw_bs,
                dummy_page=int(getattr(self, "_target_fia_dummy_page", 0)),
            )
            metadata.sr_target_tree_fia = md
            self.forward_metadata = metadata
            self.graph_mode = True
            return
        if self._use_tree_shared_prefix() and (
            draft or (forward_mode.is_target_verify() and self.verify_tree_topk > 1)
        ):
            queries = self.draft_topk if draft else int(spec_info.draft_token_num)
            width = int(self._replay_tree_s_cap or self._active_tree_s_cap)
            key = (bs, queries, width, draft)
            if key not in self._shared_graph_metadata:
                raise NpuGraphPreparationError(
                    "missing captured shared-prefix buffers", scope="graph"
                )
            md = self._shared_graph_metadata[key]
            metadata.tree_shared = md
            self.forward_metadata = metadata
            self.graph_mode = True
            if draft:
                # Filled centrally with raw_bs, never the padded graph batch size.
                return
            lengths, raw_bs = self._tree_verify_mask_layout(spec_info, seq_lens_cpu)
            lengths = cpu_prefix_lengths(lengths, raw_bs)
            loc = self._graph_out_cache_loc(raw_bs * queries)
            if loc is None:
                raise NpuGraphPreparationError(
                    "shared-prefix verify missing out_cache_loc", scope="graph"
                )
            fill_shared_verify_(
                md,
                self.req_to_token,
                req_pool_indices,
                lengths,
                loc,
                spec_info.custom_mask,
                queries,
            )
            return
        replay_seq_lens = seq_lens_cpu[:bs] if seq_lens_cpu is not None else seq_lens[:bs]
        if self._paged_impl_selected() and self._sr_tree_paged_meta is not None:
            metadata.sr_tree_paged = self._sr_tree_paged_meta
            metadata.block_tables = self._sr_tree_paged_meta.block_tables
            if forward_mode.is_target_verify():
                seq_lens = seq_lens + self.speculative_num_draft_tokens
            elif forward_mode.is_decode_or_idle() and spec_info is not None:
                seq_lens = seq_lens + self.speculative_step_offset_npu
            metadata.seq_lens[:bs].copy_(seq_lens[:bs])
            self.forward_metadata = metadata
            self.graph_mode = True
            return
        use_tree_draft_slots = (
            self.draft_topk > 1
            and self.page_size > 1
            and forward_mode.is_decode_or_idle()
            and spec_info is not None
        )
        skip_block_tables = use_tree_draft_slots and self._use_tree_compact_fia()

        if not skip_block_tables:
            max_len = seq_lens_cpu[:bs].max().item()
            if forward_mode.is_target_verify():
                max_len += self.speculative_num_draft_tokens
            elif forward_mode.is_decode_or_idle() and spec_info is not None:
                max_len += self.speculative_step_id + 1
            max_seq_pages = (max_len + self.page_size - 1) // self.page_size

            if self.draft_topk > 1 and forward_mode.is_decode_or_idle() and spec_info is not None:
                if self.is_hybrid_swa:
                    self._copy_tree_draft_block_tables(
                        metadata.block_tables_swa,
                        req_pool_indices[:bs],
                        replay_seq_lens,
                        index_mapping=self.full_to_swa_index_mapping,
                    )
                self._copy_tree_draft_block_tables(
                    metadata.block_tables,
                    req_pool_indices[:bs],
                    replay_seq_lens,
                )
            else:
                if self.is_hybrid_swa:
                    metadata.block_tables_swa[:bs, :max_seq_pages].copy_(
                        self.full_to_swa_index_mapping[
                            self.req_to_token[req_pool_indices[:bs], :max_len]
                        ][:, :: self.page_size]
                        // self.page_size
                    )
                    metadata.block_tables_swa[:bs, max_seq_pages:].fill_(0)
                    metadata.block_tables_swa[bs:, :].fill_(0)
                metadata.block_tables[:bs, :max_seq_pages].copy_(
                    self.req_to_token[req_pool_indices[:bs], :max_len][:, :: self.page_size]
                    // self.page_size
                )

                metadata.block_tables[:bs, max_seq_pages:].fill_(0)
                metadata.block_tables[bs:, :].fill_(0)

        orig_seq_lens = seq_lens[:bs]
        if forward_mode.is_target_verify():
            seq_lens = seq_lens + self.speculative_num_draft_tokens
        elif forward_mode.is_decode_or_idle() and spec_info is not None:
            seq_lens = seq_lens + self.speculative_step_offset_npu
        metadata.seq_lens[:bs].copy_(seq_lens[:bs])

        self.forward_metadata = metadata
        self.graph_mode = True

        if use_tree_draft_slots:
            self._restore_graph_draft_slot_views(metadata, bs)
            self.forward_metadata = metadata
            if not getattr(self, "_central_tree_draft_fill", False):
                self._fill_tree_draft_kv_slots(req_pool_indices[:bs], replay_seq_lens)

        if forward_mode.is_target_verify() and spec_info is not None:
            dummy_batch = type("ForwardBatchLite", (), {})()
            dummy_batch.spec_info = spec_info
            dummy_batch.forward_mode = forward_mode
            num_draft = int(
                getattr(spec_info, "draft_token_num", None)
                or self.speculative_num_draft_tokens
                or 1
            )
            seq_for_mask, raw_bs = self._tree_verify_mask_layout(
                spec_info, orig_seq_lens
            )
            dummy_batch.seq_lens = seq_for_mask
            dummy_batch.req_pool_indices = req_pool_indices[:raw_bs]
            dummy_batch.req_to_token_pool = type("PoolLite", (), {})()
            dummy_batch.req_to_token_pool.req_to_token = self.req_to_token
            dummy_batch.out_cache_loc = self._graph_out_cache_loc(
                int(raw_bs) * num_draft
            )
            self._fill_tree_verify_mask(dummy_batch)
            if self.verify_tree_topk > 1:
                self._restore_graph_verify_slot_views(metadata, bs)
                self.forward_metadata = metadata
                if dummy_batch.out_cache_loc is None:
                    raise NpuGraphPreparationError(
                        "tree verify graph replay missing out_cache_loc",
                        scope="graph",
                    )
                self._fill_tree_verify_kv_slots(dummy_batch)

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def _generate_alibi_bias(
        self,
        seq_len: int,
        slopes: torch.Tensor,
        num_heads: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        position_point = (
            torch.arange(seq_len).view(1, 1, -1).expand(num_heads, -1, -1).to(device)
        )
        alibi = slopes.view(-1, 1, 1) * position_point
        alibi_bias = alibi.view(num_heads, 1, seq_len).to(device).to(dtype)
        return alibi_bias

    def generate_alibi_bias(
        self,
        q_seq_len: int,
        kv_seq_len: int,
        slopes: torch.Tensor,
        num_heads: int,
        device: torch.device,
        is_extend: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        MAX_LEN_ALB = 5000
        max_seq_len = max(kv_seq_len, q_seq_len, MAX_LEN_ALB)
        if getattr(self, "alibi_bias", None) is None:
            self.alibi_bias = self._generate_alibi_bias(
                max_seq_len, slopes, num_heads, device, dtype
            )

        if getattr(self, "super_mask", None) is None:
            super_mask = torch.ones(size=(1, max_seq_len, max_seq_len), dtype=dtype)
            super_mask = super_mask.float().fill_(float("-inf")).type_as(super_mask)
            super_mask = torch.triu(super_mask, 1).to(device)
            self.super_mask = super_mask
        if is_extend:
            return (
                self.alibi_bias[:, :q_seq_len, :kv_seq_len]
                + self.super_mask[:, :q_seq_len, :kv_seq_len]
            )
        else:
            return self.alibi_bias[:, :q_seq_len, :kv_seq_len]

    def attn_alibi(
        self,
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        query_lens,
        scale_value,
        num_heads,
        slopes,
        is_extend,
    ):
        curr = 0
        num_prompts = query_lens.shape[0]
        head_size = k_cache.shape[3]
        head_size_v = v_cache.shape[3]
        block_size = k_cache.shape[1]
        attn_output = []
        for i in range(num_prompts):
            seq_len = seq_lens[i].item()
            block_table = block_tables[i]

            j = torch.arange(seq_len, device=block_table.device)

            block_number = block_table[j // block_size]
            block_offset = j % block_size

            k = k_cache[block_number, block_offset]
            v = v_cache[block_number, block_offset]
            k = k.view(seq_len, num_heads, head_size)
            v = v.view(seq_len, num_heads, head_size_v)

            if is_extend:
                q_len = query_lens[i].item()
                query = q[curr : curr + q_len]
            else:
                q_len = 1
                query = q[curr : curr + 1]

            query = query.to(torch.float32)
            query = query * scale_value
            query = query.permute(1, 0, 2)
            k = k.permute(1, 2, 0)

            score = torch.bmm(query, k)
            score = score.to(torch.float32)
            if slopes is not None:
                alibi_bias = self.generate_alibi_bias(
                    q_seq_len=q_len,
                    kv_seq_len=seq_len,
                    slopes=slopes,
                    num_heads=num_heads,
                    device=q.device,
                    is_extend=is_extend,
                    dtype=query.dtype,
                )
                score = score + alibi_bias
            score = torch.max(score, torch.tensor(torch.finfo(score.dtype).min))
            p = torch.nn.functional.softmax(score, dim=-1)
            v = v.permute(1, 0, 2)
            out = torch.bmm(p, v)
            out = out.permute(1, 0, 2)
            out = out.reshape(-1, num_heads * head_size_v)
            attn_output.append(out)
            curr += q_len
        attn_output = torch.cat(attn_output, dim=0).to(q.dtype).to(q.device)
        attn_output = attn_output.view(-1, num_heads * head_size)
        return attn_output

    def do_cp_balance_attn(
        self,
        q_nope,
        k_nope,
        q_pe,
        k_pe,
        topk_indices,
        layer,
        actual_seq_qlen,
        actual_seq_lengths_kv,
    ):
        seq_len = q_nope.shape[0]
        split_len = (seq_len + 1) // 2
        q_nope_prev, q_nope_next = torch.split(q_nope, split_len, dim=0)
        q_rope_prev, q_rope_next = torch.split(q_pe, split_len, dim=0)
        q_nope_prev = q_nope_prev.contiguous()
        q_nope_next = q_nope_next.contiguous()
        q_rope_prev = q_rope_prev.contiguous()
        q_rope_next = q_rope_next.contiguous()
        topk_indices_prev, topk_indices_next = topk_indices

        actual_seq_qlen_prev, actual_seq_qlen_next = actual_seq_qlen
        actual_seq_lengths_kv_prev, actual_seq_lengths_kv_next = actual_seq_lengths_kv

        attn_out_prev, _, _ = torch_npu.npu_sparse_flash_attention(
            query=q_nope_prev,
            key=k_nope,
            value=k_nope,
            query_rope=q_rope_prev,
            key_rope=k_pe,
            sparse_indices=topk_indices_prev,
            scale_value=layer.scaling,
            actual_seq_lengths_query=actual_seq_qlen_prev.to(
                device=q_nope.device, dtype=torch.int32
            ),
            actual_seq_lengths_kv=actual_seq_lengths_kv_prev.to(
                device=q_nope.device, dtype=torch.int32
            ),
            block_table=self.forward_metadata.block_tables,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=False,
        )
        attn_out_next, _, _ = torch_npu.npu_sparse_flash_attention(
            query=q_nope_next,
            key=k_nope,
            value=k_nope,
            query_rope=q_rope_next,
            key_rope=k_pe,
            sparse_indices=topk_indices_next,
            scale_value=layer.scaling,
            actual_seq_lengths_query=actual_seq_qlen_next.to(
                device=q_nope.device, dtype=torch.int32
            ),
            actual_seq_lengths_kv=actual_seq_lengths_kv_next.to(
                device=q_nope.device, dtype=torch.int32
            ),
            block_table=self.forward_metadata.block_tables,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=False,
        )
        return torch.cat([attn_out_prev, attn_out_next], dim=0)

    def forward_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi_head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: torch.Tensor = None,
    ):

        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_draft_extend()
            and not forward_batch.forward_mode.is_target_verify()
        )

        if save_kv_cache:
            k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
            k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, k_rope
            )
        q_nope, q_pe = q, q_rope
        k_nope, k_pe = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        if is_prefill:
            if self.forward_metadata.actual_seq_lengths_q is not None:
                actual_seq_qlen = self.forward_metadata.actual_seq_lengths_q
            else:
                actual_seq_qlen = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
        else:
            if self.forward_metadata.actual_seq_lengths_q is None:
                if (
                    forward_batch.forward_mode.is_draft_extend_v2()
                    or forward_batch.forward_mode.is_target_verify()
                ):
                    actual_seq_qlen = (
                        torch.arange(
                            self.speculative_num_draft_tokens,
                            self.speculative_num_draft_tokens + q.shape[0],
                            self.speculative_num_draft_tokens,
                            dtype=torch.int32,
                        )
                        .to(q.device)
                        .to(torch.int32)
                    )
                elif forward_batch.forward_mode.is_draft_extend():
                    actual_seq_qlen = (
                        forward_batch.extend_seq_lens.cumsum()
                        .to(q.device)
                        .to(torch.int32)
                    )
                else:
                    actual_seq_qlen = (
                        torch.arange(1, q.shape[0] + 1).to(q.device).to(torch.int32)
                    )
            else:
                actual_seq_qlen = self.forward_metadata.actual_seq_lengths_q

        if self.forward_metadata.actual_seq_lengths_kv is not None:
            actual_seq_lengths_kv = self.forward_metadata.actual_seq_lengths_kv
        elif self.forward_metadata.seq_lens_cpu_int is not None:
            actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_int
        else:
            actual_seq_lengths_kv = self.forward_metadata.seq_lens

        if (
            is_prefill
            and is_nsa_enable_prefill_cp()
            and forward_batch.nsa_cp_metadata is not None
        ):
            attn_out = self.do_cp_balance_attn(
                q_nope,
                k_nope,
                q_pe,
                k_pe,
                topk_indices,
                layer,
                actual_seq_qlen,
                actual_seq_lengths_kv,
            )
        else:
            attn_out, _, _ = torch_npu.npu_sparse_flash_attention(
                query=q_nope,
                key=k_nope,
                value=k_nope,
                query_rope=q_pe,
                key_rope=k_pe,
                sparse_indices=topk_indices,
                scale_value=layer.scaling,
                actual_seq_lengths_query=actual_seq_qlen.to(
                    device=q_nope.device, dtype=torch.int32
                ),
                actual_seq_lengths_kv=actual_seq_lengths_kv.to(
                    device=q_nope.device, dtype=torch.int32
                ),
                block_table=self.forward_metadata.block_tables,
                sparse_block_size=1,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
                return_softmax_lse=False,
            )

        return attn_out

    def _can_run_sr_tree_paged(self, layer, forward_batch, sinks, slopes):
        if not self._paged_impl_selected():
            return False
        meta = None
        if self.forward_metadata is not None:
            meta = getattr(self.forward_metadata, "sr_tree_paged", None)
        if meta is None:
            meta = self._sr_tree_paged_meta
        if meta is None:
            return False
        return (
            self.draft_topk > 1
            and int(self.page_size) > 1
            and not self.use_mla
            and not self.use_alibi
            and not layer.is_cross_attention
            and layer.attn_type != AttentionType.ENCODER_ONLY
            and forward_batch.encoder_lens is None
            and layer.sliding_window_size == -1
            and layer.logit_cap == 0
            and sinks is None
            and slopes is None
        )

    def bind_sr_tree_paged_metadata(self, meta: SRTreePagedMetadata) -> None:
        self._sr_tree_paged_meta = meta

    def _run_sr_tree_paged_attention(self, q, k_cache, v_cache, layer):
        """Read raw paged cache. Caller already wrote this step's K/V."""
        metadata = None
        if self.forward_metadata is not None:
            metadata = getattr(self.forward_metadata, "sr_tree_paged", None)
        if metadata is None:
            metadata = self._sr_tree_paged_meta
        query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
        if metadata is None or query.shape[0] != int(metadata.block_tables.shape[0]):
            raise ValueError("SR tree paged attention query/metadata row mismatch")
        if self.use_fia:
            output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                query.unsqueeze(1),
                k_cache.view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim),
                v_cache.view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim),
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="BSND",
                atten_mask=None,
                block_size=self.page_size,
                block_table=metadata.block_tables,
                actual_seq_lengths_kv=metadata.context_lens_list,
                scale=layer.scaling,
            )
        else:
            output = query.new_empty(
                (query.shape[0], layer.tp_q_head_num, layer.v_head_dim)
            )
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=k_cache,
                value_cache=v_cache,
                num_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
                scale_value=layer.scaling,
                block_table=metadata.block_tables,
                context_lens=metadata.context_lens_cpu,
                out=output,
            )
        return output.reshape(query.shape[0], layer.tp_q_head_num * layer.v_head_dim)

    def _can_run_sr_tail_paged(self, layer, forward_batch, sinks, slopes):
        return (
            getattr(forward_batch, "is_sr_tail_extend", False)
            and not self.use_mla
            and not self.use_alibi
            and not layer.is_cross_attention
            and layer.attn_type != AttentionType.ENCODER_ONLY
            and forward_batch.encoder_lens is None
            and layer.sliding_window_size == -1
            and layer.logit_cap == 0
            and sinks is None
            and slopes is None
        )

    def _run_sr_tail_paged(self, q, k_cache, v_cache, layer):
        """All tail queries in one paged call; KV has already been written.

        Only attention sees one row per token. The model and logits processor
        retain the original EXTEND batch and its per-request last positions.
        """
        metadata = self.forward_metadata.sr_tail
        query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
        if metadata is None or query.shape[0] != len(metadata.context_lens_list):
            raise ValueError("SR tail attention query/metadata row mismatch")
        if self.use_fia:
            self.sr_tail_attention_paths.add("paged_fia")
            output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                query.unsqueeze(1),
                k_cache.view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim),
                v_cache.view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim),
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="BSND",
                atten_mask=None,
                block_size=self.page_size,
                block_table=metadata.block_tables,
                actual_seq_lengths_kv=metadata.context_lens_list,
                scale=layer.scaling,
            )
        else:
            self.sr_tail_attention_paths.add("paged_atb")
            output = query.new_empty(
                (query.shape[0], layer.tp_q_head_num, layer.v_head_dim)
            )
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=k_cache,
                value_cache=v_cache,
                num_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
                scale_value=layer.scaling,
                block_table=metadata.block_tables,
                context_lens=metadata.context_lens_cpu,
                out=output,
            )
        return output.reshape(query.shape[0], layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi_head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        slopes: Optional[torch.Tensor] = None,
    ):
        if is_mla_preprocess_enabled():
            # MLAPO and MLAPROLOG do save kv_cache
            save_kv_cache = False
        if self.is_dllm_model:
            return self.forward_dllm(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
            )
        if topk_indices is not None:
            return self.forward_sparse(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
            )
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            return self.forward_mtp(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
            )

        if not self.use_mla:
            # In cross attention layer, when there is no vision input,the values of k and v is None
            if save_kv_cache and k is not None and v is not None:
                # support cross attention
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            if sinks is not None:
                # Use SWA block tables if hybrid SWA is enabled for this layer
                if self.is_hybrid_swa and layer.sliding_window_size != -1:
                    block_tables = self.forward_metadata.block_tables_swa
                else:
                    block_tables = self.forward_metadata.block_tables
                attn_out = attention_sinks_prefill_triton(
                    q,
                    k_cache,
                    v_cache,
                    sinks,
                    self.forward_metadata.extend_seq_lens,
                    block_tables,
                    self.forward_metadata.seq_lens,
                    layer.scaling,
                    layer.sliding_window_size,
                    layer.tp_q_head_num,
                    layer.tp_k_head_num,
                )
                return attn_out

            if self._can_run_sr_tail_paged(layer, forward_batch, sinks, slopes):
                return self._run_sr_tail_paged(q, k_cache, v_cache, layer)
            if getattr(forward_batch, "is_sr_tail_extend", False):
                self.sr_tail_attention_paths.add("ordinary_extend_specialized")

            if self.use_fia:
                """FIA will support multi-bs in the later version of CANN"""
                q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                attn_output = torch.empty(
                    (q.size(0), layer.tp_q_head_num, layer.v_head_dim),
                    device=q.device,
                    dtype=q.dtype,
                )
                q_len_offset = 0
                for q_len in forward_batch.extend_seq_lens_cpu:
                    attn_output[q_len_offset : q_len_offset + q_len] = (
                        torch.ops.npu.npu_fused_infer_attention_score(
                            q[None, q_len_offset : q_len_offset + q_len],
                            k[None, q_len_offset : q_len_offset + q_len],
                            v[None, q_len_offset : q_len_offset + q_len],
                            num_heads=layer.tp_q_head_num,
                            num_key_value_heads=layer.tp_k_head_num,
                            input_layout="BSND",  # todo, TND not supports q_heads!=k_heads
                            atten_mask=self.fia_mask.unsqueeze(0),
                            sparse_mode=3 if q_len != 1 else 0,
                            scale=layer.scaling,
                            next_tokens=0,
                        )[0]
                    )
                    q_len_offset += q_len
                attn_output = attn_output.view(
                    -1, layer.tp_q_head_num * layer.v_head_dim
                )

            else:
                causal = True
                if (
                    layer.is_cross_attention
                    or layer.attn_type == AttentionType.ENCODER_ONLY
                ):
                    causal = False
                # there are some accuracy issues in cross attention scene to use torch_npu._npu_flash_attention_qlens
                # forward_batch.encoder_lens is not None in cross attention scend, we add native attn to solve accuracy issues
                # Model skywork-reward-gemma2-2-27B also suffers from precision anomalies, thus the torch native backend becomes beneficial approach.
                if (
                    layer.qk_head_dim <= 128
                    and causal
                    and forward_batch.encoder_lens is None
                    and layer.logit_cap == 0
                    and not getattr(self, "use_native_sdpa", False)
                ):
                    if not self.use_alibi:
                        query = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
                        attn_output = torch.empty(
                            (query.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                            dtype=query.dtype,
                            device=query.device,
                        )
                        torch_npu._npu_flash_attention_qlens(
                            query=query,
                            key_cache=k_cache,
                            value_cache=v_cache,
                            mask=self.mask,
                            block_table=self.forward_metadata.block_tables,
                            seq_len=self.forward_metadata.extend_seq_lens_cpu_int,
                            context_lens=self.forward_metadata.seq_lens_cpu_int,
                            scale_value=layer.scaling,
                            num_heads=layer.tp_q_head_num,
                            num_kv_heads=layer.tp_k_head_num,
                            out=attn_output,
                        )
                    else:
                        attn_output = self.attn_alibi(
                            q=q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim),
                            k_cache=k_cache,
                            v_cache=v_cache,
                            block_tables=self.forward_metadata.block_tables,
                            seq_lens=self.forward_metadata.seq_lens_cpu_int,
                            query_lens=self.forward_metadata.extend_seq_lens_cpu_int,
                            scale_value=layer.scaling,
                            num_heads=layer.tp_q_head_num,
                            slopes=slopes,
                            is_extend=True,
                        )
                else:
                    if layer.qk_head_dim != layer.v_head_dim:
                        attn_output = q.new_empty(
                            (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                        )
                    else:
                        attn_output = torch.empty_like(q)

                    use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

                    q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                    o_ = attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)

                    # add forward_batch.encoder_lens and is_cross_attention arguments for cross attention scene
                    attn_output = self.native_attn.run_sdpa_forward_extend(
                        q_,
                        o_,
                        k_cache.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                        v_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                        forward_batch.req_to_token_pool.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.extend_prefix_lens,
                        forward_batch.extend_seq_lens,
                        forward_batch.encoder_lens,
                        is_cross_attention=layer.is_cross_attention,
                        scaling=layer.scaling,
                        enable_gqa=use_gqa,
                        causal=causal,
                        logit_cap=layer.logit_cap,
                        logit_capping_method=layer.logit_capping_method,
                    )
                    attn_output = attn_output.view(
                        -1, layer.tp_q_head_num * layer.v_head_dim
                    )
        elif sum(forward_batch.extend_prefix_lens_cpu) > 0:
            # This branch adds support for prefix cache for GLM-4.7-Flash.
            # When using the MLA architecture, if qk head dim equals v head dim and the head count is not a power of 2,
            # we use the FIA kernel for computation.
            if layer.qk_head_dim == layer.v_head_dim:
                q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)

                k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(
                    layer.layer_id
                )
                kv_cached = torch.index_select(
                    k_buffer, 0, self.forward_metadata.flatten_prefix_block_tables
                )
                k_rope_cached = torch.index_select(
                    v_buffer, 0, self.forward_metadata.flatten_prefix_block_tables
                ).flatten(0, 1)

                assert layer.kv_b_proj is not None
                kv = layer.kv_b_proj(kv_cached)[0].view(
                    -1, layer.tp_k_head_num, self.qk_nope_head_dim + layer.v_head_dim
                )
                k_nope, v_pre = kv.split(
                    [self.qk_nope_head_dim, layer.v_head_dim], dim=-1
                )

                k_rope = k_rope_cached.expand(-1, layer.tp_k_head_num, -1)
                k_pre = torch.cat([k_nope, k_rope], dim=-1)

                attn_output = torch.empty(
                    (q.size(0), layer.tp_q_head_num, layer.v_head_dim),
                    device=q.device,
                    dtype=q.dtype,
                )
                q_len_offset = 0
                prefix_len_offset = 0
                for q_len, prefix_len in zip(
                    self.forward_metadata.extend_seq_lens_cpu_int,
                    self.forward_metadata.prefix_lens,
                ):
                    k_cur_slice = k[None, q_len_offset : q_len_offset + q_len]
                    v_cur_slice = v[None, q_len_offset : q_len_offset + q_len]
                    k_pre_slice = k_pre[
                        None, prefix_len_offset : prefix_len_offset + prefix_len
                    ]
                    v_pre_slice = v_pre[
                        None, prefix_len_offset : prefix_len_offset + prefix_len
                    ]

                    k_full = torch.cat([k_pre_slice, k_cur_slice], dim=1)
                    v_full = torch.cat([v_pre_slice, v_cur_slice], dim=1)

                    attn_output[q_len_offset : q_len_offset + q_len] = (
                        torch.ops.npu.npu_fused_infer_attention_score(
                            q[None, q_len_offset : q_len_offset + q_len],
                            k_full,
                            v_full,
                            num_heads=layer.tp_q_head_num,
                            num_key_value_heads=layer.tp_k_head_num,
                            input_layout="BSND",  # todo, TND not supports q_heads!=k_heads
                            atten_mask=self.fia_mask,
                            sparse_mode=3,
                            scale=layer.scaling,
                            next_tokens=0,
                        )[0]
                    )
                    q_len_offset += q_len
                    prefix_len_offset += prefix_len
                attn_output = attn_output.view(
                    -1, layer.tp_q_head_num * layer.v_head_dim
                )
            else:
                num_token_padding = q.shape[0]
                q, k, v = [
                    data[: forward_batch.num_token_non_padded_cpu] for data in [q, k, v]
                ]
                q_nope, q_rope = q.split(
                    [layer.v_head_dim, self.qk_rope_head_dim], dim=-1
                )
                k_nope, k_rope = k.split(
                    [layer.v_head_dim, self.qk_rope_head_dim], dim=-1
                )

                # 1st, compute extend tokens to get attn_output and attn_lse
                num_tokens = q_nope.size(0)
                attn_output = torch.zeros(
                    num_tokens,
                    layer.tp_q_head_num,
                    layer.v_head_dim,
                    dtype=q_nope.dtype,
                    device=q_nope.device,
                )
                attn_lse = torch.zeros(
                    layer.tp_q_head_num,
                    num_tokens,
                    dtype=torch.float32,
                    device=q_nope.device,
                )
                torch_npu.atb.npu_ring_mla(
                    q_nope=q_nope,
                    q_rope=q_rope,
                    k_nope=k_nope,
                    k_rope=k_rope,
                    value=v,
                    mask=self.ringmla_mask,
                    seqlen=self.forward_metadata.extend_seq_lens_cpu_int,
                    head_num=layer.tp_q_head_num,
                    kv_head_num=layer.tp_k_head_num,
                    pre_out=None,
                    prev_lse=None,
                    qk_scale=layer.scaling,
                    kernel_type="kernel_type_high_precision",
                    mask_type="mask_type_triu",
                    calc_type="calc_type_first_ring",
                    output=attn_output,
                    softmax_lse=attn_lse,
                )

                # 2nd, load history kvcache(kv_a and k_pe) and calculate k_nope
                k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(
                    layer.layer_id
                )
                kv_cached = torch.index_select(
                    k_buffer, 0, self.forward_metadata.flatten_prefix_block_tables
                )
                k_rope_cached = torch.index_select(
                    v_buffer, 0, self.forward_metadata.flatten_prefix_block_tables
                ).flatten(0, 1)

                assert layer.kv_b_proj is not None
                kv = layer.kv_b_proj(kv_cached)[0].view(
                    -1, layer.tp_k_head_num, self.qk_nope_head_dim + layer.v_head_dim
                )
                k_nope, v = kv.split([self.qk_nope_head_dim, layer.v_head_dim], dim=-1)

                # 3rd, compute history kv to attn_out
                k_rope = k_rope_cached.expand(-1, layer.tp_k_head_num, -1)
                seq_len = torch.stack(
                    [
                        self.forward_metadata.extend_seq_lens_cpu_int,
                        self.forward_metadata.prefix_lens,
                    ]
                )
                torch_npu.atb.npu_ring_mla(
                    q_nope=q_nope,
                    q_rope=q_rope,
                    k_nope=k_nope,
                    k_rope=k_rope,
                    value=v,
                    mask=self.ringmla_mask,
                    seqlen=seq_len,
                    head_num=layer.tp_q_head_num,
                    kv_head_num=layer.tp_k_head_num,
                    pre_out=attn_output,
                    prev_lse=attn_lse,
                    qk_scale=layer.scaling,
                    kernel_type="kernel_type_high_precision",
                    mask_type="no_mask",
                    calc_type="calc_type_default",
                    output=attn_output,
                    softmax_lse=attn_lse,
                )
                attn_output = attn_output.reshape(
                    [-1, layer.tp_q_head_num, layer.v_head_dim]
                )
                if num_token_padding != forward_batch.num_token_non_padded_cpu:
                    attn_output = torch.cat(
                        [
                            attn_output,
                            attn_output.new_zeros(
                                num_token_padding - attn_output.shape[0],
                                *attn_output.shape[1:],
                            ),
                        ],
                        dim=0,
                    )
        else:
            if layer.qk_head_dim == layer.v_head_dim:
                """FIA will support multi-bs in the later version of CANN"""
                q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                attn_output = torch.empty(
                    (q.size(0), layer.tp_q_head_num, layer.v_head_dim),
                    device=q.device,
                    dtype=q.dtype,
                )
                q_len_offset = 0
                for q_len in forward_batch.extend_seq_lens_cpu:
                    attn_output[q_len_offset : q_len_offset + q_len] = (
                        torch.ops.npu.npu_fused_infer_attention_score(
                            q[None, q_len_offset : q_len_offset + q_len],
                            k[None, q_len_offset : q_len_offset + q_len],
                            v[None, q_len_offset : q_len_offset + q_len],
                            num_heads=layer.tp_q_head_num,
                            num_key_value_heads=layer.tp_k_head_num,
                            input_layout="BSND",  # todo, TND not supports q_heads!=k_heads
                            atten_mask=self.fia_mask.unsqueeze(0),
                            sparse_mode=3 if q_len != 1 else 0,
                            scale=layer.scaling,
                            next_tokens=0,
                        )[0]
                    )
                    q_len_offset += q_len
                attn_output = attn_output.view(
                    -1, layer.tp_q_head_num * layer.v_head_dim
                )
            elif layer.v_head_dim in [256]:
                """Currently, in NO_QUANT situation, qk_nope_head_dim == v_head_dim, and rope exists, v_head_dim only support 512 and 128"""
                kv_lora_rank = k.shape[-1] - self.qk_rope_head_dim
                kv_c, k_rope = k.split([kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, forward_batch.out_cache_loc, kv_c, k_rope
                    )
                attn_output = q.new_empty(
                    (q.shape[0], layer.tp_q_head_num, kv_lora_rank)
                )
                use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

                k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                    layer.layer_id
                )
                kv_cache = torch.cat([k_cache, v_cache], dim=-1)
                attn_output = self.native_attn.run_sdpa_forward_extend(
                    q,
                    attn_output,
                    kv_cache.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                    k_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    forward_batch.req_to_token_pool.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.extend_prefix_lens,
                    forward_batch.extend_seq_lens,
                    scaling=layer.scaling,
                    enable_gqa=use_gqa,
                    causal=True,
                )
            else:
                num_token_padding = q.shape[0]
                q, k, v = [
                    data[: forward_batch.num_token_non_padded_cpu] for data in [q, k, v]
                ]

                q_nope, q_rope = q.split(
                    [layer.v_head_dim, self.qk_rope_head_dim], dim=-1
                )
                k_nope, k_rope = k.split(
                    [layer.v_head_dim, self.qk_rope_head_dim], dim=-1
                )

                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q_nope,
                    k_nope,
                    v,
                    query_rope=q_rope,
                    key_rope=k_rope,
                    num_heads=layer.tp_q_head_num,
                    input_layout="TND",
                    atten_mask=self.fia_mask,
                    sparse_mode=3,
                    actual_seq_lengths=self.forward_metadata.seq_lens_list_cumsum,
                    actual_seq_lengths_kv=self.forward_metadata.seq_lens_list_cumsum,
                    scale=layer.scaling,
                    next_tokens=0,
                )

                attn_output = attn_output.reshape(
                    -1, layer.tp_q_head_num, layer.v_head_dim
                )
                if num_token_padding != forward_batch.num_token_non_padded_cpu:
                    attn_output = torch.cat(
                        [
                            attn_output,
                            attn_output.new_zeros(
                                num_token_padding - attn_output.shape[0],
                                *attn_output.shape[1:],
                            ),
                        ],
                        dim=0,
                    )

        return attn_output

    def forward_dllm(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi_head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)

        if self.forward_metadata.seq_lens_cpu_int is None:
            # capture
            actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
        else:
            # eagle
            actual_seq_lengths_kv = (
                self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
            )

        if self.forward_metadata.extend_seq_lens_cpu_int is None:
            # capture & replay
            actual_seq_lengths = self.forward_metadata.seq_lens_list_cumsum
        else:
            actual_seq_lengths = (
                torch.cumsum(self.forward_metadata.extend_seq_lens_cpu_int, dim=0)
                .int()
                .tolist()
            )

        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query,
            k_cache.view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim),
            v_cache.view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim),
            block_table=self.forward_metadata.block_tables,
            block_size=self.page_size,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            atten_mask=None,
            scale=layer.scaling,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
        )
        attn_output = attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

        return attn_output

    def forward_mtp(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if save_kv_cache:
            if self.use_mla:
                k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
                k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            else:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

        if (
            self._use_target_tree_paged_fia()
            and forward_batch.forward_mode.is_target_verify()
            and getattr(self.forward_metadata, "sr_target_tree_fia", None) is not None
        ):
            return self._run_sr_target_tree_fia(
                q,
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                layer,
            )

        if (
            self._use_tree_shared_prefix()
            and forward_batch.forward_mode.is_target_verify()
            and self.forward_metadata.tree_shared is not None
        ):
            return self._run_tree_shared_prefix_attention(
                q,
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                layer,
            )

        if not self.use_mla:
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                layer.layer_id
            ).view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                layer.layer_id
            ).view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim)
            query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim).contiguous()
            if not self.graph_mode:
                num_token_padding = query.shape[0]
                query = query[: forward_batch.num_token_non_padded_cpu]
            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_lengths_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )
            if forward_batch.forward_mode.is_draft_extend():
                actual_seq_lengths = (
                    np.array(forward_batch.extend_seq_lens_cpu).cumsum().tolist()
                )
            else:
                actual_seq_lengths = np.arange(
                    self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens + query.shape[0],
                    self.speculative_num_draft_tokens,
                )

            custom_mask = getattr(
                getattr(forward_batch, "spec_info", None), "custom_mask", None
            )
            if use_tree_verify_fallback(
                forward_batch.forward_mode.is_target_verify(),
                self.verify_tree_topk,
                custom_mask,
            ):
                attn_output = self._run_tree_verify_slot_gather(
                    query,
                    k_cache,
                    v_cache,
                    layer,
                    forward_batch,
                    custom_mask,
                    qk_head_dim=layer.qk_head_dim,
                    v_head_dim=layer.v_head_dim,
                )
            else:
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    query,
                    k_cache,
                    v_cache,
                    block_table=self.forward_metadata.block_tables,
                    block_size=self.page_size,
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="TND",
                    atten_mask=self.mtp_mask,
                    scale=layer.scaling,
                    actual_seq_lengths=actual_seq_lengths,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    sparse_mode=3,
                )
                attn_output = attn_output.view(
                    -1, layer.tp_q_head_num * layer.v_head_dim
                )
            if (
                not self.graph_mode
                and forward_batch.num_token_non_padded_cpu != num_token_padding
            ):
                attn_output = torch.cat(
                    [
                        attn_output,
                        attn_output.new_zeros(
                            num_token_padding - forward_batch.num_token_non_padded_cpu,
                            *attn_output.shape[1:],
                        ),
                    ],
                    dim=0,
                )
            return attn_output
        else:
            c_kv, k_rope = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            if is_fia_nz():
                k_rope_cache = _reshape_kv_for_fia_nz(
                    k_rope, layer.tp_k_head_num, self.qk_rope_head_dim, self.page_size
                )
                c_kv_cache = _reshape_kv_for_fia_nz(
                    c_kv, layer.tp_v_head_num, self.kv_lora_rank, self.page_size
                )
            else:
                k_rope_cache = k_rope.view(
                    -1, layer.tp_k_head_num, self.page_size, self.qk_rope_head_dim
                )
                c_kv_cache = c_kv.view(
                    -1, layer.tp_v_head_num, self.page_size, self.kv_lora_rank
                )

            q_nope = q.view(-1, layer.tp_q_head_num, self.kv_lora_rank).contiguous()
            q_rope = q_rope.view(-1, layer.tp_q_head_num, self.qk_rope_head_dim)
            if not self.graph_mode:
                num_token_padding = q.shape[0]
                q_nope = q_nope[: forward_batch.num_token_non_padded_cpu]
                q_rope = q_rope[: forward_batch.num_token_non_padded_cpu]
            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_lengths_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_lengths_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )
            if forward_batch.forward_mode.is_draft_extend():
                actual_seq_lengths = (
                    np.array(forward_batch.extend_seq_lens_cpu).cumsum().tolist()
                )
            else:
                actual_seq_lengths = np.arange(
                    self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens + q_nope.shape[0],
                    self.speculative_num_draft_tokens,
                )

            custom_mask = getattr(
                getattr(forward_batch, "spec_info", None), "custom_mask", None
            )
            if use_tree_verify_fallback(
                forward_batch.forward_mode.is_target_verify(),
                self.verify_tree_topk,
                custom_mask,
            ):
                attn_output = self._run_tree_verify_slot_gather(
                    q_nope,
                    c_kv,
                    c_kv,
                    layer,
                    forward_batch,
                    custom_mask,
                    qk_head_dim=self.kv_lora_rank,
                    v_head_dim=self.kv_lora_rank,
                    q_rope=q_rope,
                    k_rope_cache=k_rope,
                    rope_head_dim=self.qk_rope_head_dim,
                )
            else:
                workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    q_nope,
                    c_kv_cache,
                    c_kv_cache,
                    query_rope=q_rope,
                    key_rope=k_rope_cache,
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="TND",
                    scale=layer.scaling,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    block_table=self.forward_metadata.block_tables,
                    block_size=self.page_size,
                    sparse_mode=3,
                    atten_mask=self.mtp_mask,
                    actual_seq_lengths=actual_seq_lengths,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                )
                attn_output = torch.empty_like(q_nope, dtype=q.dtype, device=q.device)
                softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)
                torch_npu.npu_fused_infer_attention_score.out(
                    q_nope,
                    c_kv_cache,
                    c_kv_cache,
                    query_rope=q_rope,
                    key_rope=k_rope_cache,
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="TND",
                    scale=layer.scaling,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    block_table=self.forward_metadata.block_tables,
                    block_size=self.page_size,
                    sparse_mode=3,
                    atten_mask=self.mtp_mask,
                    actual_seq_lengths=actual_seq_lengths,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    workspace=workspace,
                    out=[attn_output, softmax_lse],
                )
                attn_output = attn_output.view(
                    -1, layer.tp_q_head_num * layer.v_head_dim
                )
            if (
                not self.graph_mode
                and forward_batch.num_token_non_padded_cpu != num_token_padding
            ):
                attn_output = torch.cat(
                    [
                        attn_output,
                        attn_output.new_zeros(
                            num_token_padding - attn_output.shape[0],
                            *attn_output.shape[1:],
                        ),
                    ],
                    dim=0,
                )
            return attn_output

    def forward_decode_graph(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ):
        if save_kv_cache:
            if self.use_mla:
                k = k.view(-1, layer.tp_k_head_num, self.kv_lora_rank)
                k_rope = k_rope.view(-1, layer.tp_k_head_num, self.qk_rope_head_dim)
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            else:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

        if sinks is not None:
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            # Use SWA block tables if hybrid SWA is enabled for this layer
            if self.is_hybrid_swa and layer.sliding_window_size != -1:
                block_tables = self.forward_metadata.block_tables_swa
            else:
                block_tables = self.forward_metadata.block_tables
            attn_out = attention_sinks_triton(
                q,
                k_cache,
                v_cache,
                sinks,
                block_tables,
                self.forward_metadata.seq_lens,
                layer.scaling,
                layer.sliding_window_size,
                layer.tp_q_head_num,
                layer.tp_k_head_num,
            )
            return attn_out

        if not self.use_mla:
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                layer.layer_id
            ).view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                layer.layer_id
            ).view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim)
            query = q.reshape(-1, 1, layer.tp_q_head_num * layer.qk_head_dim)
            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_len_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )
            num_tokens = query.shape[0]
            actual_seq_len_kv = self._tree_draft_kv_lens(actual_seq_len_kv, num_tokens)
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query,
                k_cache,
                v_cache,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="BSH",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
            )
            output = torch.empty(
                (num_tokens, 1, layer.tp_q_head_num * layer.v_head_dim),
                dtype=q.dtype,
                device=q.device,
            )
            softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)
            torch_npu.npu_fused_infer_attention_score.out(
                query,
                k_cache,
                v_cache,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_k_head_num,
                input_layout="BSH",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                workspace=workspace,
                out=[output, softmax_lse],
            )
            return output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
        else:
            c_kv, k_rope = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            if is_fia_nz():
                k_rope_cache = _reshape_kv_for_fia_nz(
                    k_rope, layer.tp_k_head_num, self.qk_rope_head_dim, self.page_size
                )
                c_kv_cache = _reshape_kv_for_fia_nz(
                    c_kv, layer.tp_v_head_num, self.kv_lora_rank, self.page_size
                )
            else:
                k_rope_cache = k_rope.view(
                    -1, self.page_size, layer.tp_k_head_num * self.qk_rope_head_dim
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_k_head_num * self.kv_lora_rank
                )

            q_nope = q.view(-1, 1, layer.tp_q_head_num, self.kv_lora_rank).contiguous()
            q_rope = q_rope.view(-1, 1, layer.tp_q_head_num, self.qk_rope_head_dim)

            assert (
                self.q_head_num_padding is None
                or self.q_head_num_padding >= layer.tp_q_head_num
            )

            if (
                self.q_head_num_padding is not None
                and self.q_head_num_padding > layer.tp_q_head_num
            ):
                # The FIA kernel only supports head counts that are powers of 2.
                # Therefore, we pad the head dimension when it is not a power of 2.
                q_nope = torch.cat(
                    [q_nope, self.forward_metadata.nope_padding], dim=2
                ).contiguous()
                q_rope = torch.cat(
                    [q_rope, self.forward_metadata.rope_padding], dim=2
                ).contiguous()

            if self.forward_metadata.seq_lens_cpu_int is None:
                actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
            else:
                actual_seq_len_kv = (
                    self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                )

            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=self.q_head_num_padding,
                num_key_value_heads=layer.tp_k_head_num,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                input_layout="BSND",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                antiquant_mode=0,
                antiquant_scale=None,
                sparse_mode=0,
            )
            output = torch.empty_like(q_nope, dtype=q.dtype, device=q.device)
            softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)

            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                c_kv_cache,
                c_kv_cache,
                query_rope=q_rope,
                key_rope=k_rope_cache,
                num_heads=self.q_head_num_padding,
                num_key_value_heads=layer.tp_k_head_num,
                block_table=self.forward_metadata.block_tables,
                block_size=self.page_size,
                input_layout="BSND",
                scale=layer.scaling,
                actual_seq_lengths_kv=actual_seq_len_kv,
                antiquant_mode=0,
                antiquant_scale=None,
                sparse_mode=0,
                workspace=workspace,
                out=[output, softmax_lse],
            )

            output = output[:, :, : layer.tp_q_head_num, :]
            return output.view(-1, layer.tp_q_head_num * self.kv_lora_rank)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        slopes: Optional[torch.Tensor] = None,
    ):
        if is_mla_preprocess_enabled():
            # MLAPO does saving kv_cache
            save_kv_cache = False
        if topk_indices is not None:
            return self.forward_sparse(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
            )

        if not self.use_mla and self._can_run_sr_tree_paged(
            layer, forward_batch, sinks, slopes
        ):
            if save_kv_cache and k is not None and v is not None:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            attn_output = self._run_sr_tree_paged_attention(
                q, k_cache, v_cache, layer
            )
            return attn_output.view(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

        if self.graph_mode and (not self.enable_torch_compile):
            if not self._use_tree_draft_slot_gather(
                forward_batch
            ) and not self._paged_impl_selected():
                return self.forward_decode_graph(
                    q,
                    k,
                    v,
                    layer,
                    forward_batch,
                    save_kv_cache,
                    q_rope=q_rope,
                    k_rope=k_rope,
                    sinks=sinks,
                )

        if not self.use_mla:
            # In cross attention layer, when there is no vision input,the values of k and v is None
            if save_kv_cache and k is not None and v is not None:
                # support cross attention
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            num_tokens = q.shape[0]
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            if self._use_tree_draft_slot_gather(forward_batch):
                attn_output = self._run_tree_draft_slot_gather(
                    q,
                    k_cache,
                    v_cache,
                    layer,
                    qk_head_dim=layer.qk_head_dim,
                    v_head_dim=layer.v_head_dim,
                )
                return attn_output.view(
                    num_tokens, layer.tp_q_head_num * layer.v_head_dim
                )

            if sinks is not None:
                # Use SWA block tables if hybrid SWA is enabled for this layer
                if self.is_hybrid_swa and layer.sliding_window_size != -1:
                    block_tables = self.forward_metadata.block_tables_swa
                else:
                    block_tables = self.forward_metadata.block_tables
                attn_out = attention_sinks_triton(
                    q,
                    k_cache,
                    v_cache,
                    sinks,
                    block_tables,
                    self.forward_metadata.seq_lens,
                    layer.scaling,
                    layer.sliding_window_size,
                    layer.tp_q_head_num,
                    layer.tp_k_head_num,
                )
                return attn_out

            if self.use_fia:
                if self.forward_metadata.seq_lens_cpu_int is None:
                    actual_seq_len_kv = self.forward_metadata.seq_lens_cpu_list
                else:
                    actual_seq_len_kv = (
                        self.forward_metadata.seq_lens_cpu_int.cpu().int().tolist()
                    )
                is_tree_draft = self._is_tree_draft(forward_batch)
                if is_tree_draft:
                    query = q.reshape(
                        -1, 1, layer.tp_q_head_num, layer.qk_head_dim
                    )
                    actual_seq_len_kv = normalize_tree_draft_kv_lens(
                        actual_seq_len_kv, query.shape[0], self.draft_topk
                    )
                else:
                    query = q.view(
                        forward_batch.batch_size,
                        -1,
                        layer.tp_q_head_num,
                        layer.qk_head_dim,
                    )
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    query,
                    k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim
                    ),
                    v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num * layer.qk_head_dim
                    ),
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="BSND",
                    atten_mask=None,
                    block_size=self.page_size,
                    block_table=self.forward_metadata.block_tables,
                    actual_seq_lengths_kv=actual_seq_len_kv,
                    scale=layer.scaling,
                )
            # there are some accuracy issues in cross attention scene to use torch_npu._npu_flash_attention_qlens
            # forward_batch.encoder_lens is not None in cross attention scend, we add native attn to solve accuracy issues
            elif forward_batch.encoder_lens is None and layer.logit_cap == 0:
                query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
                num_tokens = query.shape[0]
                if not self.use_alibi:
                    attn_output = torch.empty(
                        (num_tokens, layer.tp_q_head_num, layer.v_head_dim),
                        dtype=query.dtype,
                        device=query.device,
                    )

                    context_lens = self.forward_metadata.seq_lens_cpu_int
                    if self._is_tree_draft(forward_batch):
                        kv_lens = normalize_tree_draft_kv_lens(
                            context_lens.cpu().int().tolist(),
                            num_tokens,
                            self.draft_topk,
                        )
                        context_lens = torch.tensor(
                            kv_lens,
                            dtype=torch.int32,
                            device=context_lens.device,
                        )
                    torch_npu._npu_paged_attention(
                        query=query,
                        key_cache=k_cache,
                        value_cache=v_cache,
                        num_heads=layer.tp_q_head_num,
                        num_kv_heads=layer.tp_k_head_num,
                        scale_value=layer.scaling,
                        block_table=self.forward_metadata.block_tables,
                        context_lens=context_lens,
                        out=attn_output,
                    )
                else:
                    attn_output = self.attn_alibi(
                        q=query,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        block_tables=self.forward_metadata.block_tables,
                        seq_lens=self.forward_metadata.seq_lens_cpu_int,
                        query_lens=torch.ones(num_tokens, dtype=torch.int32),
                        scale_value=layer.scaling,
                        num_heads=layer.tp_q_head_num,
                        slopes=slopes,
                        is_extend=False,
                    )
            else:
                if layer.qk_head_dim != layer.v_head_dim:
                    attn_output = q.new_empty(
                        (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                    )
                else:
                    attn_output = torch.empty_like(q)

                use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

                q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                o_ = attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)

                attn_output = self.native_attn.run_sdpa_forward_decode(
                    q_,
                    o_,
                    k_cache.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                    v_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    forward_batch.req_to_token_pool.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.encoder_lens,
                    is_cross_attention=layer.is_cross_attention,
                    scaling=layer.scaling,
                    enable_gqa=use_gqa,
                    causal=False,
                    logit_cap=layer.logit_cap,
                    logit_capping_method=layer.logit_capping_method,
                )
            return attn_output.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)
        else:
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )
            num_tokens = q.shape[0]
            kv_c = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            k_pe = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            if self._use_tree_draft_slot_gather(forward_batch):
                attn_output = self._run_tree_draft_slot_gather(
                    q,
                    kv_c,
                    kv_c,
                    layer,
                    qk_head_dim=self.kv_lora_rank,
                    v_head_dim=self.kv_lora_rank,
                    q_rope=q_rope,
                    k_rope_cache=k_pe,
                    rope_head_dim=self.qk_rope_head_dim,
                )
                return attn_output.view(
                    num_tokens, layer.tp_q_head_num * self.kv_lora_rank
                )

            if self.use_fia and (layer.tp_q_head_num // layer.tp_k_head_num) >= 8:
                """layer.tp_q_head_num // layer.tp_k_head_num < 8 will support in the later version of CANN"""
                if is_fia_nz():
                    kv_c = _reshape_kv_for_fia_nz(
                        kv_c, layer.tp_k_head_num, self.kv_lora_rank, self.page_size
                    )
                    k_pe = _reshape_kv_for_fia_nz(
                        k_pe, layer.tp_k_head_num, self.qk_rope_head_dim, self.page_size
                    )
                else:
                    kv_c = kv_c.view(
                        -1, self.page_size, layer.tp_k_head_num * self.kv_lora_rank
                    )
                    k_pe = k_pe.view(
                        -1, self.page_size, layer.tp_k_head_num * self.qk_rope_head_dim
                    )
                q = q.view(
                    forward_batch.batch_size, -1, layer.tp_q_head_num, self.kv_lora_rank
                )
                q_rope = q_rope.view(
                    forward_batch.batch_size,
                    -1,
                    layer.tp_q_head_num,
                    self.qk_rope_head_dim,
                )
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q,
                    kv_c,
                    kv_c,
                    query_rope=q_rope,
                    key_rope=k_pe,
                    num_heads=layer.tp_q_head_num,
                    num_key_value_heads=layer.tp_k_head_num,
                    input_layout="BSND",
                    atten_mask=None,
                    sparse_mode=0,
                    scale=layer.scaling,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    block_table=self.forward_metadata.block_tables,
                    block_size=self.page_size,
                    actual_seq_lengths_kv=self.forward_metadata.seq_lens_cpu_int,
                )
            else:
                assert (
                    self.graph_mode == False
                )  # _npu_paged_attention_mla not support graph mode
                if q_rope is not None:
                    q = torch.cat([q, q_rope], dim=-1)
                query = q.view(-1, layer.tp_q_head_num, layer.head_dim)
                kv_c_and_k_pe_cache = torch.cat([kv_c, k_pe], dim=-1)
                kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                )
                attn_output = torch.empty(
                    [num_tokens, layer.tp_q_head_num, self.kv_lora_rank],
                    dtype=q.dtype,
                    device=q.device,
                )
                torch_npu._npu_paged_attention_mla(
                    query=query,
                    key_cache=kv_c_and_k_pe_cache,
                    num_kv_heads=layer.tp_k_head_num,
                    num_heads=layer.tp_q_head_num,
                    scale_value=layer.scaling,
                    block_table=self.forward_metadata.block_tables,
                    context_lens=self.forward_metadata.seq_lens_cpu_int,
                    mla_vheadsize=self.kv_lora_rank,
                    out=attn_output,
                )
            return attn_output.view(num_tokens, layer.tp_q_head_num * self.kv_lora_rank)

    def forward_mixed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if (
            topk_indices is not None
            or self.use_mla
            or (not self.use_fia and layer.qk_head_dim > 128)
        ):
            raise NotImplementedError(
                "The 'enable-mixed-chunk' feature is currently unsupported in the following scenarios: "
                "1. When using the MLA backend on Ascend NPU devices, "
                "2. When using the deepseekv3.2 model on Ascend NPU devices, "
                "3. When the environment variable ASCEND_USE_FIA is set to 0 and qk_head_dim exceeds 128 on Ascend NPU devices."
            )
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        num_block, block_size, _, _ = k_cache.shape
        key = k_cache.view(num_block, block_size, -1)
        value = v_cache.view(num_block, block_size, -1)

        query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)

        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            block_size=block_size,
            block_table=self.forward_metadata.block_tables,
            atten_mask=self.mix_mask,
            sparse_mode=3,
            actual_seq_lengths=self.forward_metadata.seq_lens_list_cumsum,
            actual_seq_lengths_kv=self.forward_metadata.seq_lens_cpu_int,
            scale=layer.scaling,
        )

        return attn_output.view(
            attn_output.shape[0], layer.tp_q_head_num * layer.v_head_dim
        )


class AscendAttnMultiStepDraftBackend:
    """
    Wrap multiple Ascend attention backends as one for multiple consecutive
    draft decoding steps
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.page_size = int(getattr(model_runner, "page_size", 1) or 1)

        self.attn_backends = []
        for step_id in range(self.speculative_num_steps):
            self.attn_backends.append(
                AscendAttnBackend(
                    model_runner,
                    speculative_step_id=step_id,
                    draft_topk=topk,
                    draft_num_steps=speculative_num_steps,
                    graph_roles=frozenset({AttnGraphRole.TREE_DRAFT}),
                )
            )
        self._central_tree_draft_fill = (
            int(topk) > 1
            and self.page_size > 1
            and bool(self.attn_backends)
            and self.attn_backends[0]._use_tree_compact_fia()
            and not self.attn_backends[0]._use_tree_shared_prefix()
            and not getattr(
                self.attn_backends[0], "_paged_impl_selected", lambda: False
            )()
        )
        self._paged_prep_count = 0
        self._paged_copy_count = 0
        self._paged_round_tables = None
        self._paged_round_active = None
        self._paged_round_prefix = None
        self._paged_round_dummy = 0
        self._paged_round_impl = None
        self._paged_dummy_page = 0
        if self._central_tree_draft_fill:
            for inner in self.attn_backends:
                inner._central_tree_draft_fill = True

    def tree_slot_graph_can_run(self, forward_batch: ForwardBatch) -> bool:
        """Admit draft graph using the captured slot width of step 0.

        Every in-graph step shares the same S_cap. Needed KV is the
        conservative ``max(seq)+num_steps`` bound on the inner backend.
        """
        if not self.attn_backends:
            return True
        ok = self.attn_backends[0].tree_slot_graph_can_run(forward_batch)
        s_cap = self.attn_backends[0]._replay_tree_s_cap
        for inner in self.attn_backends[1:]:
            inner._replay_tree_s_cap = s_cap
        return ok

    def paged_impl_selected(self) -> bool:
        return bool(self.attn_backends) and self.attn_backends[0]._paged_impl_selected()

    def _sr_clear_paged_round_state(self) -> None:
        """Drop warmup/round tables so the next request rebuilds them."""
        self._paged_round_tables = None
        self._paged_round_active = None
        self._paged_round_prefix = None
        self._paged_round_dummy = 0
        self._paged_round_impl = None
        for inner in self.attn_backends:
            inner._sr_tree_paged_meta = None
            metadata = getattr(inner, "forward_metadata", None)
            if metadata is None:
                continue
            metadata.sr_tree_paged = None
            metadata.block_tables = None

    def prepare_sr_tree_paged_eager(
        self,
        forward_batch: ForwardBatch,
        compact_slots: torch.Tensor,
        prefix_lens_cpu,
        allocation_kind: str,
        kv_pool,
        dummy_page: int = 0,
        metrics=None,
    ) -> bool:
        """Build page tables once and optionally submit prefix-tail copy."""
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_paged_layout import (
            materialize_prefix_tail_copy_slots,
            plan_prefix_tail_copy_indices,
        )
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            copy_kv_pool_by_slot,
        )

        if not self.paged_impl_selected():
            return False
        inner0 = self.attn_backends[0]
        inner0._sr_tree_paged_prep_count += 1
        self._paged_prep_count += 1
        raw_bs = int(forward_batch.batch_size)
        topk = int(self.topk)
        slots = compact_slots.reshape(raw_bs, topk, self.speculative_num_steps)
        needed_pages = max(
            max_query_pages_for_tree(
                prefix_lens_cpu, self.speculative_num_steps, self.page_size
            )
            or [1]
        )
        page_buckets = resolve_eager_page_buckets(
            getattr(inner0, "tree_kv_buckets", None),
            self.page_size,
            getattr(inner0, "_paged_graph_max_pages", None),
        )
        max_pages = quantize_page_width(needed_pages, page_buckets)
        t_view = time.perf_counter()
        tables, _shared, branch, active, _n_sh, _n_q = prepare_tree_paged_view(
            inner0.req_to_token,
            forward_batch.req_pool_indices[:raw_bs],
            slots,
            prefix_lens_cpu,
            self.page_size,
            topk,
            self.speculative_num_steps,
            dummy_page=dummy_page,
            max_pages=max_pages,
        )
        if metrics is not None:
            metrics.add_host("tree_paged_view", time.perf_counter() - t_view)
        # Graph buffers belong to bind_sr_tree_paged_replay; eager binds fresh tables.
        dummy = int(dummy_page)
        self._paged_round_tables = tables
        self._paged_round_active = active
        self._paged_round_prefix = prefix_lens_cpu
        self._paged_round_dummy = dummy
        self._paged_round_impl = inner0.tree_attention_impl
        copied = False
        t_copy = time.perf_counter()
        indices = plan_prefix_tail_copy_indices(
            prefix_lens_cpu, allocation_kind, topk, self.page_size
        )
        if len(indices):
            src, dst = materialize_prefix_tail_copy_slots(
                inner0.req_to_token,
                forward_batch.req_pool_indices[:raw_bs],
                branch,
                indices,
                self.page_size,
            )
            if int(src.numel()) > 0:
                inner0._sr_tree_paged_copy_count += 1
                self._paged_copy_count += 1
                copy_kv_pool_by_slot(kv_pool, src, dst)
                copied = True
        if metrics is not None:
            metrics.add_host("tree_paged_copy", time.perf_counter() - t_copy)
        impl = inner0.tree_attention_impl
        n_rows = int(tables.shape[0])
        n_fwd = max(int(self.speculative_num_steps) - 1, 0)
        t_bind = time.perf_counter()
        for inner in self.attn_backends:
            step = int(inner.speculative_step_id)
            if n_fwd:
                step = min(step, n_fwd - 1)
            lens = build_step_context_lens(prefix_lens_cpu, topk, step, n_rows)
            meta = SRTreePagedMetadata(
                block_tables=tables,
                active_rows=active,
                context_lens_cpu=lens,
                context_lens_list=context_lens_list(lens),
                dummy_page=int(dummy_page),
                max_pages=int(tables.shape[1]) if tables.ndim == 2 else 0,
                impl=impl,
            )
            inner.bind_sr_tree_paged_metadata(meta)
            if inner.forward_metadata is None:
                inner.forward_metadata = ForwardMetadata()
            inner.forward_metadata.sr_tree_paged = meta
            inner.forward_metadata.block_tables = tables
        if metrics is not None:
            metrics.add_host("tree_paged_bind", time.perf_counter() - t_bind)
        return copied

    def _paged_graph_table_view(
        self,
        capture_bs: int,
        max_pages: int,
        dummy_page: int,
        allow_alloc: bool = False,
    ):
        """Return the exact-shape contiguous buffer for (rows, pages).

        Capture may allocate. Replay must reuse the captured tensor so ATB
        sees the same storage it captured; a slice of a larger buffer is a
        strided view and is rejected.
        """
        del dummy_page
        rows = int(capture_bs) * int(self.topk)
        pages = max(int(max_pages), 1)
        inner0 = self.attn_backends[0]
        tables_map = getattr(inner0, "cuda_graph_paged_tables", None)
        actives_map = getattr(inner0, "cuda_graph_paged_actives", None)
        if tables_map is None or actives_map is None:
            if not allow_alloc:
                raise NpuGraphPreparationError(
                    f"paged tree graph buffer missing for ({rows}, {pages})",
                    scope="graph",
                )
            tables_map = {}
            actives_map = {}
            inner0.cuda_graph_paged_tables = tables_map
            inner0.cuda_graph_paged_actives = actives_map
        key = (rows, pages)
        tables = tables_map.get(key)
        active = actives_map.get(key)
        if tables is None or active is None:
            if not allow_alloc:
                raise NpuGraphPreparationError(
                    f"paged tree graph buffer missing for ({rows}, {pages})",
                    scope="graph",
                )
            device = getattr(inner0, "device", None)
            if device is None:
                raise NpuGraphPreparationError(
                    "paged tree graph buffer device missing", scope="graph"
                )
            if tables is None:
                tables = torch.zeros((rows, pages), dtype=torch.int32, device=device)
                tables_map[key] = tables
            if active is None:
                active = torch.zeros((rows,), dtype=torch.bool, device=device)
                actives_map[key] = active
        if (
            int(tables.ndim) != 2
            or tuple(int(x) for x in tables.shape) != (rows, pages)
            or (not tables.is_contiguous())
            or tuple(int(x) for x in tables.stride()) != (pages, 1)
        ):
            raise NpuGraphPreparationError(
                f"paged tree graph tables shape {tuple(tables.shape)} "
                f"stride {tuple(tables.stride())} violate contiguous "
                f"({rows}, {pages})",
                scope="graph",
            )
        if int(active.numel()) != rows or (not active.is_contiguous()):
            raise NpuGraphPreparationError(
                f"paged tree graph active {int(active.numel())} "
                f"contiguous={active.is_contiguous()} != {rows}",
                scope="graph",
            )
        return tables, active

    def bind_sr_tree_paged_capture(self, capture_bs: int, max_pages: int, dummy_page: int):
        if not self.paged_impl_selected():
            return
        dummy = int(dummy_page)
        tables, active = self._paged_graph_table_view(
            capture_bs, max_pages, dummy, allow_alloc=True
        )
        tables.fill_(dummy)
        active.zero_()
        impl = self.attn_backends[0].tree_attention_impl
        n_fwd = max(int(self.speculative_num_steps) - 1, 0)
        rows = int(tables.shape[0])
        pages = int(tables.shape[1]) if tables.ndim == 2 else 0
        for inner in self.attn_backends:
            step = int(inner.speculative_step_id)
            if n_fwd:
                step = min(step, n_fwd - 1)
            lens = torch.ones((rows,), dtype=torch.int32)
            meta = SRTreePagedMetadata(
                block_tables=tables,
                active_rows=active,
                context_lens_cpu=lens,
                context_lens_list=context_lens_list(lens),
                dummy_page=dummy,
                max_pages=pages,
                impl=impl,
            )
            inner.bind_sr_tree_paged_metadata(meta)
            if inner.forward_metadata is None:
                inner.forward_metadata = ForwardMetadata()
            inner.forward_metadata.sr_tree_paged = meta
            inner.forward_metadata.block_tables = tables

    def _validate_sr_tree_paged_replay(self, capture_bs: int, max_pages: int):
        """Host-only checks. Must run before any graph buffer write."""
        raw_bs = int(getattr(self, "_tree_replay_raw_bs", 0) or 0)
        capture_bs = int(capture_bs)
        max_pages = int(max_pages)
        topk = int(self.topk)
        prefix = getattr(self, "_paged_round_prefix", None)
        if prefix is None:
            raise NpuGraphPreparationError(
                "missing paged tree prefix lengths", scope="graph"
            )
        if not torch.is_tensor(prefix):
            raise NpuGraphPreparationError(
                "paged tree prefix must be a CPU tensor", scope="graph"
            )
        if (
            prefix.device.type != "cpu"
            or int(prefix.ndim) != 1
            or prefix.dtype not in (torch.int32, torch.int64)
        ):
            raise NpuGraphPreparationError(
                "paged tree prefix must be rank-1 CPU int32/int64",
                scope="graph",
            )
        if not (0 < raw_bs <= capture_bs) or int(prefix.numel()) != raw_bs:
            raise NpuGraphPreparationError(
                f"paged tree prefix numel {int(prefix.numel())} "
                f"raw_bs {raw_bs} capture_bs {capture_bs}",
                scope="graph",
            )
        src = getattr(self, "_paged_round_tables", None)
        src_act = getattr(self, "_paged_round_active", None)
        if src is None or src_act is None:
            raise NpuGraphPreparationError(
                "missing paged tree source tables", scope="graph"
            )
        need_src = raw_bs * topk
        if int(src.dim()) != 2 or int(src.shape[0]) != need_src:
            raise NpuGraphPreparationError(
                f"paged tree source rows {tuple(src.shape)} != {need_src}",
                scope="graph",
            )
        if int(src_act.numel()) != need_src:
            raise NpuGraphPreparationError(
                f"paged tree source active {int(src_act.numel())} != {need_src}",
                scope="graph",
            )
        if int(src.shape[1]) > max_pages:
            raise NpuGraphPreparationError(
                f"paged tree source cols {int(src.shape[1])} > bucket {max_pages}",
                scope="graph",
            )
        dummy = int(
            getattr(self, "_paged_round_dummy", getattr(self, "_paged_dummy_page", 0))
        )
        tables, active = self._paged_graph_table_view(
            capture_bs, max_pages, dummy, allow_alloc=False
        )
        return prefix, src, src_act, tables, active, dummy, raw_bs, capture_bs, max_pages, topk

    def bind_sr_tree_paged_replay(self, capture_bs: int, max_pages: int):
        if not self.paged_impl_selected():
            return
        (
            prefix,
            src,
            src_act,
            tables,
            active,
            dummy,
            _raw_bs,
            capture_bs,
            max_pages,
            topk,
        ) = self._validate_sr_tree_paged_replay(capture_bs, max_pages)
        need_src = int(src.shape[0])
        src_cols = int(src.shape[1])
        tables.fill_(dummy)
        active.zero_()
        tables[:need_src, :src_cols].copy_(src)
        active[:need_src].copy_(src_act.to(device=active.device))
        impl = getattr(self, "_paged_round_impl", self.attn_backends[0].tree_attention_impl)
        n_rows = int(tables.shape[0])
        n_fwd = max(int(self.speculative_num_steps) - 1, 0)
        for inner in self.attn_backends:
            step = int(inner.speculative_step_id)
            if n_fwd:
                step = min(step, n_fwd - 1)
            lens = build_step_context_lens(prefix, topk, step, n_rows)
            meta = SRTreePagedMetadata(
                block_tables=tables,
                active_rows=active,
                context_lens_cpu=lens,
                context_lens_list=context_lens_list(lens),
                dummy_page=dummy,
                max_pages=int(tables.shape[1]) if tables.ndim == 2 else 0,
                impl=impl,
            )
            inner.bind_sr_tree_paged_metadata(meta)
            if inner.forward_metadata is None:
                inner.forward_metadata = ForwardMetadata()
            inner.forward_metadata.sr_tree_paged = meta
            inner.forward_metadata.block_tables = tables

    def common_template(self, forward_batch: ForwardBatch, call_fn: int):
        assert forward_batch.spec_info is not None

        for i in range(self.speculative_num_steps - 1):
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        if self.paged_impl_selected():
            extra = getattr(self, "_paged_capture_max_pages", 1)
            dummy = int(getattr(self, "_paged_dummy_page", 0) or 0)
            self.bind_sr_tree_paged_capture(
                int(forward_batch.batch_size), extra, dummy
            )
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, call_fn)
        self._fill_central_tree_draft_graph_metadata(forward_batch)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, call_fn)
        self._fill_central_tree_draft_graph_metadata(forward_batch, capture_bs=bs)

    def _fill_central_tree_draft_graph_metadata(
        self, forward_batch: ForwardBatch, capture_bs: Optional[int] = None
    ):
        if self.attn_backends and self.attn_backends[0]._use_tree_shared_prefix():
            raw_bs = getattr(self, "_tree_replay_raw_bs", None)
            if raw_bs is None:
                raw_bs = forward_batch.batch_size
            lengths = cpu_prefix_lengths(forward_batch.seq_lens_cpu, int(raw_bs))
            for inner in self.attn_backends[: max(self.speculative_num_steps - 1, 0)]:
                inner._fill_shared_metadata(
                    inner.forward_metadata.tree_shared,
                    forward_batch.req_pool_indices,
                    lengths,
                    draft=True,
                )
            return
        if not self._central_tree_draft_fill:
            return
        n_forward = max(int(self.speculative_num_steps) - 1, 0)
        inners = self.attn_backends[:n_forward]
        if not inners:
            return
        slots_out = []
        lens_out = []
        for inner in inners:
            md = inner.forward_metadata
            dest = getattr(md, "tree_draft_kv_slots", None)
            dest_lens = getattr(md, "tree_draft_kv_lens_t", None)
            if dest is None or dest_lens is None:
                return
            slots_out.append(dest)
            lens_out.append(dest_lens)
        inner0 = inners[0]
        kv_bucket = getattr(inner0, "_replay_tree_s_cap", None) or getattr(
            inner0, "_active_tree_s_cap", None
        )
        if kv_bucket is None:
            kv_bucket = int(slots_out[0].shape[1])
        prefix = getattr(forward_batch, "seq_lens_cpu", None)
        raw_bs = getattr(self, "_tree_replay_raw_bs", None)
        if raw_bs is None:
            raw_bs = int(getattr(forward_batch, "batch_size", 0) or 0)
        if capture_bs is None:
            capture_bs = getattr(self, "_tree_replay_capture_bs", None)
        if capture_bs is None:
            capture_bs = int(getattr(forward_batch, "batch_size", raw_bs) or raw_bs)
        if prefix is None:
            prefix = getattr(forward_batch, "seq_lens", None)
        try:
            cpu_lens_by_step = fill_tree_draft_metadata_(
                inner0.req_to_token,
                forward_batch.req_pool_indices,
                prefix,
                slots_out,
                lens_out,
                raw_bs=int(raw_bs),
                capture_bs=int(capture_bs),
                page_size=self.page_size,
                topk=self.topk,
                speculative_num_steps=self.speculative_num_steps,
                kv_bucket=int(kv_bucket),
            )
        except (RuntimeError, ValueError) as e:
            raise NpuGraphPreparationError(str(e), scope="graph") from e
        capture_rows = int(capture_bs) * max(int(self.topk), 1)
        for inner, cpu_lens in zip(inners, cpu_lens_by_step):
            inner._store_tree_fia_kv_lens_cpu(cpu_lens, capture_rows)
