from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
    from sglang.srt.speculative.ngram_worker import NGRAMWorker
    from sglang.srt.speculative.spectre.verifier.spectre_worker import (
        SpectreWorker,
    )
    from sglang.srt.speculative.standalone_remote.verifier.sr_worker import (
        StandaloneRemoteWorker,
    )


def remote_spec_role(server_args):
    """Role shared by remote speculative servers.

    ``STANDALONE_REMOTE`` stores it on ``standalone_remote_role``.
    ``OPT_TREE_REMOTE`` stores it on ``opt_tree_role``. The SR field wins when
    both are set. Neither algorithm is treated as the other.
    """
    if server_args is None:
        return None
    return getattr(server_args, "standalone_remote_role", None) or getattr(
        server_args, "opt_tree_role", None
    )


class SpeculativeAlgorithm(Enum):
    """Enumeration of speculative decoding algorithms."""

    EAGLE = auto()
    EAGLE3 = auto()
    STANDALONE = auto()
    NGRAM = auto()
    SPECTRE = auto()
    STANDALONE_REMOTE = auto()
    OPT_TREE_REMOTE = auto()
    NONE = auto()

    @classmethod
    def from_string(cls, name: Optional[str]) -> SpeculativeAlgorithm:
        if name is None:
            return cls.NONE
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Unknown speculative algorithm name: {name}")

    def is_none(self) -> bool:
        return self == SpeculativeAlgorithm.NONE

    def is_eagle(self) -> bool:
        # NOTE: EAGLE3 is a variant of EAGLE
        return self == SpeculativeAlgorithm.EAGLE or self == SpeculativeAlgorithm.EAGLE3

    def is_eagle3(self) -> bool:
        return self == SpeculativeAlgorithm.EAGLE3

    def is_standalone(self) -> bool:
        return self == SpeculativeAlgorithm.STANDALONE

    def is_ngram(self) -> bool:
        return self == SpeculativeAlgorithm.NGRAM

    def is_spectre(self) -> bool:
        return self == SpeculativeAlgorithm.SPECTRE

    def is_standalone_remote(self) -> bool:
        return self == SpeculativeAlgorithm.STANDALONE_REMOTE

    def is_opt_tree_remote(self) -> bool:
        return self == SpeculativeAlgorithm.OPT_TREE_REMOTE

    def skips_colocated_draft_worker(self, server_args) -> bool:
        """Draft role runs the target model itself and does not launch a colocated worker."""
        if self.is_spectre() and getattr(server_args, "spectre_role", None) == "draft":
            return True
        if (
            self.is_standalone_remote()
            and getattr(server_args, "standalone_remote_role", None) == "draft"
        ):
            return True
        if (
            self.is_opt_tree_remote()
            and getattr(server_args, "opt_tree_role", None) == "draft"
        ):
            return True
        return False

    def needs_draft_multimodal_processor(self, server_args) -> bool:
        """Remote draft must load its own multimodal processor.

        ``spectre_role=draft`` keeps the historical check even when the
        algorithm is not SPECTRE. SR and OPT drafts are included through
        ``skips_colocated_draft_worker``.
        """
        if getattr(server_args, "spectre_role", None) == "draft":
            return True
        return self.skips_colocated_draft_worker(server_args)

    def uses_draft_paused_reqs(self, server_args) -> bool:
        """Paused requests live on ``draft_paused_reqs`` rather than the SPECTRE lock."""
        if (
            self.is_standalone_remote()
            and getattr(server_args, "standalone_remote_role", None) == "draft"
        ):
            return True
        if (
            self.is_opt_tree_remote()
            and getattr(server_args, "opt_tree_role", None) == "draft"
        ):
            return True
        return False

    def uses_ordinary_decode_prepare(self, server_args) -> bool:
        """Draft decode collapses to the last token in ``prepare_for_decode``."""
        if self.is_spectre() and getattr(server_args, "spectre_role", None) == "draft":
            return True
        if (
            self.is_standalone_remote()
            and getattr(server_args, "standalone_remote_role", None) == "draft"
        ):
            return True
        if (
            self.is_opt_tree_remote()
            and getattr(server_args, "opt_tree_role", None) == "draft"
        ):
            return True
        return False

    def guess_tokens_per_step(self, server_args) -> int:
        """Draft tokens counted in the acceptance-rate denominator.

        EAGLE-style algorithms store the target-sampled token inside
        ``speculative_num_draft_tokens``, so the guess count is ``n - 1``.
        OPT's denominator is the tree width N. Target's
        ``speculative_num_draft_tokens`` is the verify length N + 1.
        """
        if self.is_opt_tree_remote():
            return int(getattr(server_args, "opt_tree_width", 0) or 0)
        n = int(getattr(server_args, "speculative_num_draft_tokens", 0) or 0)
        return n - 1

    def reports_remote_spec_metrics(self) -> bool:
        """Zero-verify and fallback counts are part of the client totals."""
        return self.is_standalone_remote() or self.is_opt_tree_remote()

    def uses_spec_topk_cuda_graph_layout(self) -> bool:
        """Draft-tree CUDA graph pads tokens as ``bs * topk`` (EAGLE / STANDALONE / SR)."""
        return self.is_eagle() or self.is_standalone() or self.is_standalone_remote()

    def captures_target_verify_cuda_graph(
        self, server_args: "ServerArgs", is_draft_worker: bool = False
    ) -> bool:
        """Whether CudaGraphRunner should capture TARGET_VERIFY (not DECODE).

        STANDALONE_REMOTE and OPT_TREE_REMOTE Draft stay on DECODE so chain
        ``run_batch`` can replay ordinary decode graphs. Only the Target role
        captures verify.
        """
        if is_draft_worker:
            return False
        if self.is_eagle() or self.is_standalone() or self.is_ngram():
            return True
        if self.is_spectre() and getattr(server_args, "spectre_role", None) != "draft":
            return True
        if (
            self.is_standalone_remote()
            and getattr(server_args, "standalone_remote_role", None) == "target"
        ):
            return True
        if (
            self.is_opt_tree_remote()
            and getattr(server_args, "opt_tree_role", None) == "target"
        ):
            return True
        return False

    def uses_dual_ntpb_cuda_graph(
        self, server_args: "ServerArgs", is_draft_worker: bool = False
    ) -> bool:
        """Capture both TARGET_VERIFY and ntpb=1 DECODE graphs.

        Covers SPECTRE, STANDALONE_REMOTE Target, and OPT_TREE_REMOTE Target.
        OPT Target options are ``{1, N + 1}``. DECODE graphs cover 1-token AR.
        STANDALONE_REMOTE and OPT Draft stay on a single ordinary DECODE capture.
        """
        if is_draft_worker:
            return False
        if self.is_spectre() and getattr(server_args, "spectre_role", None) != "draft":
            return True
        if (
            self.is_standalone_remote()
            and getattr(server_args, "standalone_remote_role", None) == "target"
        ):
            return True
        if (
            self.is_opt_tree_remote()
            and getattr(server_args, "opt_tree_role", None) == "target"
        ):
            return True
        return False

    def dual_ntpb_cuda_graph_options(
        self, server_args: "ServerArgs", is_draft_worker: bool = False
    ) -> Optional[List[int]]:
        if not self.uses_dual_ntpb_cuda_graph(server_args, is_draft_worker):
            return None
        ntpb = self.target_verify_cuda_graph_num_tokens_per_bs(
            server_args, is_draft_worker
        )
        return sorted(set([1, ntpb]), reverse=True)

    def target_verify_cuda_graph_num_tokens_per_bs(
        self, server_args: "ServerArgs", is_draft_worker: bool = False
    ) -> int:
        """Tokens per sequence captured into the Target CUDA graph.

        TARGET_VERIFY graphs use ``speculative_num_draft_tokens``. DECODE
        (including STANDALONE_REMOTE Draft chain) stays at 1.
        """
        if self.captures_target_verify_cuda_graph(server_args, is_draft_worker):
            return int(
                getattr(server_args, "speculative_num_draft_tokens", 1) or 1
            )
        return 1

    def decode_cuda_graph_hidden_mode(
        self, server_args: "ServerArgs", is_draft_worker: bool = False
    ) -> int:
        """Hidden mode value for ordinary DECODE CUDA graphs.

        Returns a ``CaptureHiddenMode`` integer (NULL=0, LAST=1). STANDALONE_REMOTE
        Draft does not consume recurrent hidden for tree seeds, so ordinary
        DECODE graphs stay NULL. HTTP ``--enable-return-hidden-states`` still
        requests FULL through the resolver. In-process EAGLE draft workers and
        other algorithms also stay NULL.
        """
        return 0  # CaptureHiddenMode.NULL

    def supports_spec_v2(self) -> bool:
        return self.is_eagle() or self.is_standalone()

    def create_worker(self, server_args: ServerArgs) -> Optional[
        Union[
            Type[BaseSpecWorker],
            Type[TpModelWorker],
            Type[NGRAMWorker],
            Type[SpectreWorker],
            Type[StandaloneRemoteWorker],
        ]
    ]:
        assert (
            not self.is_none()
        ), "Cannot create worker for NONE speculative algorithm."

        enable_overlap = not server_args.disable_overlap_schedule
        if self.is_eagle() and server_args.enable_multi_layer_eagle:
            # FIXME: migrate to EagleWorker
            if enable_overlap:
                from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
                    MultiLayerEagleWorkerV2,
                )

                return MultiLayerEagleWorkerV2

            from sglang.srt.speculative.multi_layer_eagle_worker import (
                MultiLayerEagleWorker,
            )

            return MultiLayerEagleWorker

        elif self.is_eagle():
            if enable_overlap:
                from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

                return EAGLEWorkerV2

            from sglang.srt.speculative.eagle_worker import EAGLEWorker

            return EAGLEWorker
        elif self.is_standalone():
            if enable_overlap:
                from sglang.srt.speculative.standalone_worker_v2 import (
                    StandaloneWorkerV2,
                )

                return StandaloneWorkerV2

            from sglang.srt.speculative.standalone_worker import StandaloneWorker

            return StandaloneWorker
        elif self.is_ngram():
            if enable_overlap:
                raise ValueError(
                    f"Speculative algorithm {self.name} does not support overlap worker creation."
                )

            from sglang.srt.speculative.ngram_worker import NGRAMWorker

            return NGRAMWorker
        elif self.is_spectre():
            if enable_overlap:
                raise ValueError(
                    f"Speculative algorithm {self.name} does not support overlap worker creation."
                )

            from sglang.srt.speculative.spectre.verifier.spectre_worker import (
                SpectreWorker,
            )

            # Target-only verifier. SPECTRE draft skips create_worker in
            # Scheduler.maybe_init_draft_worker and uses TpModelWorker.
            return SpectreWorker
        elif self.is_standalone_remote():
            if enable_overlap:
                raise ValueError(
                    f"Speculative algorithm {self.name} does not support overlap worker creation."
                )

            from sglang.srt.speculative.standalone_remote.verifier.sr_worker import (
                StandaloneRemoteWorker,
            )

            # Target-only verifier. STANDALONE_REMOTE draft skips create_worker
            # in Scheduler.maybe_init_draft_worker and uses TpModelWorker.
            return StandaloneRemoteWorker
        elif self.is_opt_tree_remote():
            if enable_overlap:
                raise ValueError(
                    f"Speculative algorithm {self.name} does not support overlap worker creation."
                )
            if getattr(server_args, "opt_tree_role", None) != "target":
                raise ValueError(
                    "OPT_TREE_REMOTE draft uses TpModelWorker. "
                    "create_worker is only valid for --opt-tree-role target."
                )

            from sglang.srt.speculative.opt_tree_remote.opt_tree_worker import (
                OptTreeRemoteWorker,
            )

            return OptTreeRemoteWorker

        raise ValueError("Unreachable code path in create_worker.")


def resolve_cuda_graph_capture_hidden_mode(current, spec_info):
    """Hidden mode used while recording CUDA graphs.

    FULL is sticky (``--enable-return-hidden-states``). Otherwise keep
    ``current`` and raise to ``spec_info.capture_hidden_mode`` when present.
    A missing spec_info must not reset a captured mode to NULL.
    """
    full = 2  # CaptureHiddenMode.FULL
    if current is not None and int(current) >= full:
        return current
    spec_mode = (
        getattr(spec_info, "capture_hidden_mode", None)
        if spec_info is not None
        else None
    )
    if spec_mode is None:
        return current
    if current is None:
        return spec_mode
    return max(current, spec_mode)


def cuda_graph_hidden_mode_can_run(requested, captured) -> bool:
    """Whether ``CudaGraphRunner.can_run`` should accept this hidden-mode pair.

    Captured graphs can emulate weaker modes (``requested <= captured``).
    Stronger requests must not replay: ``recapture_if_needed`` is not reachable
    from ``can_run``, and forcing True made tree-draft capture replay DECODE
    graphs before ``raw_num_token`` existed.
    """
    return requested <= captured


def cuda_graph_hidden_mode_needs_recapture(requested, captured) -> bool:
    """True when replay must recapture because the request is stronger."""
    return requested > captured


def decode_cuda_graph_accepts_spec_info(spec_info) -> bool:
    """Ordinary DECODE graphs are 1-token AR.

    Tree ``EagleDraftInput`` and OPT layer verify must not replay them.
    OPT's ``seq_lens`` is the prefix; DECODE includes the token being written.
    """
    if spec_info is None:
        return True
    if getattr(spec_info, "forbids_decode_graph", False):
        return False
    is_draft = getattr(spec_info, "is_draft_input", None)
    if callable(is_draft) and is_draft():
        return False
    ntpb = getattr(spec_info, "num_tokens_per_req", None)
    if ntpb is not None and int(ntpb) > 1:
        return False
    return True


class SpecInputType(IntEnum):
    # NOTE: introduce this to distinguish the SpecInput types of multiple algorithms when asserting in attention backends.
    # If all algorithms can share the same datastrucutre of draft_input and verify_input, consider simplify it
    EAGLE_DRAFT = auto()
    EAGLE_VERIFY = auto()
    NGRAM_VERIFY = auto()


class SpecInput(ABC):
    def __init__(self, spec_input_type: SpecInputType):
        self.spec_input_type = spec_input_type

    def is_draft_input(self) -> bool:
        # FIXME: remove this function which is only used for assertion
        # or use another variable name like `draft_input` to substitute `spec_info`
        return self.spec_input_type == SpecInputType.EAGLE_DRAFT

    def is_verify_input(self) -> bool:
        return self.spec_input_type in {
            SpecInputType.EAGLE_VERIFY,
            SpecInputType.NGRAM_VERIFY,
        }

    @abstractmethod
    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        pass

    def get_spec_adjusted_global_num_tokens(
        self, forward_batch: ModelWorkerBatch
    ) -> Tuple[List[int], List[int]]:
        c1, c2 = self.get_spec_adjust_token_coefficient()
        global_num_tokens = [x * c1 for x in forward_batch.global_num_tokens]
        global_num_tokens_for_logprob = [
            x * c2 for x in forward_batch.global_num_tokens_for_logprob
        ]
        return global_num_tokens, global_num_tokens_for_logprob
