"""CPU tests for original OPT-Tree expand, stop, and parent-pointer layout.

The oracle ``_ReferenceTree`` is ``opt_classic_qwen3/tree.py`` ``Tree``, copied
here so the test does not import that repository.
"""

import unittest

import torch

from sglang.srt.speculative.opt_tree_remote.opt_tree_config import (
    OptTreeConfig,
    longest_draft_chain,
)
from sglang.srt.speculative.opt_tree_remote.opt_tree_layout import (
    build_tree_from_parents,
    pack_verify_draft_token,
)
from sglang.srt.speculative.opt_tree_remote.opt_tree_ops import (
    OptTreeState,
    _ranked_verify_indices,
    ancestor_mask,
    depth_rule,
    expand_layer,
    finish,
    seed,
    topk_probs_from_logits,
    update,
)
from sglang.srt.speculative.tree_attn_mask import iter_full_mask_rows
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=15, suite="stage-a-test-cpu")


class _ReferenceTree:
    """Copy of opt_classic_qwen3/tree.py Tree.update / add / initialize."""

    def __init__(self, nnodes, device, threshold, max_depth):
        self.nnodes = nnodes
        self.device = device
        self.threshold = threshold
        self.depth = 0
        self.weight = 0
        self.max_depth = max_depth
        self.weight_matrix = torch.zeros([max_depth, self.nnodes], device=self.device)
        self.input_ids_matrix = torch.zeros(
            [max_depth, self.nnodes], dtype=torch.long, device=self.device
        )
        self.parents_matrix = torch.zeros(
            [max_depth, self.nnodes], dtype=torch.long, device=self.device
        )
        self.kv_mask = torch.zeros([self.nnodes, 0], dtype=torch.int8, device=self.device)
        self.tri = torch.eye(self.nnodes, dtype=torch.int8, device=self.device)
        self.rows = torch.arange(self.nnodes, device=self.device)
        self.position_id = torch.zeros([self.nnodes], dtype=torch.long, device=self.device)
        self.kv_cache_mask = torch.zeros(
            [self.nnodes, self.nnodes], dtype=torch.int8, device=self.device
        )
        self.last_draft_mask = self.tri.clone()

    def initialize(self, logits):
        logits, ids = torch.topk(logits[0][-1], k=self.nnodes, dim=-1)
        self.weight_matrix[self.depth].copy_(logits)
        self.input_ids_matrix[self.depth].copy_(ids)
        self.parents_matrix[0].copy_(self.rows)
        output_dict = {
            "input_ids": ids,
            "position_ids": self.position_id + 1,
            "attention_mask": self.tri,
            "parent_last": self.rows,
            "is_final": False,
        }
        self.last_draft_mask = output_dict["attention_mask"].clone()
        self.depth += 1
        return output_dict

    def add(self, logits):
        logits, ids = torch.topk(logits[0], k=self.nnodes, dim=-1)
        last_layer_weights = self.weight_matrix[self.depth - 1].unsqueeze(1)
        logits = logits * last_layer_weights
        flat_logits, flat_ids = logits.view(-1), ids.view(-1)
        global_top_logits, global_top_idx = torch.topk(flat_logits, k=self.nnodes, dim=-1)
        input_ids = flat_ids[global_top_idx]
        parents = global_top_idx // self.nnodes
        self.parents_matrix[self.depth].copy_(parents)
        self.weight_matrix[self.depth].copy_(global_top_logits)
        self.input_ids_matrix[self.depth].copy_(input_ids)
        self.kv_cache_mask[self.rows, parents] = 1
        self.kv_mask = torch.cat([self.kv_mask[parents], self.kv_cache_mask], dim=1)
        self.kv_cache_mask[self.rows, parents] = 1
        attention_mask = torch.cat([self.kv_mask, self.tri], dim=1)
        output_dict = {
            "input_ids": input_ids,
            "position_ids": self.position_id + (self.depth + 1),
            "attention_mask": attention_mask,
            "parent_last": parents,
            "is_final": False,
        }
        self.last_draft_mask = attention_mask.clone()
        self.depth += 1
        return output_dict

    def generate_attention_mask(self, parents):
        attention_mask = self.tri.clone()
        grandp = parents.clone()
        for _ in range(self.depth - 2):
            attention_mask[self.rows, grandp] = 1
            grandp = grandp[parents]
        return attention_mask

    def update(self, logits):
        if self.depth == 0:
            return self.initialize(logits)
        outputs = self.add(logits)
        top_weights, top_index = torch.topk(
            self.weight_matrix[: self.depth - 1].view(-1), k=self.nnodes, dim=-1
        )
        weight = top_weights.sum()
        if weight - self.weight > self.threshold and self.depth < self.max_depth:
            self.weight = weight
        else:
            rows = top_index // self.nnodes
            cols = top_index % self.nnodes
            sorted_indices = torch.argsort(rows)
            rows = rows[sorted_indices]
            cols = cols[sorted_indices]
            index = torch.arange(self.nnodes * (self.depth - 1), device=self.device).view(
                -1, self.nnodes
            )
            input_ids = self.input_ids_matrix[rows, cols]
            position_ids = rows + 1
            parents = self.parents_matrix[rows, cols]
            source_idx = index[rows, cols]
            parent_idx = index[torch.clamp_min(rows - 1, 0), parents]
            parents_index = torch.nonzero(
                source_idx.unsqueeze(0) == parent_idx.unsqueeze(1)
            )[:, 1]
            attention_mask = self.generate_attention_mask(parents_index)
            outputs = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "parent_last": parents_index,
                "is_final": True,
            }
        return outputs

    def draft_mask(self):
        """Mask from the last seed or ``add``, before stop rewrites ``tri``."""
        return self.last_draft_mask


def _node_pairs(tokens, positions):
    return sorted(
        (int(tok), int(pos))
        for tok, pos in zip(tokens.reshape(-1).tolist(), positions.reshape(-1).tolist())
    )


def _verify_triples(verify):
    """(token, depth, parent column). Layer 0's parent column is -1."""
    cols = verify.cols.tolist()
    triples = []
    for i, (tok, depth, parent) in enumerate(
        zip(verify.tokens.tolist(), verify.depths.tolist(), verify.parents.tolist())
    ):
        parent_col = -1 if int(depth) == 0 else int(cols[parent])
        triples.append((int(tok), int(depth), parent_col))
    return sorted(triples)


def _reference_triples(ref):
    """Same total order, with parent columns read from the reference matrix."""
    n = ref.nnodes
    layer_count = ref.depth - 1
    flat = ref.weight_matrix[:layer_count].reshape(-1)
    selected = _ranked_verify_indices(flat, layer_count, n)
    triples = []
    for idx in selected.tolist():
        row, col = divmod(int(idx), n)
        token = int(ref.input_ids_matrix[row, col])
        parent_col = -1 if row == 0 else int(ref.parents_matrix[row, col])
        triples.append((token, row, parent_col))
    return sorted(triples)


class OptTreeRemoteTest(CustomTestCase):
    def test_config_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            OptTreeConfig(width=0, tau=0.1, max_depth=4)
        with self.assertRaises(ValueError):
            OptTreeConfig(width=4, tau=0.1, max_depth=1)
        with self.assertRaises(ValueError):
            OptTreeConfig(width=4, tau=float("nan"), max_depth=4)
        config = OptTreeConfig(width=4, tau=0.5, max_depth=10)
        self.assertEqual(longest_draft_chain(config), 9)

    def test_topk_probs_match_full_softmax(self):
        logits = torch.tensor(
            [
                [0.2, 1.5, -0.4, 3.0, 0.1, 2.2],
                [4.0, 0.0, 0.5, -1.0, 1.0, 0.25],
            ]
        )
        got_p, got_i = topk_probs_from_logits(logits, k=3)
        ref = torch.softmax(logits, dim=-1)
        ref_p, ref_i = torch.topk(ref, k=3, dim=-1)
        self.assertTrue(torch.equal(got_i, ref_i))
        self.assertTrue(torch.allclose(got_p, ref_p, atol=1e-5, rtol=0))

    def test_random_rounds_match_reference_tree(self):
        config = OptTreeConfig(width=4, tau=0.05, max_depth=6)
        for seed in range(5):
            gen = torch.Generator().manual_seed(seed)
            ref = _ReferenceTree(config.width, "cpu", config.tau, config.max_depth)
            our = OptTreeState.create(config)
            vocab = 16
            our_out = None
            ref_out = None
            for step in range(config.max_depth + 1):
                if ref.depth == 0:
                    logits = torch.randn(1, 1, vocab, generator=gen)
                else:
                    logits = torch.randn(1, config.width, vocab, generator=gen)
                probs = torch.softmax(logits, dim=-1)
                if our_out is None or not our_out.is_final:
                    our_out = update(our, probs)
                ref_out = ref.update(probs)
                if not our_out.is_final:
                    self.assertFalse(
                        ref_out["is_final"], msg=f"seed {seed} step {step}"
                    )
                    self.assertTrue(torch.equal(our_out.tokens, ref_out["input_ids"]))
                    self.assertTrue(torch.equal(our_out.positions, ref_out["position_ids"]))
                    self.assertTrue(torch.equal(our_out.leak_mask, ref_out["attention_mask"]))
                    self.assertTrue(
                        torch.equal(our_out.ancestor_mask, ancestor_mask(our, leak=False))
                    )
                    self.assertTrue(torch.equal(our.tokens[our.depth - 1], our_out.tokens))
                if ref_out["is_final"]:
                    break
            else:
                self.fail(f"seed {seed} did not stop")
            self.assertIsNotNone(our_out)
            self.assertTrue(our_out.is_final)
            self.assertTrue(ref_out["is_final"])
            self.assertEqual(our.depth, ref.depth - 1)
            self.assertEqual(
                _node_pairs(our_out.verify.tokens, our_out.verify.positions),
                _node_pairs(ref_out["input_ids"], ref_out["position_ids"]),
            )
            self.assertEqual(_verify_triples(our_out.verify), _reference_triples(ref))
            self.assertTrue(torch.equal(our_out.positions, our_out.verify.positions))
            self.assertTrue((our_out.verify.depths < our.depth).all())
            for i, parent in enumerate(our_out.verify.parents.tolist()):
                self.assertTrue(parent == -1 or parent < i)

    def test_continue_until_cap_keeps_filled_layers(self):
        config = OptTreeConfig(width=2, tau=-1.0, max_depth=4)
        gen = torch.Generator().manual_seed(7)
        vocab = 8
        state = OptTreeState.create(config)
        seed_probs = torch.softmax(torch.randn(1, 1, vocab, generator=gen), dim=-1)
        first = update(state, seed_probs)
        self.assertFalse(first.is_final)
        self.assertEqual(state.depth, 1)
        self.assertTrue(torch.equal(first.positions, torch.ones(config.width, dtype=torch.long)))
        self.assertTrue(torch.equal(state.tokens[0], first.tokens))

        grow_probs = torch.softmax(torch.randn(1, config.width, vocab, generator=gen), dim=-1)
        second = update(state, grow_probs)
        self.assertFalse(second.is_final)
        self.assertEqual(state.depth, 2)
        self.assertTrue(
            torch.equal(second.positions, torch.full((config.width,), 2, dtype=torch.long))
        )
        self.assertTrue(torch.equal(state.tokens[1], second.tokens))
        kept_tokens = state.tokens[1].clone()
        kept_weight = state.weight[1].clone()

        while not state.finished:
            probs = torch.softmax(torch.randn(1, config.width, vocab, generator=gen), dim=-1)
            step = update(state, probs)
        self.assertTrue(step.is_final)
        self.assertEqual(state.depth, config.max_depth - 1)
        self.assertTrue(torch.equal(state.tokens[1], kept_tokens))
        self.assertTrue(torch.equal(state.weight[1], kept_weight))
        self.assertTrue((step.verify.depths <= config.max_depth - 2).all())
        self.assertNotIn(config.max_depth - 1, step.verify.depths.tolist())

    def test_nonfinal_leak_mask_matches_reference(self):
        config = OptTreeConfig(width=2, tau=-1.0, max_depth=6)
        seed_probs = torch.tensor([[[0.50, 0.40, 0.05, 0.05]]], dtype=torch.float32)
        expand1 = torch.tensor(
            [[[0.05, 0.05, 0.60, 0.30], [0.40, 0.05, 0.50, 0.05]]],
            dtype=torch.float32,
        )
        expand2 = torch.tensor(
            [[[0.50, 0.49, 0.005, 0.005], [0.10, 0.10, 0.10, 0.70]]],
            dtype=torch.float32,
        )
        ref = _ReferenceTree(config.width, "cpu", config.tau, config.max_depth)
        state = OptTreeState.create(config)
        saw_stale = False
        for probs in (seed_probs, expand1, expand2):
            ref_out = ref.update(probs)
            our = update(state, probs)
            self.assertFalse(our.is_final)
            self.assertFalse(ref_out["is_final"])
            self.assertTrue(torch.equal(our.leak_mask, ref_out["attention_mask"]))
            self.assertTrue(
                torch.equal(our.ancestor_mask, ancestor_mask(state, leak=False))
            )
            if not torch.equal(our.leak_mask, our.ancestor_mask):
                saw_stale = True
                stale = (our.leak_mask == 1) & (our.ancestor_mask == 0)
                self.assertGreater(int(stale.sum()), 0)
                self.assertEqual(int((our.ancestor_mask == 1).sum()), 2 * 3)
        self.assertTrue(saw_stale)

    def test_tied_scores_keep_parent_and_do_not_insert_weaker_node(self):
        config = OptTreeConfig(width=3, tau=0.0, max_depth=4)
        state = OptTreeState.create(config)
        state.depth = 2
        state.weight[0] = torch.tensor([0.5, 0.5, 0.2])
        state.weight[1] = torch.tensor([0.5, 0.1, 0.05])
        state.tokens[0] = torch.tensor([10, 11, 12])
        state.tokens[1] = torch.tensor([20, 21, 22])
        # (1, 0) ties its parent (0, 1). The 0.2 node must not replace the child.
        state.parents[1] = torch.tensor([1, 0, 0])
        verify = finish(state)
        self.assertEqual(int(verify.tokens.numel()), config.width)
        self.assertEqual(
            sorted(
                zip(
                    verify.tokens.tolist(),
                    verify.depths.tolist(),
                    verify.cols.tolist(),
                )
            ),
            [(10, 0, 0), (11, 0, 1), (20, 1, 0)],
        )
        self.assertNotIn(12, verify.tokens.tolist())
        self.assertIn(11, verify.tokens.tolist())
        self.assertIn(20, verify.tokens.tolist())

    def test_depth_rule_does_not_change_prev_e(self):
        config = OptTreeConfig(width=2, tau=0.05, max_depth=6)
        state = OptTreeState.create(config)
        seed(state, torch.tensor([0.5, 0.4, 0.1]))
        expand_layer(
            state,
            torch.tensor([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1]]),
        )
        before = state.prev_e.clone()
        first = depth_rule(state)
        second = depth_rule(state)
        self.assertTrue(first)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(state.prev_e, before))

    def test_layout_from_parents_rebuilds_mask_and_retrive(self):
        parents = torch.tensor(
            [
                [-1, 0],
                [-1, -1],
            ],
            dtype=torch.long,
        )
        seq_lens = torch.tensor([5, 2], dtype=torch.long)
        layout = build_tree_from_parents(parents, seq_lens)
        self.assertEqual(layout.num_verify_tokens, 3)
        self.assertTrue(
            torch.equal(
                layout.positions,
                torch.tensor([5, 6, 7, 2, 3, 3], dtype=torch.long),
            )
        )
        rows = {
            (b, t): row.tolist()
            for b, t, row in iter_full_mask_rows(layout.tree_mask, [5, 2], 3)
        }
        self.assertEqual(rows[(0, 0)], [True] * 5 + [True, False, False])
        self.assertEqual(rows[(0, 1)], [True] * 5 + [True, True, False])
        self.assertEqual(rows[(0, 2)], [True] * 5 + [True, True, True])
        self.assertEqual(rows[(1, 0)], [True, True, True, False, False])
        self.assertEqual(rows[(1, 1)], [True, True, True, True, False])
        self.assertEqual(rows[(1, 2)], [True, True, True, False, True])
        self.assertEqual(layout.retrive_next_token[0].tolist(), [1, 2, -1])
        self.assertEqual(layout.retrive_next_sibling[0].tolist(), [-1, -1, -1])
        self.assertEqual(layout.retrive_next_token[1].tolist(), [1, -1, -1])
        self.assertEqual(layout.retrive_next_sibling[1].tolist(), [-1, 2, -1])
        self.assertTrue(torch.equal(layout.retrive_index[1], torch.tensor([3, 4, 5])))

        with self.assertRaises(ValueError):
            build_tree_from_parents(torch.tensor([0, -1]), seq_lens[:1])
        with self.assertRaises(ValueError):
            build_tree_from_parents(torch.tensor([[1, -1]]), seq_lens[:1])

    def test_finished_verify_parents_roundtrip_layout(self):
        config = OptTreeConfig(width=3, tau=0.0, max_depth=5)
        gen = torch.Generator().manual_seed(11)
        state = OptTreeState.create(config)
        vocab = 12
        probs = torch.softmax(torch.randn(1, 1, vocab, generator=gen), dim=-1)
        step = update(state, probs)
        while not step.is_final:
            probs = torch.softmax(
                torch.randn(1, config.width, vocab, generator=gen), dim=-1
            )
            step = update(state, probs)
        verify = step.verify
        for i, parent in enumerate(verify.parents.tolist()):
            self.assertTrue(parent == -1 or parent < i)
        seq_len = 9
        layout = build_tree_from_parents(verify.parents, torch.tensor([seq_len]))
        self.assertEqual(layout.num_verify_tokens, config.width + 1)
        children = {}
        for parent in range(layout.num_verify_tokens):
            child = int(layout.retrive_next_token[0, parent])
            while child != -1:
                children[child] = parent
                child = int(layout.retrive_next_sibling[0, child])
        for i, parent in enumerate(verify.parents.tolist()):
            verify_i = i + 1
            if parent == -1:
                self.assertEqual(children[verify_i], 0)
            else:
                self.assertEqual(children[verify_i], parent + 1)
        rows = list(
            iter_full_mask_rows(layout.tree_mask, [seq_len], layout.num_verify_tokens)
        )
        for i in range(config.width):
            _, _, row = rows[i + 1]
            expect = [False] * layout.num_verify_tokens
            node = i + 1
            seen = set()
            while node >= 0 and node not in seen:
                seen.add(node)
                expect[node] = True
                if node == 0:
                    break
                node = children[node]
            self.assertEqual(row[seq_len:].tolist(), expect)
            self.assertTrue(all(row[:seq_len].tolist()))

    def test_grow_stops_before_the_extra_layer(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_grow import grow_opt_tree

        config = OptTreeConfig(width=2, tau=0.5, max_depth=2)
        seen = []

        def next_probs(step, state):
            seen.append(None if step is None else [int(x) for x in step.tokens.tolist()])
            if step is None:
                probs = torch.zeros(8)
                probs[1] = 0.6
                probs[2] = 0.3
                probs[3] = 0.1
                return probs
            return torch.full((config.width, 8), 1.0 / 8)

        tokens, parents, cols = grow_opt_tree(config, next_probs)
        self.assertEqual(seen, [None])
        self.assertEqual(tokens, [1, 2])
        self.assertEqual(parents, [-1, -1])
        self.assertEqual(len(cols), config.width)

    def test_verify_draft_token_is_flat_for_kv_alloc(self):
        width = 4
        roots = [11, 21]
        token_rows = [[1, 2, 3, 4], [5, 6, 7, 8]]
        packed = pack_verify_draft_token(roots, token_rows)
        num_verify = width + 1
        self.assertEqual(packed.dim(), 1)
        self.assertTrue(packed.is_contiguous())
        self.assertEqual(packed.numel(), len(roots) * num_verify)
        self.assertEqual(len(packed), len(roots) * num_verify)
        self.assertEqual(packed.view(len(roots), num_verify)[:, 0].tolist(), roots)
        rectangular = packed.view(len(roots), num_verify)
        self.assertEqual(len(rectangular), len(roots))
        self.assertEqual(rectangular.numel(), packed.numel())

    def test_algorithm_is_not_standalone_remote(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        algo = SpeculativeAlgorithm.from_string("OPT_TREE_REMOTE")
        self.assertTrue(algo.is_opt_tree_remote())
        self.assertFalse(algo.is_standalone_remote())

    def test_cli_registers_opt_tree_remote(self):
        from pathlib import Path

        text = Path("python/sglang/srt/server_args.py").read_text()
        self.assertIn('"OPT_TREE_REMOTE"', text)
        for flag in (
            '"--opt-tree-role"',
            '"--opt-tree-width"',
            '"--opt-tree-tau"',
            '"--opt-tree-max-depth"',
            '"--opt-tree-addr"',
            '"--opt-tree-port"',
        ):
            self.assertIn(flag, text)
        scheduler = Path("python/sglang/srt/managers/scheduler.py").read_text()
        sr_pos = scheduler.find("event_loop_normal_standalone_remote_draft")
        opt_pos = scheduler.find("event_loop_normal_opt_tree_draft")
        self.assertGreater(sr_pos, 0)
        self.assertGreater(opt_pos, sr_pos)
        self.assertIn("event_loop_normal_opt_tree_target", scheduler)
        self.assertIn("is_opt_tree_remote()", scheduler)
        self.assertIn("grow_opt_tree", Path(
            "python/sglang/srt/speculative/opt_tree_remote/opt_tree_forward.py"
        ).read_text())

    def test_abort_request_has_no_sr_target_abort_name(self):
        from pathlib import Path

        scheduler = Path("python/sglang/srt/managers/scheduler.py").read_text()
        self.assertNotIn("sr_target_abort", scheduler)
        self.assertIn("maybe_notify_remote_draft_aborted", scheduler)
        mixin = Path(
            "python/sglang/srt/speculative/standalone_remote/verifier/"
            "sr_target_scheduler_mixin.py"
        ).read_text()
        self.assertIn("def maybe_notify_remote_draft_aborted", mixin)
        opt = Path(
            "python/sglang/srt/speculative/opt_tree_remote/opt_tree_scheduler.py"
        ).read_text()
        self.assertIn("def maybe_notify_remote_draft_aborted", opt)

    def _opt_server_args(self, **overrides):
        from types import SimpleNamespace

        args = SimpleNamespace(
            speculative_verify_mode="auto",
            speculative_num_draft_tokens=5,
            speculative_num_steps=10,
            speculative_eagle_topk=None,
            opt_tree_role="target",
            opt_tree_width=4,
            opt_tree_tau=0.5,
            opt_tree_max_depth=10,
            page_size=1,
            disable_overlap_schedule=False,
            enable_mixed_chunk=True,
            max_running_requests=None,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_opt_tree_validation(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_args import (
            apply_opt_tree_settings,
        )

        ok = self._opt_server_args()
        apply_opt_tree_settings(ok, on_npu=False)
        self.assertTrue(ok.disable_overlap_schedule)
        self.assertFalse(ok.enable_mixed_chunk)
        self.assertEqual(ok.max_running_requests, 48)
        self.assertEqual(ok.speculative_eagle_topk, 4)
        self.assertEqual(ok.speculative_num_draft_tokens, ok.opt_tree_width + 1)

        rpd = self._opt_server_args(speculative_verify_mode="rpd")
        with self.assertRaises(ValueError):
            apply_opt_tree_settings(rpd, on_npu=False)

        target_short = self._opt_server_args(speculative_num_draft_tokens=4)
        with self.assertRaises(ValueError):
            apply_opt_tree_settings(target_short, on_npu=False)

        target_width = self._opt_server_args(
            opt_tree_width=3, speculative_num_draft_tokens=4
        )
        apply_opt_tree_settings(target_width, on_npu=False)
        self.assertEqual(target_width.speculative_num_draft_tokens, 4)

        draft_long = self._opt_server_args(
            opt_tree_role="draft", speculative_num_draft_tokens=5
        )
        with self.assertRaises(ValueError):
            apply_opt_tree_settings(draft_long, on_npu=False)

        draft_ok = self._opt_server_args(
            opt_tree_role="draft", speculative_num_draft_tokens=4
        )
        apply_opt_tree_settings(draft_ok, on_npu=False)
        self.assertEqual(draft_ok.speculative_num_draft_tokens, draft_ok.opt_tree_width)

        depth = self._opt_server_args(opt_tree_max_depth=9)
        with self.assertRaises(ValueError):
            apply_opt_tree_settings(depth, on_npu=False)

        cuda_page = self._opt_server_args(page_size=128)
        with self.assertRaises(ValueError):
            apply_opt_tree_settings(cuda_page, on_npu=False)

        npu = self._opt_server_args(page_size=128)
        apply_opt_tree_settings(npu, on_npu=True)
        npu_page = self._opt_server_args(page_size=1)
        with self.assertRaises(ValueError):
            apply_opt_tree_settings(npu_page, on_npu=True)

        filled = self._opt_server_args(
            speculative_num_draft_tokens=None,
            speculative_num_steps=None,
        )
        apply_opt_tree_settings(filled, on_npu=False)
        self.assertEqual(
            filled.speculative_num_draft_tokens, filled.opt_tree_width + 1
        )
        self.assertEqual(filled.speculative_num_steps, filled.opt_tree_max_depth)

        filled_draft = self._opt_server_args(
            opt_tree_role="draft",
            speculative_num_draft_tokens=None,
            speculative_num_steps=None,
        )
        apply_opt_tree_settings(filled_draft, on_npu=False)
        self.assertEqual(
            filled_draft.speculative_num_draft_tokens, filled_draft.opt_tree_width
        )
        self.assertEqual(
            filled_draft.speculative_num_steps, filled_draft.opt_tree_max_depth
        )

    def test_local_spec_shape_opt_uses_tree_width(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.srt.speculative.standalone_remote.sr_protocol import (
            local_spec_shape,
        )

        args = SimpleNamespace(
            speculative_algorithm="OPT_TREE_REMOTE",
            speculative_eagle_topk=8,
            speculative_num_steps=7,
            speculative_num_draft_tokens=5,
            opt_tree_width=4,
            opt_tree_max_depth=10,
        )
        self.assertEqual(local_spec_shape(args), (4, 10, 4))
        enum_args = SimpleNamespace(
            speculative_algorithm=SpeculativeAlgorithm.OPT_TREE_REMOTE,
            speculative_eagle_topk=8,
            speculative_num_steps=7,
            speculative_num_draft_tokens=5,
            opt_tree_width=4,
            opt_tree_max_depth=10,
        )
        self.assertEqual(local_spec_shape(enum_args), (4, 10, 4))
        self.assertEqual(
            local_spec_shape(
                SimpleNamespace(
                    speculative_algorithm="STANDALONE_REMOTE",
                    speculative_eagle_topk=4,
                    speculative_num_steps=10,
                    speculative_num_draft_tokens=5,
                    opt_tree_width=4,
                    opt_tree_max_depth=10,
                )
            ),
            (4, 10, 5),
        )

    def test_guess_tokens_per_step_uses_opt_width(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        args = SimpleNamespace(
            speculative_num_draft_tokens=5,
            opt_tree_width=4,
        )
        self.assertEqual(
            SpeculativeAlgorithm.OPT_TREE_REMOTE.guess_tokens_per_step(args), 4
        )
        eagle = SimpleNamespace(speculative_num_draft_tokens=5, opt_tree_width=4)
        self.assertEqual(SpeculativeAlgorithm.EAGLE.guess_tokens_per_step(eagle), 4)

    def test_opt_target_captures_verify_cuda_graph(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        opt = SpeculativeAlgorithm.OPT_TREE_REMOTE
        width = 4
        target = SimpleNamespace(
            opt_tree_role="target",
            speculative_num_draft_tokens=width + 1,
            spectre_role=None,
            standalone_remote_role=None,
        )
        self.assertTrue(opt.captures_target_verify_cuda_graph(target))
        self.assertTrue(opt.uses_dual_ntpb_cuda_graph(target))
        self.assertEqual(opt.dual_ntpb_cuda_graph_options(target), [width + 1, 1])
        self.assertEqual(
            opt.target_verify_cuda_graph_num_tokens_per_bs(target), width + 1
        )
        self.assertFalse(
            opt.captures_target_verify_cuda_graph(target, is_draft_worker=True)
        )
        self.assertFalse(
            opt.uses_dual_ntpb_cuda_graph(target, is_draft_worker=True)
        )
        self.assertEqual(
            opt.target_verify_cuda_graph_num_tokens_per_bs(
                target, is_draft_worker=True
            ),
            1,
        )

        draft = SimpleNamespace(
            opt_tree_role="draft",
            speculative_num_draft_tokens=width,
            spectre_role=None,
            standalone_remote_role=None,
        )
        self.assertFalse(opt.captures_target_verify_cuda_graph(draft))
        self.assertFalse(opt.uses_dual_ntpb_cuda_graph(draft))
        self.assertIsNone(opt.dual_ntpb_cuda_graph_options(draft))
        self.assertEqual(opt.target_verify_cuda_graph_num_tokens_per_bs(draft), 1)

    def test_extend_past_prompt_mrope_keeps_token_width(self):
        import ast
        from pathlib import Path
        from types import SimpleNamespace

        from sglang.srt.speculative.standalone_remote.drafter.sr_tail_extend import (
            tail_mrope_positions,
        )

        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/model_executor/forward_batch_info.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(
            child
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ForwardBatch"
            for child in node.body
            if isinstance(child, ast.FunctionDef)
            and child.name == "_compute_mrope_positions"
        )
        for arg in (*fn.args.args, *fn.args.kwonlyargs):
            arg.annotation = None
        fn.returns = None
        namespace = {
            "torch": torch,
            "get_global_server_args": lambda: SimpleNamespace(
                rl_on_policy_target=None
            ),
        }
        exec(compile(ast.Module([fn], type_ignores=[]), str(source), "exec"), namespace)

        class _ExtendMode:
            def is_decode(self):
                return False

            def is_extend(self, include_draft_extend_v2=False):
                return True

        prompt_width = 10
        extend_len = 4
        mm = SimpleNamespace(
            mrope_positions=torch.arange(prompt_width).unsqueeze(0).repeat(3, 1),
            mrope_position_delta=torch.tensor([2]),
        )
        fb = SimpleNamespace(
            seq_lens_cpu=torch.tensor([prompt_width + extend_len]),
            forward_mode=_ExtendMode(),
            is_sr_tail_extend=False,
            input_ids=torch.arange(extend_len, dtype=torch.int64),
        )
        batch = SimpleNamespace(
            multimodal_inputs=[mm],
            extend_seq_lens=[extend_len],
            extend_prefix_lens=[prompt_width],
        )
        namespace["_compute_mrope_positions"](
            fb, SimpleNamespace(device="cpu"), batch
        )
        expected = tail_mrope_positions(mm, prompt_width, extend_len)
        self.assertEqual(tuple(fb.mrope_positions.shape), (3, extend_len))
        self.assertTrue(torch.equal(fb.mrope_positions.cpu(), expected))

    def test_layer_positions_and_mask(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            layer_attention_mask,
            layer_positions,
        )

        first = layer_positions(10, 1, 4)
        second = layer_positions(10, 2, 4)
        self.assertTrue(torch.equal(first, torch.full((4,), 10)))
        self.assertTrue(torch.equal(second, torch.full((4,), 11)))

        ancestor = torch.tensor([[1, 0], [0, 1]], dtype=torch.int8)
        mask = layer_attention_mask(3, ancestor)
        self.assertEqual(mask.numel(), 2 * (3 + 2))
        rows = mask.view(2, 5)
        self.assertTrue(torch.equal(rows[:, :3], torch.ones(2, 3, dtype=torch.bool)))
        self.assertTrue(torch.equal(rows[:, 3:], ancestor.to(dtype=torch.bool)))

    def test_forward_once_per_kept_layer(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            grow_with_forwards,
        )

        config = OptTreeConfig(width=2, tau=-1.0, max_depth=4)
        calls = []

        def forward_layer(step):
            self.assertFalse(step.is_final)
            self.assertEqual(int(step.tokens.shape[0]), config.width)
            self.assertEqual(int(step.ancestor_mask.shape[0]), config.width)
            self.assertEqual(int(step.positions.min()), int(step.positions.max()))
            calls.append(
                (
                    int(step.positions[0]),
                    step.tokens.clone(),
                    step.ancestor_mask.clone(),
                )
            )
            return torch.randn(config.width, 16)

        grow_with_forwards(config, torch.randn(16), forward_layer)
        self.assertEqual(len(calls), config.max_depth - 2)
        self.assertEqual([depth for depth, _, _ in calls], [1, 2])
        self.assertTrue((calls[-1][2] == 0).any())

        stopped = []

        def forward_once(step):
            stopped.append(step.ancestor_mask.clone())
            return torch.rand(2, 16)

        grow_with_forwards(
            OptTreeConfig(width=2, tau=1.0, max_depth=10),
            torch.rand(16),
            forward_once,
        )
        self.assertEqual(len(stopped), 0)

    def test_prefill_materializes_missing_pool(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            produce_opt_windows,
        )
        from sglang.srt.speculative.standalone_remote.sr_protocol import SRAction

        host = _OptWindowHost(assign_pool=True)
        req = _OptReq()
        windows = produce_opt_windows(host, SRAction.PREFILL, [(None, req)])
        self.assertEqual(host.calls, ["materialize"])
        self.assertEqual(windows, [([], None, None)])
        self.assertEqual(req.req_pool_idx, 3)

        bare = _OptWindowHost(assign_pool=False)
        with self.assertRaises(RuntimeError):
            produce_opt_windows(bare, SRAction.PREFILL, [(None, _OptReq())])
        self.assertEqual(bare.calls, ["materialize"])

    def test_step_materializes_before_expand(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            produce_opt_windows,
        )
        from sglang.srt.speculative.standalone_remote.sr_protocol import SRAction

        missing = _OptReq()
        host = _OptWindowHost(assign_pool=True)
        windows = produce_opt_windows(host, SRAction.STEP, [(None, missing)])
        self.assertEqual(host.calls, ["materialize", "grammar", "expand"])
        self.assertEqual(windows, [([7], [-1], [0])])

        ready = _OptReq()
        ready.req_pool_idx = 2
        again = _OptWindowHost(assign_pool=True)
        produce_opt_windows(again, SRAction.STEP, [(None, ready)])
        self.assertEqual(again.calls, ["grammar", "expand"])

        blocked = _OptWindowHost(assign_pool=False)
        with self.assertRaises(RuntimeError):
            produce_opt_windows(blocked, SRAction.STEP, [(None, _OptReq())])
        self.assertEqual(blocked.calls, ["materialize"])

    def test_layer_spec_forbids_decode_graph_and_keeps_raw_logits(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            OptLayerSpec,
            _keep_raw_logits,
        )

        self.assertTrue(OptLayerSpec.forbids_decode_graph)
        info = SimpleNamespace(
            is_all_greedy=False,
            need_top_p_sampling=True,
            need_top_k_sampling=True,
            need_min_p_sampling=True,
            temperatures=torch.tensor([0.7]),
            acc_additive_penalties=torch.ones(1),
            acc_scaling_penalties=torch.ones(1),
            penalizer_orchestrator=object(),
            logit_bias=torch.ones(1),
            vocab_mask=torch.ones(1),
        )
        batch = SimpleNamespace(
            sampling_info=info,
            has_grammar=True,
            return_logprob=True,
            return_hidden_states=True,
        )
        _keep_raw_logits(batch)
        self.assertIsNone(info.acc_additive_penalties)
        self.assertIsNone(info.acc_scaling_penalties)
        self.assertIsNone(info.penalizer_orchestrator)
        self.assertIsNone(info.logit_bias)
        self.assertIsNone(info.vocab_mask)
        self.assertFalse(batch.has_grammar)


class _OptReq:
    def __init__(self):
        self.rid = "opt-req"
        self.req_pool_idx = None
        self.origin_input_ids = [1, 2, 3]
        self.output_ids = []


class _OptWindowHost:
    def __init__(self, assign_pool):
        self.assign_pool = assign_pool
        self.calls = []

    def _sr_materialize_prefix_batch(self, reqs):
        self.calls.append("materialize")
        if not self.assign_pool:
            return
        for req in reqs:
            req.req_pool_idx = 3

    def _sr_replay_grammars(self, reqs):
        self.calls.append("grammar")

    def _opt_expand_one(self, req):
        self.calls.append("expand")
        return [7], [-1], [0]


if __name__ == "__main__":
    unittest.main()
