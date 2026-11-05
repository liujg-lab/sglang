"""CPU tests for original OPT-Tree expand, stop, and parent-pointer layout.

The oracle ``_ReferenceTree`` is ``opt_classic_qwen3/tree.py`` ``Tree``, copied
here so the test does not import that repository.
"""

import unittest
from types import SimpleNamespace

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
    _pack_scores_and_ids,
    _ranked_verify_indices,
    ancestor_mask,
    depth_rule,
    expand_layer,
    finish,
    seed,
    topk_probs_from_logits,
    update,
)
from sglang.srt.speculative.tree_attn_mask import full_mask_numel, iter_full_mask_rows
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


def _reference_ancestor_mask(state):
    """Parent walk that ``ancestor_mask(leak=False)`` used to run on device."""
    n = state.config.width
    d = state.depth
    parents = state.parents[:d].tolist()
    mask = torch.zeros(n, d * n, dtype=torch.int8)
    for i in range(n):
        row = d - 1
        col = i
        while True:
            mask[i, row * n + col] = 1
            if row == 0:
                break
            col = int(parents[row][col])
            row -= 1
    return mask


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

    def test_score_id_pack_keeps_int64_ids(self):
        scores = torch.tensor([0.5, 0.25, 0.125], dtype=torch.float32)
        ids = torch.tensor([3, 2**24 + 3, 7], dtype=torch.long)
        got_scores, got_ids = _pack_scores_and_ids(scores, ids)
        self.assertEqual(got_scores.device.type, "cpu")
        self.assertEqual(got_ids.dtype, torch.int64)
        self.assertTrue(torch.equal(got_scores, scores))
        self.assertTrue(torch.equal(got_ids, ids))
        wide_scores = torch.arange(9, dtype=torch.float32).reshape(3, 3) / 9
        wide_ids = torch.arange(9, dtype=torch.long).reshape(3, 3) + 2**24
        got_scores, got_ids = _pack_scores_and_ids(wide_scores, wide_ids)
        self.assertTrue(torch.equal(got_scores, wide_scores))
        self.assertTrue(torch.equal(got_ids, wide_ids))

    def test_state_stays_on_cpu(self):
        config = OptTreeConfig(width=3, tau=-1.0, max_depth=4)
        state = OptTreeState.create(config)
        seed(state, torch.tensor([0.5, 0.4, 0.1, 0.0]))
        expand_layer(state, torch.tensor([[0.6, 0.3, 0.1, 0.0]] * 3))
        for tensor in (state.weight, state.tokens, state.parents, state.anc_mask):
            self.assertEqual(tensor.device.type, "cpu")
        self.assertEqual(tuple(state.anc_mask.shape), (3, 6))

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
                    self.assertIs(our_out.ancestor_mask, our.anc_mask)
                    self.assertTrue(
                        torch.equal(our.anc_mask, _reference_ancestor_mask(our))
                    )
                    self.assertTrue(
                        torch.equal(
                            ancestor_mask(our, leak=True), ref_out["attention_mask"]
                        )
                    )
                    self.assertIsNone(our_out.leak_mask)
                    self.assertTrue(torch.equal(our.tokens[our.depth - 1], our_out.tokens))
                if ref_out["is_final"]:
                    break
            else:
                self.fail(f"seed {seed} did not stop")
            self.assertIsNotNone(our_out)
            self.assertTrue(our_out.is_final)
            self.assertTrue(ref_out["is_final"])
            self.assertIsNone(our_out.ancestor_mask)
            self.assertIsNone(our_out.leak_mask)
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
        self.assertIsNone(step.ancestor_mask)
        self.assertIsNone(step.leak_mask)
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
            leak = ancestor_mask(state, leak=True)
            self.assertTrue(torch.equal(leak, ref_out["attention_mask"]))
            self.assertIs(our.ancestor_mask, state.anc_mask)
            self.assertIs(ancestor_mask(state, leak=False), state.anc_mask)
            self.assertTrue(torch.equal(state.anc_mask, _reference_ancestor_mask(state)))
            if not torch.equal(leak, our.ancestor_mask):
                saw_stale = True
                stale = (leak == 1) & (our.ancestor_mask == 0)
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
        self.assertIsNone(step.ancestor_mask)
        self.assertIsNone(step.leak_mask)
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

    def test_remote_spec_role_reads_opt_when_sr_is_empty(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.spec_info import remote_spec_role

        self.assertEqual(
            remote_spec_role(SimpleNamespace(standalone_remote_role="target")),
            "target",
        )
        self.assertEqual(
            remote_spec_role(
                SimpleNamespace(standalone_remote_role=None, opt_tree_role="draft")
            ),
            "draft",
        )
        self.assertIsNone(
            remote_spec_role(
                SimpleNamespace(standalone_remote_role=None, opt_tree_role=None)
            )
        )
        self.assertEqual(
            remote_spec_role(
                SimpleNamespace(
                    standalone_remote_role="target", opt_tree_role="draft"
                )
            ),
            "target",
        )
        self.assertIsNone(remote_spec_role(None))

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
        self.assertTrue(opt.captures_target_verify_cuda_graph(draft))
        self.assertTrue(opt.uses_dual_ntpb_cuda_graph(draft))
        self.assertEqual(opt.dual_ntpb_cuda_graph_options(draft), [width, 1])
        self.assertEqual(
            opt.target_verify_cuda_graph_num_tokens_per_bs(draft), width
        )
        self.assertFalse(
            opt.captures_target_verify_cuda_graph(draft, is_draft_worker=True)
        )
        self.assertFalse(
            opt.uses_dual_ntpb_cuda_graph(draft, is_draft_worker=True)
        )
        self.assertEqual(
            opt.target_verify_cuda_graph_num_tokens_per_bs(
                draft, is_draft_worker=True
            ),
            1,
        )
        narrow = SimpleNamespace(
            opt_tree_role="draft",
            speculative_num_draft_tokens=1,
            spectre_role=None,
            standalone_remote_role=None,
        )
        self.assertEqual(opt.dual_ntpb_cuda_graph_options(narrow), [1])

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
        mask = layer_attention_mask(3, ancestor, "cpu")
        self.assertEqual(mask.numel(), 2 * (3 + 2))
        rows = mask.view(2, 5)
        self.assertTrue(torch.equal(rows[:, :3], torch.ones(2, 3, dtype=torch.bool)))
        self.assertTrue(torch.equal(rows[:, 3:], ancestor.to(dtype=torch.bool)))

        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            layer_verify_mask,
        )

        runner = SimpleNamespace(prefix_len=3)
        with patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_attention._attention_for",
            return_value=object(),
        ):
            cascade = layer_verify_mask(runner, ancestor, "cpu")
        self.assertEqual(cascade.numel(), ancestor.numel())
        with patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_attention._attention_for",
            return_value=None,
        ):
            ordinary = layer_verify_mask(runner, ancestor, "cpu")
        self.assertEqual(ordinary.numel(), 2 * (3 + 2))

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

    def test_extend_logits_are_one_row_per_request(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _extend_request_rows,
        )

        logits = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
        rows = _extend_request_rows(logits, 3)
        self.assertEqual(len(rows), 3)
        for i, row in enumerate(rows):
            self.assertTrue(torch.equal(row, logits[i]))
        # Lengths [2, 5, 1] used to address a per-token logit tensor at [1, 6, 7].
        self.assertGreater(7, int(logits.shape[0]) - 1)
        with self.assertRaises(IndexError):
            _ = logits[7]
        with self.assertRaises(RuntimeError):
            _extend_request_rows(logits[:2], 3)

    def test_pack_verify_attention_splits_by_request(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            pack_verify_attention,
        )

        width = 2
        eye = torch.eye(width, dtype=torch.int8)
        mask, positions = pack_verify_attention(
            [3, 5], [eye, eye], depth=1, width=width, device="cpu"
        )
        self.assertEqual(int(mask.numel()), full_mask_numel([3, 5], width))
        self.assertTrue(torch.equal(positions, torch.tensor([3, 3, 5, 5])))
        rows = {
            (b, t): row.tolist()
            for b, t, row in iter_full_mask_rows(mask, [3, 5], width)
        }
        self.assertEqual(rows[(0, 0)], [True, True, True, True, False])
        self.assertEqual(rows[(0, 1)], [True, True, True, False, True])
        self.assertEqual(rows[(1, 0)], [True] * 5 + [True, False])
        self.assertEqual(rows[(1, 1)], [True] * 5 + [False, True])

        deeper = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1]],
            dtype=torch.int8,
        )
        mask, positions = pack_verify_attention(
            [4], [deeper], depth=2, width=width, device="cpu"
        )
        cached = 4 + width
        self.assertEqual(int(mask.numel()), full_mask_numel([cached], width))
        self.assertTrue(torch.equal(positions, torch.tensor([5, 5])))
        deeper_rows = [
            row.tolist() for _, _, row in iter_full_mask_rows(mask, [cached], width)
        ]
        self.assertEqual(deeper_rows[0], [True] * 4 + [True, False, True, False])
        self.assertEqual(deeper_rows[1], [True] * 4 + [False, True, False, True])

    def test_grow_active_layers_matches_serial_and_drops_stopped(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            grow_active_layers,
            grow_with_forwards,
        )

        def serial(config, tail, layers):
            queue = list(layers)

            def forward_layer(step):
                return queue.pop(0)

            return grow_with_forwards(config, tail, forward_layer)

        def batched(config, tails, layers):
            queues = [list(rows) for rows in layers]
            seen = []

            def forward_active(steps, indices):
                seen.append(list(indices))
                return torch.stack([queues[i].pop(0) for i in indices])

            got = grow_active_layers(config, tails, forward_active)
            return got, seen

        config = OptTreeConfig(width=2, tau=-1.0, max_depth=4)
        gen = torch.Generator().manual_seed(3)
        tails = [torch.randn(8, generator=gen) for _ in range(2)]
        layers = [
            [torch.randn(2, 8, generator=gen) for _ in range(2)] for _ in range(2)
        ]
        got, seen = batched(config, tails, layers)
        self.assertEqual(seen, [[0, 1], [0, 1]])
        for i in range(2):
            self.assertEqual(got[i], serial(config, tails[i], layers[i]))

        flat = torch.log(torch.full((2, 4), 0.25))
        peaked = torch.log(
            torch.tensor([[0.9, 0.05, 0.03, 0.02], [0.9, 0.05, 0.03, 0.02]])
        )
        stop = OptTreeConfig(width=2, tau=0.3, max_depth=6)
        stop_tails = [
            torch.log(torch.full((4,), 0.25)),
            torch.log(torch.tensor([0.8, 0.1, 0.05, 0.05])),
        ]
        stop_layers = [[flat], [peaked, flat]]
        got, seen = batched(stop, stop_tails, stop_layers)
        self.assertEqual(seen[0], [0, 1])
        self.assertNotIn(0, seen[-1])
        self.assertEqual(seen[-1], [1])
        for i in range(2):
            self.assertEqual(got[i], serial(stop, stop_tails[i], stop_layers[i]))

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

    def test_opt_layer_verify_graph_matches_width_and_blocks_decode(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            OptLayerSpec,
            opt_bonus_graph_extra,
            opt_layer_verify_graph_matches,
        )
        from sglang.srt.speculative.spec_info import (
            decode_cuda_graph_accepts_spec_info,
        )

        spec = OptLayerSpec(
            custom_mask=torch.ones(4, dtype=torch.bool),
            positions=torch.zeros(4, dtype=torch.long),
            draft_token_num=4,
            num_tokens_per_req=4,
            capture_hidden_mode=0,
        )
        self.assertTrue(opt_layer_verify_graph_matches(spec, 4))
        self.assertFalse(opt_layer_verify_graph_matches(spec, 1))
        mismatch = OptLayerSpec(
            custom_mask=spec.custom_mask,
            positions=spec.positions,
            draft_token_num=4,
            num_tokens_per_req=2,
            capture_hidden_mode=0,
        )
        self.assertFalse(opt_layer_verify_graph_matches(mismatch, 4))
        width_one = OptLayerSpec(
            custom_mask=spec.custom_mask[:1],
            positions=spec.positions[:1],
            draft_token_num=1,
            num_tokens_per_req=1,
            capture_hidden_mode=0,
        )
        self.assertFalse(opt_layer_verify_graph_matches(width_one, 1))
        self.assertFalse(opt_layer_verify_graph_matches(None, 4))
        bonus = OptLayerSpec(
            custom_mask=spec.custom_mask,
            positions=spec.positions,
            draft_token_num=4,
            num_tokens_per_req=4,
            capture_hidden_mode=0,
            is_bonus=True,
        )
        self.assertTrue(spec.apply_prefix_custom_mask)
        self.assertTrue(opt_layer_verify_graph_matches(bonus, 4))
        self.assertEqual(opt_bonus_graph_extra(bonus), 0)
        self.assertIsNone(opt_bonus_graph_extra(spec))
        self.assertFalse(decode_cuda_graph_accepts_spec_info(spec))
        self.assertFalse(decode_cuda_graph_accepts_spec_info(width_one))

    def test_layer_and_bonus_batches_apply_prefix_mask(self):
        import sys
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            OptLayerForward,
            _BonusCandidate,
            _forward_one_bonus_group,
            opt_bonus_graph_extra,
            opt_layer_verify_graph_matches,
        )

        seen = []
        pool = torch.full((1, 16), -1, dtype=torch.int64)

        def make_batch(_reqs):
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
            return SimpleNamespace(
                seq_lens=torch.tensor([4], dtype=torch.int64),
                sampling_info=info,
                has_grammar=True,
                return_logprob=True,
                return_hidden_states=True,
            )

        def run_batch(batch):
            seen.append(batch)
            width = int(batch.spec_info.num_tokens_per_req)
            return SimpleNamespace(
                logits_output=SimpleNamespace(
                    next_token_logits=torch.zeros(width, 3)
                )
            )

        host = SimpleNamespace(
            _sr_make_decode_batch=make_batch,
            run_batch=run_batch,
            cur_batch=None,
            req_to_token_pool=SimpleNamespace(device="cpu", req_to_token=pool),
        )
        req = SimpleNamespace(req_pool_idx=0, kv_allocated_len=8)
        runner = OptLayerForward.__new__(OptLayerForward)
        runner.scheduler = host
        runner.req = req
        runner.cached_len = 4
        runner.prefix_len = 4
        runner._alloc_layer_slots = lambda n, start=None: torch.arange(
            8, 8 + int(n), dtype=torch.int64
        )

        forward_info = SimpleNamespace(
            ForwardMode=SimpleNamespace(TARGET_VERIFY=object()),
            CaptureHiddenMode=SimpleNamespace(NULL=object()),
        )
        with patch.dict(
            sys.modules,
            {"sglang.srt.model_executor.forward_batch_info": forward_info},
        ):
            layer = runner._verify_batch(
                torch.tensor([1, 2, 3, 4]),
                torch.tensor([4, 5, 6, 7]),
                torch.ones(4, 8, dtype=torch.bool),
                torch.arange(4, dtype=torch.int64),
            )
            guesses = _forward_one_bonus_group(
                host,
                runner,
                [
                    _BonusCandidate(-1, 7, 4, ()),
                    _BonusCandidate(0, 8, 5, (4,)),
                    _BonusCandidate(1, 9, 6, (4, 5)),
                    _BonusCandidate(2, 10, 7, (4, 5, 6)),
                ],
                8,
                4,
            )
        self.assertTrue(layer.spec_info.apply_prefix_custom_mask)
        self.assertFalse(layer.spec_info.is_bonus)
        self.assertTrue(opt_layer_verify_graph_matches(layer.spec_info, 4))
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].spec_info.apply_prefix_custom_mask)
        self.assertTrue(seen[0].spec_info.is_bonus)
        self.assertEqual(int(seen[0].spec_info.num_tokens_per_req), 4)
        self.assertTrue(opt_layer_verify_graph_matches(seen[0].spec_info, 4))
        self.assertEqual(opt_bonus_graph_extra(seen[0].spec_info), 0)
        self.assertIsNone(opt_bonus_graph_extra(layer.spec_info))
        self.assertEqual([guess.token for guess in guesses], [7, 8, 9, 10])

    def test_bonus_slot_ids_share_one_host_copy(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _read_bonus_slot_ids,
        )

        rows = _read_bonus_slot_ids(
            [torch.tensor([8, 9]), torch.tensor([10, 11, 12])]
        )
        self.assertEqual(rows, [[8, 9], [10, 11, 12]])

    def test_bonus_template_forward_skips_run_batch(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _BonusCandidate,
            _forward_one_bonus_group,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        seen = []

        def make_batch(reqs):
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
            return SimpleNamespace(
                reqs=list(reqs),
                seq_lens=torch.tensor([8], dtype=torch.int64),
                orig_seq_lens=torch.tensor([8], dtype=torch.int32),
                sampling_info=info,
                has_grammar=False,
                return_logprob=False,
                return_hidden_states=False,
            )

        def run_batch(_batch):
            raise AssertionError("run_batch")

        def forward(forward_batch):
            seen.append(forward_batch)
            width = int(forward_batch.spec_info.num_tokens_per_req)
            return SimpleNamespace(
                can_run_graph=True,
                logits_output=SimpleNamespace(
                    next_token_logits=torch.zeros(width, 3)
                ),
            )

        host = SimpleNamespace(
            _sr_make_decode_batch=make_batch,
            run_batch=run_batch,
            cur_batch=None,
            req_to_token_pool=SimpleNamespace(
                device="cpu",
                req_to_token=torch.full((1, 16), -1, dtype=torch.int64),
            ),
            model_worker=SimpleNamespace(
                model_runner=SimpleNamespace(forward=forward)
            ),
        )
        req = SimpleNamespace(req_pool_idx=0, kv_allocated_len=8, rid="bonus")
        runner = SimpleNamespace(
            scheduler=host,
            req=req,
            cached_len=4,
            prefix_len=4,
        )
        runner._alloc_layer_slots = lambda n, start=None: torch.arange(
            8, 8 + int(n), dtype=torch.int64
        )
        chunk = [
            _BonusCandidate(-1, 7, 4, ()),
            _BonusCandidate(0, 8, 5, (4,)),
            _BonusCandidate(1, 9, 6, (4, 5)),
            _BonusCandidate(2, 10, 7, (4, 5, 6)),
        ]
        with patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward._lookahead_forward_template",
            return_value=SimpleNamespace(),
        ):
            guesses = _forward_one_bonus_group(host, runner, chunk, 8, 4)
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].spec_info.is_bonus)
        self.assertEqual(int(seen[0].spec_info.num_tokens_per_req), 4)
        self.assertEqual([guess.token for guess in guesses], [7, 8, 9, 10])
        self.assertEqual([guess.slot for guess in guesses], [8, 9, 10, 11])
        counts = get_sr_round_metrics(host, "Draft").counts
        self.assertEqual(counts["opt_bonus_template_fallbacks"], 0)
        self.assertEqual(counts["opt_bonus_eager_groups"], 0)

    def test_bonus_template_reused_until_pool_or_finish(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _BonusCandidate,
            _forward_one_bonus_group,
            drop_lookahead_request,
        )

        _import_or_stub_forward_batch()
        admits = {"n": 0}
        forwards = {"n": 0}

        def make_batch(reqs):
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
            return SimpleNamespace(
                reqs=list(reqs),
                seq_lens=torch.tensor([8], dtype=torch.int64),
                orig_seq_lens=torch.tensor([8], dtype=torch.int32),
                sampling_info=info,
                has_grammar=False,
                return_logprob=False,
                return_hidden_states=False,
            )

        def run_batch(_batch):
            raise AssertionError("run_batch")

        def forward(forward_batch):
            forwards["n"] += 1
            width = int(forward_batch.spec_info.num_tokens_per_req)
            return SimpleNamespace(
                can_run_graph=True,
                logits_output=SimpleNamespace(
                    next_token_logits=torch.zeros(width, 3)
                ),
            )

        def admit(_scheduler, _batch):
            admits["n"] += 1
            return SimpleNamespace()

        host = SimpleNamespace(
            _sr_make_decode_batch=make_batch,
            run_batch=run_batch,
            cur_batch=None,
            req_to_token_pool=SimpleNamespace(
                device="cpu",
                req_to_token=torch.full((1, 16), -1, dtype=torch.int64),
            ),
            model_worker=SimpleNamespace(
                model_runner=SimpleNamespace(forward=forward)
            ),
        )
        req = SimpleNamespace(req_pool_idx=0, kv_allocated_len=8, rid="bonus")
        runner = SimpleNamespace(
            scheduler=host,
            req=req,
            cached_len=4,
            prefix_len=4,
        )
        runner._alloc_layer_slots = lambda n, start=None: torch.arange(
            8, 8 + int(n), dtype=torch.int64
        )
        chunk = [
            _BonusCandidate(-1, 7, 4, ()),
            _BonusCandidate(0, 8, 5, (4,)),
            _BonusCandidate(1, 9, 6, (4, 5)),
            _BonusCandidate(2, 10, 7, (4, 5, 6)),
        ]
        with patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward._lookahead_forward_template",
            admit,
        ):
            _forward_one_bonus_group(host, runner, chunk, 8, 4)
            _forward_one_bonus_group(host, runner, chunk, 8, 4)
            self.assertEqual(admits["n"], 1)
            self.assertEqual(forwards["n"], 2)

            req.req_pool_idx = 1
            _forward_one_bonus_group(host, runner, chunk, 8, 4)
            self.assertEqual(admits["n"], 2)

            drop_lookahead_request(host, req.rid)
            req.req_pool_idx = 0
            _forward_one_bonus_group(host, runner, chunk, 8, 4)
            self.assertEqual(admits["n"], 3)
            self.assertEqual(forwards["n"], 4)

    def test_layer_template_reused_until_pool_or_finish(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            LookaheadFallback,
            _admit_lookahead,
            _lookahead_batch,
            drop_lookahead_request,
        )

        _import_or_stub_forward_batch()
        builds = {"n": 0}
        evict = {"on": False}
        width = 4

        def make_batch(reqs):
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
            return SimpleNamespace(
                reqs=list(reqs),
                sampling_info=info,
                has_grammar=False,
                return_logprob=False,
                return_hidden_states=False,
            )

        def build_template(_scheduler, _batch):
            builds["n"] += 1
            return SimpleNamespace(tag=builds["n"])

        def would_evict(_scheduler, _ingest=0):
            return evict["on"]

        host = SimpleNamespace(
            _sr_make_decode_batch=make_batch,
            server_args=SimpleNamespace(opt_tree_width=width),
            req_to_token_pool=SimpleNamespace(device="cpu"),
        )
        req = SimpleNamespace(rid="layer", req_pool_idx=0, kv_committed_len=8)
        runner = SimpleNamespace(
            req=req,
            prefix_len=8,
            cached_len=8,
            _la_positions=torch.empty(width, dtype=torch.long),
            _la_seq=torch.empty(1, dtype=torch.int64),
            _la_seq_cpu=torch.empty(1, dtype=torch.int64),
            _la_orig=torch.empty(1, dtype=torch.int32),
            _la_dummy_tokens=torch.zeros(width, dtype=torch.long),
            _la_forward_template=None,
            _la_mrope_delta=None,
        )

        def admit():
            batch = _lookahead_batch(host, req)
            _admit_lookahead(host, runner, batch)

        with patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward._lookahead_forward_template",
            build_template,
        ), patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward._lookahead_would_evict",
            would_evict,
        ):
            admit()
            first = runner._la_forward_template
            admit()
            self.assertEqual(builds["n"], 1)
            self.assertIs(runner._la_forward_template, first)
            self.assertIs(host._opt_la_batches[req.rid]["layer_template"], first)
            self.assertIsNone(runner._la_mrope_delta)

            evict["on"] = True
            with self.assertRaises(LookaheadFallback) as caught:
                admit()
            self.assertIn("kv evict", str(caught.exception))
            self.assertEqual(builds["n"], 1)
            self.assertIs(host._opt_la_batches[req.rid]["layer_template"], first)

            evict["on"] = False
            req.req_pool_idx = 1
            admit()
            self.assertEqual(builds["n"], 2)
            self.assertIsNot(runner._la_forward_template, first)

            second = runner._la_forward_template
            drop_lookahead_request(host, req.rid)
            req.req_pool_idx = 0
            admit()
            self.assertEqual(builds["n"], 3)
            self.assertIsNot(runner._la_forward_template, second)
            self.assertIs(
                host._opt_la_batches[req.rid]["layer_template"],
                runner._la_forward_template,
            )

    def test_bonus_group_logits_survive_next_replay(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _BonusCandidate,
            _forward_bonus_candidates,
        )

        _import_or_stub_forward_batch()

        patterns = [
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
        ]
        buffer = torch.empty(2, 3)
        calls = {"n": 0}

        def make_batch(reqs):
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
            return SimpleNamespace(
                reqs=list(reqs),
                seq_lens=torch.tensor([8], dtype=torch.int64),
                orig_seq_lens=torch.tensor([8], dtype=torch.int32),
                sampling_info=info,
                has_grammar=False,
                return_logprob=False,
                return_hidden_states=False,
            )

        def run_batch(_batch):
            raise AssertionError("run_batch")

        def forward(_forward_batch):
            pattern = patterns[calls["n"]]
            calls["n"] += 1
            buffer.copy_(pattern)
            return SimpleNamespace(
                can_run_graph=True,
                logits_output=SimpleNamespace(next_token_logits=buffer),
            )

        host = SimpleNamespace(
            server_args=SimpleNamespace(speculative_num_draft_tokens=1),
            _sr_make_decode_batch=make_batch,
            run_batch=run_batch,
            cur_batch=None,
            req_to_token_pool=SimpleNamespace(
                device="cpu",
                req_to_token=torch.full((1, 16), -1, dtype=torch.int64),
            ),
            model_worker=SimpleNamespace(
                model_runner=SimpleNamespace(forward=forward)
            ),
        )
        req = SimpleNamespace(req_pool_idx=0, kv_allocated_len=8, rid="bonus")
        runner = SimpleNamespace(
            scheduler=host,
            req=req,
            cached_len=4,
            prefix_len=4,
        )
        next_slot = {"n": 8}

        def alloc(n, start=None):
            begin = next_slot["n"]
            next_slot["n"] += int(n)
            return torch.arange(begin, begin + int(n), dtype=torch.int64)

        runner._alloc_layer_slots = alloc
        plans = [
            _BonusCandidate(index, 10 + index, 20 + index, (index,))
            for index in range(4)
        ]
        with patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward._lookahead_forward_template",
            return_value=SimpleNamespace(),
        ):
            _forward_bonus_candidates(host, runner, plans)
        guesses = req.opt_bonus_guesses
        self.assertEqual(calls["n"], 2)
        self.assertEqual(len(guesses), 4)
        for index, guess in enumerate(guesses):
            row = patterns[index // 2][index % 2]
            self.assertTrue(torch.equal(guess.logits, row))
            self.assertNotEqual(guess.logits.data_ptr(), buffer.data_ptr())
        buffer.fill_(-1.0)
        for index, guess in enumerate(guesses):
            row = patterns[index // 2][index % 2]
            self.assertTrue(torch.equal(guess.logits, row))

    def test_bonus_fallback_binds_cascade(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _bonus_fallback_logits,
            _fill_bonus_schedule_batch,
        )

        _import_or_stub_forward_batch()
        width = 3
        tree_end = 8
        prefix = 4
        mask = torch.arange(width * (tree_end + width), dtype=torch.uint8).reshape(
            width, tree_end + width
        )
        positions = torch.arange(tree_end, tree_end + width)
        tokens = torch.arange(width, dtype=torch.long)
        loc = torch.arange(20, 20 + width)

        def make_batch(reqs):
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
            return SimpleNamespace(
                reqs=list(reqs),
                seq_lens=torch.tensor([tree_end], dtype=torch.int64),
                orig_seq_lens=torch.tensor([tree_end], dtype=torch.int32),
                sampling_info=info,
                has_grammar=False,
                return_logprob=False,
                return_hidden_states=False,
            )

        def run_batch(_batch):
            return SimpleNamespace(
                logits_output=SimpleNamespace(next_token_logits=torch.zeros(width, 2))
            )

        host = SimpleNamespace(
            _sr_make_decode_batch=make_batch,
            run_batch=run_batch,
            cur_batch=None,
        )
        req = SimpleNamespace(req_pool_idx=0, rid="bonus")
        runner = SimpleNamespace(scheduler=host, req=req, prefix_len=prefix)
        calls = []

        def record(
            spec, bound_runner, bound_loc, tree_mask, prefix_rows=None, cached_len=None
        ):
            calls.append(
                (
                    spec,
                    bound_runner,
                    bound_loc.detach().clone(),
                    tree_mask.detach().clone(),
                    None if prefix_rows is None else prefix_rows.detach().clone(),
                    cached_len,
                )
            )

        with patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_attention.bind_opt_cascade",
            side_effect=record,
        ):
            _bonus_fallback_logits(
                host, runner, tokens, positions, mask, loc, tree_end, width
            )
            _fill_bonus_schedule_batch(
                host, runner, tokens, positions, mask, loc, tree_end, width
            )

        self.assertEqual(len(calls), 2)
        expected_tree = mask[:, prefix:]
        expected_rows = mask[:, 0]
        for spec, bound_runner, bound_loc, tree_mask, prefix_rows, cached_len in calls:
            self.assertTrue(spec.is_bonus)
            self.assertIs(bound_runner, runner)
            self.assertTrue(torch.equal(bound_loc, loc))
            self.assertTrue(torch.equal(tree_mask, expected_tree))
            self.assertTrue(torch.equal(prefix_rows, expected_rows))
            self.assertEqual(cached_len, tree_end)

    def test_bonus_graph_key_does_not_reuse_layer_graph(self):
        from pathlib import Path

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            OPT_BONUS_GRAPH_EXTRA,
            OptLayerSpec,
            opt_bonus_graph_extra,
            opt_draft_bonus_ntpb,
            opt_draft_capture_extra_keys,
            opt_layer_verify_graph_matches,
        )

        mask = torch.ones(4, dtype=torch.bool)
        positions = torch.zeros(4, dtype=torch.long)
        layer = OptLayerSpec(
            custom_mask=mask,
            positions=positions,
            draft_token_num=4,
            num_tokens_per_req=4,
            capture_hidden_mode=0,
        )
        bonus = OptLayerSpec(
            custom_mask=mask,
            positions=positions,
            draft_token_num=4,
            num_tokens_per_req=4,
            capture_hidden_mode=0,
            is_bonus=True,
        )
        draft = SimpleNamespace(opt_tree_role="draft", opt_tree_width=4)
        target = SimpleNamespace(opt_tree_role="target", opt_tree_width=5)
        self.assertEqual(opt_draft_bonus_ntpb(draft), 5)
        self.assertIsNone(opt_draft_bonus_ntpb(target))
        self.assertEqual(opt_draft_capture_extra_keys(draft, 4), [None])
        self.assertEqual(
            opt_draft_capture_extra_keys(draft, 5), [OPT_BONUS_GRAPH_EXTRA]
        )
        self.assertEqual(opt_draft_capture_extra_keys(draft, 1), [None])
        self.assertEqual(opt_draft_capture_extra_keys(target, 5), [None])
        self.assertTrue(layer.apply_prefix_custom_mask)
        self.assertIsNone(opt_bonus_graph_extra(layer))
        self.assertEqual(opt_bonus_graph_extra(bonus), OPT_BONUS_GRAPH_EXTRA)
        wide = OptLayerSpec(
            custom_mask=torch.ones(5, dtype=torch.bool),
            positions=torch.zeros(5, dtype=torch.long),
            draft_token_num=5,
            num_tokens_per_req=5,
            capture_hidden_mode=0,
            apply_prefix_custom_mask=True,
            is_bonus=True,
        )
        self.assertTrue(opt_layer_verify_graph_matches(wide, 5))
        self.assertFalse(opt_layer_verify_graph_matches(layer, 5))
        layer_key = "r4_1"
        bonus_key = f"r5_1_s{opt_bonus_graph_extra(wide)}"
        graphs = {layer_key: object()}
        self.assertEqual(bonus_key, "r5_1_s0")
        self.assertNotIn(bonus_key, graphs)
        graphs[bonus_key] = object()
        self.assertIn(bonus_key, graphs)

        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/model_executor/cuda_graph_runner.py"
        ).read_text(encoding="utf-8")
        self.assertIn("opt_draft_capture_extra_keys", source)
        self.assertEqual(source.count("_replay_graph_extra(forward_batch)"), 3)
        self.assertIn("== OPT_BONUS_GRAPH_EXTRA", source)

    def test_other_algorithms_keep_verify_graph_when_opt_role_is_draft(self):
        from unittest.mock import patch
        from types import SimpleNamespace

        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        both = SimpleNamespace(
            standalone_remote_role="target",
            spectre_role="target",
            opt_tree_role="draft",
            speculative_num_draft_tokens=5,
        )
        opt_draft = SimpleNamespace(
            opt_tree_role="draft",
            speculative_num_draft_tokens=4,
            spectre_role=None,
            standalone_remote_role=None,
        )
        opt_target = SimpleNamespace(
            opt_tree_role="target",
            speculative_num_draft_tokens=5,
            spectre_role=None,
            standalone_remote_role=None,
        )
        with patch("sglang.srt.utils.is_npu", return_value=True):
            self.assertTrue(
                SpeculativeAlgorithm.STANDALONE_REMOTE.captures_target_verify_cuda_graph(
                    both
                )
            )
            self.assertTrue(
                SpeculativeAlgorithm.STANDALONE_REMOTE.uses_dual_ntpb_cuda_graph(
                    both
                )
            )
            self.assertTrue(
                SpeculativeAlgorithm.SPECTRE.captures_target_verify_cuda_graph(both)
            )
            self.assertTrue(
                SpeculativeAlgorithm.SPECTRE.uses_dual_ntpb_cuda_graph(both)
            )
            self.assertFalse(
                SpeculativeAlgorithm.OPT_TREE_REMOTE.captures_target_verify_cuda_graph(
                    opt_draft
                )
            )
            self.assertFalse(
                SpeculativeAlgorithm.OPT_TREE_REMOTE.uses_dual_ntpb_cuda_graph(
                    opt_draft
                )
            )
            self.assertTrue(
                SpeculativeAlgorithm.OPT_TREE_REMOTE.captures_target_verify_cuda_graph(
                    opt_target
                )
            )
            self.assertTrue(
                SpeculativeAlgorithm.OPT_TREE_REMOTE.uses_dual_ntpb_cuda_graph(
                    opt_target
                )
            )

    def test_release_uncommitted_rolls_back_to_kv_committed_len(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            release_uncommitted,
        )

        host = _ReleaseHost()
        # Logical length is 5, but slots only through kv_committed_len are kept.
        gap = SimpleNamespace(
            rid="gap",
            kv_committed_len=2,
            kv_allocated_len=4,
            origin_input_ids=[1, 2, 3],
            output_ids=[4, 5],
        )
        cached = release_uncommitted(gap, host._opt_rollback)
        self.assertEqual(cached, 2)
        self.assertEqual(host.rolls, [2])

        past = SimpleNamespace(
            rid="past",
            kv_committed_len=5,
            kv_allocated_len=8,
            origin_input_ids=[1, 2, 3, 4, 5],
            output_ids=[],
        )
        host.rolls.clear()
        cached = release_uncommitted(past, host._opt_rollback)
        self.assertEqual(cached, 5)
        self.assertEqual(host.rolls, [5])

        equal = SimpleNamespace(
            rid="equal",
            kv_committed_len=5,
            kv_allocated_len=5,
            origin_input_ids=[1, 2, 3, 4, 5],
            output_ids=[],
        )
        host.rolls.clear()
        cached = release_uncommitted(equal, host._opt_rollback)
        self.assertEqual(cached, 5)
        self.assertEqual(host.rolls, [])

    def test_plan_opt_tail_extends_instead_of_reprefill_threshold(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            extend_chunk_bounds,
            plan_opt_tail,
        )

        def req(origin, output, kv, pool=1):
            return SimpleNamespace(
                rid="opt-tail",
                req_pool_idx=pool,
                origin_input_ids=list(range(origin)),
                output_ids=list(range(output)),
                kv_committed_len=kv,
            )

        # 20 missing outputs used to exceed the per-token decode threshold.
        self.assertEqual(plan_opt_tail(req(10, 20, 10), 2048), "extend")
        self.assertEqual(plan_opt_tail(req(10, 0, 10), 2048), "noop")
        self.assertEqual(plan_opt_tail(req(10, 1, 10), 2048), "decode")
        self.assertEqual(plan_opt_tail(req(10, 2, 10), 2048), "decode")
        self.assertEqual(plan_opt_tail(req(10, 1, 10), 2048, page_size=128), "extend")
        self.assertEqual(plan_opt_tail(req(10, 2, 10), 2048, page_size=128), "extend")
        self.assertEqual(plan_opt_tail(req(0, 3000, 0), 2048), "extend_chunked")
        self.assertEqual(
            extend_chunk_bounds(0, 3000, 2048),
            [(0, 2048), (2048, 3000)],
        )
        self.assertEqual(plan_opt_tail(req(5, 1, 2), 2048), "reprefill")
        self.assertEqual(plan_opt_tail(req(4, 0, 4, pool=None), 2048), "reprefill")
        self.assertEqual(plan_opt_tail(req(0, 5000, 0), None), "extend")
        self.assertEqual(plan_opt_tail(req(0, 5000, 0), 0), "extend")

    def test_prefill_callback_skips_logits_rewritten_by_sampling(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            raw_logits_survive_sampling,
        )

        raw = torch.tensor([[1.0, 3.0, 0.5, -1.0]])

        def sampling_info(*, greedy, temperature, penalty):
            return SimpleNamespace(
                is_all_greedy=greedy,
                temperatures=torch.tensor([[temperature]]),
                logit_bias=None,
                has_custom_logit_processor=False,
                penalizer_orchestrator=SimpleNamespace(is_required=penalty),
            )

        def capture(info):
            mixin = _materialize_host()
            mixin.draft_paused_reqs = []
            mixin.waiting_queue = []
            mixin.running_batch = SimpleNamespace(is_empty=lambda: True, reqs=[])
            mixin.server_args = SimpleNamespace(speculative_eagle_topk=1)
            mixin.chunked_req = None
            origin = [1, 2, 3]
            req = SimpleNamespace(
                rid="opt-raw",
                is_sr_draft=True,
                origin_input_ids=list(origin),
                output_ids=[],
                fill_ids=list(origin),
                req_pool_idx=None,
                kv_committed_len=0,
                draft_is_paused=False,
                draft_generation_start_len=0,
                finished=lambda: False,
                multimodal_inputs=None,
            )
            seen = []

            class _Extend:
                def is_extend(self):
                    return True

            def next_batch():
                req.req_pool_idx = 1
                req.kv_committed_len = len(origin)
                mixin.chunked_req = None
                return SimpleNamespace(
                    reqs=[req],
                    forward_mode=_Extend(),
                    is_extend_in_batch=False,
                    sampling_info=info,
                    has_grammar=False,
                )

            def run_batch(batch):
                sampling = batch.sampling_info
                logits = raw.clone()
                orch = getattr(sampling, "penalizer_orchestrator", None)
                if orch is not None and orch.is_required:
                    logits.add_(1.0)
                if not sampling.is_all_greedy:
                    logits.div_(sampling.temperatures)
                    logits[:] = torch.softmax(logits, dim=-1)
                return SimpleNamespace(
                    logits_output=SimpleNamespace(next_token_logits=logits)
                )

            def on_extend_logits(_batch, logits):
                seen.append(logits.detach().clone())

            mixin.get_next_batch_to_run = next_batch
            mixin.run_batch = run_batch
            mixin.process_batch_result = lambda *_args, **_kwargs: None
            mixin._sr_materialize_prefix_batch(
                [req],
                on_extend_logits=on_extend_logits,
                logits_survive=raw_logits_survive_sampling,
            )
            return seen

        self.assertEqual(
            capture(sampling_info(greedy=False, temperature=0.7, penalty=False)),
            [],
        )
        self.assertEqual(
            capture(sampling_info(greedy=True, temperature=1.0, penalty=True)),
            [],
        )
        greedy = capture(sampling_info(greedy=True, temperature=1.0, penalty=False))
        self.assertEqual(len(greedy), 1)
        self.assertTrue(torch.equal(greedy[0], raw))

    def test_reprefill_reads_raw_tail_when_prefill_cache_is_skipped(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.opt_tree_remote import opt_tree_forward as forward

        req = SimpleNamespace(
            rid="opt-reprefill",
            origin_input_ids=[1, 2],
            output_ids=[3],
            kv_committed_len=3,
            req_pool_idx=1,
            opt_prefill_tail_logits=None,
        )
        seen = []

        def committed(_reqs, on_extend_logits=None, logits_survive=None):
            self.assertIsNotNone(on_extend_logits)
            self.assertIs(logits_survive, forward.raw_logits_survive_sampling)

        def tail(_scheduler, runners):
            seen.append([runner.prefix_len for runner in runners])
            return [torch.tensor([4.0, 5.0])]

        scheduler = SimpleNamespace(_sr_reprefill_committed=committed)
        original = forward._logits_at_tail_batch
        forward._logits_at_tail_batch = tail
        try:
            rows = forward.reprefill_tail_logits(scheduler, [req])
        finally:
            forward._logits_at_tail_batch = original
        self.assertEqual(seen, [[3]])
        self.assertTrue(torch.equal(rows[0], torch.tensor([4.0, 5.0])))

    def test_verify_groups_follow_page_aligned_budget(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _page_aligned_layer_tokens,
            _take_verify_group,
        )

        self.assertEqual(_page_aligned_layer_tokens(100, 60, 128), 128)
        self.assertEqual(_page_aligned_layer_tokens(100, 60, 1), 60)
        runners = [
            SimpleNamespace(cached_len=100),
            SimpleNamespace(cached_len=100),
        ]
        pending = list(runners)
        groups = []
        while pending:
            group = _take_verify_group(pending, 200, 128, 60)
            self.assertGreater(len(group), 0)
            groups.append(group)
            pending = pending[len(group) :]
        self.assertEqual([len(group) for group in groups], [1, 1])

    def test_verify_budget_includes_evictable_cache(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.opt_tree_remote import opt_tree_forward as forward

        def scheduler(available, evictable):
            return SimpleNamespace(
                token_to_kv_pool_allocator=SimpleNamespace(
                    available_size=lambda: available
                ),
                tree_cache=SimpleNamespace(page_size=1, evictable_size=evictable),
                server_args=SimpleNamespace(opt_tree_width=60),
            )

        radix = scheduler(70, lambda: 100)
        chunk = scheduler(70, lambda: 0)

        def raise_evictable():
            raise NotImplementedError

        hybrid = scheduler(70, raise_evictable)
        self.assertEqual(forward._verify_layer_budget(radix), 170)
        self.assertEqual(forward._verify_layer_budget(chunk), 70)
        self.assertEqual(forward._verify_layer_budget(hybrid), 70)

        runners = [SimpleNamespace(cached_len=0) for _ in range(3)]
        steps = [object(), object(), object()]
        seen = []

        def fake_batch(_scheduler, group_runners, _steps):
            seen.append(len(group_runners))
            return [torch.zeros(1) for _ in group_runners]

        original = forward._forward_verify_batch
        try:
            forward._forward_verify_batch = fake_batch
            rows, forwards = forward._forward_verify_fitting(radix, runners, steps)
            self.assertEqual(seen, [2, 1])
            self.assertEqual(len(rows), 3)
            self.assertEqual(forwards, 2)
            seen.clear()
            _rows, forwards = forward._forward_verify_fitting(chunk, runners, steps)
            self.assertEqual(seen, [1, 1, 1])
            self.assertEqual(forwards, 3)
        finally:
            forward._forward_verify_batch = original

    def test_split_verify_replays_match_one_batch(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.opt_tree_remote import opt_tree_forward as forward
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            grow_active_layers,
        )

        width = 2
        vocab = 4
        config = OptTreeConfig(width=width, tau=-1.0, max_depth=4)

        def peaked(token):
            row = torch.full((width, vocab), -8.0)
            row[:, int(token)] = 8.0
            return row

        tails = [
            torch.tensor([8.0, -8.0, -8.0, -8.0]),
            torch.tensor([-8.0, 8.0, -8.0, -8.0]),
        ]
        layers = [
            [peaked(0), peaked(2)],
            [peaked(1), peaked(3)],
        ]

        def run(budget):
            queues = [[row.clone() for row in rows] for rows in layers]
            runners = [SimpleNamespace(cached_len=0, idx=i) for i in range(2)]
            scheduler = SimpleNamespace(
                token_to_kv_pool_allocator=SimpleNamespace(
                    available_size=lambda: budget
                ),
                tree_cache=SimpleNamespace(page_size=1, evictable_size=lambda: 0),
                server_args=SimpleNamespace(opt_tree_width=width),
            )
            calls = []
            buf = torch.empty(2, width, vocab)

            def fake_batch(_scheduler, group_runners, _steps):
                count = len(group_runners)
                calls.append(count)
                for row, runner in enumerate(group_runners):
                    buf[row].copy_(queues[runner.idx].pop(0))
                return buf[:count]

            def forward_active(steps, indices):
                chosen = [runners[i] for i in indices]
                rows, _n_forwards = forward._forward_verify_fitting(
                    scheduler, chosen, steps
                )
                return rows

            original = forward._forward_verify_batch
            try:
                forward._forward_verify_batch = fake_batch
                got = grow_active_layers(config, tails, forward_active)
            finally:
                forward._forward_verify_batch = original
            return got, calls

        split, split_calls = run(width)
        whole, whole_calls = run(width * 2)
        self.assertEqual(split_calls, [1, 1, 1, 1])
        self.assertEqual(whole_calls, [2, 2])
        self.assertEqual(split, whole)

    def test_tail_logits_survive_a_later_replay(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _logits_at_tail_batch,
        )

        class _Info:
            is_all_greedy = False
            temperatures = torch.ones(2, 1)
            logit_bias = None
            has_custom_logit_processor = False
            penalizer_orchestrator = None
            need_top_p_sampling = True
            need_top_k_sampling = True
            need_min_p_sampling = True
            acc_additive_penalties = None
            acc_scaling_penalties = None
            vocab_mask = None

        buffer = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        batch = SimpleNamespace(
            output_ids=torch.tensor([7, 8]),
            sampling_info=_Info(),
            has_grammar=False,
            return_logprob=True,
            return_hidden_states=True,
        )
        scheduler = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.tensor([[4, 5]])
            ),
            _sr_make_decode_batch=lambda _reqs: batch,
            run_batch=lambda _batch: SimpleNamespace(
                logits_output=SimpleNamespace(next_token_logits=buffer)
            ),
            cur_batch=None,
        )
        runners = [
            SimpleNamespace(
                prefix_len=2,
                req=SimpleNamespace(rid=f"opt-tail-{i}", req_pool_idx=0),
            )
            for i in range(2)
        ]
        rows = _logits_at_tail_batch(scheduler, runners)
        buffer.zero_()
        self.assertEqual(rows[0].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(rows[1].tolist(), [0.0, 1.0, 0.0])

    def test_batch_fallback_expands_only_missing_windows(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote import opt_tree_forward as forward
        from sglang.srt.speculative.standalone_remote.sr_protocol import SRAction
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        host = _OptWindowHost(assign_pool=True)
        first = _OptReq()
        second = _OptReq()
        second.rid = "opt-missing"

        def grow(_scheduler, _reqs):
            return [([1], [-1], [0]), None]

        with patch.object(forward, "grow_batch", grow):
            windows = forward.produce_opt_windows(
                host, SRAction.STEP, [(None, first), (None, second)]
            )
        self.assertEqual(windows[0], ([1], [-1], [0]))
        self.assertEqual(windows[1], ([7], [-1], [0]))
        self.assertEqual(host.calls.count("expand"), 1)
        self.assertEqual(host._sr_round_metrics.counts["opt_batch_fallbacks"], 1)

        alone = _OptWindowHost(assign_pool=True)
        get_sr_round_metrics(alone, "Draft")
        forward.produce_opt_windows(alone, SRAction.STEP, [(None, _OptReq())])
        self.assertEqual(alone._sr_round_metrics.counts["opt_batch_fallbacks"], 0)

    def test_device_context_error_skips_batch_fallback(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote import opt_tree_forward as forward
        from sglang.srt.speculative.standalone_remote.sr_protocol import SRAction

        cleared = []
        host = _OptWindowHost(assign_pool=True)
        host._sr_device_poisoned = False
        host._sr_clear_stamps_after_poison = lambda: cleared.append(1)

        def grow(_scheduler, _reqs):
            raise RuntimeError("cudaErrorIllegalAddress")

        with patch.object(forward, "grow_batch", grow):
            with self.assertRaises(RuntimeError) as ctx:
                forward.produce_opt_windows(
                    host,
                    SRAction.STEP,
                    [(None, _OptReq()), (None, _OptReq())],
                )
        self.assertIn("cudaErrorIllegalAddress", str(ctx.exception))
        self.assertNotIn("expand", host.calls)
        self.assertTrue(host._sr_device_poisoned)
        self.assertEqual(cleared, [1])

        plain = _OptWindowHost(assign_pool=True)
        plain._sr_device_poisoned = False

        def ordinary(_scheduler, _reqs):
            raise RuntimeError("shape mismatch")

        with patch.object(forward, "grow_batch", ordinary):
            windows = forward.produce_opt_windows(
                plain,
                SRAction.STEP,
                [(None, _OptReq()), (None, _OptReq())],
            )
        self.assertEqual(windows, [([7], [-1], [0]), ([7], [-1], [0])])
        self.assertEqual(plain.calls.count("expand"), 2)
        self.assertFalse(plain._sr_device_poisoned)

    def test_layer_forward_device_error_poisons(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            grow_active_layers,
        )

        config = OptTreeConfig(width=2, tau=-1.0, max_depth=4)
        tails = [torch.zeros(4)]
        cleared = []
        scheduler = SimpleNamespace(
            _sr_device_poisoned=False,
            _sr_clear_stamps_after_poison=lambda: cleared.append(1),
        )

        def device_forward(_steps, _indices):
            raise RuntimeError("cudaErrorIllegalAddress")

        with self.assertRaises(RuntimeError) as ctx:
            grow_active_layers(config, tails, device_forward, scheduler)
        self.assertIn("cudaErrorIllegalAddress", str(ctx.exception))
        self.assertTrue(scheduler._sr_device_poisoned)
        self.assertEqual(cleared, [1])

        with self.assertRaises(RuntimeError):
            grow_active_layers(config, tails, device_forward, None)

        plain = SimpleNamespace(
            _sr_device_poisoned=False,
            _sr_clear_stamps_after_poison=lambda: cleared.append(1),
        )

        def ordinary_forward(_steps, _indices):
            raise RuntimeError("shape mismatch")

        results = grow_active_layers(config, tails, ordinary_forward, plain)
        self.assertEqual(results, [None])
        self.assertFalse(plain._sr_device_poisoned)
        self.assertEqual(cleared, [1])

    def test_expand_failure_counts_empty_draft(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        expand = _load_opt_expand_one()

        def release(_req):
            raise RuntimeError("kv")

        host = SimpleNamespace(
            _sr_resume_req=lambda _req: None,
            _sr_isolate_need=lambda _rids: None,
            _opt_release_uncommitted=release,
            _opt_rollback=lambda _req, _keep: None,
            _sr_is_finished=lambda _req: False,
            _sr_pause_req=lambda _req: None,
            last_batch=None,
            cur_batch=None,
        )
        req = SimpleNamespace(rid="opt-fail", origin_input_ids=[1], output_ids=[])
        window = expand(host, req)
        self.assertEqual(window, ([], None, None))
        self.assertEqual(
            get_sr_round_metrics(host, "Draft").counts["opt_expand_failures"], 1
        )
        self.assertIsNone(getattr(req, "opt_tree_lease", None))

    def test_expand_device_error_reraises_through_rollback(self):
        from types import SimpleNamespace

        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        expand = _load_opt_expand_one()
        cleared = []
        paused = []

        def release(_req):
            raise RuntimeError("cudaErrorIllegalAddress")

        def rollback(_req, _keep):
            raise RuntimeError("acl stream synchronize failed")

        host = SimpleNamespace(
            _sr_resume_req=lambda _req: None,
            _sr_isolate_need=lambda _rids: None,
            _opt_release_uncommitted=release,
            _opt_rollback=rollback,
            _sr_is_finished=lambda _req: False,
            _sr_pause_req=lambda _req: paused.append(1),
            _sr_device_poisoned=False,
            _sr_clear_stamps_after_poison=lambda: cleared.append(1),
            last_batch="stale",
            cur_batch="stale",
        )
        req = SimpleNamespace(rid="opt-device", origin_input_ids=[1], output_ids=[])
        with self.assertRaises(RuntimeError) as ctx:
            expand(host, req)
        self.assertIn("cudaErrorIllegalAddress", str(ctx.exception))
        self.assertNotIn("acl stream synchronize failed", str(ctx.exception))
        self.assertTrue(host._sr_device_poisoned)
        self.assertEqual(cleared, [1])
        self.assertEqual(paused, [1])
        self.assertIsNone(host.last_batch)
        self.assertIsNone(host.cur_batch)
        self.assertEqual(
            get_sr_round_metrics(host, "Draft").counts["opt_expand_failures"], 0
        )
        self.assertIsNone(getattr(req, "opt_tree_lease", None))

    def test_grow_batch_records_draft_stage_times(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote import opt_tree_forward as forward
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        syncs = []
        host = _opt_stage_host()
        req = _opt_stage_req("opt-batch")

        def ingest(_scheduler, _runners, phases=None):
            return [torch.zeros(4)]

        def grow(_config, _tails, forward_active, _scheduler, states_out=None, **_kwargs):
            forward_active(["layer-0"], [0])
            forward_active(["layer-1"], [0])
            return [([1], [-1], [0])]

        def fitting(_scheduler, _runners, _steps, **_kwargs):
            return [torch.zeros(1)], 1

        metrics = get_sr_round_metrics(host, "Draft")
        with patch.object(forward, "ingest_tail_logits", ingest), patch.object(
            forward, "grow_active_layers", grow
        ), patch.object(forward, "_forward_verify_fitting", fitting), patch.object(
            forward.torch, "get_device_module", lambda _device: _SyncProbe(syncs)
        ):
            with metrics.round():
                windows = forward.grow_batch(host, [req])
        self.assertEqual(windows, [([1], [-1], [0])])
        self.assertEqual(syncs, [1])
        self.assertIn("opt_ingest", metrics.host)
        self.assertIn("opt_grow", metrics.host)
        self.assertGreaterEqual(metrics.host["opt_ingest"], 0)
        self.assertGreaterEqual(metrics.host["opt_grow"], 0)
        self.assertEqual(metrics.counts["opt_layer_forwards"], 2)
        self.assertEqual(metrics.counts["opt_rounds"], 1)

    def test_expand_records_draft_stage_times(self):
        import time
        from types import SimpleNamespace

        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        syncs = []
        expand = _load_opt_expand_one()
        globals_ = expand.__globals__
        globals_["time"] = time
        globals_["torch"] = SimpleNamespace(
            get_device_module=lambda _device: _SyncProbe(syncs)
        )
        globals_["OptTreeConfig"] = OptTreeConfig

        class _Runner:
            def __init__(self, _scheduler, _req, _prefix_len):
                self.steps = []

            def forward_layer(self, step):
                self.steps.append(step)
                return torch.zeros(1)

        globals_["OptLayerForward"] = _Runner

        def ingest(_scheduler, _runners, phases=None):
            return [torch.zeros(4)]

        def grow(
            _config, _tails, forward_active, _scheduler=None, states_out=None, **_kwargs
        ):
            forward_active(["layer-0"], [0])
            forward_active(["layer-1"], [0])
            return [([1], [-1], [0])]

        globals_["ingest_tail_logits"] = ingest
        globals_["grow_active_layers"] = grow
        host = _opt_stage_host()
        req = _opt_stage_req("opt-expand")
        metrics = get_sr_round_metrics(host, "Draft")
        with metrics.round():
            window = expand(host, req)
        self.assertEqual(window, ([1], [-1], [0]))
        self.assertEqual(syncs, [1])
        self.assertIn("opt_ingest", metrics.host)
        self.assertIn("opt_grow", metrics.host)
        self.assertGreaterEqual(metrics.host["opt_ingest"], 0)
        self.assertGreaterEqual(metrics.host["opt_grow"], 0)
        self.assertEqual(metrics.counts["opt_layer_forwards"], 2)
        self.assertEqual(metrics.counts["opt_rounds"], 1)

    def test_admit_ordinary_error_falls_back_to_layer_grow(self):
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        def admit(*_args, **_kwargs):
            raise RuntimeError("CUDA out of memory")

        expand, called = _load_expand_with_admit(admit)
        host = _expand_admit_host()
        req = _opt_stage_req("opt-admit-oom")
        metrics = get_sr_round_metrics(host, "Draft")
        with metrics.round():
            window = expand(host, req)
        self.assertEqual(window, ([1], [-1], [0]))
        self.assertEqual(called["grow"], 1)
        self.assertEqual(called["ingest"], 1)
        self.assertEqual(called["lookahead"], 0)
        self.assertEqual(metrics.counts["opt_lookahead_fallbacks"], 1)
        self.assertEqual(metrics.counts["opt_expand_failures"], 0)
        self.assertFalse(host._sr_device_poisoned)
        self.assertFalse(host._opt_lookahead_active)

    def test_admit_device_error_reraises(self):
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        def admit(*_args, **_kwargs):
            raise RuntimeError("cudaErrorIllegalAddress")

        expand, called = _load_expand_with_admit(admit)
        host = _expand_admit_host()
        cleared = []
        host._sr_clear_stamps_after_poison = lambda: cleared.append(1)
        req = _opt_stage_req("opt-admit-device")
        metrics = get_sr_round_metrics(host, "Draft")
        with self.assertRaises(RuntimeError) as ctx:
            with metrics.round():
                expand(host, req)
        self.assertIn("cudaErrorIllegalAddress", str(ctx.exception))
        self.assertTrue(host._sr_device_poisoned)
        self.assertEqual(cleared, [1, 1])
        self.assertEqual(called["grow"], 0)
        self.assertEqual(called["ingest"], 0)
        self.assertEqual(called["lookahead"], 0)
        self.assertEqual(metrics.counts["opt_expand_failures"], 0)
        self.assertEqual(metrics.counts["opt_lookahead_fallbacks"], 0)

    def test_layer_timing_stays_off_without_an_active_round(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote import opt_tree_forward as forward
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _commit_prob_rows,
            _forward_verify_batch,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        touched = []
        host = _layer_timing_host()
        metrics = get_sr_round_metrics(host, "Draft")
        state = OptTreeState.create(OptTreeConfig(width=2, tau=0.1, max_depth=4))
        with patch.object(
            forward.torch,
            "get_device_module",
            lambda _device: touched.append(1),
        ), patch.object(
            forward, "_verify_batch_many", lambda *_args, **_kwargs: "batch"
        ), patch.object(
            forward, "_run_batch", lambda _scheduler, _batch: torch.zeros(2, 3)
        ):
            logits = _forward_verify_batch(
                host, [_LayerSlots()], [_layer_timing_step()]
            )
            _commit_prob_rows(
                [state],
                [0],
                [torch.tensor([0.1, 0.2, 0.3, 0.4])],
                seed=True,
                metrics=metrics,
            )
        self.assertEqual(touched, [])
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(metrics.counts["opt_layer_host_us"], 0)
        self.assertEqual(metrics.counts["opt_layer_launch_us"], 0)
        self.assertEqual(metrics.counts["opt_layer_gpu_us"], 0)
        self.assertEqual(metrics.counts["opt_layer_d2h_wait_us"], 0)
        self.assertEqual(metrics.counts["opt_seed_d2h_wait_us"], 0)
        self.assertIsNone(getattr(host, "_opt_layer_events", None))

    def test_layer_timing_splits_host_gpu_and_d2h(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote import opt_tree_forward as forward
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            OptLayerForward,
            _commit_prob_rows,
            _forward_verify_batch,
            drain_opt_layer_gpu_us,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        samples = [
            10.0,
            10.0 + 250e-6,
            10.0 + 250e-6 + 1800e-6,
            20.0,
            20.0 + 250e-6,
            20.0 + 250e-6 + 1800e-6,
            30.0,
            30.0 + 400e-6,
            40.0,
            40.0 + 150e-6,
        ]
        clock = _SampleClock(samples)
        syncs = []
        created = []
        device = _TimingDevice(syncs, created)
        host = _layer_timing_host()
        metrics = get_sr_round_metrics(host, "Draft")
        runner = _LayerSlots()
        layer = OptLayerForward.__new__(OptLayerForward)
        layer.scheduler = host
        layer.req = SimpleNamespace(kv_allocated_len=0)
        layer.prefix_len = 4
        layer.cached_len = 4
        layer._alloc_layer_slots = lambda n, start=None: torch.arange(n)
        layer._verify_batch = lambda *_args, **_kwargs: "batch"
        layer._run = lambda _batch: torch.zeros(2, 4)

        def batch_many(*_args, **_kwargs):
            return "batch"

        def run_batch(_scheduler, _batch):
            return torch.zeros(2, 3)

        metrics.active = True
        try:
            with patch.object(forward.time, "perf_counter", clock), patch.object(
                forward.torch, "get_device_module", lambda _device: device
            ), patch.object(forward, "_verify_batch_many", batch_many), patch.object(
                forward, "_run_batch", run_batch
            ):
                _forward_verify_batch(host, [runner], [_layer_timing_step()])
                layer.forward_layer(_layer_timing_step())
                state = OptTreeState.create(OptTreeConfig(width=2, tau=0.1, max_depth=4))
                _commit_prob_rows(
                    [state],
                    [0],
                    [torch.tensor([0.1, 0.2, 0.3, 0.4])],
                    seed=True,
                    metrics=metrics,
                )
                _commit_prob_rows(
                    [state],
                    [0],
                    [torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])],
                    seed=False,
                    metrics=metrics,
                )
                self.assertEqual(len(host._opt_layer_events), 2)
                self.assertEqual(syncs, [])
                drain_opt_layer_gpu_us(host, metrics)
        finally:
            metrics.active = False
        self.assertEqual(clock.calls, len(samples))
        self.assertEqual(
            metrics.counts["opt_layer_host_us"],
            int((samples[1] - samples[0]) * 1e6)
            + int((samples[4] - samples[3]) * 1e6),
        )
        self.assertEqual(
            metrics.counts["opt_layer_launch_us"],
            int((samples[2] - samples[1]) * 1e6)
            + int((samples[5] - samples[4]) * 1e6),
        )
        self.assertEqual(
            metrics.counts["opt_seed_d2h_wait_us"],
            int((samples[7] - samples[6]) * 1e6),
        )
        self.assertEqual(
            metrics.counts["opt_layer_d2h_wait_us"],
            int((samples[9] - samples[8]) * 1e6),
        )
        self.assertEqual(metrics.counts["opt_layer_gpu_us"], 3000)
        self.assertEqual(syncs, [1])
        self.assertEqual(host._opt_layer_events, [])
        self.assertEqual(runner.cached_len, 6)
        self.assertEqual(layer.cached_len, 6)
        self.assertTrue(all(event.recorded == 1 for event in created))

    def test_layer_timing_replay_error_keeps_the_event_pair_off_the_queue(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote import opt_tree_forward as forward
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _forward_verify_batch,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        samples = [10.0, 10.0 + 250e-6]
        clock = _SampleClock(samples)
        device = _TimingDevice([], [])
        host = _layer_timing_host()
        metrics = get_sr_round_metrics(host, "Draft")
        runner = _LayerSlots()

        def run_batch(_scheduler, _batch):
            raise RuntimeError("replay failed")

        metrics.active = True
        try:
            with patch.object(forward.time, "perf_counter", clock), patch.object(
                forward.torch, "get_device_module", lambda _device: device
            ), patch.object(
                forward, "_verify_batch_many", lambda *_a, **_k: "batch"
            ), patch.object(forward, "_run_batch", run_batch):
                with self.assertRaisesRegex(RuntimeError, "replay failed"):
                    _forward_verify_batch(host, [runner], [_layer_timing_step()])
        finally:
            metrics.active = False
        self.assertEqual(clock.calls, len(samples))
        self.assertEqual(
            metrics.counts["opt_layer_host_us"],
            int((samples[1] - samples[0]) * 1e6),
        )
        self.assertEqual(metrics.counts["opt_layer_launch_us"], 0)
        self.assertIsNone(getattr(host, "_opt_layer_events", None))
        self.assertEqual(runner.cached_len, 4)

    def test_layer_timing_drain_skips_sync_when_idle_or_poisoned(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote import opt_tree_forward as forward
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            drain_opt_layer_gpu_us,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        syncs = []
        host = _layer_timing_host()
        metrics = get_sr_round_metrics(host, "Draft")

        class _Device:
            def synchronize(self):
                syncs.append(1)

        with patch.object(forward.torch, "get_device_module", lambda _device: _Device()):
            drain_opt_layer_gpu_us(host, metrics)
            host._opt_layer_events = []
            drain_opt_layer_gpu_us(host, metrics)
            host._sr_device_poisoned = True
            host._opt_layer_events = [(object(), object())]
            drain_opt_layer_gpu_us(host, metrics)
        self.assertEqual(syncs, [])
        self.assertEqual(host._opt_layer_events, [])
        self.assertEqual(metrics.counts["opt_layer_gpu_us"], 0)

        host._sr_device_poisoned = False
        host._opt_layer_events = [(object(), object())]

        class _Dead:
            def synchronize(self):
                raise RuntimeError("cudaErrorIllegalAddress")

        with patch.object(forward.torch, "get_device_module", lambda _device: _Dead()):
            with self.assertRaisesRegex(RuntimeError, "cudaErrorIllegalAddress"):
                drain_opt_layer_gpu_us(host, metrics)
        self.assertEqual(metrics.counts["device_timing_dropped_opt_layer"], 0)
        self.assertEqual(host._opt_layer_events, [])

        host._opt_layer_events = [(object(), object())]

        class _ClockSkew:
            def synchronize(self):
                raise RuntimeError("clock skew")

        with patch.object(
            forward.torch, "get_device_module", lambda _device: _ClockSkew()
        ):
            drain_opt_layer_gpu_us(host, metrics)
        self.assertEqual(metrics.counts["device_timing_dropped_opt_layer"], 1)
        self.assertEqual(metrics.counts["opt_layer_gpu_us"], 0)

    def test_ingest_phases_sum_to_opt_ingest_and_other_is_nonnegative(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            OPT_INGEST_PHASES,
            flush_opt_ingest_phases,
            opt_other_seconds,
        )

        class _Metrics:
            def __init__(self):
                self.active = True
                self.host = {}

            def add_host(self, name, seconds):
                self.host[name] = self.host.get(name, 0.0) + float(seconds)

        metrics = _Metrics()
        parts = {
            "opt_ingest_plan": 0.001,
            "opt_ingest_alloc": 0.002,
            "opt_ingest_batch": 0.003,
            "opt_ingest_forward": 0.004,
        }
        flush_opt_ingest_phases(metrics, parts, 0.010)
        self.assertAlmostEqual(
            sum(metrics.host[name] for name in OPT_INGEST_PHASES),
            metrics.host["opt_ingest"],
        )
        self.assertAlmostEqual(metrics.host["opt_ingest"], 0.010)
        self.assertAlmostEqual(metrics.host["opt_ingest_batch"], 0.003)

        self.assertAlmostEqual(opt_other_seconds(0.020, 0.008, 0.011), 0.001)
        self.assertEqual(opt_other_seconds(0.001, 0.008, 0.011), 0.0)

        inactive = _Metrics()
        inactive.active = False
        flush_opt_ingest_phases(inactive, parts, 0.010)
        self.assertEqual(inactive.host, {})

    def test_materialize_tree_seed_switch(self):
        from types import SimpleNamespace

        def run(capture_tree_seed=None):
            host = _materialize_host_from_source()
            calls = []
            host._sr_enable_tree_seed_hidden = lambda _batch: calls.append("enable")
            host._sr_cache_tree_seeds = lambda *_args, **_kwargs: calls.append("cache")
            origin = [1, 2, 3]
            req = SimpleNamespace(
                rid="opt-seed",
                origin_input_ids=list(origin),
                output_ids=[],
                req_pool_idx=None,
                kv_committed_len=0,
                draft_is_paused=False,
                draft_generation_start_len=0,
                finished=lambda: False,
            )

            class _Extend:
                def is_extend(self):
                    return True

            def next_batch():
                req.req_pool_idx = 1
                req.kv_committed_len = len(origin)
                host.chunked_req = None
                return SimpleNamespace(
                    reqs=[req],
                    forward_mode=_Extend(),
                    is_extend_in_batch=False,
                )

            host.get_next_batch_to_run = next_batch
            host.run_batch = lambda _batch: SimpleNamespace(logits_output=None)
            host.process_batch_result = lambda *_args, **_kwargs: None
            kwargs = {}
            if capture_tree_seed is not None:
                kwargs["capture_tree_seed"] = capture_tree_seed
            host._sr_materialize_prefix_batch([req], **kwargs)
            return calls

        self.assertEqual(run(), ["enable", "cache"])
        self.assertEqual(run(False), [])

    def test_opt_materialize_forces_tree_seed_off(self):
        from types import SimpleNamespace

        opt_cls = _load_opt_materialize_override()

        def captured(is_opt, requested):
            host = opt_cls()
            host.spec_algorithm = SimpleNamespace(
                is_opt_tree_remote=lambda: is_opt
            )
            host._sr_materialize_prefix_batch([], capture_tree_seed=requested)
            return host.captured

        self.assertFalse(captured(True, True))
        self.assertTrue(captured(False, True))
        self.assertFalse(captured(False, False))

    def test_layer_alloc_records_slots_and_extend_does_not(self):
        import sys
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            OptLayerForward,
        )

        runner = OptLayerForward.__new__(OptLayerForward)
        runner.layer_slots = []
        runner.cached_len = 4
        runner.req = SimpleNamespace(req_pool_idx=0)
        runner.scheduler = SimpleNamespace(
            tree_cache=SimpleNamespace(page_size=1),
            req_to_token_pool=SimpleNamespace(
                device="cpu",
                req_to_token=torch.full((1, 8), -1, dtype=torch.int32),
            ),
        )
        loc = torch.tensor([5, 6, 7])
        common = SimpleNamespace(
            alloc_token_slots=lambda _tree, n: loc[: int(n)].clone(),
            alloc_paged_token_slots_extend=lambda *_args, **_kwargs: loc,
            get_last_loc=lambda *_args, **_kwargs: None,
        )
        spec_utils = SimpleNamespace(
            assign_req_to_token_pool_func=lambda *_args, **_kwargs: None
        )
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.mem_cache.common": common,
                "sglang.srt.speculative.spec_utils": spec_utils,
            },
        ):
            recorded = runner._alloc_layer_slots(3)
            skipped = runner._alloc_layer_slots(1, start=4)
        self.assertEqual(recorded.tolist(), [5, 6, 7])
        self.assertEqual(len(runner.layer_slots), 1)
        self.assertEqual(runner.layer_slots[0].tolist(), [5, 6, 7])
        self.assertEqual(skipped.tolist(), [5])

    def test_decode_append_replays_one_token_without_spec_info(self):
        import sys
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _decode_append_batch,
            _decode_uncommitted_batch,
        )

        class _DecodeMode:
            name = "DECODE"

        forward_info = SimpleNamespace(ForwardMode=SimpleNamespace(DECODE=_DecodeMode()))

        class _Info:
            is_all_greedy = False
            temperatures = torch.ones(1, 1)
            logit_bias = None
            has_custom_logit_processor = False
            penalizer_orchestrator = None
            need_top_p_sampling = True
            need_top_k_sampling = True
            need_min_p_sampling = True
            acc_additive_penalties = None
            acc_scaling_penalties = None
            vocab_mask = None

        class _Runner:
            def __init__(self, prefix, origin, output, kv):
                self.prefix_len = prefix
                self.cached_len = kv
                self.layer_slots = []
                self.alloc_starts = []
                self.req = SimpleNamespace(
                    rid="decode",
                    req_pool_idx=0,
                    kv_committed_len=kv,
                    kv_allocated_len=kv,
                    origin_input_ids=list(origin),
                    output_ids=list(output),
                )

            def _alloc_layer_slots(self, n, start=None):
                self.alloc_starts.append(start)
                return torch.tensor([40 + int(start)])

        def run(runners):
            batch = SimpleNamespace(
                seq_lens=torch.tensor([runners[0].req.kv_committed_len]),
                sampling_info=_Info(),
                has_grammar=False,
                return_logprob=True,
                return_hidden_states=True,
                spec_info="verify",
                forward_mode=None,
                global_forward_mode=None,
            )
            seen = []

            def make_batch(_reqs):
                return batch

            def run_batch(built):
                seen.append(
                    {
                        "input_ids": [int(item) for item in built.input_ids],
                        "seq_lens": [int(item) for item in built.seq_lens],
                        "spec_info": built.spec_info,
                        "forward_mode": built.forward_mode,
                    }
                )
                return SimpleNamespace(
                    logits_output=SimpleNamespace(
                        next_token_logits=torch.zeros(len(runners), 4)
                    )
                )

            host = SimpleNamespace(
                _sr_make_decode_batch=make_batch,
                run_batch=run_batch,
                cur_batch=None,
                _opt_rollback=lambda *_args, **_kwargs: None,
            )
            return host, batch, seen

        one = _Runner(4, [1, 2, 3], [9], 3)
        host, batch, seen = run([one])
        with patch.dict(
            sys.modules,
            {"sglang.srt.model_executor.forward_batch_info": forward_info},
        ):
            rows = _decode_append_batch(host, [one])
        self.assertEqual(tuple(rows[0].shape), (4,))
        self.assertIsNone(batch.spec_info)
        self.assertEqual(batch.forward_mode.name, "DECODE")
        self.assertEqual(batch.global_forward_mode.name, "DECODE")
        self.assertEqual(int(batch.seq_lens[0]), 4)
        self.assertEqual(int(batch.input_ids[0]), 9)
        self.assertEqual(int(batch.out_cache_loc[0]), 43)
        self.assertEqual(one.req.kv_committed_len, 4)
        self.assertEqual(one.req.kv_allocated_len, 4)
        self.assertEqual(one.cached_len, 4)
        self.assertEqual(one.layer_slots, [])
        self.assertEqual(one.alloc_starts, [3])
        self.assertEqual(len(seen), 1)

        two = _Runner(5, [1, 2, 3], [8, 9], 3)
        host, batch, seen = run([two])
        with patch.dict(
            sys.modules,
            {"sglang.srt.model_executor.forward_batch_info": forward_info},
        ):
            rows = _decode_uncommitted_batch(host, [two])
        self.assertEqual(tuple(rows[0].shape), (4,))
        self.assertEqual(two.alloc_starts, [3, 4])
        self.assertEqual(seen[0]["input_ids"], [8])
        self.assertEqual(seen[1]["input_ids"], [9])
        self.assertEqual(seen[0]["seq_lens"], [4])
        self.assertEqual(seen[1]["seq_lens"], [5])
        self.assertIsNone(seen[0]["spec_info"])
        self.assertIsNone(seen[1]["spec_info"])
        self.assertEqual(seen[0]["forward_mode"].name, "DECODE")
        self.assertEqual(seen[1]["forward_mode"].name, "DECODE")
        self.assertEqual(two.req.kv_committed_len, 5)
        self.assertEqual(two.layer_slots, [])

    def test_publish_keeps_page1_slots_and_skips_other_pages(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import (
            publish_opt_tree_lease,
        )

        host, pool, _freed = _lease_host()
        req = _lease_req(keep=3, allocated=5)
        runner = SimpleNamespace(layer_slots=[torch.tensor([10, 11])])
        published = publish_opt_tree_lease(
            host, req, runner, 3, ([8, 9], [-1, 0], [1, 0])
        )
        self.assertTrue(published)
        self.assertEqual(req.kv_allocated_len, 3)
        self.assertEqual(req.kv_committed_len, 3)
        self.assertEqual(pool[0, 3:5].tolist(), [-1, -1])
        self.assertEqual(req.opt_tree_lease.layer_slots, ((10, 11),))
        self.assertEqual(req.opt_tree_lease.depths, (0, 1))

        npu, npu_pool, npu_freed = _lease_host(page=128)
        npu_req = _lease_req(keep=3, allocated=5)
        self.assertFalse(
            publish_opt_tree_lease(
                npu, npu_req, runner, 3, ([8], [-1], [0])
            )
        )
        self.assertIsNone(getattr(npu_req, "opt_tree_lease", None))
        self.assertEqual(npu_req.kv_allocated_len, 5)
        self.assertEqual(npu_pool[0, 3:5].tolist(), [7, 7])
        self.assertEqual(npu_freed, [])

        poisoned, _poison_pool, _poison_freed = _lease_host()
        poisoned._sr_device_poisoned = True
        poison_req = _lease_req(keep=3, allocated=5)
        self.assertFalse(
            publish_opt_tree_lease(
                poisoned, poison_req, runner, 3, ([8], [-1], [0])
            )
        )
        self.assertIsNone(getattr(poison_req, "opt_tree_lease", None))

    def test_apply_rewrites_accepted_slots_and_frees_the_rest(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import (
            OptTreeKVLease,
            consume_opt_tree_lease,
            release_opt_tree_lease,
        )

        host, pool, freed = _lease_host(width=8)
        req = _lease_req(keep=3, allocated=3)
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = [20, 21, 99]
        req.sr_align_result = SimpleNamespace(kind="append_n", old_local_len=3)
        req.opt_tree_lease = OptTreeKVLease(
            keep=3,
            layer_slots=((100, 101, 102), (200, 201, 202)),
            tokens=(20, 21, 30),
            parents=(-1, 0, -1),
            cols=(1, 0, 2),
            depths=(0, 1, 0),
        )
        consume_opt_tree_lease(host, req)
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        counts = get_sr_round_metrics(host, "Draft").counts
        self.assertEqual(counts["opt_tree_depth_2"], 1)
        self.assertEqual(counts["opt_accept_depth_2"], 1)
        self.assertEqual(counts["opt_lease_dropped_misaligned"], 0)
        self.assertEqual(counts["opt_lease_dropped_path"], 0)
        self.assertEqual(pool[0, 3].item(), 101)
        self.assertEqual(pool[0, 4].item(), 200)
        self.assertEqual(req.kv_committed_len, 5)
        self.assertEqual(req.kv_allocated_len, 5)
        self.assertIsNone(req.opt_tree_lease)
        self.assertCountEqual(freed, [100, 102, 201, 202])

        host, pool, freed = _lease_host()
        req = _lease_req(keep=3, allocated=3)
        req.output_ids = [7]
        req.sr_align_result = SimpleNamespace(kind="append_one", old_local_len=3)
        req.opt_tree_lease = _sample_lease(3)
        pool[0, 3] = 4
        consume_opt_tree_lease(host, req)
        self.assertEqual(pool[0, 3].item(), 4)
        self.assertEqual(req.kv_committed_len, 3)
        self.assertEqual(req.kv_allocated_len, 3)
        self.assertIsNone(req.opt_tree_lease)
        self.assertCountEqual(freed, [100, 101])
        release_opt_tree_lease(host, req)
        self.assertEqual(freed, [100, 101])

    def test_apply_stops_before_unforwarded_node(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import (
            OptTreeKVLease,
            consume_opt_tree_lease,
        )

        host, pool, freed = _lease_host()
        req = _lease_req(keep=2, allocated=2)
        req.origin_input_ids = [1, 2]
        req.output_ids = [20, 21]
        req.sr_align_result = SimpleNamespace(kind="append_n", old_local_len=2)
        req.opt_tree_lease = OptTreeKVLease(
            keep=2,
            layer_slots=((100, 101),),
            tokens=(20, 21),
            parents=(-1, 0),
            cols=(0, 1),
            depths=(0, 1),
        )
        consume_opt_tree_lease(host, req)
        self.assertEqual(pool[0, 2].item(), 100)
        self.assertEqual(req.kv_committed_len, 3)
        self.assertEqual(req.kv_allocated_len, 3)
        self.assertCountEqual(freed, [101])

    def test_apply_drops_lease_when_depth_does_not_match_step(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import (
            OptTreeKVLease,
            consume_opt_tree_lease,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        host, pool, freed = _lease_host()
        req = _lease_req(keep=3, allocated=3)
        req.output_ids = [20]
        req.sr_align_result = SimpleNamespace(kind="append_one", old_local_len=3)
        req.opt_tree_lease = OptTreeKVLease(
            keep=3,
            layer_slots=((100, 101),),
            tokens=(20,),
            parents=(-1,),
            cols=(0,),
            depths=(1,),
        )
        pool[0, 3] = 4
        consume_opt_tree_lease(host, req)
        self.assertEqual(pool[0, 3].item(), 4)
        self.assertEqual(req.kv_committed_len, 3)
        self.assertIsNone(req.opt_tree_lease)
        self.assertCountEqual(freed, [100, 101])
        path_counts = get_sr_round_metrics(host, "Draft").counts
        self.assertEqual(path_counts["opt_lease_dropped_path"], 1)
        self.assertEqual(path_counts["opt_lease_dropped_misaligned"], 0)

        host, _pool, freed = _lease_host()
        req = _lease_req(keep=3, allocated=3)
        req.output_ids = [20]
        req.sr_align_result = SimpleNamespace(kind="equal", old_local_len=3)
        req.opt_tree_lease = _sample_lease(3)
        consume_opt_tree_lease(host, req)
        self.assertIsNone(req.opt_tree_lease)
        self.assertCountEqual(freed, [100, 101])
        align_counts = get_sr_round_metrics(host, "Draft").counts
        self.assertEqual(align_counts["opt_lease_dropped_misaligned"], 1)
        self.assertEqual(align_counts["opt_lease_dropped_path"], 0)

    def test_grow_batch_publishes_on_success_and_rolls_back_on_failure(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            grow_batch,
        )

        host, pool, _freed = _lease_host()
        rolls = []
        host.device = "cpu"
        host.chunked_prefill_size = None
        host.server_args = SimpleNamespace(
            opt_tree_width=2, opt_tree_tau=0.1, opt_tree_max_depth=4
        )
        host._sr_resume_req = lambda _req: None
        host._sr_isolate_need = lambda _rids: None
        host._opt_release_uncommitted = lambda _req: None
        host._opt_rollback = lambda _req, keep: rolls.append(int(keep))
        host._sr_is_finished = lambda _req: False
        host._sr_pause_req = lambda _req: None
        host.last_batch = "stale"
        host.cur_batch = "stale"
        req = _lease_req(keep=3, allocated=5)

        def ingest(_scheduler, runners, phases=None):
            runners[0].layer_slots.append(torch.tensor([10, 11]))
            return [torch.zeros(4)]

        def grow(
            _config, _tails, _forward_active, _scheduler, states_out=None, **_kwargs
        ):
            return [([8], [-1], [1])]

        with patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward.ingest_tail_logits",
            ingest,
        ), patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward.grow_active_layers",
            grow,
        ):
            windows = grow_batch(host, [req])
        self.assertEqual(windows, [([8], [-1], [1])])
        self.assertEqual(rolls, [])
        self.assertEqual(req.kv_allocated_len, 3)
        self.assertEqual(pool[0, 3:5].tolist(), [-1, -1])
        self.assertEqual(req.opt_tree_lease.layer_slots, ((10, 11),))

        rolls.clear()
        failed = _lease_req(keep=3, allocated=5)

        def boom(*_args, **_kwargs):
            raise RuntimeError("layer failed")

        with patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward.ingest_tail_logits",
            ingest,
        ), patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward.grow_active_layers",
            boom,
        ):
            with self.assertRaises(RuntimeError):
                grow_batch(host, [failed])
        self.assertEqual(rolls, [3])
        self.assertIsNone(getattr(failed, "opt_tree_lease", None))
        self.assertEqual(failed.kv_allocated_len, 5)

    def test_cleanup_paths_release_or_retain_the_lease(self):
        opt_cls = _load_opt_lease_overrides()
        host = opt_cls()
        host.events = []
        host._sr_device_poisoned = False
        host.spec_algorithm = SimpleNamespace(is_opt_tree_remote=lambda: True)
        host.server_args = SimpleNamespace(opt_tree_role="draft")
        _freed = []
        host.token_to_kv_pool_allocator = _RecordingAllocator(_freed)
        host.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.full((1, 4), -1, dtype=torch.int32)
        )
        host.tree_cache = SimpleNamespace(page_size=1)
        req = _lease_req(keep=2, allocated=2)
        req.opt_tree_lease = _sample_lease(2)
        state = SimpleNamespace(req_object=req)
        host._opt_la_batches = {
            "rid": {"pool_idx": 0, "batch": object()},
            "other": {"pool_idx": 1, "batch": object()},
        }

        host._sr_finish_rid("rid", state=state)
        self.assertEqual(host.events, ["finish"])
        self.assertNotIn("rid", host._opt_la_batches)
        self.assertIn("other", host._opt_la_batches)
        self.assertIsNone(req.opt_tree_lease)
        self.assertCountEqual(_freed, [100, 101])

        _freed.clear()
        host.events.clear()
        req.opt_tree_lease = _sample_lease(2)
        host._sr_rollback_else_reprefill(req, 1, 2, lambda: None)
        self.assertEqual(host.events, ["rollback"])
        self.assertIsNone(req.opt_tree_lease)
        self.assertCountEqual(_freed, [100, 101])

        _freed.clear()
        host.events.clear()
        req.opt_tree_lease = _sample_lease(2)
        host._sr_reset_linear_kv_state(req, [1, 2])
        self.assertEqual(host.events, ["reset"])
        self.assertIsNone(req.opt_tree_lease)
        self.assertCountEqual(_freed, [100, 101])

        host._sr_device_poisoned = True
        req.opt_tree_lease = _sample_lease(2)
        before = list(_freed)
        host._sr_finish_rid("rid", state=state)
        self.assertIsNotNone(req.opt_tree_lease)
        self.assertEqual(_freed, before)

        host._sr_abandon_poisoned_kv(req)
        self.assertIsNone(req.opt_tree_lease)
        self.assertEqual(_freed, before)
        owners = [record.owner for record in host._sr_kv_quarantine.records]
        self.assertTrue(any(isinstance(owner, tuple) and owner[0] is req for owner in owners))

    def test_lease_cleanup_stays_off_other_algorithms(self):
        opt_cls = _load_opt_lease_overrides()

        def assert_untouched(host):
            freed = []
            host.events = []
            host._sr_device_poisoned = False
            host.token_to_kv_pool_allocator = _RecordingAllocator(freed)
            host.req_to_token_pool = SimpleNamespace(
                req_to_token=torch.full((1, 4), -1, dtype=torch.int32)
            )
            host.tree_cache = SimpleNamespace(page_size=1)
            req = _lease_req(keep=2, allocated=2)
            lease = _sample_lease(2)
            req.opt_tree_lease = lease
            state = SimpleNamespace(req_object=req)
            host._sr_finish_rid("rid", state=state)
            host._sr_rollback_else_reprefill(req, 1, 2, lambda: None)
            host._sr_reset_linear_kv_state(req, [1, 2])
            host._sr_abandon_poisoned_kv(req)
            self.assertIs(req.opt_tree_lease, lease)
            self.assertFalse(lease.released)
            self.assertEqual(freed, [])
            self.assertIsNone(getattr(host, "_sr_kv_quarantine", None))
            self.assertEqual(host.events, ["finish", "rollback", "reset", "abandon"])

        other = opt_cls()
        other.spec_algorithm = SimpleNamespace(is_opt_tree_remote=lambda: False)
        other.server_args = SimpleNamespace(opt_tree_role="draft")
        assert_untouched(other)

        target = opt_cls()
        target.spec_algorithm = SimpleNamespace(is_opt_tree_remote=lambda: True)
        target.server_args = SimpleNamespace(opt_tree_role="target")
        assert_untouched(target)

        missing = opt_cls()
        assert_untouched(missing)

    def test_grow_active_layers_exports_unforwarded_last_layer(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            grow_active_layers,
        )

        calls = []

        def forward_active(steps, indices):
            calls.append(len(indices))
            return torch.zeros(len(indices), 2, 4)

        states = []
        windows = grow_active_layers(
            OptTreeConfig(width=2, tau=-1.0, max_depth=3),
            [torch.zeros(4)],
            forward_active,
            states_out=states,
        )
        self.assertEqual(calls, [1])
        self.assertEqual(len(windows), 1)
        self.assertIsNotNone(windows[0])
        self.assertEqual(len(states), 1)
        self.assertTrue(states[0].finished)
        self.assertEqual(states[0].depth, 2)
        self.assertEqual(tuple(states[0].anc_mask.shape), (2, 4))

    def test_deferred_last_layer_leaves_one_bonus_token(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            forward_deferred_final_layers,
        )
        from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import (
            OptTreeKVLease,
            consume_opt_tree_lease,
        )

        host, pool, _freed = _lease_host(width=8)
        host._sr_is_finished = lambda _req: False
        host._sr_pause_req = lambda _req: None
        host._opt_rollback = lambda _req, _keep: None
        host.last_batch = "stale"
        host.cur_batch = "stale"
        req = _lease_req(keep=3, allocated=5)
        req.kv_committed_len = 3
        runner = SimpleNamespace(
            req=req,
            prefix_len=3,
            cached_len=5,
            layer_slots=[torch.tensor([100, 101])],
        )
        state = _finished_two_layer_state()
        job = SimpleNamespace(
            req=req,
            runner=runner,
            keep=3,
            window=([20, 21], [-1, 0], [0, 0]),
            state=state,
        )

        def fitting(_scheduler, runners, steps, **_kwargs):
            step = steps[0]
            active = runners[0]
            n = int(step.tokens.shape[0])
            depth = int(step.positions.reshape(-1)[0].item())
            self.assertEqual(active.cached_len + n, active.prefix_len + depth * n)
            active.layer_slots.append(torch.tensor([200, 201]))
            active.req.kv_allocated_len = active.cached_len + n
            active.cached_len += n
            return [torch.zeros(n, 4)], 1

        with patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward._forward_verify_fitting",
            fitting,
        ):
            forward_deferred_final_layers(host, [job])
        self.assertEqual(req.opt_tree_lease.layer_slots, ((100, 101), (200, 201)))
        req.output_ids = [20, 21, 99]
        req.sr_align_result = SimpleNamespace(kind="append_n", old_local_len=3)
        consume_opt_tree_lease(host, req)
        self.assertEqual(len(req.origin_input_ids) + len(req.output_ids) - req.kv_committed_len, 1)
        self.assertEqual(pool[0, 3].item(), 100)
        self.assertEqual(pool[0, 4].item(), 200)

        bare, bare_pool, bare_freed = _lease_host(width=8)
        bare_req = _lease_req(keep=3, allocated=3)
        bare_req.origin_input_ids = [1, 2, 3]
        bare_req.output_ids = [20, 21, 99]
        bare_req.sr_align_result = SimpleNamespace(kind="append_n", old_local_len=3)
        bare_req.opt_tree_lease = OptTreeKVLease(
            keep=3,
            layer_slots=((100, 101),),
            tokens=(20, 21),
            parents=(-1, 0),
            cols=(0, 0),
            depths=(0, 1),
        )
        consume_opt_tree_lease(bare, bare_req)
        self.assertEqual(
            len(bare_req.origin_input_ids) + len(bare_req.output_ids) - bare_req.kv_committed_len,
            2,
        )
        self.assertEqual(bare_pool[0, 3].item(), 100)
        self.assertCountEqual(bare_freed, [101])

    def test_deferred_forward_failure_pops_the_last_layer(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            forward_deferred_final_layers,
        )

        host, pool, freed = _lease_host(width=8)
        host._sr_is_finished = lambda _req: False
        host._sr_pause_req = lambda _req: None
        host.last_batch = None
        host.cur_batch = None
        req = _lease_req(keep=3, allocated=5)
        runner = SimpleNamespace(
            req=req,
            prefix_len=3,
            cached_len=5,
            layer_slots=[torch.tensor([100, 101])],
        )

        def rollback(target, keep_len):
            allocated = int(target.kv_allocated_len)
            if allocated <= int(keep_len):
                return
            freed.extend(int(slot) for slot in pool[0, keep_len:allocated].tolist())
            target.kv_committed_len = int(keep_len)
            target.kv_allocated_len = int(keep_len)

        host._opt_rollback = rollback

        def fitting(_scheduler, runners, _steps, **_kwargs):
            active = runners[0]
            active.layer_slots.append(torch.tensor([200, 201]))
            active.req.kv_allocated_len = active.cached_len + 2
            pool[0, active.cached_len : active.cached_len + 2] = torch.tensor(
                [200, 201], dtype=pool.dtype
            )
            raise RuntimeError("final layer failed")

        job = SimpleNamespace(
            req=req,
            runner=runner,
            keep=3,
            window=([20, 21], [-1, 0], [0, 0]),
            state=_finished_two_layer_state(),
        )
        with patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward._forward_verify_fitting",
            fitting,
        ):
            forward_deferred_final_layers(host, [job])
        self.assertEqual(req.opt_tree_lease.layer_slots, ((100, 101),))
        self.assertNotIn(200, req.opt_tree_lease.all_slot_ids())
        self.assertNotIn(201, req.opt_tree_lease.all_slot_ids())
        self.assertEqual(req.kv_committed_len, 3)
        self.assertEqual(pool[0, 5:7].tolist(), [-1, -1])
        self.assertCountEqual(freed, [200, 201])

    def test_deferred_finish_releases_tree_slots(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            forward_deferred_final_layers,
        )

        host, pool, freed = _lease_host(width=8)
        req = _lease_req(keep=3, allocated=5)
        req.finished = lambda: True
        host._sr_is_finished = lambda target: target.finished()
        host._sr_pause_req = lambda _req: None
        host.last_batch = None
        host.cur_batch = None
        pool[0, 3:5] = torch.tensor([100, 101], dtype=pool.dtype)
        runner = SimpleNamespace(
            req=req,
            prefix_len=3,
            cached_len=5,
            layer_slots=[torch.tensor([100, 101])],
        )

        def rollback(target, keep_len):
            allocated = int(target.kv_allocated_len)
            if allocated <= int(keep_len):
                return
            freed.extend(int(slot) for slot in pool[0, keep_len:allocated].tolist())
            target.kv_committed_len = int(keep_len)
            target.kv_allocated_len = int(keep_len)

        host._opt_rollback = rollback
        called = []
        job = SimpleNamespace(
            req=req,
            runner=runner,
            keep=3,
            window=([20], [-1], [0]),
            state=_finished_two_layer_state(),
        )
        with patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward._forward_verify_fitting",
            lambda *_args, **_kwargs: called.append(1),
        ):
            forward_deferred_final_layers(host, [job])
        self.assertEqual(called, [])
        self.assertIsNone(req.opt_tree_lease)
        self.assertCountEqual(freed, [100, 101])
        self.assertEqual(req.kv_allocated_len, 3)

    def test_deferred_layer_forwards_count_one_per_depth_group(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            forward_deferred_final_layers,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        groups = []
        forwards = {"n": 1}

        def fitting(_scheduler, runners, _steps, **_kwargs):
            groups.append([runner.req.rid for runner in runners])
            return [], forwards["n"]

        def host():
            return SimpleNamespace(
                _sr_device_poisoned=False,
                _sr_is_finished=lambda _req: False,
                _sr_pause_req=lambda _req: None,
                _opt_rollback=lambda _req, _keep: None,
                last_batch=None,
                cur_batch=None,
            )

        def job(rid, depth):
            req = SimpleNamespace(rid=rid)
            runner = SimpleNamespace(
                req=req,
                lookahead_inflight=False,
                lookahead_dropped=False,
                _lookahead_satisfied=False,
                prefix_len=1,
                cached_len=1,
                layer_slots=[],
            )
            return SimpleNamespace(
                req=req,
                runner=runner,
                keep=1,
                window=None,
                state=SimpleNamespace(depth=depth, config=SimpleNamespace(width=2)),
            )

        with patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward._forward_verify_fitting",
            fitting,
        ), patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward.final_layer_step",
            lambda state: state,
        ):
            shared = host()
            forward_deferred_final_layers(shared, [job("a", 2), job("b", 2)])
            self.assertEqual(groups, [["a", "b"]])
            self.assertEqual(
                get_sr_round_metrics(shared, "Draft").counts[
                    "opt_layer_forwards_deferred"
                ],
                1,
            )
            groups.clear()
            split = host()
            forward_deferred_final_layers(split, [job("a", 1), job("b", 2)])
            self.assertEqual(groups, [["a"], ["b"]])
            self.assertEqual(
                get_sr_round_metrics(split, "Draft").counts[
                    "opt_layer_forwards_deferred"
                ],
                2,
            )
            groups.clear()
            forwards["n"] = 2
            tight = host()
            forward_deferred_final_layers(tight, [job("a", 2), job("b", 2)])
            self.assertEqual(groups, [["a", "b"]])
            self.assertEqual(
                get_sr_round_metrics(tight, "Draft").counts[
                    "opt_layer_forwards_deferred"
                ],
                2,
            )

    def test_deferred_layer_forwards_skip_unforwarded_jobs(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            forward_deferred_final_layers,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        calls = []
        fail = {"on": False}

        def fitting(_scheduler, _runners, _steps, **_kwargs):
            calls.append(1)
            if fail["on"]:
                raise RuntimeError("final layer failed")

        def host(finished=False):
            return SimpleNamespace(
                _sr_device_poisoned=False,
                _sr_is_finished=lambda _req: finished,
                _sr_pause_req=lambda _req: None,
                _opt_rollback=lambda _req, _keep: None,
                last_batch=None,
                cur_batch=None,
            )

        def job(*, satisfied=False):
            req = SimpleNamespace(rid="r")
            runner = SimpleNamespace(
                req=req,
                lookahead_inflight=False,
                lookahead_dropped=False,
                _lookahead_satisfied=satisfied,
                prefix_len=3,
                cached_len=5,
                layer_slots=[],
            )
            return SimpleNamespace(
                req=req,
                runner=runner,
                keep=3,
                window=None,
                state=SimpleNamespace(depth=2, config=SimpleNamespace(width=2)),
            )

        with patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward._forward_verify_fitting",
            fitting,
        ), patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward.final_layer_step",
            lambda state: state,
        ):
            satisfied = host()
            forward_deferred_final_layers(satisfied, [job(satisfied=True)])
            finished = host(finished=True)
            forward_deferred_final_layers(finished, [job()])
            self.assertEqual(calls, [])
            fail["on"] = True
            failed = host()
            forward_deferred_final_layers(failed, [job()])

        self.assertEqual(calls, [1])
        for scheduler in (satisfied, finished, failed):
            self.assertEqual(
                get_sr_round_metrics(scheduler, "Draft").counts[
                    "opt_layer_forwards_deferred"
                ],
                0,
            )

    def test_deferred_queue_stays_empty_when_reuse_is_off(self):
        from sglang.srt.environ import envs
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            commit_deferred_expands,
            finish_opt_expand_req,
        )

        state = _finished_two_layer_state()
        window = ([20, 21], [-1, 0], [0, 0])

        def host_for(page):
            host, _pool, _freed = _lease_host(page=page)
            host._opt_deferred = []
            host._sr_is_finished = lambda _req: False
            host._sr_pause_req = lambda _req: None
            host._opt_rollback = lambda _req, _keep: None
            req = _lease_req(keep=3, allocated=3)
            runner = SimpleNamespace(
                req=req, prefix_len=3, cached_len=3, layer_slots=[]
            )
            return host, req, runner

        npu, npu_req, npu_runner = host_for(128)
        self.assertIsNone(
            finish_opt_expand_req(npu, npu_req, npu_runner, 3, window, True, state)
        )
        self.assertEqual(npu._opt_deferred, [])

        disabled, disabled_req, disabled_runner = host_for(1)
        with envs.SGLANG_OPT_TREE_KV_REUSE.override(False):
            self.assertIsNone(
                finish_opt_expand_req(
                    disabled, disabled_req, disabled_runner, 3, window, True, state
                )
            )
        self.assertEqual(disabled._opt_deferred, [])

        enabled, enabled_req, enabled_runner = host_for(1)
        job = finish_opt_expand_req(
            enabled, enabled_req, enabled_runner, 3, window, True, state
        )
        self.assertIsNotNone(job)
        self.assertEqual(enabled._opt_deferred, [])
        commit_deferred_expands(enabled, [job], keep=True)
        self.assertEqual(enabled._opt_deferred, [job])

    def test_bonus_argmax_excludes_verify_children(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _first_unbanned,
            ancestor_columns,
            select_bonus_candidates,
        )

        self.assertEqual(_first_unbanned(torch.tensor([1, 3, 2]), [1]), 3)
        self.assertEqual(_first_unbanned(torch.tensor([1, 3, 2]), []), 1)
        self.assertEqual(_first_unbanned(torch.tensor([4, 5, 6]), [4, 5]), 6)
        self.assertIsNone(_first_unbanned(torch.tensor([4, 5]), [4, 5]))
        self.assertIsNone(_first_unbanned(torch.empty(0, dtype=torch.long), []))

        state = _finished_two_layer_state()
        runner = SimpleNamespace(
            prefix_len=3,
            bonus_ids=[
                torch.tensor([2, 3, 1]),
                torch.tensor([[1, 3, 2], [0, 1, 2]]),
                torch.tensor([[3, 2, 0], [0, 1, 2]]),
            ],
        )
        # Token 2 is the root's verify child. Token 1 is node 0's child.
        plans = select_bonus_candidates(state, ([2, 1], [-1, 0], [0, 0]), runner)
        chosen = {plan.parent: plan.token for plan in plans}
        self.assertEqual(chosen[-1], 3)
        self.assertEqual(chosen[0], 3)
        self.assertEqual(chosen[1], 3)
        self.assertEqual([plan.parent for plan in plans], [-1, 0, 1])
        self.assertEqual(
            {plan.parent: plan.end_rank for plan in plans}, {-1: 0, 0: 1, 1: 2}
        )
        self.assertEqual(plans[0].position, 3)
        self.assertEqual([plan.position for plan in plans if plan.parent == 0], [4])
        self.assertEqual([plan.position for plan in plans if plan.parent == 1], [5])
        self.assertEqual(ancestor_columns(state, 3, 0, 0), (3,))

        exhausted = SimpleNamespace(
            prefix_len=3,
            bonus_ids=[torch.tensor([7, 8, 9])],
        )
        extra = select_bonus_candidates(state, ([7, 8], [-1, -1], [0, 1]), exhausted)
        self.assertEqual({plan.parent: plan.token for plan in extra}, {-1: 9})
        self.assertEqual(extra[0].end_rank, 0)

    def test_bonus_mask_sees_prefix_ancestors_and_self(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            bonus_attention_mask,
        )

        mask = bonus_attention_mask(4, 8, [[4, 6], []], device="cpu")
        self.assertEqual(mask.shape, (2, 10))
        self.assertEqual(
            mask[0].tolist(),
            [True, True, True, True, True, False, True, False, True, False],
        )
        self.assertEqual(
            mask[1].tolist(),
            [True, True, True, True, False, False, False, False, False, True],
        )

    def test_bonus_forward_chunks_match_verify_width(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _BonusCandidate,
            bonus_attention_mask,
            bonus_verify_chunks,
        )

        plans = [
            _BonusCandidate(index, 10 + index, 20 + index, (index,))
            for index in range(5)
        ]
        chunks = bonus_verify_chunks(plans, 4)
        self.assertEqual(len(chunks), 2)
        self.assertEqual([len(chunk) for chunk in chunks], [4, 4])
        self.assertTrue(all(item.keep for item in chunks[0]))
        self.assertIs(chunks[1][0], plans[4])
        self.assertTrue(chunks[1][0].keep)
        for pad in chunks[1][1:]:
            self.assertEqual(pad.token, chunks[1][0].token)
            self.assertEqual(pad.position, chunks[1][0].position)
            self.assertEqual(pad.chain, chunks[1][0].chain)
            self.assertNotEqual(pad.token, 0)
            self.assertFalse(pad.keep)
        exact = bonus_verify_chunks(plans[:4], 4)
        self.assertEqual(len(exact), 1)
        self.assertTrue(all(item.keep for item in exact[0]))
        one_group = bonus_verify_chunks(plans, 5)
        self.assertEqual(len(one_group), 1)
        self.assertEqual(len(one_group[0]), 5)
        self.assertTrue(all(item.keep for item in one_group[0]))

        mask = bonus_attention_mask(4, 8, [[4, 6]], device="cpu", query_width=4)
        self.assertEqual(mask.shape, (4, 12))
        self.assertEqual(
            mask[0].tolist(),
            [True, True, True, True, True, False, True, False, True, False, False, False],
        )
        for index in range(1, 4):
            row = [False] * 12
            row[8 + index] = True
            self.assertEqual(mask[index].tolist(), row)

    def test_bonus_hit_installs_slot_and_seeds_ingest(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            take_cached_tail_logits,
        )
        from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import (
            OptBonusGuess,
            OptTreeKVLease,
            consume_opt_tree_lease,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        host, pool, freed = _lease_host()
        req = _lease_req(keep=3, allocated=3)
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = [20, 30]
        req.sr_align_result = SimpleNamespace(kind="append_n", old_local_len=3)
        req.opt_tree_lease = OptTreeKVLease(
            keep=3,
            layer_slots=((100, 101),),
            tokens=(20,),
            parents=(-1,),
            cols=(0,),
            depths=(0,),
        )
        seed = torch.tensor([1.0, 2.0, 3.0])
        req.opt_bonus_guesses = [
            OptBonusGuess(parent=0, token=30, slot=200, logits=seed, end_rank=4),
            OptBonusGuess(parent=-1, token=8, slot=201, logits=torch.zeros(3), end_rank=1),
        ]
        consume_opt_tree_lease(host, req)
        counts = get_sr_round_metrics(host, "Draft").counts
        self.assertEqual(counts["opt_bonus_hits"], 1)
        self.assertEqual(counts["opt_bonus_hit_rank_3"], 1)
        self.assertEqual(counts["opt_bonus_hit_rank_0"], 0)
        self.assertEqual(counts["opt_end_rank_3"], 1)
        self.assertEqual(req.kv_committed_len, 5)
        self.assertEqual(req.kv_allocated_len, 5)
        self.assertEqual(pool[0, 3].item(), 100)
        self.assertEqual(pool[0, 4].item(), 200)
        self.assertIsNone(req.opt_bonus_guesses)
        self.assertTrue(torch.equal(take_cached_tail_logits(req, 5), seed))
        self.assertCountEqual(freed, [101, 201])
        self.assertEqual(
            len(req.origin_input_ids) + len(req.output_ids) - req.kv_committed_len, 0
        )

    def test_bonus_miss_and_finish_release_slots(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import (
            OptBonusGuess,
            OptTreeKVLease,
            consume_opt_tree_lease,
            release_opt_bonus_guesses,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        host, pool, freed = _lease_host()
        req = _lease_req(keep=3, allocated=3)
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = [20, 99]
        req.sr_align_result = SimpleNamespace(kind="append_n", old_local_len=3)
        req.opt_tree_lease = OptTreeKVLease(
            keep=3,
            layer_slots=((100, 101),),
            tokens=(20,),
            parents=(-1,),
            cols=(0,),
            depths=(0,),
        )
        req.opt_bonus_guesses = [
            OptBonusGuess(parent=0, token=30, slot=200, logits=torch.zeros(2), end_rank=0)
        ]
        consume_opt_tree_lease(host, req)
        counts = get_sr_round_metrics(host, "Draft").counts
        self.assertEqual(counts["opt_bonus_hits"], 0)
        for rank in range(4):
            self.assertEqual(counts[f"opt_bonus_hit_rank_{rank}"], 0)
        self.assertEqual(counts["opt_end_rank_0"], 1)
        self.assertEqual(req.kv_committed_len, 4)
        self.assertEqual(pool[0, 3].item(), 100)
        self.assertIsNone(req.opt_bonus_guesses)
        self.assertIsNone(getattr(req, "opt_prefill_tail_logits", None))
        self.assertCountEqual(freed, [101, 200])

        host, _pool, freed = _lease_host()
        finished = _lease_req(keep=3, allocated=3)
        finished.opt_bonus_guesses = [
            OptBonusGuess(parent=-1, token=7, slot=50, logits=torch.zeros(1)),
            OptBonusGuess(parent=0, token=8, slot=51, logits=torch.zeros(1)),
        ]
        release_opt_bonus_guesses(host, finished)
        self.assertCountEqual(freed, [50, 51])
        self.assertIsNone(finished.opt_bonus_guesses)
        release_opt_bonus_guesses(host, finished)
        self.assertEqual(freed, [50, 51])

    def test_bonus_forward_stays_off_when_page_is_not_1(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            maybe_attach_bonus_guesses,
        )

        host, _pool, freed = _lease_host(page=128)
        calls = []
        runner = SimpleNamespace(
            prefix_len=3,
            cached_len=5,
            bonus_ids=[torch.tensor([1, 2, 3])],
            _alloc_layer_slots=lambda *_args, **_kwargs: calls.append(1),
        )
        req = _lease_req(keep=3, allocated=5)
        job = SimpleNamespace(
            req=req,
            runner=runner,
            keep=3,
            window=([1], [-1], [0]),
            state=_finished_two_layer_state(),
        )
        maybe_attach_bonus_guesses(host, job)
        self.assertEqual(calls, [])
        self.assertIsNone(getattr(req, "opt_bonus_guesses", None))
        self.assertEqual(freed, [])
        self.assertEqual(runner.bonus_ids, [])

    def test_poisoned_bonus_forward_does_not_free(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _BonusCandidate,
            _forward_bonus_candidates,
            _publish_or_rollback,
        )
        from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import OptBonusGuess

        freed = []
        host, _pool, _allocator_freed = _lease_host()
        host._sr_device_poisoned = False
        host._sr_clear_stamps_after_poison = lambda: None
        host.server_args = SimpleNamespace(speculative_num_draft_tokens=4)
        host.token_to_kv_pool_allocator.free = lambda index: freed.extend(
            int(slot) for slot in index.detach().cpu().tolist()
        )
        host._opt_rollback = lambda *_args, **_kwargs: freed.append(-1)
        runner = SimpleNamespace(cached_len=8, prefix_len=3, req=_lease_req(3, 8))
        plans = [
            _BonusCandidate(index, 10 + index, 20 + index, (index,))
            for index in range(6)
        ]
        calls = {"n": 0}

        def group(*_args, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return [
                    OptBonusGuess(
                        parent=-1, token=3, slot=9, logits=torch.zeros(1)
                    )
                ]
            raise RuntimeError(
                "CUDA error: an illegal memory access was encountered"
            )

        with patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward._launch_bonus_group",
            group,
        ):
            with self.assertRaises(RuntimeError):
                _forward_bonus_candidates(host, runner, plans)
        self.assertTrue(host._sr_device_poisoned)
        self.assertEqual(freed, [])

        req = _lease_req(keep=3, allocated=7)
        req.opt_bonus_guesses = [
            OptBonusGuess(parent=-1, token=3, slot=9, logits=torch.zeros(1))
        ]
        _publish_or_rollback(host, req, SimpleNamespace(layer_slots=[]), 3, ([1], [-1], [0]), True)
        self.assertEqual(freed, [])
        self.assertIsNotNone(req.opt_bonus_guesses)


def _finished_two_layer_state():
    from sglang.srt.speculative.opt_tree_remote.opt_tree_ops import (
        OptTreeState,
        expand_from_topk,
        seed_from_topk,
    )

    state = OptTreeState.create(OptTreeConfig(width=2, tau=-1.0, max_depth=4))
    seed_from_topk(state, torch.tensor([0.6, 0.4]), torch.tensor([20, 30]))
    expand_from_topk(
        state,
        torch.tensor([[0.5, 0.2], [0.4, 0.1]]),
        torch.tensor([[21, 31], [22, 32]]),
    )
    state.finished = True
    return state


def _load_opt_materialize_override():
    """Load OPT's materialize override with a recording ``super()``."""
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/speculative/opt_tree_remote/opt_tree_scheduler.py"
    )
    src = path.read_text()
    start = src.index("    def _sr_materialize_prefix_batch(")
    end = src.index("\n    def maybe_notify_remote_draft_finished(")
    code = (
        "class _Super:\n"
        "    def _sr_materialize_prefix_batch(\n"
        "        self, reqs, on_extend_logits=None, logits_survive=None,\n"
        "        capture_tree_seed=True,\n"
        "    ):\n"
        "        self.captured = capture_tree_seed\n"
        "\n"
        "class _Opt(_Super):\n"
        + src[start:end]
    )
    ns = {}
    exec(compile(code, str(path), "exec"), ns)
    return ns["_Opt"]


class _SyncProbe:
    def __init__(self, calls):
        self.calls = calls

    def synchronize(self):
        self.calls.append(1)


def _import_or_stub_forward_batch():
    """Load ``ForwardMode``, or a stand-in when this interpreter lacks ``dill``.

    ``launch_layer`` imports it lazily. The CPU suite on the editing machine
    does not install that dependency. The stub is only inserted when the real
    module is absent, and it stays for the rest of the process.
    """
    import sys
    from enum import IntEnum
    from types import ModuleType

    key = "sglang.srt.model_executor.forward_batch_info"
    if key in sys.modules:
        return
    try:
        import sglang.srt.model_executor.forward_batch_info  # noqa: F401
        return
    except ModuleNotFoundError:
        sys.modules.pop(key, None)

    class _Mode(IntEnum):
        TARGET_VERIFY = 8

    class _Hidden:
        NULL = 0

    module = ModuleType(key)
    module.ForwardMode = _Mode
    module.CaptureHiddenMode = _Hidden
    sys.modules[key] = module


class _SampleClock:
    def __init__(self, samples):
        self.samples = list(samples)
        self.calls = 0

    def __call__(self):
        value = self.samples[self.calls]
        self.calls += 1
        return value


class _TimingEvent:
    def __init__(self, enable_timing=False):
        self.recorded = 0
        self.enable_timing = enable_timing

    def record(self):
        self.recorded += 1

    def elapsed_time(self, _end):
        return 1.5


class _TimingDevice:
    def __init__(self, syncs, events):
        self.syncs = syncs
        self.events = events

    def Event(self, enable_timing=False):
        event = _TimingEvent(enable_timing=enable_timing)
        self.events.append(event)
        return event

    def synchronize(self):
        self.syncs.append(1)


class _LayerSlots:
    def __init__(self):
        self.prefix_len = 4
        self.cached_len = 4
        self.req = SimpleNamespace(kv_allocated_len=0)

    def _alloc_layer_slots(self, n, start=None):
        return torch.arange(n)


def _layer_timing_host():
    return SimpleNamespace(
        device="cpu",
        server_args=SimpleNamespace(opt_tree_width=2),
        req_to_token_pool=SimpleNamespace(device=torch.device("cpu")),
        _sr_device_poisoned=False,
    )


def _layer_timing_step(width=2, depth=1):
    return SimpleNamespace(
        tokens=torch.arange(width),
        positions=torch.full((width,), depth),
        ancestor_mask=torch.ones((width, depth * width), dtype=torch.int8),
    )


def _opt_stage_host():
    from types import SimpleNamespace

    return SimpleNamespace(
        device="cpu",
        server_args=SimpleNamespace(
            opt_tree_width=4, opt_tree_tau=0.1, opt_tree_max_depth=8
        ),
        _sr_resume_req=lambda _req: None,
        _sr_isolate_need=lambda _rids: None,
        _opt_release_uncommitted=lambda _req: None,
        _opt_rollback=lambda _req, _keep: None,
        _sr_is_finished=lambda _req: False,
        _sr_pause_req=lambda _req: None,
        last_batch=None,
        cur_batch=None,
    )


def _opt_stage_req(rid):
    from types import SimpleNamespace

    return SimpleNamespace(
        rid=rid,
        origin_input_ids=[1, 2],
        output_ids=[3],
        kv_committed_len=3,
    )


def _lease_host(page=1, width=8):
    pool = torch.full((1, width), 7, dtype=torch.int32)
    freed = []
    return (
        SimpleNamespace(
            tree_cache=SimpleNamespace(page_size=page),
            req_to_token_pool=SimpleNamespace(req_to_token=pool, device=pool.device),
            token_to_kv_pool_allocator=_RecordingAllocator(freed),
            _sr_device_poisoned=False,
        ),
        pool,
        freed,
    )


def _lease_req(keep, allocated):
    return SimpleNamespace(
        rid="lease",
        req_pool_idx=0,
        origin_input_ids=[1, 2, 3][:keep],
        output_ids=[],
        kv_committed_len=keep,
        kv_allocated_len=allocated,
        opt_tree_lease=None,
    )


def _sample_lease(keep):
    from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import OptTreeKVLease

    return OptTreeKVLease(
        keep=keep,
        layer_slots=((100, 101),),
        tokens=(20,),
        parents=(-1,),
        cols=(0,),
        depths=(0,),
    )


class _RecordingAllocator:
    def __init__(self, freed):
        self.freed = freed

    def free(self, index):
        self.freed.extend(int(slot) for slot in index.detach().cpu().tolist())


def _load_opt_lease_overrides():
    """Load the lease cleanup overrides with a recording ``super()``."""
    from pathlib import Path

    from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
        drop_lookahead_request,
    )

    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/speculative/opt_tree_remote/opt_tree_scheduler.py"
    )
    src = path.read_text()
    start = src.index("    def _opt_draft_lease_active(")
    end = src.index("    def _opt_expand_one(")
    code = (
        "class _Super:\n"
        "    def _sr_finish_rid(self, *args, **kwargs):\n"
        "        self.events.append('finish')\n"
        "    def _sr_rollback_else_reprefill(self, *args, **kwargs):\n"
        "        self.events.append('rollback')\n"
        "        return True\n"
        "    def _sr_reset_linear_kv_state(self, *args, **kwargs):\n"
        "        self.events.append('reset')\n"
        "    def _sr_abandon_poisoned_kv(self, *args, **kwargs):\n"
        "        self.events.append('abandon')\n"
        "class _Opt(_Super):\n"
        + src[start:end]
    )
    ns = {"drop_lookahead_request": drop_lookahead_request}
    exec(compile(code, str(path), "exec"), ns)
    return ns["_Opt"]


def _load_run_opt_draft_round():
    """Load ``run_opt_draft_round`` without importing the scheduler module.

    That import pulls sampler and quantization kernels the CPU suite lacks.
    """
    import textwrap
    from contextlib import nullcontext
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/speculative/opt_tree_remote/opt_tree_scheduler.py"
    )
    src = path.read_text()
    start = src.index("def run_opt_draft_round(")
    end = src.index("\nclass OptTreeSchedulerMixin:")
    ns = {"nullcontext": nullcontext}
    exec(compile(textwrap.dedent(src[start:end]), str(path), "exec"), ns)
    return ns["run_opt_draft_round"]


def _load_drain_deferred_layer_events():
    """Load the post-reply drain without the scheduler's kernel imports."""
    import textwrap
    from contextlib import nullcontext
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/speculative/opt_tree_remote/opt_tree_scheduler.py"
    )
    src = path.read_text()
    start = src.index("def run_opt_draft_round(")
    end = src.index("\nclass OptTreeSchedulerMixin:")
    calls = []

    def drain(scheduler, metrics):
        calls.append((scheduler, metrics))

    ns = {"nullcontext": nullcontext, "drain_opt_layer_gpu_us": drain}
    exec(compile(textwrap.dedent(src[start:end]), str(path), "exec"), ns)
    return ns["drain_deferred_layer_events"], calls


def _load_opt_expand_one():
    """Load ``_opt_expand_one`` from source.

    Importing the scheduler module pulls sampler and quantization kernels
    that the CPU suite does not provide.
    """
    import logging
    import textwrap
    from pathlib import Path

    from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
        _reraise_device_context,
        clear_prefill_tail_logits,
        critical_layer_forwards,
    )
    from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
        get_sr_round_metrics,
    )

    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/speculative/opt_tree_remote/opt_tree_scheduler.py"
    )
    src = path.read_text()
    start = src.index("    def _opt_expand_one(")
    end = src.index("\n    def _opt_handle_batch_impl(")
    ns = {
        "get_sr_round_metrics": get_sr_round_metrics,
        "logger": logging.getLogger("opt-expand-test"),
        "clear_prefill_tail_logits": clear_prefill_tail_logits,
        "critical_layer_forwards": critical_layer_forwards,
        "_reraise_device_context": _reraise_device_context,
        # The extracted method stays on grow_active_layers. Lookahead is off.
        "lookahead_supported": lambda _scheduler: False,
        "prepare_lookahead_runtime": lambda *_args, **_kwargs: None,
        "grow_lookahead": lambda *_args, **_kwargs: None,
        "LookaheadFallback": RuntimeError,
    }
    code = "from __future__ import annotations\n" + textwrap.dedent(src[start:end])
    exec(compile(code, str(path), "exec"), ns)
    return ns["_opt_expand_one"]


def _load_opt_step_rtt_host():
    """Load the STEP RTT overrides without the scheduler's kernel imports."""
    import time
    from pathlib import Path

    from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
        get_sr_round_metrics,
    )

    class _Parent:
        def _sr_send_step(self, reqs):
            return self.sent

        def _sr_recv(self, inflight):
            return inflight

    class _Clock:
        perf_counter = time.perf_counter

    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/speculative/opt_tree_remote/opt_tree_scheduler.py"
    )
    src = path.read_text()
    start = src.index("    def _opt_step_rtt_enabled(")
    end = src.index("\n    @property\n    def is_remote_spec_draft(")
    ns = {
        "_Parent": _Parent,
        "time": _Clock,
        "get_sr_round_metrics": get_sr_round_metrics,
    }
    code = "class _Host(_Parent):\n" + src[start:end]
    exec(compile(code, str(path), "exec"), ns)
    return ns["_Host"], _Clock


def _expand_admit_host():
    host = _opt_stage_host()
    host._sr_device_poisoned = False
    host._sr_clear_stamps_after_poison = lambda: None
    return host


def _load_expand_with_admit(admit):
    """Load ``_opt_expand_one`` with lookahead admission under test control."""
    import time

    from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
        LookaheadFallback,
    )

    expand = _load_opt_expand_one()
    globals_ = expand.__globals__
    globals_["time"] = time
    globals_["torch"] = SimpleNamespace(
        get_device_module=lambda _device: _SyncProbe([])
    )
    globals_["OptTreeConfig"] = OptTreeConfig
    globals_["LookaheadFallback"] = LookaheadFallback
    globals_["lookahead_supported"] = lambda _scheduler: True
    globals_["prepare_lookahead_runtime"] = lambda *_args, **_kwargs: (object(),)
    globals_["_admit_lookahead"] = admit
    called = {"grow": 0, "lookahead": 0, "ingest": 0}

    class _Runner:
        def __init__(self, _scheduler, _req, _prefix_len):
            self.steps = []

        def forward_layer(self, step):
            self.steps.append(step)
            return torch.zeros(1)

    def ingest(_scheduler, _runners, phases=None):
        called["ingest"] += 1
        return [torch.zeros(4)]

    def grow(
        _config, _tails, forward_active, _scheduler=None, states_out=None, **_kwargs
    ):
        called["grow"] += 1
        forward_active(["layer-0"], [0])
        return [([1], [-1], [0])]

    def grow_lookahead(*_args, **_kwargs):
        called["lookahead"] += 1
        raise AssertionError("lookahead grow should not run after admit fails")

    globals_["OptLayerForward"] = _Runner
    globals_["ingest_tail_logits"] = ingest
    globals_["grow_active_layers"] = grow
    globals_["grow_lookahead"] = grow_lookahead
    return expand, called


def _materialize_host():
    """Host whose ``_sr_materialize_prefix_batch`` is the real method.

    The mixin module imports sampler and quantization kernels. The CPU suite
    does not provide those, so this loads the method from source and stubs
    the scheduler helpers it calls.
    """
    try:
        from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
            StandaloneRemoteDraftSchedulerMixin,
        )
    except Exception:
        return _materialize_host_from_source()
    return StandaloneRemoteDraftSchedulerMixin()


def _materialize_host_from_source():
    import textwrap
    from pathlib import Path
    from types import SimpleNamespace
    from typing import Callable, Optional

    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/speculative/standalone_remote/drafter/"
        "sr_draft_scheduler_mixin.py"
    )
    src = path.read_text()
    start = src.index("    def _sr_materialize_prefix_batch(")
    end = src.index("\n    def _sr_kv_len(")
    ns = {
        "List": list,
        "Optional": Optional,
        "Callable": Callable,
        "Req": object,
    }
    exec(compile(textwrap.dedent(src[start:end]), str(path), "exec"), ns)
    host = SimpleNamespace()
    host._sr_materialize_prefix_batch = ns["_sr_materialize_prefix_batch"].__get__(
        host, SimpleNamespace
    )

    def prefix_pending(req):
        if req.req_pool_idx is None:
            return True
        if getattr(host, "chunked_req", None) is req:
            return True
        target = len(getattr(req, "origin_input_ids", None) or [])
        committed = int(getattr(req, "kv_committed_len", 0) or 0)
        return committed < target

    host._sr_prefix_pending = prefix_pending
    host._sr_resume_req = lambda req: setattr(req, "draft_is_paused", False)
    host._sr_isolate_need = lambda *_args, **_kwargs: None
    host._sr_tree_mode = lambda: False
    host._sr_enable_tree_seed_hidden = lambda _batch: None
    host._sr_cache_tree_seeds = lambda *_args, **_kwargs: None
    host._sr_park_finished_prefill_embeddings = lambda *_args, **_kwargs: None
    host._sr_is_finished = lambda _req: False
    host._sr_pause_req = lambda req: setattr(req, "draft_is_paused", True)
    return host


class _ReleaseHost:
    def __init__(self):
        self.rolls = []

    def _opt_rollback(self, req, keep_len):
        self.rolls.append(int(keep_len))


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

    def _sr_materialize_prefix_batch(
        self, reqs, on_extend_logits=None, logits_survive=None
    ):
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


class OptLookaheadOpsTest(unittest.TestCase):
    def test_env_default_off(self):
        import os

        from sglang.srt.speculative.opt_tree_remote.opt_tree_config import (
            opt_draft_lookahead_enabled,
            opt_probe_layer_wait_enabled,
        )

        old = os.environ.pop("SGLANG_OPT_DRAFT_LOOKAHEAD", None)
        probe = os.environ.pop("SGLANG_OPT_PROBE_LAYER_WAIT", None)
        try:
            self.assertFalse(opt_draft_lookahead_enabled())
            os.environ["SGLANG_OPT_DRAFT_LOOKAHEAD"] = "1"
            self.assertTrue(opt_draft_lookahead_enabled())
            os.environ["SGLANG_OPT_DRAFT_LOOKAHEAD"] = "no"
            self.assertFalse(opt_draft_lookahead_enabled())
            self.assertFalse(opt_probe_layer_wait_enabled())
            os.environ["SGLANG_OPT_PROBE_LAYER_WAIT"] = "1"
            self.assertTrue(opt_probe_layer_wait_enabled())
            os.environ["SGLANG_OPT_PROBE_LAYER_WAIT"] = "no"
            self.assertFalse(opt_probe_layer_wait_enabled())
        finally:
            if old is None:
                os.environ.pop("SGLANG_OPT_DRAFT_LOOKAHEAD", None)
            else:
                os.environ["SGLANG_OPT_DRAFT_LOOKAHEAD"] = old
            if probe is None:
                os.environ.pop("SGLANG_OPT_PROBE_LAYER_WAIT", None)
            else:
                os.environ["SGLANG_OPT_PROBE_LAYER_WAIT"] = probe

    def test_expand_on_device_matches_cpu_expand_including_ties(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_ops import (
            expand_from_topk,
            expand_on_device,
            mirror_layer,
            seed_from_topk,
        )

        scores = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        ids = torch.tensor([[10, 11], [12, 13]])
        weight = torch.tensor([0.4, 0.4])
        anc = torch.eye(2, dtype=torch.int8)
        top_scores, tokens, parents, anc_new = expand_on_device(
            weight, anc, scores, ids, 2
        )

        device_state = OptTreeState.create(OptTreeConfig(width=2, tau=0.0, max_depth=4))
        seed_from_topk(device_state, weight.clone(), torch.tensor([1, 2]))
        device_state.anc_mask = anc.clone()
        mirror_layer(device_state, top_scores, tokens, parents, seed=False)

        cpu_state = OptTreeState.create(OptTreeConfig(width=2, tau=0.0, max_depth=4))
        seed_from_topk(cpu_state, weight.clone(), torch.tensor([1, 2]))
        cpu_state.anc_mask = anc.clone()
        expand_from_topk(cpu_state, scores, ids)

        self.assertTrue(torch.equal(device_state.tokens[1], cpu_state.tokens[1]))
        self.assertTrue(torch.equal(device_state.parents[1], cpu_state.parents[1]))
        self.assertTrue(torch.equal(device_state.weight[1], cpu_state.weight[1]))
        self.assertTrue(torch.equal(device_state.anc_mask, cpu_state.anc_mask))
        self.assertTrue(torch.equal(device_state.anc_mask, anc_new))
        self.assertEqual(int(device_state.depth), 2)

    def test_mirror_rebuilds_ancestor_mask_from_parent_cols(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_ops import (
            mirror_layer,
            seed_from_topk,
        )

        state = OptTreeState.create(OptTreeConfig(width=2, tau=0.0, max_depth=4))
        seed_from_topk(state, torch.tensor([0.6, 0.3]), torch.tensor([4, 5]))
        parents = torch.tensor([1, 0])
        mirror_layer(
            state,
            torch.tensor([0.2, 0.1]),
            torch.tensor([8, 9]),
            parents,
            seed=False,
        )
        eye = torch.eye(2, dtype=torch.int8)
        base = torch.eye(2, dtype=torch.int8)
        expect = torch.cat([base[parents], eye], dim=1)
        self.assertTrue(torch.equal(state.anc_mask, expect))
        self.assertTrue(torch.equal(state.tokens[1], torch.tensor([8, 9])))
        self.assertTrue(torch.equal(state.parents[1], parents))

    def test_lookahead_batch_reuses_until_rid_or_pool_changes(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _lookahead_batch,
            drop_lookahead_request,
        )

        made = []

        class _Info:
            def __init__(self):
                self.temperatures = torch.ones(1)

        class _Batch:
            def __init__(self, reqs):
                self.reqs = list(reqs)
                self.sampling_info = _Info()
                self.has_grammar = True
                self.return_logprob = True
                self.return_hidden_states = True

        sched = SimpleNamespace()

        def _make(reqs):
            made.append(reqs[0])
            return _Batch(reqs)

        sched._sr_make_decode_batch = _make
        req = SimpleNamespace(rid="r", req_pool_idx=3)
        first = _lookahead_batch(sched, req)
        second = _lookahead_batch(sched, req)
        self.assertIs(first, second)
        self.assertEqual(len(made), 1)

        rebound = SimpleNamespace(rid="r", req_pool_idx=3)
        third = _lookahead_batch(sched, rebound)
        self.assertIs(third, first)
        self.assertIs(third.reqs[0], rebound)
        self.assertEqual(len(made), 1)

        moved = SimpleNamespace(rid="r", req_pool_idx=4)
        fourth = _lookahead_batch(sched, moved)
        self.assertIsNot(fourth, first)
        self.assertEqual(len(made), 2)
        drop_lookahead_request(sched, "r")
        self.assertNotIn("r", sched._opt_la_batches)

    @unittest.skipUnless(torch.cuda.is_available(), "lookahead pin cache needs CUDA")
    def test_lookahead_width_cache_allocates_once(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _lookahead_width_cache,
        )

        sched = SimpleNamespace()
        device = torch.device("cuda")
        first = _lookahead_width_cache(sched, 4, device)
        second = _lookahead_width_cache(sched, 4, device)
        self.assertIs(first, second)
        self.assertIs(first["slots"][0], second["slots"][0])
        self.assertIs(first["bonus"], second["bonus"])
        self.assertIs(first["positions"], second["positions"])
        self.assertIs(first["dummy_tokens"], second["dummy_tokens"])
        self.assertIs(first["launch_event"], second["launch_event"])
        wider = _lookahead_width_cache(sched, 8, device)
        self.assertIsNot(wider, first)

    def test_bonus_len_cache_reused_across_runners(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _bonus_len_cache,
        )

        width_hit = {"width": True}
        sched = SimpleNamespace(_opt_la_cache={4: width_hit})
        cpu = torch.device("cpu")
        first = _bonus_len_cache(sched, cpu)
        second = _bonus_len_cache(sched, "cpu")
        self.assertIs(first["seq"], second["seq"])
        self.assertIs(first["pool"], second["pool"])
        self.assertIsNone(first["pool_idx"])
        other = _bonus_len_cache(sched, torch.device("meta"))
        self.assertIsNot(other["seq"], first["seq"])
        self.assertIsNot(other["pool"], first["pool"])
        self.assertIs(sched._opt_la_cache[4], width_hit)
        self.assertIs(sched._opt_la_cache["_bonus_len"][cpu]["seq"], first["seq"])

    def test_lookahead_kv_budget_covers_tree_and_bonus(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _lookahead_ingest_slots,
            _lookahead_kv_need,
            _lookahead_would_evict,
        )

        class _Alloc:
            def __init__(self, free):
                self.free = free

            def available_size(self):
                return self.free

        class _Tree:
            def __init__(self, free, chunk=False):
                self.token_to_kv_pool_allocator = _Alloc(free)
                self._chunk = chunk

            def is_chunk_cache(self):
                return self._chunk

        args = SimpleNamespace(opt_tree_width=10, opt_tree_max_depth=10)
        need = (10 - 1) * 10 + (10 + 1)
        host_args = SimpleNamespace(server_args=args)
        self.assertEqual(_lookahead_kv_need(host_args), need)
        self.assertEqual(_lookahead_kv_need(host_args, 0), need)
        self.assertEqual(_lookahead_kv_need(host_args, 3), need + 3)

        def host(free, chunk=False):
            return SimpleNamespace(
                server_args=args, tree_cache=_Tree(free, chunk=chunk)
            )

        self.assertTrue(_lookahead_would_evict(host(10)))
        self.assertTrue(_lookahead_would_evict(host(need - 1)))
        self.assertFalse(_lookahead_would_evict(host(need)))
        self.assertTrue(_lookahead_would_evict(host(need - 1, chunk=True)))
        self.assertFalse(_lookahead_would_evict(host(need, chunk=True)))
        self.assertTrue(_lookahead_would_evict(host(need), ingest_slots=1))
        self.assertFalse(_lookahead_would_evict(host(need + 1), ingest_slots=1))
        self.assertEqual(
            _lookahead_ingest_slots(
                SimpleNamespace(prefix_len=12, req=SimpleNamespace(kv_committed_len=9))
            ),
            3,
        )
        self.assertEqual(
            _lookahead_ingest_slots(
                SimpleNamespace(prefix_len=12, req=SimpleNamespace(kv_committed_len=12))
            ),
            0,
        )

    def test_settle_clone_failure_drops_bonus_without_a_second_forward(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _settle_lookahead_job,
        )

        class _Bonus:
            def clone_cpu(self):
                raise RuntimeError("bonus copy failed")

        bonus_ids = [torch.tensor([1])]
        runner = SimpleNamespace(
            req=SimpleNamespace(rid="r"),
            prefix_len=4,
            cached_len=6,
            lookahead_inflight=True,
            lookahead_dropped=False,
            _lookahead_satisfied=False,
            bonus_ids=bonus_ids,
            _lookahead_bonus=_Bonus(),
        )
        job = SimpleNamespace(
            req=runner.req,
            runner=runner,
            state=SimpleNamespace(depth=1, config=SimpleNamespace(width=2)),
        )
        scheduler = SimpleNamespace(
            _sr_device_poisoned=False,
            _sr_is_finished=lambda _req: False,
            tree_cache=SimpleNamespace(page_size=1),
        )

        _settle_lookahead_job(scheduler, job)

        self.assertTrue(runner.lookahead_dropped)
        self.assertTrue(runner._lookahead_satisfied)
        self.assertFalse(runner.lookahead_inflight)
        self.assertEqual(len(bonus_ids), 1)

    def test_settle_drop_syncs_bonus_and_clears_ready(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _settle_lookahead_job,
            finish_opt_expand_req,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        class _Event:
            def __init__(self):
                self.syncs = 0

            def synchronize(self):
                self.syncs += 1

        class _Bonus:
            def __init__(self):
                self.event = _Event()
                self.ready = True

            def clone_cpu(self):
                if not self.ready:
                    raise RuntimeError("lookahead bonus ids were read before the copy")
                self.event.synchronize()
                self.ready = False
                return torch.tensor([9])

        def dropped_runner(bonus):
            req = SimpleNamespace(
                rid="r",
                kv_allocated_len=8,
                kv_committed_len=8,
            )
            return req, SimpleNamespace(
                req=req,
                prefix_len=4,
                cached_len=8,
                lookahead_inflight=True,
                lookahead_dropped=False,
                _lookahead_launch_event=_Event(),
                _lookahead_bonus=bonus,
                layer_slots=[torch.tensor([1, 2])],
            )

        bonus = _Bonus()
        req, runner = dropped_runner(bonus)
        rolls = []
        job = SimpleNamespace(
            req=req,
            runner=runner,
            keep=4,
            state=SimpleNamespace(depth=1, config=SimpleNamespace(width=2)),
        )
        scheduler = SimpleNamespace(
            _sr_device_poisoned=False,
            _sr_is_finished=lambda _req: False,
            tree_cache=SimpleNamespace(page_size=1),
            _opt_rollback=lambda _req, keep: rolls.append(int(keep)),
        )

        _settle_lookahead_job(scheduler, job)

        self.assertEqual(bonus.event.syncs, 1)
        self.assertFalse(bonus.ready)
        self.assertEqual(rolls, [4])
        self.assertEqual(
            get_sr_round_metrics(scheduler, "Draft").counts["opt_lookahead_rollbacks"],
            1,
        )
        self.assertTrue(runner.lookahead_dropped)
        self.assertFalse(runner.lookahead_inflight)
        with self.assertRaises(RuntimeError):
            bonus.clone_cpu()

        finish_bonus = _Bonus()
        finish_req, finish_runner = dropped_runner(finish_bonus)
        finish_rolls = []
        finish_host = SimpleNamespace(
            _sr_device_poisoned=False,
            _sr_is_finished=lambda _req: False,
            _sr_pause_req=lambda _req: None,
            tree_cache=SimpleNamespace(page_size=1),
            _opt_rollback=lambda _req, keep: finish_rolls.append(int(keep)),
        )
        self.assertIsNone(
            finish_opt_expand_req(
                finish_host, finish_req, finish_runner, 4, None, False, None
            )
        )
        self.assertEqual(finish_bonus.event.syncs, 1)
        self.assertFalse(finish_bonus.ready)
        self.assertTrue(finish_runner.lookahead_dropped)
        self.assertFalse(finish_runner.lookahead_inflight)
        with self.assertRaises(RuntimeError):
            finish_bonus.clone_cpu()

    def test_settle_drop_clears_req_to_token_span(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _settle_lookahead_job,
        )

        class _Event:
            def synchronize(self):
                return None

        class _Bonus:
            def __init__(self):
                self.event = _Event()
                self.ready = True

        def drop_job():
            req = SimpleNamespace(
                rid="r",
                req_pool_idx=0,
                kv_allocated_len=8,
                kv_committed_len=8,
            )
            runner = SimpleNamespace(
                req=req,
                prefix_len=4,
                cached_len=8,
                lookahead_inflight=True,
                lookahead_dropped=False,
                _lookahead_launch_event=_Event(),
                _lookahead_bonus=_Bonus(),
                layer_slots=[torch.tensor([1, 2])],
            )
            job = SimpleNamespace(
                req=req,
                runner=runner,
                keep=4,
                state=SimpleNamespace(depth=1, config=SimpleNamespace(width=2)),
            )
            return req, job

        prefix = [10, 11, 12, 13]
        tree = [20, 21, 22, 23]
        pool = torch.tensor([prefix + tree], dtype=torch.int64)
        _req, job = drop_job()
        rolls = []

        def rollback(rolled, keep):
            rolls.append(int(keep))
            rolled.kv_allocated_len = int(keep)

        scheduler = SimpleNamespace(
            _sr_device_poisoned=False,
            _sr_is_finished=lambda _req: False,
            tree_cache=SimpleNamespace(page_size=1),
            req_to_token_pool=SimpleNamespace(req_to_token=pool),
            _opt_rollback=rollback,
        )

        _settle_lookahead_job(scheduler, job)

        self.assertEqual(rolls, [4])
        self.assertEqual(pool[0, :4].tolist(), prefix)
        self.assertEqual(pool[0, 4:].tolist(), [-1, -1, -1, -1])

        stale = torch.tensor([prefix + tree], dtype=torch.int64)
        stale_req, stale_job = drop_job()

        def fail_rollback(_req, _keep):
            raise RuntimeError("rollback failed")

        fail_scheduler = SimpleNamespace(
            _sr_device_poisoned=False,
            _sr_is_finished=lambda _req: False,
            tree_cache=SimpleNamespace(page_size=1),
            req_to_token_pool=SimpleNamespace(req_to_token=stale),
            _opt_rollback=fail_rollback,
        )

        _settle_lookahead_job(fail_scheduler, stale_job)

        self.assertEqual(stale[0].tolist(), prefix + tree)
        self.assertEqual(stale_req.kv_allocated_len, 4)

    def test_bonus_attach_failure_is_counted_and_publish_still_runs(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            forward_deferred_final_layers,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        published = []
        paused = []
        host = SimpleNamespace(
            _sr_device_poisoned=False,
            _sr_is_finished=lambda _req: False,
            _sr_pause_req=lambda req: paused.append(req.rid),
            tree_cache=SimpleNamespace(page_size=1),
            last_batch="stale",
            cur_batch="stale",
        )
        req = SimpleNamespace(rid="r")
        runner = SimpleNamespace(
            req=req,
            lookahead_inflight=False,
            lookahead_dropped=False,
            _lookahead_satisfied=True,
        )
        job = SimpleNamespace(req=req, runner=runner, keep=1, window=None)

        def boom(*_args, **_kwargs):
            raise RuntimeError("attach failed")

        def publish(_scheduler, published_req, *_args, **_kwargs):
            published.append(published_req.rid)
            return True

        with patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_forward.maybe_attach_bonus_guesses",
            boom,
        ), patch(
            "sglang.srt.speculative.opt_tree_remote.opt_tree_lease.publish_opt_tree_lease",
            publish,
        ):
            forward_deferred_final_layers(host, [job])

        self.assertEqual(published, ["r"])
        self.assertEqual(paused, ["r"])
        self.assertEqual(
            get_sr_round_metrics(host, "Draft").counts["opt_bonus_attach_failed"], 1
        )

    def test_critical_layer_forwards_drop_the_early_final_launch(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            critical_layer_forwards,
        )

        self.assertEqual(critical_layer_forwards(6), 5)
        self.assertEqual(critical_layer_forwards(1), 0)
        self.assertEqual(critical_layer_forwards(0), 0)

    def test_drain_reads_ingest_gpu_with_the_layer_events(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote import opt_tree_forward as forward
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            drain_opt_layer_gpu_us,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        syncs = []
        host = _layer_timing_host()
        metrics = get_sr_round_metrics(host, "Draft")
        class _IngestEvent(_TimingEvent):
            def elapsed_time(self, _end):
                return 4.0

        host._opt_layer_events = [(_TimingEvent(), _TimingEvent())]
        host._opt_ingest_events = [(_IngestEvent(), _IngestEvent())]
        host._opt_ingest_open = _TimingEvent()

        with patch.object(
            forward.torch, "get_device_module", lambda _device: _TimingDevice(syncs, [])
        ):
            drain_opt_layer_gpu_us(host, metrics)
            self.assertEqual(syncs, [1])
            self.assertEqual(metrics.counts["opt_layer_gpu_us"], 1500)
            self.assertEqual(metrics.counts["opt_ingest_gpu_us"], 4000)
            self.assertEqual(host._opt_layer_events, [])
            self.assertEqual(host._opt_ingest_events, [])
            self.assertIsNone(host._opt_ingest_open)

            metrics.counts["opt_ingest_gpu_us"] = 0
            host._opt_ingest_events = [
                (_IngestEvent(), _IngestEvent()),
                (_TimingEvent(), _TimingEvent()),
            ]
            drain_opt_layer_gpu_us(host, metrics)

        self.assertEqual(syncs, [1, 1])
        self.assertEqual(metrics.counts["opt_ingest_gpu_us"], 5500)
        self.assertEqual(host._opt_ingest_events, [])

    def test_drain_reads_select_events_and_drops_them_when_poisoned(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote import opt_tree_forward as forward
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            drain_opt_layer_gpu_us,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        syncs = []
        host = _layer_timing_host()
        metrics = get_sr_round_metrics(host, "Draft")
        host._opt_select_events = [(_TimingEvent(), _TimingEvent())]

        with patch.object(
            forward.torch, "get_device_module", lambda _device: _TimingDevice(syncs, [])
        ):
            drain_opt_layer_gpu_us(host, metrics)
        self.assertEqual(syncs, [1])
        self.assertEqual(metrics.counts["opt_layer_select_gpu_us"], 1500)
        self.assertEqual(host._opt_select_events, [])
        self.assertEqual(metrics.counts["opt_layer_gpu_us"], 0)

        syncs.clear()
        drain_opt_layer_gpu_us(host, metrics)
        self.assertEqual(syncs, [])
        self.assertEqual(metrics.counts["opt_layer_select_gpu_us"], 1500)

        host._sr_device_poisoned = True
        host._opt_select_events = [(_TimingEvent(), _TimingEvent())]
        with patch.object(
            forward.torch, "get_device_module", lambda _device: _TimingDevice(syncs, [])
        ):
            drain_opt_layer_gpu_us(host, metrics)
        self.assertEqual(syncs, [])
        self.assertEqual(metrics.counts["opt_layer_select_gpu_us"], 1500)
        self.assertEqual(host._opt_select_events, [])

    def test_opt_step_rtt_counts_only_a_successful_target_step(self):
        from unittest.mock import patch

        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        Host, clock = _load_opt_step_rtt_host()

        def make(opt, role):
            host = Host()
            host.spec_algorithm = SimpleNamespace(is_opt_tree_remote=lambda: opt)
            host.server_args = SimpleNamespace(opt_tree_role=role)
            return host

        inflight = SimpleNamespace(send_ok=True, wait_reply=True)
        host = make(True, "target")
        host.sent = ([], [], inflight)
        with patch.object(clock, "perf_counter", _SampleClock([1.0, 1.0 + 250e-6])):
            self.assertIs(host._sr_send_step([])[2], inflight)
            self.assertIs(host._sr_recv(inflight), inflight)
        metrics = get_sr_round_metrics(host, "Target")
        self.assertEqual(metrics.counts["opt_step_rtt_us"], 250)
        self.assertIsNone(host._opt_step_sent_at)

        failed = SimpleNamespace(send_ok=False, wait_reply=True)
        host.sent = ([], [], failed)
        host._sr_send_step([])
        host._sr_recv(failed)
        self.assertEqual(metrics.counts["opt_step_rtt_us"], 250)
        self.assertIsNone(host._opt_step_sent_at)

        quiet = SimpleNamespace(send_ok=True, wait_reply=False)
        host.sent = ([], [], quiet)
        with patch.object(clock, "perf_counter", _SampleClock([5.0])):
            host._sr_send_step([])
        host._sr_recv(quiet)
        self.assertEqual(metrics.counts["opt_step_rtt_us"], 250)
        self.assertEqual(host._opt_step_sent_at, 5.0)

        other_inflight = SimpleNamespace(send_ok=True, wait_reply=True)
        step = SimpleNamespace(send_ok=True, wait_reply=True)
        paired = make(True, "target")
        paired.sent = ([], [], step)
        with patch.object(clock, "perf_counter", _SampleClock([1.0, 1.0 + 250e-6])):
            paired._sr_send_step([])
            paired._sr_recv(other_inflight)
            paired._sr_recv(step)
        self.assertEqual(
            get_sr_round_metrics(paired, "Target").counts["opt_step_rtt_us"], 250
        )
        self.assertIsNone(paired._opt_step_sent_at)

        for opt, role in ((False, "target"), (True, "draft")):
            skipped = make(opt, role)
            skipped.sent = ([], [], inflight)
            skipped._sr_send_step([])
            skipped._sr_recv(inflight)
            self.assertEqual(
                get_sr_round_metrics(skipped, "Target").counts["opt_step_rtt_us"],
                0,
            )
            self.assertFalse(hasattr(skipped, "_opt_step_sent_at"))

    def test_launch_wait_syncs_only_after_a_previous_layer(self):
        import os
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote import opt_tree_forward as forward
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _note_lookahead_launch_wait,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        syncs = []

        class _Stream:
            def synchronize(self):
                syncs.append(1)

        class _Device:
            def current_stream(self):
                return _Stream()

        host = _layer_timing_host()
        metrics = get_sr_round_metrics(host, "Draft")
        runner = SimpleNamespace(scheduler=host, _lookahead_queued=True)
        samples = [1.0, 1.0 + 400e-6]
        probe = os.environ.pop("SGLANG_OPT_PROBE_LAYER_WAIT", None)
        try:
            with patch.object(
                forward.torch, "get_device_module", lambda _device: _Device()
            ), patch.object(forward.time, "perf_counter", _SampleClock(samples)):
                metrics.active = True
                _note_lookahead_launch_wait(runner)
                self.assertEqual(syncs, [])
                self.assertEqual(metrics.counts["opt_lookahead_launch_wait_us"], 0)
                os.environ["SGLANG_OPT_PROBE_LAYER_WAIT"] = "1"
                runner._lookahead_queued = False
                _note_lookahead_launch_wait(runner)
                self.assertEqual(syncs, [])
                runner._lookahead_queued = True
                _note_lookahead_launch_wait(runner)
                metrics.active = False
                _note_lookahead_launch_wait(runner)
        finally:
            metrics.active = False
            if probe is None:
                os.environ.pop("SGLANG_OPT_PROBE_LAYER_WAIT", None)
            else:
                os.environ["SGLANG_OPT_PROBE_LAYER_WAIT"] = probe
        self.assertEqual(syncs, [1])
        self.assertEqual(
            metrics.counts["opt_lookahead_launch_wait_us"],
            int((samples[1] - samples[0]) * 1e6),
        )

    def test_layer_wait_is_outside_host_and_launch(self):
        import os
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote import opt_tree_forward as forward
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _LayerTiming,
            _close_layer_host,
            _finish_layer_timing,
            _mark_layer_launch,
            _note_lookahead_launch_wait,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        class _Stream:
            def synchronize(self):
                return None

        class _Device:
            def current_stream(self):
                return _Stream()

            def synchronize(self):
                return None

        host = _layer_timing_host()
        metrics = get_sr_round_metrics(host, "Draft")
        metrics.active = True
        start = 10.0
        prep = 300e-6
        wait = 4000e-6
        launch = 800e-6
        samples = [
            start + prep,
            start + prep,
            start + prep + wait,
            start + prep + wait,
            start + prep + wait + launch,
        ]
        timing = _LayerTiming(metrics, start, _TimingEvent(), _TimingEvent())
        runner = SimpleNamespace(scheduler=host, _lookahead_queued=True)
        probe = os.environ.pop("SGLANG_OPT_PROBE_LAYER_WAIT", None)
        os.environ["SGLANG_OPT_PROBE_LAYER_WAIT"] = "1"
        try:
            with patch.object(
                forward.torch, "get_device_module", lambda _device: _Device()
            ), patch.object(forward.time, "perf_counter", _SampleClock(samples)):
                _close_layer_host(timing)
                _note_lookahead_launch_wait(runner)
                _mark_layer_launch(timing)
                _finish_layer_timing(host, timing)
        finally:
            metrics.active = False
            if probe is None:
                os.environ.pop("SGLANG_OPT_PROBE_LAYER_WAIT", None)
            else:
                os.environ["SGLANG_OPT_PROBE_LAYER_WAIT"] = probe
        self.assertEqual(
            metrics.counts["opt_layer_host_us"], int((samples[0] - start) * 1e6)
        )
        self.assertEqual(
            metrics.counts["opt_lookahead_launch_wait_us"],
            int((samples[2] - samples[1]) * 1e6),
        )
        self.assertEqual(
            metrics.counts["opt_layer_launch_us"],
            int((samples[4] - samples[3]) * 1e6),
        )
        self.assertLess(
            metrics.counts["opt_layer_host_us"],
            metrics.counts["opt_lookahead_launch_wait_us"],
        )
        self.assertLess(
            metrics.counts["opt_layer_launch_us"],
            metrics.counts["opt_lookahead_launch_wait_us"],
        )

    def test_lookahead_template_copy_keeps_the_admission_batch(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            OptLayerSpec,
            _refresh_lookahead_forward,
        )

        old_ids = torch.zeros(2, dtype=torch.long)
        old_loc = torch.zeros(2, dtype=torch.long)
        old_mask = torch.zeros(6, dtype=torch.bool)
        template = SimpleNamespace(
            input_ids=old_ids,
            seq_lens=torch.zeros(1, dtype=torch.long),
            seq_lens_cpu=torch.zeros(1, dtype=torch.long),
            orig_seq_lens=torch.zeros(1, dtype=torch.long),
            seq_lens_sum=0,
            out_cache_loc=old_loc,
            spec_info=OptLayerSpec(
                custom_mask=old_mask,
                positions=torch.zeros(2, dtype=torch.long),
                draft_token_num=2,
                num_tokens_per_req=2,
            ),
            positions=torch.zeros(2, dtype=torch.long),
            extend_num_tokens=0,
            extend_seq_lens_cpu=None,
            extend_prefix_lens_cpu=None,
            extend_logprob_start_lens_cpu=None,
            mrope_positions=None,
            req_pool_indices=torch.tensor([7]),
            forward_mode="verify",
            sampling_info="greedy",
        )
        new_ids = torch.tensor([8, 9])
        new_loc = torch.tensor([3, 4])
        new_mask = torch.ones(6, dtype=torch.bool)
        positions = torch.tensor([11, 11])
        batch = SimpleNamespace(
            input_ids=new_ids,
            seq_lens=torch.tensor([6]),
            seq_lens_cpu=torch.tensor([6]),
            orig_seq_lens=torch.tensor([6]),
            seq_lens_sum=6,
            out_cache_loc=new_loc,
            spec_info=OptLayerSpec(
                custom_mask=new_mask,
                positions=positions,
                draft_token_num=2,
                num_tokens_per_req=2,
            ),
            extend_num_tokens=2,
            extend_lens=[2],
            prefix_lens=[6],
            extend_logprob_start_lens=[6],
        )
        fresh = _refresh_lookahead_forward(template, batch, None)
        self.assertTrue(torch.equal(fresh.input_ids, new_ids))
        self.assertTrue(torch.equal(fresh.spec_info.custom_mask, new_mask))
        self.assertEqual(fresh.seq_lens_sum, 6)
        self.assertTrue(torch.equal(fresh.out_cache_loc, new_loc))
        self.assertEqual(fresh.extend_num_tokens, 0)
        self.assertIsNone(fresh.extend_seq_lens_cpu)
        self.assertIsNone(fresh.extend_prefix_lens_cpu)
        self.assertIsNone(fresh.extend_logprob_start_lens_cpu)
        self.assertIs(fresh.req_pool_indices, template.req_pool_indices)
        self.assertIs(fresh.sampling_info, template.sampling_info)
        self.assertTrue(torch.equal(template.input_ids, old_ids))
        self.assertTrue(torch.equal(template.spec_info.custom_mask, old_mask))
        self.assertEqual(template.seq_lens_sum, 0)
        self.assertTrue(torch.equal(template.out_cache_loc, old_loc))

    def test_lookahead_mrope_reuses_the_device_delta(self):
        from unittest.mock import patch

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _lookahead_mrope_positions,
            _mrope_delta_from_template,
        )

        positions = torch.tensor([10, 10])
        mrope = torch.tensor([[13, 13], [13, 13], [13, 13]])
        template = SimpleNamespace(positions=positions, mrope_positions=mrope)
        delta = _mrope_delta_from_template(template)
        self.assertTrue(torch.equal(delta, torch.tensor([[3]])))
        later = torch.tensor([20, 20])
        calls = []
        real_to = torch.Tensor.to

        def _spy_to(tensor, *args, **kwargs):
            calls.append(args)
            return real_to(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "to", _spy_to):
            rows = _lookahead_mrope_positions(later, delta)
        self.assertEqual(calls, [])
        self.assertEqual(tuple(rows.shape), (3, 2))
        self.assertTrue(torch.equal(rows, torch.full((3, 2), 23)))

    def test_mrope_delta_uses_only_the_first_text_offset(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _lookahead_mrope_positions,
            _mrope_delta_from_template,
        )

        self.assertIsNone(_mrope_delta_from_template(SimpleNamespace()))
        self.assertIsNone(
            _mrope_delta_from_template(SimpleNamespace(positions=torch.tensor([1])))
        )
        self.assertIsNone(
            _mrope_delta_from_template(
                SimpleNamespace(
                    mrope_positions=torch.zeros(3, 1),
                    positions=torch.empty(0, dtype=torch.long),
                )
            )
        )
        self.assertIsNone(
            _mrope_delta_from_template(
                SimpleNamespace(
                    mrope_positions=torch.zeros(3, 0),
                    positions=torch.tensor([1]),
                )
            )
        )
        positions = torch.tensor([4, 4])
        mrope = torch.tensor(
            [
                [10, 11],
                [20, 21],
                [30, 31],
            ]
        )
        delta = _mrope_delta_from_template(
            SimpleNamespace(positions=positions, mrope_positions=mrope)
        )
        self.assertTrue(torch.equal(delta, torch.tensor([[6]])))
        rows = _lookahead_mrope_positions(torch.tensor([7, 8]), delta)
        self.assertTrue(
            torch.equal(
                rows,
                torch.tensor([[13, 14], [13, 14], [13, 14]]),
            )
        )

    def test_launch_layer_forwards_the_template_without_sampling(self):
        _import_or_stub_forward_batch()
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            OptLayerForward,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        host = _layer_timing_host()
        seen = {}

        class _ModelRunner:
            def forward(self, forward_batch):
                seen["batch"] = forward_batch
                return SimpleNamespace(
                    logits_output=SimpleNamespace(
                        next_token_logits=torch.zeros(2, 4)
                    )
                )

            def sample(self, *_args, **_kwargs):
                raise AssertionError("sample")

        def run_batch(_batch):
            raise AssertionError("run_batch")

        host.run_batch = run_batch
        host.model_worker = SimpleNamespace(model_runner=_ModelRunner())
        runner = OptLayerForward.__new__(OptLayerForward)
        runner.scheduler = host
        runner.prefix_len = 4
        runner.cached_len = 4
        runner.req = SimpleNamespace(kv_allocated_len=0)
        runner._la_positions = torch.empty(2, dtype=torch.long)
        runner._la_seq = torch.zeros(1, dtype=torch.long)
        runner._la_orig = torch.zeros(1, dtype=torch.long)
        runner._lookahead_launch_event = None
        runner._lookahead_queued = False
        runner._alloc_lookahead_slots = lambda n: torch.arange(n)
        old_ids = torch.zeros(2, dtype=torch.long)
        runner._la_forward_template = SimpleNamespace(
            input_ids=old_ids,
            seq_lens=torch.zeros(1, dtype=torch.long),
            seq_lens_cpu=torch.zeros(1, dtype=torch.long),
            orig_seq_lens=torch.zeros(1, dtype=torch.long),
            seq_lens_sum=0,
            out_cache_loc=torch.zeros(2, dtype=torch.long),
            spec_info=None,
            positions=torch.zeros(2, dtype=torch.long),
            extend_num_tokens=0,
            mrope_positions=torch.zeros(3, 2, dtype=torch.long),
            req_pool_indices=torch.tensor([7]),
            forward_mode="verify",
            sampling_info="greedy",
        )
        runner._la_mrope_delta = torch.tensor([[5]])
        tokens = torch.tensor([8, 9])
        ancestor = torch.ones(2, 2, dtype=torch.int8)
        batch = SimpleNamespace()
        logits = runner.launch_layer(tokens, ancestor, 1, batch)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertIs(seen["batch"].input_ids, batch.input_ids)
        self.assertTrue(torch.equal(runner._la_forward_template.input_ids, old_ids))
        self.assertEqual(runner._la_forward_template.seq_lens_sum, 0)
        self.assertTrue(
            torch.equal(seen["batch"].mrope_positions, torch.full((3, 2), 9))
        )
        counts = get_sr_round_metrics(host, "Draft").counts
        self.assertEqual(counts["opt_lookahead_template_fallbacks"], 0)
        self.assertEqual(counts["opt_lookahead_eager_layers"], 0)

    def test_launch_layer_counts_an_eager_graph_miss(self):
        _import_or_stub_forward_batch()
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            OptLayerForward,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        host = _layer_timing_host()

        class _ModelRunner:
            def forward(self, _forward_batch):
                return SimpleNamespace(
                    can_run_graph=False,
                    logits_output=SimpleNamespace(
                        next_token_logits=torch.zeros(2, 4)
                    ),
                )

            def sample(self, *_args, **_kwargs):
                raise AssertionError("sample")

        def run_batch(_batch):
            raise AssertionError("run_batch")

        host.run_batch = run_batch
        host.model_worker = SimpleNamespace(model_runner=_ModelRunner())
        runner = OptLayerForward.__new__(OptLayerForward)
        runner.scheduler = host
        runner.prefix_len = 4
        runner.cached_len = 4
        runner.req = SimpleNamespace(kv_allocated_len=0)
        runner._la_positions = torch.empty(2, dtype=torch.long)
        runner._la_seq = torch.zeros(1, dtype=torch.long)
        runner._la_orig = torch.zeros(1, dtype=torch.long)
        runner._lookahead_launch_event = None
        runner._lookahead_queued = False
        runner._alloc_lookahead_slots = lambda n: torch.arange(n)
        runner._la_forward_template = SimpleNamespace(
            input_ids=torch.zeros(2, dtype=torch.long),
            seq_lens=torch.zeros(1, dtype=torch.long),
            seq_lens_cpu=torch.zeros(1, dtype=torch.long),
            orig_seq_lens=torch.zeros(1, dtype=torch.long),
            seq_lens_sum=0,
            out_cache_loc=torch.zeros(2, dtype=torch.long),
            spec_info=None,
            positions=torch.zeros(2, dtype=torch.long),
            extend_num_tokens=0,
            mrope_positions=None,
            req_pool_indices=torch.tensor([7]),
            forward_mode="verify",
            sampling_info="greedy",
        )
        runner._la_mrope_delta = None
        logits = runner.launch_layer(
            torch.tensor([8, 9]),
            torch.ones(2, 2, dtype=torch.int8),
            1,
            SimpleNamespace(),
        )
        self.assertEqual(tuple(logits.shape), (2, 4))
        counts = get_sr_round_metrics(host, "Draft").counts
        self.assertEqual(counts["opt_lookahead_eager_layers"], 1)
        self.assertEqual(counts["opt_lookahead_template_fallbacks"], 0)

    def test_launch_layer_without_a_template_uses_run_batch(self):
        _import_or_stub_forward_batch()
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            OptLayerForward,
        )
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            get_sr_round_metrics,
        )

        host = _layer_timing_host()
        seen = {"forward": 0, "run_batch": 0}

        class _ModelRunner:
            def forward(self, _forward_batch):
                seen["forward"] += 1
                raise AssertionError("forward")

            def sample(self, *_args, **_kwargs):
                raise AssertionError("sample")

        def run_batch(_batch):
            seen["run_batch"] += 1
            return SimpleNamespace(
                logits_output=SimpleNamespace(next_token_logits=torch.zeros(2, 4))
            )

        host.run_batch = run_batch
        host.model_worker = SimpleNamespace(model_runner=_ModelRunner())
        runner = OptLayerForward.__new__(OptLayerForward)
        runner.scheduler = host
        runner.prefix_len = 4
        runner.cached_len = 4
        runner.req = SimpleNamespace(kv_allocated_len=0)
        runner._la_positions = torch.empty(2, dtype=torch.long)
        runner._la_seq = torch.zeros(1, dtype=torch.long)
        runner._la_orig = torch.zeros(1, dtype=torch.long)
        runner._lookahead_launch_event = None
        runner._lookahead_queued = False
        runner._alloc_lookahead_slots = lambda n: torch.arange(n)
        runner._la_forward_template = None
        runner._la_mrope_delta = None
        logits = runner.launch_layer(
            torch.tensor([8, 9]), torch.ones(2, 2, dtype=torch.int8), 1, SimpleNamespace()
        )
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(seen, {"forward": 0, "run_batch": 1})
        self.assertEqual(
            get_sr_round_metrics(host, "Draft").counts["opt_lookahead_template_fallbacks"],
            1,
        )

    def test_post_reply_counts_stay_in_closed_window(self):
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            SRRoundMetrics,
            WINDOW_LIMIT,
        )

        run_opt_draft_round = _load_run_opt_draft_round()

        keys = ("opt_post_reply_us", "opt_bonus_rounds", "opt_layer_gpu_us")
        metrics = SRRoundMetrics("Draft")
        flushed = []
        original = metrics._emit_window

        def capture(reason):
            flushed.append({key: metrics.counts[key] for key in keys})
            original(reason)

        metrics._emit_window = capture

        def one_step(counts):
            for key in keys:
                counts[key] += 1

        for _ in range(WINDOW_LIMIT):
            run_opt_draft_round(metrics, True, one_step)

        self.assertEqual(flushed, [{key: WINDOW_LIMIT for key in keys}])
        for key in keys:
            self.assertEqual(metrics.counts[key], 0)

        run_opt_draft_round(metrics, True, one_step)
        for key in keys:
            self.assertEqual(metrics.counts[key], 1)

    def test_post_reply_drain_follows_deferred_jobs(self):
        drain, calls = _load_drain_deferred_layer_events()
        host = object()
        metrics = object()
        drain(host, metrics, defer_drain=False, jobs=[object()])
        self.assertEqual(calls, [(host, metrics)])
        calls.clear()
        drain(host, metrics, defer_drain=True, jobs=[])
        self.assertEqual(calls, [(host, metrics)])
        calls.clear()
        drain(host, metrics, defer_drain=False, jobs=[])
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
