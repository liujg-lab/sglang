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
        mask = layer_attention_mask(3, ancestor, "cpu")
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
            return [object() for _ in group_runners]

        original = forward._forward_verify_batch
        try:
            forward._forward_verify_batch = fake_batch
            rows = forward._forward_verify_fitting(radix, runners, steps)
            self.assertEqual(seen, [2, 1])
            self.assertEqual(len(rows), 3)
            seen.clear()
            forward._forward_verify_fitting(chunk, runners, steps)
            self.assertEqual(seen, [1, 1, 1])
        finally:
            forward._forward_verify_batch = original

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
        "_reraise_device_context": _reraise_device_context,
    }
    code = "from __future__ import annotations\n" + textwrap.dedent(src[start:end])
    exec(compile(code, str(path), "exec"), ns)
    return ns["_opt_expand_one"]


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


if __name__ == "__main__":
    unittest.main()
