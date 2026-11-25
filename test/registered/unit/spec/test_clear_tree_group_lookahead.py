"""CPU checks for ClearTree's batched Draft layer lookahead.

``grow_lookahead_many`` must make the same trees, the same K and the same
bonus ids as the layer-by-layer ``grow_active_layers`` when every layer's
logits are a function of that layer's tokens. The device work is replaced by
CPU tensors; nothing here needs CUDA.
"""

import os
import unittest
from collections import Counter
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.speculative.clear_tree_remote import clear_tree_forward as forward
from sglang.srt.speculative.clear_tree_remote import clear_tree_ops as ops
from sglang.srt.speculative.clear_tree_remote.clear_tree_config import (
    ClearTreeConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

VOCAB = 50


def _row_logits(token: int, depth: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(int(token) * 131 + int(depth) * 7 + 3)
    return torch.randn(VOCAB, generator=gen)


def _layer_logits(tokens: torch.Tensor, depth: int) -> torch.Tensor:
    return torch.stack([_row_logits(int(t), depth) for t in tokens.reshape(-1)])


class _Event:
    def record(self):
        pass

    def synchronize(self):
        pass


class _Bonus:
    def __init__(self):
        self.ids = None
        self.event = _Event()
        self.ready = False

    def copy(self, ids):
        self.ids = ids.clone()
        self.ready = True


class _Plan:
    """Stops each rid at a fixed depth and cuts every tree to one K."""

    def __init__(self, stops, k, defer=True):
        self.stops = dict(stops)
        self.k = int(k)
        self.use_tau = False
        self.defer = bool(defer)
        self._frozen_k = None
        self.begin_calls = []
        self.noted = []

    @property
    def defer_finish(self):
        return self.defer

    def bind_states(self, states):
        self._all = list(states)

    def begin_layer(self, states):
        self.begin_calls.append([(state.rid, int(state.depth)) for state in states])

    def should_grow(self, state):
        return int(state.depth) < self.stops[state.rid]

    def verify_k(self, state):
        return self.k

    def freeze_k(self, states):
        self._frozen_k = self.k
        return self.k

    def note(self, state, keep):
        self.noted.append((state.rid, int(keep)))


def _scheduler(plan, rids, prefixes):
    return SimpleNamespace(
        _clear_round_plan=plan,
        _clear_round_rids=list(rids),
        _clear_round_prefixes=list(prefixes),
    )


def _runners(prefixes):
    return [
        SimpleNamespace(prefix_len=p, cached_len=p, bonus_ids=[], layer_slots=[])
        for p in prefixes
    ]


def _quiet_patches(stack):
    stack.enter_context(patch.object(forward, "_draft_metrics", return_value=None))
    stack.enter_context(patch.object(forward, "_opt_draft_count"))
    stack.enter_context(patch.object(forward, "_begin_select_timing", return_value=None))
    stack.enter_context(patch.object(forward, "_finish_select_timing"))
    stack.enter_context(patch.object(torch.cuda, "Event", _Event))


def _fake_launch(scheduler, runners, tokens_dev, anc_dev, depth, width, event):
    for runner in runners:
        runner.cached_len += int(width)
    return torch.cat([_layer_logits(tokens_dev[i], depth) for i in range(len(runners))])


def _cpu_cache(count, width, bonus):
    with patch.object(torch.cuda, "Event", _Event):
        slots = (
            forward._PinnedLayerMany(count, width, pin_memory=False),
            forward._PinnedLayerMany(count, width, pin_memory=False),
        )
    return {
        "max_bs": count,
        "slots": slots,
        "bonus": [_Bonus() for _ in range(count)] if bonus else None,
    }


class TestDeviceOpsMany(CustomTestCase):
    def test_seed_many_matches_single(self):
        scores = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
        ids = torch.tensor([[7, 8, 9], [3, 4, 5]])
        w, t, a = ops.seed_on_device_many(scores, ids, 2)
        for row in range(2):
            ws, ts, as_ = ops.seed_on_device(scores[row], ids[row], 2)
            self.assertTrue(torch.equal(w[row], ws))
            self.assertTrue(torch.equal(t[row], ts))
            self.assertTrue(torch.equal(a[row], as_))

    def test_expand_many_matches_single_on_cpu_including_ties(self):
        # On CPU each row runs the same top-k, so ties break the same way.
        # CUDA does not promise this; production does not rely on it because
        # the host mirrors whatever the device chose.
        n = 3
        weight = torch.tensor([[0.5, 0.5, 0.25], [0.6, 0.3, 0.1]])
        anc = torch.stack(
            [torch.eye(n, dtype=torch.int8), torch.eye(n, dtype=torch.int8)]
        )
        scores = torch.tensor(
            [
                [[0.5, 0.5, 0.1, 0.0], [0.5, 0.25, 0.25, 0.0], [1.0, 0.0, 0.0, 0.0]],
                [[0.7, 0.2, 0.1, 0.0], [0.5, 0.4, 0.1, 0.0], [0.9, 0.05, 0.05, 0.0]],
            ]
        )
        ids = torch.arange(2 * n * 4).reshape(2, n, 4)
        got = ops.expand_on_device_many(weight, anc, scores, ids, n)
        for row in range(2):
            want = ops.expand_on_device(weight[row], anc[row], scores[row], ids[row], n)
            for a, b in zip(got, want):
                self.assertTrue(torch.equal(a[row], b), (row, a[row], b))


class TestGroupLookahead(CustomTestCase):
    def _run_pair(self, stops, k, prefixes, bonus, defer=True):
        n = 2
        rids = [f"r{i}" for i in range(len(prefixes))]
        tails = [torch.randn(VOCAB, generator=torch.Generator().manual_seed(i)) for i in range(len(rids))]
        config = ClearTreeConfig(width=n, tau=0.0, max_depth=6)
        env = {} if bonus else {"SGLANG_CLEAR_BONUS_PREDICTION": "0"}
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, env))
            if bonus:
                os.environ.pop("SGLANG_CLEAR_BONUS_PREDICTION", None)
            _quiet_patches(stack)

            ref_plan = _Plan(dict(zip(rids, stops)), k, defer)
            ref_states = []
            topk_out = [[] for _ in rids]

            def forward_active(steps, indices):
                return [
                    _layer_logits(step.tokens, int(step.positions.reshape(-1)[0]))
                    for step in steps
                ]

            ref_windows = forward.grow_active_layers(
                config,
                [t.clone() for t in tails],
                forward_active,
                _scheduler(ref_plan, rids, prefixes),
                states_out=ref_states,
                topk_out=topk_out,
            )

            plan = _Plan(dict(zip(rids, stops)), k, defer)
            runners = _runners(prefixes)
            states = []
            cache = _cpu_cache(len(rids), n, bonus)
            stack.enter_context(
                patch.object(forward, "_lookahead_group_cache", return_value=cache)
            )
            launch = stack.enter_context(
                patch.object(forward, "_launch_layer_many", side_effect=_fake_launch)
            )
            windows, launched = forward.grow_lookahead_many(
                _scheduler(plan, rids, prefixes),
                runners,
                config,
                [t.clone() for t in tails],
                states,
            )
        return dict(
            ref_windows=ref_windows,
            ref_states=ref_states,
            ref_plan=ref_plan,
            topk_out=topk_out,
            windows=windows,
            launched=launched,
            states=states,
            plan=plan,
            runners=runners,
            cache=cache,
            launch=launch,
        )

    def test_matches_layer_by_layer_for_mixed_stops(self):
        for stops, bonus in (
            ([1, 3, 2], True),
            ([2, 2], True),
            ([3, 1, 4, 2], False),
        ):
            with self.subTest(stops=stops, bonus=bonus):
                r = self._run_pair(stops, k=4, prefixes=[10 + i for i in range(len(stops))], bonus=bonus)
                self.assertEqual(r["windows"], r["ref_windows"])
                self.assertEqual(r["plan"].noted, r["ref_plan"].noted)
                self.assertEqual(r["plan"].begin_calls, r["ref_plan"].begin_calls)
                for got, want in zip(r["states"], r["ref_states"]):
                    self.assertEqual(int(got.depth), int(want.depth))
                    d = int(got.depth)
                    self.assertTrue(torch.equal(got.tokens[:d], want.tokens[:d]))
                    self.assertTrue(torch.equal(got.parents[:d], want.parents[:d]))
                    self.assertTrue(torch.allclose(got.weight[:d], want.weight[:d]))
                # One launch per depth reached by the deepest tree.
                self.assertEqual(r["launched"], max(stops))
                for index, runner in enumerate(r["runners"]):
                    depth = stops[index]
                    self.assertTrue(runner.lookahead_inflight)
                    self.assertEqual(runner.cached_len, runner.prefix_len + depth * 2)
                    self.assertIsNotNone(runner._lookahead_launch_event)
                    if bonus:
                        self.assertEqual(len(runner.bonus_ids), len(r["topk_out"][index]))
                        for got, want in zip(runner.bonus_ids, r["topk_out"][index]):
                            self.assertTrue(torch.equal(got, want))
                        state = r["states"][index]
                        final = _layer_logits(state.tokens[depth - 1], depth)
                        _s, ids = forward._vocab_topk(final, 3)
                        self.assertTrue(torch.equal(runner._lookahead_bonus.ids, ids))
                    else:
                        self.assertEqual(runner.bonus_ids, [])
                        self.assertIsNone(runner._lookahead_bonus)

    def test_tau_style_finish_without_defer(self):
        r = self._run_pair([2, 3], k=3, prefixes=[5, 9], bonus=True, defer=False)
        self.assertEqual(r["windows"], r["ref_windows"])
        self.assertEqual(r["plan"].noted, r["ref_plan"].noted)

    def test_failure_resets_every_runner_and_keeps_private_tails(self):
        n = 2
        rids = ["a", "b"]
        tails = [torch.randn(VOCAB), torch.randn(VOCAB)]
        config = ClearTreeConfig(width=n, tau=0.0, max_depth=6)
        calls = {"n": 0}

        def launch(scheduler, runners, tokens_dev, anc_dev, depth, width, event):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            return _fake_launch(scheduler, runners, tokens_dev, anc_dev, depth, width, event)

        plan = _Plan({"a": 3, "b": 3}, 4)
        runners = _runners([4, 6])
        states = []
        with ExitStack() as stack:
            _quiet_patches(stack)
            stack.enter_context(
                patch.object(
                    forward, "_lookahead_group_cache", return_value=_cpu_cache(2, n, True)
                )
            )
            stack.enter_context(patch.object(forward, "_launch_layer_many", side_effect=launch))
            reset = stack.enter_context(patch.object(forward, "_reset_lookahead_group"))
            with self.assertRaises(forward.LookaheadFallback):
                forward.grow_lookahead_many(
                    _scheduler(plan, rids, [4, 6]), runners, config, tails, states
                )
        reset.assert_called_once()
        self.assertIs(reset.call_args.args[1], runners)
        for runner, tail in zip(runners, tails):
            self.assertTrue(torch.equal(runner._lookahead_tail, tail))
        self.assertEqual(len(states), 2)


class TestGroupLookaheadOrderAndSetup(CustomTestCase):
    def test_stop_is_marked_before_the_k_cut(self):
        from sglang.srt.speculative.clear_tree_remote import clear_tree_ledger as ledger

        n = 2
        rids = ["a", "b"]
        config = ClearTreeConfig(width=n, tau=0.0, max_depth=6)
        order = []
        real_cut = forward._cut_deferred

        def cut(*args, **kwargs):
            order.append("cut")
            return real_cut(*args, **kwargs)

        with ExitStack() as stack:
            _quiet_patches(stack)
            stack.enter_context(
                patch.object(
                    forward, "_lookahead_group_cache", return_value=_cpu_cache(2, n, True)
                )
            )
            stack.enter_context(
                patch.object(forward, "_launch_layer_many", side_effect=_fake_launch)
            )
            stack.enter_context(
                patch.object(ledger, "mark_stop", side_effect=lambda s: order.append("stop"))
            )
            stack.enter_context(patch.object(forward, "_cut_deferred", side_effect=cut))
            forward.grow_lookahead_many(
                _scheduler(_Plan({"a": 2, "b": 1}, 4), rids, [3, 4]),
                _runners([3, 4]),
                config,
                [torch.randn(VOCAB), torch.randn(VOCAB)],
                [],
            )
        self.assertEqual(order, ["stop", "cut"])

    def test_setup_failure_falls_back(self):
        config = ClearTreeConfig(width=2, tau=0.0, max_depth=6)
        runners = _runners([3, 4])
        with ExitStack() as stack:
            _quiet_patches(stack)
            stack.enter_context(
                patch.object(
                    forward,
                    "_lookahead_group_cache",
                    side_effect=RuntimeError("pinned alloc failed"),
                )
            )
            reset = stack.enter_context(patch.object(forward, "_reset_lookahead_group"))
            with self.assertRaises(forward.LookaheadFallback):
                forward.grow_lookahead_many(
                    _scheduler(_Plan({"a": 1, "b": 1}, 4), ["a", "b"], [3, 4]),
                    runners,
                    config,
                    [torch.randn(VOCAB), torch.randn(VOCAB)],
                    [],
                )
        reset.assert_called_once()

    def test_unsettled_reset_is_not_turned_into_a_fallback(self):
        config = ClearTreeConfig(width=2, tau=0.0, max_depth=6)
        with ExitStack() as stack:
            _quiet_patches(stack)
            stack.enter_context(
                patch.object(
                    forward,
                    "_lookahead_group_cache",
                    side_effect=RuntimeError("pinned alloc failed"),
                )
            )
            stack.enter_context(
                patch.object(
                    forward,
                    "_reset_lookahead_group",
                    side_effect=forward.LookaheadUnsettled("rollback failed"),
                )
            )
            with self.assertRaises(forward.LookaheadUnsettled) as caught:
                forward.grow_lookahead_many(
                    _scheduler(_Plan({"a": 1, "b": 1}, 4), ["a", "b"], [3, 4]),
                    _runners([3, 4]),
                    config,
                    [torch.randn(VOCAB), torch.randn(VOCAB)],
                    [],
                )
        self.assertNotIsInstance(caught.exception, forward.LookaheadFallback)


def _decode_batch(reqs):
    return SimpleNamespace(
        reqs=list(reqs),
        sampling_info=SimpleNamespace(temperatures=torch.ones(len(reqs))),
        get_model_worker_batch=lambda: None,
    )


def _template_host(**extra):
    return SimpleNamespace(
        req_to_token_pool=SimpleNamespace(device="cpu"),
        _sr_make_decode_batch=Mock(side_effect=_decode_batch),
        **extra,
    )


def _admission_runners(rids, pools=None):
    pools = list(range(len(rids))) if pools is None else list(pools)
    return [
        SimpleNamespace(
            req=SimpleNamespace(rid=rid, req_pool_idx=pool),
            prefix_len=8,
            cached_len=8,
        )
        for rid, pool in zip(rids, pools)
    ]


class TestGroupAdmission(CustomTestCase):
    def _admit(self, host, runners):
        forward._admit_lookahead_many(host, runners, 2)

    def _budget(self, stack):
        stack.enter_context(patch.object(forward, "_opt_draft_count"))
        stack.enter_context(
            patch.object(forward, "_lookahead_reserve_kv")
        )
        stack.enter_context(patch.object(forward, "_lookahead_tree_need", return_value=1))

    def test_groups_of_one_size_share_a_template(self):
        host = _template_host()
        template = SimpleNamespace(mrope_positions=None)
        with ExitStack() as stack:
            self._budget(stack)
            build = stack.enter_context(
                patch.object(forward, "_lookahead_forward_template", return_value=template)
            )
            self._admit(host, _admission_runners(["a", "b"], [0, 1]))
            self.assertEqual(build.call_count, 2)  # m = 1 and m = 2
            # Different members and pools, same sizes: nothing is built.
            self._admit(host, _admission_runners(["c", "d"], [4, 7]))
            self.assertEqual(build.call_count, 2)
            forward.drop_lookahead_request(host, "a")
            self.assertIs(
                forward._lookahead_group_template(host, _admission_runners(["e", "f"]), 2),
                template,
            )
            self.assertEqual(build.call_count, 2)
            # A larger batch builds only the new size.
            self._admit(host, _admission_runners(["a", "b", "c"]))
            self.assertEqual(build.call_count, 3)
        self.assertIsNone(template.mm_inputs)
        self.assertIsNone(template.rids)

    def test_admission_reads_each_request_delta(self):
        host = _template_host()
        runners = _admission_runners(["a", "b"])
        runners[0].req.multimodal_inputs = SimpleNamespace(
            mrope_position_delta=torch.tensor([[-12]])
        )
        with ExitStack() as stack:
            self._budget(stack)
            stack.enter_context(
                patch.object(
                    forward,
                    "_lookahead_forward_template",
                    return_value=SimpleNamespace(mrope_positions=None),
                )
            )
            self._admit(host, runners)
        self.assertEqual([r._la_group_delta for r in runners], [-12, 0])

    def test_only_a_full_size_miss_falls_back(self):
        template = SimpleNamespace(mrope_positions=None)
        for missing, raises in ((1, False), (2, True)):
            with self.subTest(missing=missing), ExitStack() as stack:
                self._budget(stack)
                stack.enter_context(
                    patch.object(
                        forward,
                        "_lookahead_forward_template",
                        side_effect=lambda _s, batch, k=missing, **_kw: (
                            None if len(batch.reqs) == k else template
                        ),
                    )
                )
                host = _template_host()
                if raises:
                    with self.assertRaises(forward.LookaheadFallback):
                        self._admit(host, _admission_runners(["a", "b"]))
                else:
                    self._admit(host, _admission_runners(["a", "b"]))

    def test_miss_is_cached_and_admission_falls_back(self):
        host = _template_host()
        runners = _admission_runners(["a", "b"])
        with ExitStack() as stack:
            self._budget(stack)
            build = stack.enter_context(
                patch.object(forward, "_lookahead_forward_template", return_value=None)
            )
            with self.assertRaises(forward.LookaheadFallback):
                self._admit(host, runners)
            with self.assertRaises(forward.LookaheadFallback):
                self._admit(host, runners)
            self.assertEqual(build.call_count, 2)

    def test_kv_is_reserved_last(self):
        template = SimpleNamespace(mrope_positions=None)
        for built, reserve_error, reason in (
            (template, forward.LookaheadFallback("kv short"), "kv short"),
            (None, None, "cuda graph miss"),
        ):
            with self.subTest(reason=reason), ExitStack() as stack:
                stack.enter_context(patch.object(forward, "_opt_draft_count"))
                stack.enter_context(
                    patch.object(forward, "_lookahead_tree_need", return_value=5)
                )
                build = stack.enter_context(
                    patch.object(forward, "_lookahead_forward_template", return_value=built)
                )
                reserve = stack.enter_context(
                    patch.object(forward, "_lookahead_reserve_kv", side_effect=reserve_error)
                )
                with self.assertRaises(forward.LookaheadFallback) as caught:
                    self._admit(_template_host(), _admission_runners(["a", "b"]))
                self.assertIn(reason, str(caught.exception))
                self.assertEqual(build.call_count, 2)
                # A graph miss must not evict anything.
                self.assertEqual(reserve.call_count, 1 if built is not None else 0)

    def test_kv_need_error_becomes_a_fallback(self):
        with patch.object(
            forward, "_lookahead_tree_need", side_effect=RuntimeError("no plan")
        ):
            with self.assertRaises(forward.LookaheadFallback) as caught:
                self._admit(_template_host(), _admission_runners(["a"]))
        self.assertIn("kv need unknown", str(caught.exception))


def _graph_host(can_run):
    graph_runner = SimpleNamespace(can_run=can_run)
    return _template_host(
        model_worker=SimpleNamespace(
            model_runner=SimpleNamespace(graph_runner=graph_runner)
        )
    )


def _forward_batch():
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_cuda_graph=lambda: True),
        is_sr_tail_extend=False,
        mrope_positions=None,
    )


class TestGroupTemplateCache(CustomTestCase):
    def test_init_new_or_can_run_error_is_not_cached(self):
        from sglang.srt.model_executor import forward_batch_info

        fb = _forward_batch()
        cases = {
            "init_new": (
                Mock(side_effect=[RuntimeError("one bad request"), fb]),
                Mock(return_value=True),
            ),
            "can_run": (
                Mock(return_value=fb),
                Mock(side_effect=[RuntimeError("one bad request"), True]),
            ),
        }
        for name, (init_new, can_run) in cases.items():
            with self.subTest(failing=name), ExitStack() as stack:
                count = stack.enter_context(patch.object(forward, "_opt_draft_count"))
                stack.enter_context(
                    patch.object(forward_batch_info.ForwardBatch, "init_new", init_new)
                )
                host = _graph_host(can_run)
                self.assertIsNone(
                    forward._lookahead_group_template(host, _admission_runners(["a"]), 2)
                )
                self.assertNotIn((2, 1), host._opt_la_group_templates)
                got = forward._lookahead_group_template(host, _admission_runners(["b"]), 2)
                self.assertIs(got, fb)
                names = [c.args[1] for c in count.call_args_list]
                self.assertEqual(
                    names,
                    ["opt_lookahead_group_template_errors", "opt_lookahead_group_templates"],
                )

    def test_no_graph_is_a_cached_miss_and_counted_as_one(self):
        from sglang.srt.model_executor import forward_batch_info

        can_run = Mock(return_value=False)
        with patch.object(forward, "_opt_draft_count") as count, patch.object(
            forward_batch_info.ForwardBatch, "init_new", Mock(return_value=_forward_batch())
        ):
            host = _graph_host(can_run)
            self.assertIsNone(
                forward._lookahead_group_template(host, _admission_runners(["a"]), 2)
            )
            self.assertIsNone(
                forward._lookahead_group_template(host, _admission_runners(["b"]), 2)
            )
        self.assertEqual(can_run.call_count, 1)
        self.assertEqual(
            [c.args[1] for c in count.call_args_list],
            ["opt_lookahead_group_template_misses"],
        )

    def test_single_request_path_still_swallows_errors(self):
        from sglang.srt.model_executor import forward_batch_info

        with patch.object(
            forward_batch_info.ForwardBatch,
            "init_new",
            Mock(side_effect=RuntimeError("bad")),
        ):
            host = _graph_host(Mock(return_value=True))
            self.assertIsNone(
                forward._lookahead_forward_template(host, _decode_batch([object()]))
            )

    def test_layers_only_look_up(self):
        host = _template_host()
        self.assertIsNone(forward._cached_group_template(host, 2, 1))
        template = SimpleNamespace(mrope_positions=None)
        host._opt_la_group_templates = {(2, 1): template, (2, 2): None}
        self.assertIs(forward._cached_group_template(host, 2, 1), template)
        self.assertIsNone(forward._cached_group_template(host, 2, 2))
        self.assertIsNone(forward._cached_group_template(host, 2, 3))
        host._sr_make_decode_batch.assert_not_called()

    def test_request_bound_fields_are_a_cached_miss(self):
        cases = [
            ({name: torch.tensor([1])}, {})
            for name in forward._GROUP_REQUEST_FIELDS
        ] + [({}, {"server_args": SimpleNamespace(enable_lora=True)})]
        for fields, extra in cases:
            with self.subTest(fields=list(fields), extra=list(extra)), ExitStack() as stack:
                self_host = _template_host(**extra)
                stack.enter_context(patch.object(forward, "_opt_draft_count"))
                stack.enter_context(
                    patch.object(forward, "_lookahead_reserve_kv")
                )
                stack.enter_context(
                    patch.object(forward, "_lookahead_tree_need", return_value=1)
                )
                build = stack.enter_context(
                    patch.object(
                        forward,
                        "_lookahead_forward_template",
                        side_effect=lambda *_a, f=fields, **_kw: SimpleNamespace(
                            mrope_positions=None, **f
                        ),
                    )
                )
                runners = _admission_runners(["a"])
                with self.assertRaises(forward.LookaheadFallback):
                    forward._admit_lookahead_many(self_host, runners, 2)
                with self.assertRaises(forward.LookaheadFallback):
                    forward._admit_lookahead_many(self_host, runners, 2)
                self.assertEqual(build.call_count, 1)

    def test_would_evict_checks_tree_and_allocator_before_budget(self):
        need = Mock(side_effect=RuntimeError("no plan"))
        with patch.object(forward, "_lookahead_kv_need", need):
            self.assertFalse(forward._lookahead_would_evict(SimpleNamespace(tree_cache=None)))
            self.assertTrue(
                forward._lookahead_would_evict(
                    SimpleNamespace(tree_cache=SimpleNamespace(token_to_kv_pool_allocator=None))
                )
            )
        need.assert_not_called()


class _Pool:
    """Free list and evictable radix tokens for ``_lookahead_reserve_kv``."""

    def __init__(self, free, evictable):
        self.free = int(free)
        self.evictable = int(evictable)
        self.evict_calls = []
        self.allocator = SimpleNamespace(available_size=lambda: self.free)
        self.tree = SimpleNamespace(
            token_to_kv_pool_allocator=self.allocator,
            evictable_size=Mock(side_effect=lambda: self.evictable),
            is_chunk_cache=lambda: False,
            evict=self._evict,
        )

    def _evict(self, params):
        self.evict_calls.append(int(params.num_tokens))
        moved = min(int(params.num_tokens), self.evictable)
        self.evictable -= moved
        self.free += moved

    def host(self):
        return SimpleNamespace(tree_cache=self.tree)


class TestLookaheadReserveKV(CustomTestCase):
    def _reserve(self, pool, need, *, plain=True):
        with patch.object(forward, "_plain_radix_eviction", return_value=plain), patch.object(
            forward, "_opt_draft_count"
        ) as count:
            try:
                forward._lookahead_reserve_kv(pool.host(), need)
            finally:
                self.counts = [(c.args[1], c.args[2] if len(c.args) > 2 else 1) for c in count.call_args_list]

    def test_enough_free_space_does_not_evict(self):
        pool = _Pool(free=100, evictable=1000)
        self._reserve(pool, 100)
        self.assertEqual(pool.evict_calls, [])
        pool.tree.evictable_size.assert_not_called()
        self.assertEqual(self.counts, [])

    def test_short_free_list_evicts_twice_the_need(self):
        pool = _Pool(free=10, evictable=1000)
        self._reserve(pool, 100)
        self.assertEqual(pool.evict_calls, [forward._LOOKAHEAD_EVICT_FACTOR * 100])
        self.assertEqual(pool.free, 210)
        self.assertEqual(
            self.counts,
            [("opt_lookahead_kv_evictions", 1), ("opt_lookahead_kv_evicted_tokens", 200)],
        )

    def test_evictable_covers_only_the_need(self):
        pool = _Pool(free=10, evictable=95)
        self._reserve(pool, 100)
        self.assertEqual(pool.free, 105)
        self.assertEqual(pool.evictable, 0)

    def test_free_plus_evictable_short_falls_back_with_sizes(self):
        pool = _Pool(free=10, evictable=50)
        with self.assertRaises(forward.LookaheadFallback) as caught:
            self._reserve(pool, 100)
        message = str(caught.exception)
        for part in ("need=100", "free=10", "evictable=50"):
            self.assertIn(part, message)
        self.assertEqual(pool.evict_calls, [])

    def test_other_caches_keep_falling_back(self):
        pool = _Pool(free=10, evictable=1000)
        with self.assertRaises(forward.LookaheadFallback) as caught:
            self._reserve(pool, 100, plain=False)
        self.assertIn("not-plain-radix", str(caught.exception))
        self.assertEqual(pool.evict_calls, [])
        pool.tree.evictable_size.assert_not_called()

    def test_eviction_that_frees_too_little_falls_back(self):
        pool = _Pool(free=10, evictable=1000)
        pool.tree.evict = lambda params: None  # evicts nothing
        with self.assertRaises(forward.LookaheadFallback) as caught:
            self._reserve(pool, 100)
        for part in ("after evict", "need=100", "free=10", "evictable=1000"):
            self.assertIn(part, str(caught.exception))
        self.assertEqual(self.counts, [])  # nothing freed, nothing counted

    def test_no_tree_cache_and_no_allocator(self):
        forward._lookahead_reserve_kv(SimpleNamespace(tree_cache=None), 10)
        with self.assertRaises(forward.LookaheadFallback):
            forward._lookahead_reserve_kv(
                SimpleNamespace(tree_cache=SimpleNamespace(token_to_kv_pool_allocator=None)), 10
            )

    def test_only_plain_radix_over_token_allocator_counts_as_plain(self):
        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
        from sglang.srt.mem_cache.radix_cache import RadixCache

        class _SubRadix(RadixCache):
            pass

        plain = object.__new__(RadixCache)
        plain.token_to_kv_pool_allocator = object.__new__(TokenToKVPoolAllocator)
        self.assertTrue(forward._plain_radix_eviction(plain))
        sub = object.__new__(_SubRadix)
        sub.token_to_kv_pool_allocator = object.__new__(TokenToKVPoolAllocator)
        self.assertFalse(forward._plain_radix_eviction(sub))
        other = object.__new__(RadixCache)
        other.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 0)
        self.assertFalse(forward._plain_radix_eviction(other))
        self.assertFalse(forward._plain_radix_eviction(SimpleNamespace()))

    def test_group_admission_reserves_every_tree_together(self):
        runners = _admission_runners(["a", "b"])  # prefix 8, nothing committed
        with ExitStack() as stack:
            stack.enter_context(patch.object(forward, "_opt_draft_count"))
            stack.enter_context(
                patch.object(
                    forward, "_lookahead_tree_need", side_effect=lambda _s, ingest: 10 + ingest
                )
            )
            reserve = stack.enter_context(patch.object(forward, "_lookahead_reserve_kv"))
            stack.enter_context(
                patch.object(
                    forward,
                    "_lookahead_forward_template",
                    return_value=SimpleNamespace(mrope_positions=None),
                )
            )
            host = _template_host()
            forward._admit_lookahead_many(host, runners, 2)
        reserve.assert_called_once_with(host, 36)

    def _single(self, *, cached, reserve_error=None, built=None):
        runner = SimpleNamespace(
            prefix_len=12,
            cached_len=12,
            req=SimpleNamespace(kv_committed_len=10),
            _la_positions=torch.zeros(1, dtype=torch.long),
            _la_seq=torch.zeros(1, dtype=torch.long),
            _la_seq_cpu=torch.zeros(1, dtype=torch.long),
            _la_orig=torch.zeros(1, dtype=torch.int32),
            _la_dummy_tokens=torch.zeros(4, dtype=torch.long),
            _la_forward_template="stale",
            _la_mrope_delta="stale",
        )
        host = SimpleNamespace(req_to_token_pool=SimpleNamespace(device="cpu"))
        with ExitStack() as stack:
            stack.enter_context(patch.object(forward, "_round_width", return_value=4))
            stack.enter_context(
                patch.object(
                    forward, "_lookahead_tree_need", side_effect=lambda _s, ingest: 70 + ingest
                )
            )
            reserve = stack.enter_context(
                patch.object(forward, "_lookahead_reserve_kv", side_effect=reserve_error)
            )
            stack.enter_context(
                patch.object(forward, "_load_layer_admit", return_value=cached)
            )
            stack.enter_context(
                patch.object(forward, "_lookahead_forward_template", return_value=built)
            )
            stack.enter_context(patch.object(forward, "_store_layer_admit"))
            error = None
            try:
                forward._admit_lookahead(host, runner, SimpleNamespace())
            except forward.LookaheadFallback as exc:
                error = exc
        return runner, host, reserve, error

    def test_single_admission_reserves_after_the_template(self):
        template = SimpleNamespace(mrope_positions=None)
        runner, host, reserve, error = self._single(cached=(template, None))
        self.assertIsNone(error)
        reserve.assert_called_once_with(host, 72)  # tree need only: 70 + ingest 2
        self.assertIs(runner._la_forward_template, template)

    def test_single_kv_fallback_clears_the_runner_template(self):
        runner, _host, reserve, error = self._single(
            cached=(SimpleNamespace(mrope_positions=None), None),
            reserve_error=forward.LookaheadFallback("kv short"),
        )
        self.assertIn("kv short", str(error))
        self.assertIsNone(runner._la_forward_template)
        self.assertIsNone(runner._la_mrope_delta)

    def test_single_graph_miss_does_not_evict(self):
        _runner, _host, reserve, error = self._single(cached=None, built=None)
        self.assertIn("cuda graph miss", str(error))
        reserve.assert_not_called()

    def test_eviction_error_falls_back_and_device_error_propagates(self):
        pool = _Pool(free=10, evictable=1000)

        def broken(_params):
            raise RuntimeError("radix bookkeeping")

        pool.tree.evict = broken
        with self.assertRaises(forward.LookaheadFallback) as caught:
            self._reserve(pool, 100)
        self.assertIn("kv evict failed", str(caught.exception))

        pool = _Pool(free=10, evictable=1000)

        def dead(_params):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        pool.tree.evict = dead
        host = pool.host()
        host._sr_clear_stamps_after_poison = Mock()
        with patch.object(forward, "_plain_radix_eviction", return_value=True), patch.object(
            forward, "_opt_draft_count"
        ):
            with self.assertRaises(RuntimeError) as caught:
                forward._lookahead_reserve_kv(host, 100)
        self.assertNotIsInstance(caught.exception, forward.LookaheadFallback)
        self.assertTrue(host._sr_device_poisoned)

    def test_real_radix_cache_keeps_a_locked_prefix(self):
        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

        allocator = TokenToKVPoolAllocator(
            size=16, dtype=torch.float16, device="cpu", kvcache=None, need_sort=False
        )
        cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
            )
        )
        locked_key = RadixKey(token_ids=[1, 2, 3, 4, 5, 6], extra_key=None)
        idle_key = RadixKey(token_ids=[11, 12, 13, 14, 15, 16], extra_key=None)
        cache.insert(InsertParams(key=locked_key, value=allocator.alloc(6)))
        cache.insert(InsertParams(key=idle_key, value=allocator.alloc(6)))
        node = cache.match_prefix(MatchPrefixParams(key=locked_key)).last_device_node
        cache.inc_lock_ref(node)
        self.assertEqual(allocator.available_size(), 4)
        self.assertEqual(cache.evictable_size(), 6)
        self.assertTrue(forward._plain_radix_eviction(cache))

        host = SimpleNamespace(tree_cache=cache)
        with patch.object(forward, "_opt_draft_count") as count:
            forward._lookahead_reserve_kv(host, 8)
        self.assertEqual(allocator.available_size(), 10)
        locked = cache.match_prefix(MatchPrefixParams(key=locked_key)).device_indices
        idle = cache.match_prefix(MatchPrefixParams(key=idle_key)).device_indices
        self.assertEqual(int(locked.numel()), 6)
        self.assertEqual(int(idle.numel()), 0)
        self.assertEqual(
            [(c.args[1], c.args[2] if len(c.args) > 2 else 1) for c in count.call_args_list],
            [("opt_lookahead_kv_evictions", 1), ("opt_lookahead_kv_evicted_tokens", 6)],
        )
        # The locked prefix alone cannot cover a larger need.
        with patch.object(forward, "_opt_draft_count"):
            with self.assertRaises(forward.LookaheadFallback):
                forward._lookahead_reserve_kv(host, 11)

    def test_tree_need_leaves_the_bonus_out(self):
        host = SimpleNamespace(server_args=SimpleNamespace(clear_tree_max_depth=8))
        with patch.object(forward, "_round_width", return_value=4), patch.object(
            forward, "_lookahead_bonus_slots", return_value=65
        ):
            self.assertEqual(forward._lookahead_tree_need(host, 3), 3 + 7 * 4)
            self.assertEqual(forward._lookahead_kv_need(host, 3), 3 + 7 * 4 + 65)
        with patch.object(forward, "bonus_prediction_enabled", return_value=False), patch.object(
            forward, "_bonus_width"
        ) as choose:
            self.assertEqual(forward._lookahead_bonus_slots(host), 0)
        choose.assert_not_called()


class TestRequestMropeDelta(CustomTestCase):
    def test_host_delta_absent_and_device(self):
        req = SimpleNamespace(
            multimodal_inputs=SimpleNamespace(mrope_position_delta=torch.tensor([[-37]]))
        )
        self.assertEqual(forward._request_mrope_delta(req), -37)
        self.assertEqual(forward._request_mrope_delta(SimpleNamespace()), 0)
        self.assertEqual(
            forward._request_mrope_delta(SimpleNamespace(multimodal_inputs=None)), 0
        )
        self.assertIsNone(
            forward._request_mrope_delta(
                SimpleNamespace(multimodal_inputs=SimpleNamespace(mrope_position_delta=None))
            )
        )
        for bad in (Mock(is_cuda=True), torch.tensor([[1], [2]]), torch.zeros(1, 0)):
            with self.subTest(bad=type(bad).__name__), self.assertRaises(
                forward.LookaheadFallback
            ):
                forward._request_mrope_delta(
                    SimpleNamespace(
                        multimodal_inputs=SimpleNamespace(mrope_position_delta=bad)
                    )
                )

    def test_missing_delta_falls_back_only_with_mrope(self):
        for mrope, raises in ((None, False), (torch.zeros(3, 4), True)):
            with self.subTest(mrope=mrope is not None), ExitStack() as stack:
                stack.enter_context(patch.object(forward, "_opt_draft_count"))
                stack.enter_context(
                    patch.object(forward, "_lookahead_reserve_kv")
                )
                stack.enter_context(
                    patch.object(forward, "_lookahead_tree_need", return_value=1)
                )
                stack.enter_context(
                    patch.object(
                        forward,
                        "_lookahead_forward_template",
                        return_value=SimpleNamespace(mrope_positions=mrope),
                    )
                )
                runners = _admission_runners(["a", "b"])
                runners[1].req.multimodal_inputs = SimpleNamespace(
                    mrope_position_delta=None
                )
                if raises:
                    with self.assertRaises(forward.LookaheadFallback):
                        forward._admit_lookahead_many(_template_host(), runners, 2)
                else:
                    forward._admit_lookahead_many(_template_host(), runners, 2)

    def test_group_positions_match_spec_formula(self):
        n = 3
        positions = torch.tensor([10, 10, 10, 40, 40, 40, 7, 7, 7])
        host_deltas = [torch.tensor([[-5]]), None, torch.tensor([[12]])]
        reqs = [
            SimpleNamespace(
                rid=str(i),
                multimodal_inputs=(
                    None if d is None else SimpleNamespace(mrope_position_delta=d)
                ),
            )
            for i, d in enumerate(host_deltas)
        ]
        # forward_batch_info._compute_spec_mrope_positions, target_verify branch.
        b = len(reqs)
        stacked = torch.stack(
            [
                torch.zeros(1, dtype=torch.int64)
                if r.multimodal_inputs is None
                else r.multimodal_inputs.mrope_position_delta.squeeze(0)
                for r in reqs
            ],
            dim=0,
        )
        expected = (positions.view(b, -1) + stacked).flatten().unsqueeze(0).repeat(3, 1)
        deltas = torch.tensor([forward._request_mrope_delta(r) for r in reqs])
        template = SimpleNamespace(mrope_positions=torch.zeros(3, b * n, dtype=torch.long))
        source = SimpleNamespace(
            input_ids=None,
            seq_lens=None,
            seq_lens_cpu=None,
            orig_seq_lens=None,
            seq_lens_sum=0,
            out_cache_loc=None,
            spec_info=SimpleNamespace(positions=positions),
        )
        fb = forward._refresh_group_forward(
            template, source, torch.tensor([1, 2, 3]), deltas, reqs
        )
        self.assertTrue(torch.equal(fb.mrope_positions, expected))
        self.assertEqual(fb.rids, ["0", "1", "2"])
        self.assertIs(fb.mm_inputs[0], reqs[0].multimodal_inputs)
        self.assertTrue(torch.equal(template.mrope_positions, torch.zeros(3, b * n, dtype=torch.long)))


class TestLaunchLayerMany(CustomTestCase):
    N = 2

    def _setup(self, stack, *, cascade, assign_error=None, deltas=None, admitted=True):
        n = self.N
        depth = 2
        deltas = [0, 0] if deltas is None else list(deltas)
        runners = [
            SimpleNamespace(
                prefix_len=p,
                cached_len=p + (depth - 1) * n,
                req=SimpleNamespace(
                    rid=rid,
                    req_pool_idx=pool,
                    kv_allocated_len=p + (depth - 1) * n,
                    multimodal_inputs=SimpleNamespace(tag=rid),
                ),
                layer_slots=[],
                _la_group_delta=delta,
            )
            for (p, pool, rid), delta in zip(((10, 5, "a"), (20, 6, "b")), deltas)
        ]
        if not admitted:
            # Not seen by admission: the layer reads each request's own delta.
            for runner, delta in zip(runners, deltas):
                del runner._la_group_delta
                runner.req.multimodal_inputs.mrope_position_delta = torch.tensor([[delta]])
        m = len(runners)
        logits = torch.arange(m * n * VOCAB, dtype=torch.float32).reshape(m * n, VOCAB)
        model_forward = Mock(
            return_value=SimpleNamespace(
                can_run_graph=True, logits_output=SimpleNamespace(next_token_logits=logits)
            )
        )
        allocator = Mock()
        host = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(device="cpu", req_to_token=torch.zeros(8, 64, dtype=torch.long)),
            tree_cache=object(),
            token_to_kv_pool_allocator=allocator,
            model_worker=SimpleNamespace(model_runner=SimpleNamespace(forward=model_forward)),
        )
        # Built for another group of the same size: pools 0 and 1, no M-RoPE.
        template = SimpleNamespace(
            req_pool_indices=torch.tensor([0, 1]),
            mrope_positions=torch.zeros(3, m * n, dtype=torch.long),
            positions=None,
            mm_inputs=None,
            rids=None,
        )
        loc = torch.arange(100, 100 + m * n)
        import sglang.srt.mem_cache.common as mem_common
        import sglang.srt.speculative.spec_utils as spec_utils

        stack.enter_context(patch.object(mem_common, "alloc_token_slots", return_value=loc))
        assign = stack.enter_context(
            patch.object(spec_utils, "assign_req_to_token_pool_func", side_effect=assign_error)
        )
        stack.enter_context(
            patch.object(forward, "_cached_group_template", return_value=template)
        )
        stack.enter_context(
            patch.object(
                forward,
                "_lookahead_group_template",
                side_effect=AssertionError("a layer must not build a template"),
            )
        )
        stack.enter_context(patch.object(forward, "_cascade_on", return_value=cascade))
        bind = stack.enter_context(patch.object(forward, "bind_cascade"))
        stack.enter_context(patch.object(forward, "_begin_layer_timing", return_value=None))
        stack.enter_context(patch.object(forward, "_opt_draft_count"))
        tokens = torch.tensor([[1, 2], [3, 4]])
        anc = torch.ones(m, n, depth * n, dtype=torch.int8)
        return dict(
            host=host, runners=runners, tokens=tokens, anc=anc, depth=depth, loc=loc,
            assign=assign, bind=bind, model_forward=model_forward, template=template,
            allocator=allocator, logits=logits,
        )

    def test_batch_inputs_slots_and_lengths(self):
        n = self.N
        for cascade in (True, False):
            with self.subTest(cascade=cascade), ExitStack() as stack:
                c = self._setup(stack, cascade=cascade, deltas=[7, 9])
                runners = c["runners"]
                cached = [r.cached_len for r in runners]
                out = forward._launch_layer_many(
                    c["host"], runners, c["tokens"], c["anc"], c["depth"], n, _Event()
                )
                self.assertTrue(torch.equal(out, c["logits"]))
                pool, _table, start, end, loc, bs = c["assign"].call_args.args
                self.assertEqual(pool.tolist(), [5, 6])
                self.assertEqual(start.tolist(), cached)
                self.assertEqual(end.tolist(), [x + n for x in cached])
                self.assertTrue(torch.equal(loc, c["loc"]))
                self.assertEqual(bs, 2)
                fb = c["model_forward"].call_args.args[0]
                self.assertEqual(fb.input_ids.tolist(), [1, 2, 3, 4])
                self.assertEqual(fb.seq_lens.tolist(), cached)
                self.assertEqual(fb.orig_seq_lens.dtype, torch.int32)
                self.assertEqual(fb.seq_lens_cpu.tolist(), cached)
                self.assertEqual(fb.seq_lens_sum, sum(cached))
                self.assertEqual(fb.positions.tolist(), [11, 11, 21, 21])
                self.assertEqual(
                    fb.mrope_positions.tolist(), [[18, 18, 30, 30]] * 3
                )
                self.assertEqual(fb.req_pool_indices.tolist(), [5, 6])
                self.assertEqual(fb.rids, ["a", "b"])
                self.assertEqual([mm.tag for mm in fb.mm_inputs], ["a", "b"])
                template = c["template"]
                self.assertEqual(template.req_pool_indices.tolist(), [0, 1])
                self.assertEqual(template.mrope_positions.abs().sum().item(), 0)
                self.assertIsNone(template.mm_inputs)
                self.assertIsNone(template.rids)
                self.assertIsNone(template.positions)
                mask = fb.spec_info.custom_mask
                if cascade:
                    self.assertEqual(int(mask.numel()), 2 * n * c["depth"] * n)
                else:
                    self.assertEqual(
                        int(mask.numel()),
                        sum(n * (r.prefix_len + c["depth"] * n) for r in runners),
                    )
                bargs = c["bind"].call_args
                self.assertIs(bargs.args[1], runners)
                self.assertEqual(bargs.args[3], n)
                self.assertEqual(bargs.kwargs["cached_lens"], cached)
                for i, r in enumerate(runners):
                    self.assertEqual(r.cached_len, cached[i] + n)
                    self.assertEqual(r.req.kv_allocated_len, cached[i] + n)
                    self.assertEqual(r.layer_slots[-1].tolist(), c["loc"][i * n : (i + 1) * n].tolist())

    def test_failed_assign_returns_the_slots(self):
        with ExitStack() as stack:
            c = self._setup(stack, cascade=True, assign_error=RuntimeError("assign"))
            with self.assertRaises(RuntimeError):
                forward._launch_layer_many(
                    c["host"], c["runners"], c["tokens"], c["anc"], c["depth"], self.N, _Event()
                )
        c["allocator"].free.assert_called_once()
        self.assertTrue(torch.equal(c["allocator"].free.call_args.args[0], c["loc"]))
        for r in c["runners"]:
            self.assertEqual(r.layer_slots, [])

    def test_short_row_raises_before_allocation(self):
        import sglang.srt.mem_cache.common as mem_common

        with ExitStack() as stack:
            c = self._setup(stack, cascade=True)
            # Rows start at 12 and 22; a 23-column row cannot take [22, 24).
            c["host"].req_to_token_pool.req_to_token = torch.zeros(8, 23, dtype=torch.long)
            cached = [r.cached_len for r in c["runners"]]
            with self.assertRaises(forward.RowOverflow):
                forward._launch_layer_many(
                    c["host"], c["runners"], c["tokens"], c["anc"], c["depth"], self.N, _Event()
                )
            mem_common.alloc_token_slots.assert_not_called()
            c["assign"].assert_not_called()
            c["model_forward"].assert_not_called()
        self.assertEqual([r.cached_len for r in c["runners"]], cached)
        self.assertEqual([r.layer_slots for r in c["runners"]], [[], []])

    def test_delta_read_from_request_when_not_admitted(self):
        with ExitStack() as stack:
            c = self._setup(stack, cascade=True, deltas=[7, 9], admitted=False)
            forward._launch_layer_many(
                c["host"], c["runners"], c["tokens"], c["anc"], c["depth"], self.N, _Event()
            )
            fb = c["model_forward"].call_args.args[0]
        self.assertEqual(fb.mrope_positions.tolist(), [[18, 18, 30, 30]] * 3)

    def test_missing_delta_with_mrope_template_raises(self):
        with ExitStack() as stack:
            c = self._setup(stack, cascade=True)
            c["runners"][0]._la_group_delta = None
            with self.assertRaises(RuntimeError):
                forward._launch_layer_many(
                    c["host"], c["runners"], c["tokens"], c["anc"], c["depth"], self.N, _Event()
                )
            c["model_forward"].assert_not_called()

    def test_size_without_template_uses_the_plain_forward(self):
        n = self.N
        with ExitStack() as stack:
            c = self._setup(stack, cascade=True)
            stack.enter_context(
                patch.object(forward, "_cached_group_template", return_value=None)
            )
            spec = SimpleNamespace()
            verify = stack.enter_context(
                patch.object(
                    forward, "_verify_batch_many", return_value=SimpleNamespace(spec_info=spec)
                )
            )
            run = stack.enter_context(
                patch.object(forward, "_run_batch", return_value=c["logits"])
            )
            out = forward._launch_layer_many(
                c["host"], c["runners"], c["tokens"], c["anc"], c["depth"], n, _Event()
            )
        self.assertTrue(torch.equal(out, c["logits"]))
        verify.assert_called_once()
        run.assert_called_once()
        c["model_forward"].assert_not_called()
        pool = c["assign"].call_args.args[0]
        self.assertEqual(pool.tolist(), [5, 6])


    def test_copy_failure_keeps_every_row_in_rollback(self):
        with ExitStack() as stack:
            c = self._setup(stack, cascade=True)
            cached = [r.cached_len for r in c["runners"]]
            c["runners"][1].layer_slots = _FailingList()
            with self.assertRaises(RuntimeError):
                forward._launch_layer_many(
                    c["host"], c["runners"], c["tokens"], c["anc"], c["depth"], self.N, _Event()
                )
        # Both rows are mapped, so both must already be inside the rollback range.
        for runner, before in zip(c["runners"], cached):
            self.assertEqual(runner.req.kv_allocated_len, before + self.N)
        c["allocator"].free.assert_not_called()


class _FailingList(list):
    """A ``layer_slots`` whose append fails, standing in for a failed ``clone()``."""

    def append(self, item):
        raise RuntimeError("clone failed")


def _device_module(sync=None):
    return SimpleNamespace(synchronize=Mock(side_effect=sync))


class TestSingleSlotRegistration(CustomTestCase):
    def test_copy_failure_after_the_map_is_still_rolled_back(self):
        import sglang.srt.mem_cache.common as mem_common
        import sglang.srt.speculative.spec_utils as spec_utils

        runner = SimpleNamespace(
            scheduler=SimpleNamespace(
                req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(4, 32, dtype=torch.long)),
                tree_cache=object(),
            ),
            _la_seq=torch.zeros(1, dtype=torch.long),
            _la_pool=torch.tensor([1]),
            cached_len=10,
            req=SimpleNamespace(kv_allocated_len=10),
            layer_slots=_FailingList(),
        )
        with patch.object(
            mem_common, "alloc_token_slots", return_value=torch.tensor([7, 8])
        ), patch.object(spec_utils, "assign_req_to_token_pool_func"):
            with self.assertRaises(RuntimeError):
                forward.OptLayerForward._alloc_lookahead_slots(runner, 2)
        self.assertEqual(runner.req.kv_allocated_len, 12)


def _settle_runner(rid, pool):
    return SimpleNamespace(
        prefix_len=8,
        cached_len=12,
        req=SimpleNamespace(rid=rid, kv_allocated_len=12, kv_committed_len=8, req_pool_idx=pool),
        layer_slots=[torch.tensor([20 + pool, 30 + pool])],
        lookahead_inflight=True,
        _lookahead_queued=True,
        _lookahead_bonus=None,
        _last_logits=torch.zeros(1),
        _la_forward_template="template",
        _la_mrope_delta=None,
        bonus_ids=[1],
    )


def _settle_host(rollback=None, poisoned=False):
    return SimpleNamespace(
        device="cpu",
        req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(4, 32, dtype=torch.long)),
        _clear_rollback=Mock(side_effect=rollback),
        _sr_clear_stamps_after_poison=Mock(),
        _sr_device_poisoned=poisoned,
    )


class TestSettleRollback(CustomTestCase):
    def _reset(self, host, runners, sync=None):
        module = _device_module(sync)
        with patch.object(torch, "get_device_module", return_value=module), patch.object(
            forward, "_opt_draft_count"
        ):
            error = None
            try:
                forward._reset_lookahead_group(host, runners)
            except Exception as exc:
                error = exc
        return error, module

    def _records(self, host):
        bag = getattr(host, "_sr_kv_quarantine", None)
        return list(bag.records) if bag is not None else []

    def test_clean_rollback_syncs_once_and_clears(self):
        host = _settle_host()
        runners = [_settle_runner("a", 0), _settle_runner("b", 1)]
        error, module = self._reset(host, runners)
        self.assertIsNone(error)
        module.synchronize.assert_called_once()
        self.assertEqual(host._clear_rollback.call_count, 2)
        for runner in runners:
            self.assertEqual(runner.layer_slots, [])
            self.assertEqual(runner.req.kv_allocated_len, 8)
            self.assertEqual(runner.cached_len, 8)
            self.assertFalse(runner.lookahead_inflight)
        self.assertFalse(host._sr_device_poisoned)

    def test_failed_rollback_keeps_that_request_and_stops_retrying(self):
        def rollback(req, keep):
            if req.rid == "b":
                raise RuntimeError("could not roll back")

        host = _settle_host(rollback)
        a, b = _settle_runner("a", 0), _settle_runner("b", 1)
        error, _module = self._reset(host, [a, b])
        self.assertIsInstance(error, forward.LookaheadUnsettled)
        self.assertNotIsInstance(error, forward.LookaheadFallback)
        self.assertEqual(a.layer_slots, [])
        self.assertEqual(a.req.kv_allocated_len, 8)
        self.assertEqual(len(b.layer_slots), 1)
        self.assertEqual(b.req.kv_allocated_len, 12)
        self.assertTrue(host._sr_device_poisoned)
        records = self._records(host)
        self.assertEqual(len(records), 1)
        self.assertIsNotNone(records[0].release_error)
        self.assertEqual([entry[0] for entry in records[0].owner.entries], [b.req])

    def test_ordinary_sync_failure_clears_nothing(self):
        host = _settle_host()
        runners = [_settle_runner("a", 0), _settle_runner("b", 1)]
        error, _module = self._reset(host, runners, sync=RuntimeError("stream busy"))
        self.assertIsInstance(error, forward.LookaheadUnsettled)
        host._clear_rollback.assert_not_called()
        for runner in runners:
            self.assertEqual(len(runner.layer_slots), 1)
            self.assertEqual(runner.req.kv_allocated_len, 12)
        self.assertTrue(host._sr_device_poisoned)
        self.assertEqual(len(self._records(host)), 1)

    def test_poisoned_device_does_not_roll_back(self):
        host = _settle_host(poisoned=True)
        runner = _settle_runner("a", 0)
        error, module = self._reset(host, [runner])
        self.assertIsInstance(error, forward.LookaheadUnsettled)
        module.synchronize.assert_not_called()
        host._clear_rollback.assert_not_called()
        self.assertEqual(runner.req.kv_allocated_len, 12)

    def test_device_context_error_still_propagates(self):
        host = _settle_host()
        error, _module = self._reset(
            host,
            [_settle_runner("a", 0)],
            sync=RuntimeError("CUDA error: an illegal memory access was encountered"),
        )
        self.assertIsInstance(error, RuntimeError)
        self.assertNotIsInstance(error, forward.LookaheadUnsettled)
        self.assertTrue(host._sr_device_poisoned)


class TestFinishWaitFailure(CustomTestCase):
    def _finish(self, sync=None):
        host = SimpleNamespace(
            device="cpu", _sr_pause_req=Mock(), _sr_clear_stamps_after_poison=Mock()
        )
        runner = SimpleNamespace(
            lookahead_inflight=True,
            _lookahead_launch_event=SimpleNamespace(
                synchronize=Mock(side_effect=RuntimeError("event lost"))
            ),
            _lookahead_bonus=None,
        )
        req = SimpleNamespace(rid="a")
        with patch.object(torch, "get_device_module", return_value=_device_module(sync)), patch.object(
            forward, "_publish_or_rollback"
        ) as publish, patch.object(forward, "_request_finished", return_value=False):
            error = None
            try:
                forward.finish_opt_expand_req(host, req, runner, 8, None, True, None)
            except Exception as exc:
                error = exc
        return host, runner, req, publish, error

    def test_confirmed_by_sync_publishes_and_pauses(self):
        host, runner, req, publish, error = self._finish()
        self.assertIsNone(error)
        publish.assert_called_once()
        host._sr_pause_req.assert_called_once_with(req)
        self.assertFalse(runner.lookahead_inflight)
        self.assertFalse(getattr(host, "_sr_device_poisoned", False))

    def test_unknown_completion_poisons_but_still_finishes_the_request(self):
        host, runner, req, publish, error = self._finish(sync=RuntimeError("sync failed"))
        self.assertIsInstance(error, forward.LookaheadUnsettled)
        self.assertTrue(host._sr_device_poisoned)
        # Publish runs; with the device poisoned it keeps the slots.
        publish.assert_called_once()
        host._sr_pause_req.assert_called_once_with(req)
        self.assertFalse(runner.lookahead_inflight)


class TestSettleJobDrop(CustomTestCase):
    def _job(self, host, *, launch_error=None, sync=None):
        runner = _settle_runner("a", 0)
        runner.lookahead_dropped = False
        runner._lookahead_launch_event = SimpleNamespace(
            synchronize=Mock(side_effect=launch_error)
        )
        job = SimpleNamespace(
            req=runner.req,
            runner=runner,
            keep=8,
            state=SimpleNamespace(depth=1, config=SimpleNamespace(width=4)),
        )
        with patch.object(torch, "get_device_module", return_value=_device_module(sync)), patch.object(
            forward, "_request_finished", return_value=True
        ), patch.object(forward, "_opt_page_size", return_value=1), patch.object(
            forward, "_opt_draft_count"
        ):
            forward._settle_lookahead_job(host, job)
        return runner

    def test_failed_rollback_keeps_the_tree(self):
        host = _settle_host(RuntimeError("could not roll back"))
        runner = self._job(host)
        self.assertEqual(len(runner.layer_slots), 1)
        self.assertEqual(runner.req.kv_allocated_len, 12)
        self.assertTrue(host._sr_device_poisoned)
        self.assertEqual(len(host._sr_kv_quarantine.records), 1)

    def test_unknown_completion_skips_the_rollback(self):
        host = _settle_host()
        runner = self._job(
            host, launch_error=RuntimeError("event lost"), sync=RuntimeError("sync failed")
        )
        host._clear_rollback.assert_not_called()
        self.assertEqual(len(runner.layer_slots), 1)
        self.assertTrue(host._sr_device_poisoned)

    def test_confirmed_drop_rolls_back(self):
        host = _settle_host()
        runner = self._job(host)
        host._clear_rollback.assert_called_once()
        self.assertEqual(runner.layer_slots, [])
        self.assertEqual(runner.req.kv_allocated_len, 8)
        self.assertFalse(getattr(host, "_sr_device_poisoned", False))


class TestRoundLearningSnapshot(CustomTestCase):
    def setUp(self):
        from sglang.srt.speculative.clear_tree_remote import clear_tree_policy as policy

        self.policy = policy
        saved = (policy._gain_pred, policy._gain_actual)

        def restore():
            policy._gain_pred, policy._gain_actual = saved

        self.addCleanup(restore)

    def test_abandoned_attempt_is_undone(self):
        policy = self.policy
        policy._gain_pred, policy._gain_actual = 2.0, 1.0
        plan = SimpleNamespace(
            _raw_gain={1: 0.5},
            _frozen_k=None,
            _cost_depth=None,
            _verify_quotes=None,
            _score_cache={},
            _peers=[],
            _keep_ids=set(),
            _at=None,
        )
        snapshot = policy.snapshot_round_learning(plan)
        # What a failed attempt would leave behind.
        policy._observe_gain(4.0, 3.0, 0.5)
        plan._raw_gain[99] = 7.0
        plan._raw_gain[1] = 0.9
        plan._frozen_k = 16
        plan._cost_depth = 3
        plan._score_cache[99] = ("state", 2, ())
        plan._peers = ["state"]
        plan._keep_ids = {99}
        plan._at = 3
        policy.restore_round_learning(plan, snapshot)
        self.assertEqual((policy._gain_pred, policy._gain_actual), (2.0, 1.0))
        self.assertEqual(plan._raw_gain, {1: 0.5})
        self.assertIsNone(plan._frozen_k)
        self.assertIsNone(plan._cost_depth)
        self.assertEqual(plan._score_cache, {})
        self.assertEqual(plan._peers, [])
        self.assertEqual(plan._keep_ids, set())
        self.assertIsNone(plan._at)

    def test_snapshot_does_not_alias_the_plan(self):
        policy = self.policy
        plan = SimpleNamespace(_raw_gain={1: 0.5}, _frozen_k=None, _cost_depth=None, _verify_quotes=None)
        snapshot = policy.snapshot_round_learning(plan)
        plan._raw_gain[2] = 1.0
        policy.restore_round_learning(plan, snapshot)
        self.assertEqual(plan._raw_gain, {1: 0.5})
        self.assertEqual(policy.snapshot_round_learning(None)["plan"], None)


class TestLookaheadMode(CustomTestCase):
    def test_cached_per_scheduler(self):
        host = SimpleNamespace()
        with patch.object(forward, "lookahead_supported", return_value=True) as supported:
            self.assertTrue(forward.lookahead_mode(host))
            self.assertTrue(forward.lookahead_mode(host))
        supported.assert_called_once()
        with patch.object(forward, "lookahead_supported", side_effect=RuntimeError("no")):
            self.assertFalse(forward.lookahead_mode(SimpleNamespace()))


def _mixin_method(name, namespace):
    """Run one ``ClearTreeSchedulerMixin`` method from source.

    Importing the scheduler pulls GPU dependencies into CPU checks.
    """
    import ast
    from pathlib import Path

    path = Path(forward.__file__).with_name("clear_tree_scheduler.py")
    mixin = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "ClearTreeSchedulerMixin"
    )
    method = next(
        node for node in mixin.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class TestPoisonGuards(CustomTestCase):
    def _windows(self, error, poisoned=False):
        from sglang.srt.speculative.clear_tree_remote import clear_tree_ledger as ledger
        from sglang.srt.speculative.clear_tree_remote import clear_tree_policy as policy
        from sglang.srt.speculative.standalone_remote import sr_round_metrics as srm
        from sglang.srt.speculative.standalone_remote.sr_protocol import SRAction

        metrics = SimpleNamespace(counts=Counter())
        host = SimpleNamespace(
            _sr_replay_grammars=Mock(),
            _clear_expand_one=Mock(return_value=([9], None, None)),
            _sr_device_poisoned=poisoned,
        )
        pairs = [(None, SimpleNamespace(rid="a")), (None, SimpleNamespace(rid="b"))]
        with patch.object(forward, "materialize_missing_prefix"), patch.object(
            ledger, "add_draft_prefill"
        ), patch.object(srm, "get_sr_round_metrics", return_value=metrics), patch.object(
            policy, "install_fallback_plan"
        ) as fallback_plan, patch.object(forward, "grow_batch", side_effect=error):
            windows = forward.produce_opt_windows(host, SRAction.STEP, pairs)
        return windows, host, metrics, fallback_plan

    def test_unsettled_batch_sends_empty_drafts_without_a_rebuild(self):
        windows, host, metrics, fallback_plan = self._windows(
            forward.LookaheadUnsettled("rollback failed")
        )
        self.assertEqual(windows, [([], None, None)] * 2)
        host._clear_expand_one.assert_not_called()
        fallback_plan.assert_not_called()
        self.assertEqual(metrics.counts["opt_batch_unsettled"], 1)

    def test_poisoned_device_also_skips_the_rebuild(self):
        windows, host, _metrics, _plan = self._windows(RuntimeError("publish"), poisoned=True)
        self.assertEqual(windows, [([], None, None)] * 2)
        host._clear_expand_one.assert_not_called()

    def test_ordinary_batch_failure_still_rebuilds(self):
        windows, host, metrics, fallback_plan = self._windows(RuntimeError("ordinary"))
        self.assertEqual(windows, [([9], None, None)] * 2)
        self.assertEqual(host._clear_expand_one.call_count, 2)
        fallback_plan.assert_called_once()
        self.assertEqual(metrics.counts["opt_batch_fallbacks"], 1)

    def test_rollback_is_refused_on_a_poisoned_device(self):
        rollback = _mixin_method("_clear_rollback", {})
        kv = Mock()
        kv.rollback.return_value = True
        req = SimpleNamespace(rid="a", kv_allocated_len=12)
        with self.assertRaises(RuntimeError):
            rollback(SimpleNamespace(sr_kv=kv, _sr_device_poisoned=True), req, 8)
        kv.rollback.assert_not_called()
        rollback(SimpleNamespace(sr_kv=kv, _sr_device_poisoned=False), req, 8)
        kv.rollback.assert_called_once_with(req, 8, 12)
        # Nothing past keep: no refusal even when poisoned.
        rollback(SimpleNamespace(sr_kv=kv, _sr_device_poisoned=True), req, 12)

    def test_expand_one_returns_empty_on_a_poisoned_device(self):
        metrics = SimpleNamespace(counts=Counter())
        namespace = dict(vars(forward), get_sr_round_metrics=lambda *_a: metrics)
        expand = _mixin_method("_clear_expand_one", namespace)
        host = SimpleNamespace(_sr_device_poisoned=True, _sr_resume_req=Mock())
        req = SimpleNamespace(rid="a", opt_prefill_tail_logits=torch.zeros(4))
        self.assertEqual(expand(host, req), ([], None, None))
        host._sr_resume_req.assert_not_called()
        self.assertIsNone(req.opt_prefill_tail_logits)
        self.assertEqual(metrics.counts["opt_expand_skipped_poisoned"], 1)

    def test_poisoned_batch_skips_grow_batch(self):
        from sglang.srt.speculative.clear_tree_remote import clear_tree_ledger as ledger
        from sglang.srt.speculative.standalone_remote import sr_round_metrics as srm
        from sglang.srt.speculative.standalone_remote.sr_protocol import SRAction

        metrics = SimpleNamespace(counts=Counter())
        host = SimpleNamespace(
            _sr_replay_grammars=Mock(), _clear_expand_one=Mock(), _sr_device_poisoned=True
        )
        reqs = [
            SimpleNamespace(rid=rid, opt_prefill_tail_logits=torch.zeros(4)) for rid in "ab"
        ]
        with patch.object(forward, "materialize_missing_prefix"), patch.object(
            ledger, "add_draft_prefill"
        ), patch.object(srm, "get_sr_round_metrics", return_value=metrics), patch.object(
            forward, "grow_batch"
        ) as grow:
            windows = forward.produce_opt_windows(
                host, SRAction.STEP, [(None, req) for req in reqs]
            )
        self.assertEqual(windows, [([], None, None)] * 2)
        grow.assert_not_called()
        host._clear_expand_one.assert_not_called()
        self.assertEqual(metrics.counts["opt_expand_skipped_poisoned"], 2)
        self.assertTrue(all(req.opt_prefill_tail_logits is None for req in reqs))

    def test_failed_ingest_keeps_its_error_on_a_poisoned_device(self):
        req = SimpleNamespace(rid="a")
        poisoned = SimpleNamespace(_sr_device_poisoned=True, _clear_rollback=Mock())
        forward._rollback_failed_ingest(poisoned, req, 8)
        poisoned._clear_rollback.assert_not_called()
        healthy = SimpleNamespace(_sr_device_poisoned=False, _clear_rollback=Mock())
        forward._rollback_failed_ingest(healthy, req, 8)
        healthy._clear_rollback.assert_called_once_with(req, 8)

    def test_unfinished_layer_on_a_poisoned_device_is_kept_quietly(self):
        runner = _settle_runner("a", 0)
        runner.layer_slots = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        job = SimpleNamespace(
            req=runner.req,
            runner=runner,
            keep=8,
            state=SimpleNamespace(depth=2, config=SimpleNamespace(width=4)),
        )
        host = _settle_host(poisoned=True)
        with patch.object(forward, "_opt_draft_count") as count:
            forward._drop_unfinished_layer(host, job)
        host._clear_rollback.assert_not_called()
        self.assertEqual(len(runner.layer_slots), 2)
        self.assertIsNone(getattr(host, "_sr_kv_quarantine", None))
        count.assert_not_called()

    def test_unfinished_layer_rollback_failure_is_quarantined(self):
        runner = _settle_runner("a", 0)
        runner.layer_slots = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        runner.req.kv_allocated_len = 16
        job = SimpleNamespace(
            req=runner.req,
            runner=runner,
            keep=8,
            state=SimpleNamespace(depth=2, config=SimpleNamespace(width=4)),
        )
        host = _settle_host(RuntimeError("could not roll back"))
        with patch.object(forward, "_opt_draft_count") as count:
            forward._drop_unfinished_layer(host, job)
        self.assertEqual(len(runner.layer_slots), 2)
        self.assertEqual(runner.req.kv_allocated_len, 16)
        self.assertTrue(host._sr_device_poisoned)
        self.assertEqual(len(host._sr_kv_quarantine.records), 1)
        self.assertIn("opt_deferred_unsettled", [c.args[1] for c in count.call_args_list])

    def test_lookahead_mode_ignores_a_non_bool_attribute(self):
        with patch.object(forward, "lookahead_supported", return_value=False):
            self.assertFalse(forward.lookahead_mode(Mock()))


class TestRetireBeforePublish(CustomTestCase):
    def test_failing_expand_waits_for_the_inflight_layer(self):
        order = []
        launch = SimpleNamespace(synchronize=lambda: order.append("layer"))
        bonus = SimpleNamespace(event=SimpleNamespace(synchronize=lambda: order.append("bonus")), ready=True)
        runner = SimpleNamespace(
            lookahead_inflight=True, _lookahead_launch_event=launch, _lookahead_bonus=bonus
        )
        job = SimpleNamespace(req=object(), runner=runner, keep=3, window=([], [], []))
        host = SimpleNamespace(_sr_is_finished=lambda _req: True)
        with patch.object(
            forward, "_publish_or_rollback", side_effect=lambda *a: order.append("publish")
        ):
            err = forward.commit_deferred_expands(host, [job], keep=False)
        self.assertIsNone(err)
        self.assertEqual(order, ["layer", "bonus", "publish"])
        self.assertFalse(runner.lookahead_inflight)
        self.assertTrue(runner.lookahead_dropped)
        self.assertFalse(bonus.ready)


class TestGrowBatchLookahead(CustomTestCase):
    def _req(self, rid):
        return SimpleNamespace(rid=rid, origin_input_ids=[1, 2, 3], output_ids=[4], kv_committed_len=4)

    def _run(self, *, admit_error=None, grow_error=False, catch=False):
        reqs = [self._req("a"), self._req("b")]
        tails = [torch.randn(VOCAB), torch.randn(VOCAB)]
        private = [torch.randn(VOCAB), torch.randn(VOCAB)]
        windows = [([1], [-1], [0]), ([2], [-1], [0])]
        metrics = SimpleNamespace(active=False, counts=Counter(), add_host=lambda *a: None)
        host = SimpleNamespace(
            server_args=SimpleNamespace(clear_tree_tau=0.5, clear_tree_max_depth=8),
            _sr_resume_req=Mock(),
            _sr_isolate_need=Mock(),
            _clear_release_uncommitted=Mock(),
            device="cpu",
        )
        plan = SimpleNamespace(width=2)

        def grow_many(scheduler, runners, config, tails_in, states_out):
            for runner, tail in zip(runners, private):
                runner._lookahead_tail = tail
            if isinstance(grow_error, BaseException):
                raise grow_error
            if grow_error:
                raise forward.LookaheadFallback("mid-tree")
            return windows, 3

        import sglang.srt.speculative.clear_tree_remote.clear_tree_ledger as ledger
        import sglang.srt.speculative.clear_tree_remote.clear_tree_lease as lease
        import sglang.srt.speculative.clear_tree_remote.clear_tree_policy as policy
        import sglang.srt.speculative.clear_tree_remote.clear_tree_replay as replay
        import sglang.srt.speculative.standalone_remote.sr_round_metrics as srm

        with ExitStack() as stack:
            p = lambda *a, **k: stack.enter_context(patch.object(*a, **k))
            p(srm, "get_sr_round_metrics", return_value=metrics)
            p(lease, "consume_clear_tree_lease")
            p(policy, "attach_round", return_value=plan)
            p(policy, "remember_choice")
            p(replay, "queue_hold_replay")
            p(ledger, "mark_stop")
            fallback_mark = p(ledger, "mark_round_fallback")
            p(forward, "ingest_tail_logits", return_value=tails)
            p(forward, "flush_opt_ingest_phases")
            p(forward, "lookahead_supported", return_value=True)
            admit = p(forward, "_admit_lookahead_many", side_effect=admit_error)
            many = p(forward, "grow_lookahead_many", side_effect=grow_many)
            plain = p(forward, "grow_active_layers", return_value=windows)
            drain = p(forward, "drain_opt_layer_gpu_us")
            p(forward, "clear_prefill_tail_logits")
            p(forward, "finish_opt_expand_req", return_value=None)
            p(forward, "commit_deferred_expands", return_value=None)
            restore = p(policy, "restore_round_learning")
            got = None
            error = None
            try:
                got = forward.grow_batch(host, reqs)
            except Exception as exc:
                error = exc
            if error is not None and not catch:
                raise error
        return dict(
            got=got, windows=windows, host=host, tails=tails, private=private,
            admit=admit, many=many, plain=plain, drain=drain, metrics=metrics,
            fallback_mark=fallback_mark, restore=restore, error=error,
        )

    def test_success_defers_the_drain(self):
        r = self._run()
        self.assertEqual(r["got"], r["windows"])
        r["plain"].assert_not_called()
        r["drain"].assert_not_called()
        self.assertTrue(r["host"]._opt_defer_layer_drain)
        self.assertFalse(r["host"]._opt_lookahead_active)
        self.assertEqual(r["metrics"].counts["opt_layer_forwards_critical"], 2)
        r["fallback_mark"].assert_not_called()

    def test_admission_fallback_grows_from_ingest_tails(self):
        r = self._run(admit_error=forward.LookaheadFallback("kv evict"))
        r["many"].assert_not_called()
        seed = r["plain"].call_args.args[1]
        self.assertTrue(all(a is b for a, b in zip(seed, r["tails"])))
        r["drain"].assert_called_once()
        self.assertFalse(r["host"]._opt_defer_layer_drain)
        self.assertFalse(r["host"]._opt_lookahead_active)
        r["fallback_mark"].assert_not_called()

    def test_mid_tree_fallback_uses_private_tails_and_marks_the_round(self):
        r = self._run(grow_error=True)
        seed = r["plain"].call_args.args[1]
        self.assertTrue(all(a is b for a, b in zip(seed, r["private"])))
        r["fallback_mark"].assert_called_once()
        r["drain"].assert_called_once()
        self.assertFalse(r["host"]._opt_defer_layer_drain)
        self.assertFalse(r["host"]._opt_lookahead_active)
        r["restore"].assert_called_once()
        self.assertTrue(r["host"]._clear_round_retried)

    def test_unsettled_attempt_fails_the_round_without_a_rebuild(self):
        r = self._run(grow_error=forward.LookaheadUnsettled("rollback failed"), catch=True)
        self.assertIsInstance(r["error"], forward.LookaheadUnsettled)
        r["plain"].assert_not_called()
        r["fallback_mark"].assert_called_once()
        r["restore"].assert_called_once()
        self.assertTrue(r["host"]._clear_round_retried)

    def test_success_keeps_the_learning(self):
        r = self._run()
        r["restore"].assert_not_called()
        self.assertFalse(getattr(r["host"], "_clear_round_retried", False))


class TestSingleLookaheadCallsBeginLayer(CustomTestCase):
    def test_begin_layer_runs_before_the_stop_check(self):
        config = ClearTreeConfig(width=2, tau=-1.0, max_depth=4)
        plan = _Plan({"r": 1}, 2, defer=False)
        host = _scheduler(plan, ["r"], [5])
        runner = SimpleNamespace(
            bonus_ids=[], _last_logits=torch.zeros(2, 4), launch_layer=Mock()
        )
        slot = Mock()
        slot.clone_cpu.return_value = (
            torch.tensor([0.6, 0.4]),
            torch.tensor([20, 30]),
            torch.zeros(2, dtype=torch.long),
            torch.tensor([[20, 30, 40], [0, 0, 0]]),
            "seed",
        )
        with patch.object(forward, "_opt_draft_count"):
            forward.grow_lookahead(host, runner, config, torch.zeros(4), (object(), (slot, slot), Mock()))
        self.assertEqual(plan.begin_calls, [[("r", 1)]])


class _RowPlan(_Plan):
    """``_Plan`` plus the per-tree ``max_depth`` stop the real plan applies."""

    def should_grow(self, state):
        return super().should_grow(state) and int(state.depth) < int(
            state.config.max_depth
        ) - 1


def _row_host(plan, rids, prefixes, row):
    host = _scheduler(plan, rids, prefixes)
    host.req_to_token_pool = SimpleNamespace(
        device="cpu", req_to_token=torch.zeros(len(rids), row, dtype=torch.long)
    )
    return host


class TestRowCapacity(CustomTestCase):
    """Tree layers stay inside each request's ``req_to_token`` row."""

    N = 2
    ROW = 16

    def _tails(self, count):
        return [
            torch.randn(VOCAB, generator=torch.Generator().manual_seed(i))
            for i in range(count)
        ]

    def test_row_max_depths(self):
        config = ClearTreeConfig(width=4, tau=0.0, max_depth=8)
        host = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(4, 20))
        )
        runners = _runners([2, 10, 17, 20])
        with patch.object(forward, "_opt_draft_count") as count:
            caps = forward.row_max_depths(host, runners, config)
            # Room for 4, 2, 0, 0 layers: max_depth is room + 1, at least 2.
            self.assertEqual(caps, [5, 3, 2, 2])
            wide = SimpleNamespace(
                req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(1, 64))
            )
            self.assertIsNone(forward.row_max_depths(wide, _runners([0]), config))
            # No pool: nothing is read from the runner.
            self.assertIsNone(
                forward.row_max_depths(SimpleNamespace(), [object()], config)
            )
        # Asking again after a lookahead fallback must not count twice.
        count.assert_not_called()
        self.assertEqual(forward.row_layer_cap(host, 9, 4), 2)
        self.assertIsNone(forward.row_layer_cap(SimpleNamespace(), 9, 4))

    def test_lookahead_stops_the_long_prefix_tree_inside_its_row(self):
        n, row = self.N, self.ROW
        rids, prefixes = ["a", "b"], [10, 2]
        config = ClearTreeConfig(width=n, tau=0.0, max_depth=6)
        plan = _RowPlan({"a": 5, "b": 5}, 4)
        host = _row_host(plan, rids, prefixes, row)
        runners = _runners(prefixes)
        ends = []

        def launch(scheduler, group, tokens_dev, anc_dev, depth, width, event):
            ends.extend(r.cached_len + int(width) for r in group)
            return _fake_launch(scheduler, group, tokens_dev, anc_dev, depth, width, event)

        states = []
        with ExitStack() as stack:
            _quiet_patches(stack)
            stack.enter_context(
                patch.object(
                    forward, "_lookahead_group_cache", return_value=_cpu_cache(2, n, True)
                )
            )
            stack.enter_context(
                patch.object(forward, "_launch_layer_many", side_effect=launch)
            )
            windows, launched = forward.grow_lookahead_many(
                host, runners, config, self._tails(2), states
            )
        # a: (16 - 10) // 2 = 3 layers fit; b: 7 layers fit, so no cap.
        self.assertEqual([int(s.depth) for s in states], [3, 5])
        self.assertEqual([s.config.max_depth for s in states], [4, 6])
        self.assertEqual(launched, 5)
        self.assertLessEqual(max(ends), row)
        self.assertEqual(runners[0].cached_len, 10 + 3 * n)
        self.assertTrue(all(window is not None for window in windows))

    def test_ordinary_path_stops_the_same_tree_and_its_final_layer_fits(self):
        n, row = self.N, self.ROW
        rids, prefixes = ["a", "b"], [10, 2]
        config = ClearTreeConfig(width=n, tau=0.0, max_depth=6)
        plan = _RowPlan({"a": 5, "b": 5}, 4)
        host = _row_host(plan, rids, prefixes, row)
        ends = []

        def forward_active(steps, indices):
            out = []
            for step, index in zip(steps, indices):
                depth = int(step.positions.reshape(-1)[0])
                ends.append(prefixes[index] + depth * n)
                out.append(_layer_logits(step.tokens, depth))
            return out

        states = []
        with ExitStack() as stack:
            _quiet_patches(stack)
            caps = forward.row_max_depths(host, _runners(prefixes), config)
            windows = forward.grow_active_layers(
                config,
                self._tails(2),
                forward_active,
                host,
                states_out=states,
                depth_caps=caps,
            )
        self.assertEqual(caps, [4, None])
        self.assertEqual([int(s.depth) for s in states], [3, 5])
        self.assertLessEqual(max(ends), row)
        # The deferred post-reply forward of each last layer also fits.
        for prefix, state in zip(prefixes, states):
            self.assertLessEqual(prefix + int(state.depth) * n, row)
        self.assertTrue(all(window is not None for window in windows))

    def test_admission_leaves_lookahead_when_layer_one_does_not_fit(self):
        host = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(2, 16))
        )
        runners = _runners([10, 15])
        for rid, runner in zip(("a", "b"), runners):
            runner.req = SimpleNamespace(rid=rid)
        with patch.object(forward, "_opt_draft_count") as count, patch.object(
            forward,
            "_lookahead_group_template",
            side_effect=AssertionError("must not build a template"),
        ), patch.object(
            forward,
            "_lookahead_reserve_kv",
            side_effect=AssertionError("must not evict"),
        ):
            with self.assertRaises(forward.LookaheadFallback) as ctx:
                forward._admit_lookahead_many(host, runners, 2)
            self.assertIn("row short rid=b", str(ctx.exception))
            count.assert_called_once_with(host, "opt_lookahead_row_short")
            # 14 + 2 == 16 still fits.
            forward._admit_lookahead_rows(host, _runners([10, 14]), 2)

    def test_layer_allocators_stop_before_a_short_row(self):
        import sglang.srt.mem_cache.common as mem_common

        runner = SimpleNamespace(
            scheduler=SimpleNamespace(
                req_to_token_pool=SimpleNamespace(
                    device="cpu", req_to_token=torch.zeros(2, 12, dtype=torch.long)
                ),
                tree_cache=SimpleNamespace(page_size=1),
            ),
            cached_len=11,
            req=SimpleNamespace(rid="r", req_pool_idx=0, kv_allocated_len=11),
            layer_slots=[],
            _la_seq=torch.zeros(1, dtype=torch.long),
            _la_pool=torch.tensor([0]),
        )
        with patch.object(forward, "_opt_draft_count") as count, patch.object(
            mem_common, "alloc_token_slots"
        ) as alloc:
            with self.assertRaises(forward.RowOverflow):
                forward.OptLayerForward._alloc_layer_slots(runner, 2)
            # Ingest spans pass ``start``.
            with self.assertRaises(forward.RowOverflow):
                forward.OptLayerForward._alloc_layer_slots(runner, 4, start=10)
            with self.assertRaises(forward.RowOverflow):
                forward.OptLayerForward._alloc_lookahead_slots(runner, 2)
        alloc.assert_not_called()
        self.assertEqual(count.call_count, 3)
        self.assertEqual(runner.req.kv_allocated_len, 11)
        self.assertEqual(runner.layer_slots, [])

    def test_require_row_room_checks_every_row_first(self):
        host = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(2, 16))
        )
        runners = _runners([4, 15])
        for rid, runner in zip(("a", "b"), runners):
            runner.req = SimpleNamespace(rid=rid)
        with patch.object(forward, "_opt_draft_count"):
            with self.assertRaises(forward.RowOverflow) as ctx:
                forward._require_row_room(host, runners, 2)
            forward._require_row_room(host, runners[:1], 2)
            forward._require_row_room(SimpleNamespace(), runners, 2)
        self.assertIn("[15, 17)", str(ctx.exception))

    def test_requests_past_the_row_get_empty_drafts(self):
        host = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(2, 8))
        )
        long_req = SimpleNamespace(rid="long", origin_input_ids=list(range(7)), output_ids=[1, 2])
        short_req = SimpleNamespace(rid="short", origin_input_ids=[1, 2, 3], output_ids=[4])
        pairs = [("d0", long_req), ("d1", short_req)]
        from sglang.srt.speculative.standalone_remote.sr_protocol import SRAction

        step = SRAction.STEP
        with patch.object(forward, "_opt_draft_count") as count, patch.object(
            forward, "_produce_fitting_windows", return_value=["window"]
        ) as fitting, patch.object(forward, "_drop_row_short") as drop:
            got = forward.produce_opt_windows(host, step, pairs)
            # A PREFILL frees the same way but must not touch the ledger round.
            forward.produce_opt_windows(host, SRAction.PREFILL, pairs)
        self.assertEqual(got, [([], None, None), "window"])
        self.assertEqual(
            fitting.call_args_list[0], ((host, step, [("d1", short_req)]),)
        )
        self.assertEqual(count.call_args_list[0], ((host, "opt_row_short_empty", 1),))
        self.assertEqual(
            drop.call_args_list,
            [
                ((host, [long_req], 8), {"step": True}),
                ((host, [long_req], 8), {"step": False}),
            ],
        )
        with patch.object(
            forward, "_produce_fitting_windows", return_value=["a", "b"]
        ) as fitting:
            wide = SimpleNamespace(
                req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(2, 64))
            )
            self.assertEqual(forward.produce_opt_windows(wide, step, pairs), ["a", "b"])
            fitting.assert_called_once_with(wide, step, pairs)

    def test_missing_fitting_windows_become_empty_drafts(self):
        from sglang.srt.speculative.standalone_remote.sr_protocol import SRAction

        host = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(3, 8))
        )
        long_req = SimpleNamespace(rid="long", origin_input_ids=list(range(9)), output_ids=[])
        fit_a = SimpleNamespace(rid="a", origin_input_ids=[1, 2], output_ids=[])
        fit_b = SimpleNamespace(rid="b", origin_input_ids=[1, 2, 3], output_ids=[])
        pairs = [("d0", fit_a), ("d1", long_req), ("d2", fit_b)]
        with patch.object(forward, "_opt_draft_count"), patch.object(
            forward, "_drop_row_short"
        ), patch.object(
            forward, "_produce_fitting_windows", return_value=["wa"]
        ), self.assertLogs(forward.logger, level="WARNING") as logs:
            got = forward.produce_opt_windows(host, SRAction.STEP, pairs)
        # Only one window for two fitting requests: no StopIteration, and the
        # missing one gets an empty draft like the caller's zip used to give.
        self.assertEqual(got, ["wa", ([], None, None), ([], None, None)])
        self.assertTrue(any("got no window" in line for line in logs.output))


class TestRowShortCleanup(CustomTestCase):
    def test_short_request_frees_its_lease_and_warns_once(self):
        from sglang.srt.speculative.clear_tree_remote import (
            clear_tree_lease as lease,
            clear_tree_ledger as ledger,
        )

        host = SimpleNamespace(_opt_la_batches={"long": {}, "other": {}})
        req = SimpleNamespace(
            rid="long",
            origin_input_ids=list(range(7)),
            output_ids=[1, 2],
            opt_prefill_tail_logits=torch.zeros(1),
        )
        with patch.object(
            lease, "opt_kv_release_allowed", return_value=True
        ), patch.object(lease, "release_clear_tree_lease") as free_lease, patch.object(
            lease, "release_opt_bonus_guesses"
        ) as free_bonus, patch.object(
            ledger, "mark_round_row_short"
        ) as mark, patch.object(
            forward, "_opt_draft_count"
        ) as count, self.assertLogs(forward.logger, level="WARNING") as logs:
            # PREFILL first: it frees and warns but leaves the ledger alone.
            forward._drop_row_short(host, [req], 8, step=False)
            mark.assert_not_called()
            count.assert_not_called()
            forward._drop_row_short(host, [req], 8, step=True)
            forward._drop_row_short(host, [req], 8, step=True)
        self.assertEqual(free_lease.call_args_list, [((host, req),)] * 3)
        self.assertEqual(free_bonus.call_args_list, [((host, req),)] * 3)
        self.assertEqual(mark.call_args_list, [((host, ["long"]),)] * 2)
        self.assertEqual(count.call_args_list, [((host, "opt_row_short_rounds"),)] * 2)
        self.assertEqual(len(logs.output), 1)
        self.assertIsNone(req.opt_prefill_tail_logits)
        # Its lookahead decode batch is gone; other requests keep theirs.
        self.assertEqual(set(host._opt_la_batches), {"other"})

    def test_release_waits_for_a_clean_device(self):
        from sglang.srt.speculative.clear_tree_remote import (
            clear_tree_lease as lease,
            clear_tree_ledger as ledger,
        )

        req = SimpleNamespace(rid="long", origin_input_ids=[1] * 9, output_ids=[])
        with patch.object(
            lease, "opt_kv_release_allowed", return_value=False
        ), patch.object(lease, "release_clear_tree_lease") as free_lease, patch.object(
            lease, "release_opt_bonus_guesses"
        ) as free_bonus, patch.object(ledger, "mark_round_row_short"), patch.object(
            forward, "_opt_draft_count"
        ):
            host = SimpleNamespace(_opt_la_batches={"long": {}})
            forward._drop_row_short(host, [req], 8, step=True)
        free_lease.assert_not_called()
        free_bonus.assert_not_called()
        # The decode batch is host state, not KV: it goes even then.
        self.assertEqual(host._opt_la_batches, {})


class TestRowCappedCounting(CustomTestCase):
    def test_capped_trees_are_stamped_and_counted_once(self):
        config = ClearTreeConfig(width=2, tau=0.0, max_depth=6)
        states, _plan = forward._round_states(None, config, 3, [4, None, 6])
        self.assertEqual([s.config.max_depth for s in states], [4, 6, 6])
        self.assertEqual(
            [getattr(s, "row_capped", False) for s in states], [True, False, False]
        )
        host = SimpleNamespace()
        with patch.object(forward, "_opt_draft_count") as count:
            forward.count_row_capped(host, states)
            forward.count_row_capped(host, states[1:])
        count.assert_called_once_with(host, "opt_row_depth_capped", 1)

    def test_partial_batch_counts_only_the_trees_it_keeps(self):
        # Two capped trees; the second layer forward failed, so request 1 is
        # rebuilt by ``_clear_expand_one``, which counts its own tree.
        config = ClearTreeConfig(width=2, tau=0.0, max_depth=6)
        states, _plan = forward._round_states(None, config, 2, [3, 4])
        rebuilt, _plan = forward._round_states(None, config, 1, [4])
        host = SimpleNamespace()
        with patch.object(forward, "_opt_draft_count") as count:
            forward.count_row_capped(host, states, [("window",), None])
            forward.count_row_capped(host, rebuilt)
            forward.count_row_capped(host, states, [None, None])
        self.assertEqual(
            count.call_args_list,
            [((host, "opt_row_depth_capped", 1),)] * 2,
        )

    def test_row_capped_round_is_not_a_width_pair_sample(self):
        from sglang.srt.speculative.clear_tree_remote import (
            clear_tree_replay as replay,
        )

        config = ClearTreeConfig(width=2, tau=0.0, max_depth=6)
        r = TestGroupLookahead._run_pair(
            self, [2, 3], k=4, prefixes=[10, 11], bonus=False
        )
        plan = SimpleNamespace(
            width=2,
            batch=2,
            replay_eligible=True,
            theta_value=None,
            locked_k=4,
            _frozen_k=4,
            _time=lambda k: 1.0,
            flat_k=4,
        )
        reqs = [SimpleNamespace(rid="r0"), SimpleNamespace(rid="r1")]
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=1.0,
        ):
            clean = replay.hold_replay(
                SimpleNamespace(), reqs, r["states"], plan, token=1, served=2
            )
            forward._apply_depth_cap(r["states"][0], config, 3)
            capped = replay.hold_replay(
                SimpleNamespace(), reqs, r["states"], plan, token=1, served=2
            )
        self.assertTrue(clean["eligible"])
        self.assertFalse(clean["row_capped"])
        self.assertFalse(capped["eligible"])
        self.assertTrue(capped["row_capped"])


class TestReplayRowsMatchLaunches(CustomTestCase):
    """The width replay prices each layer at the trees really launched."""

    def _rows(self, states):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_replay import (
            _freeze_layer_rows,
        )

        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=1.0,
        ):
            return _freeze_layer_rows(SimpleNamespace(), states)

    def test_lookahead_rows_follow_the_launch_sizes(self):
        stops = [1, 3, 2]
        r = TestGroupLookahead._run_pair(
            self, stops, k=4, prefixes=[10, 11, 12], bonus=False
        )
        launches = [len(call.args[1]) for call in r["launch"].call_args_list]
        # Layer d is launched for every tree with depth >= d.
        self.assertEqual(launches, [3, 2, 1])
        forward.note_forwarded_layers(r["states"], r["runners"], 2)
        self.assertEqual([s.forwarded_layers for s in r["states"]], stops)
        rows = self._rows(r["states"])
        # The deepest tree's last launch is off the critical path.
        self.assertEqual([row["active"] for row in rows], launches[:-1])
        self.assertEqual([row["sum_ell"] for row in rows], [33, 23])
        # Layer-by-layer forwards only trees that keep growing.
        ordinary = self._rows(r["ref_states"])
        self.assertEqual([row["active"] for row in ordinary], [2, 1])
        self.assertEqual([row["sum_ell"] for row in ordinary], [23, 11])

    def test_ordinary_stamps_keep_the_old_rows(self):
        stops = [1, 3, 2]
        r = TestGroupLookahead._run_pair(
            self, stops, k=4, prefixes=[10, 11, 12], bonus=False
        )
        runners = [
            SimpleNamespace(prefix_len=p, cached_len=p + (d - 1) * 2)
            for p, d in zip([10, 11, 12], stops)
        ]
        forward.note_forwarded_layers(r["ref_states"], runners, 2)
        self.assertEqual([s.forwarded_layers for s in r["ref_states"]], [0, 2, 1])
        rows = self._rows(r["ref_states"])
        self.assertEqual([row["active"] for row in rows], [2, 1])


class TestDraftHeaderLookahead(CustomTestCase):
    def _scheduler(self):
        return SimpleNamespace(
            server_args=SimpleNamespace(
                clear_tree_widths=None,
                clear_tree_max_depth=8,
                model_path="m",
                page_size=1,
                attention_backend="triton",
                disable_cuda_graph=False,
            ),
            tp_size=1,
            model_worker=None,
            _clear_layer_capture_bs=(1, 2, 4),
        )

    def test_key_only_when_lookahead_can_run(self):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_fit import draft_header

        with patch.object(forward, "lookahead_supported", return_value=True):
            self.assertIs(draft_header(self._scheduler())["draft"].get("lookahead"), True)
        with patch.object(forward, "lookahead_supported", return_value=False):
            self.assertNotIn("lookahead", draft_header(self._scheduler())["draft"])

    def test_switch_on_without_support_leaves_the_header_alone(self):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_fit import draft_header

        # No CUDA device on this scheduler, so lookahead cannot run.
        with patch.dict(os.environ, {"SGLANG_CLEAR_DRAFT_LOOKAHEAD": "1"}):
            self.assertNotIn("lookahead", draft_header(self._scheduler())["draft"])


if __name__ == "__main__":
    unittest.main()
