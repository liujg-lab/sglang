"""CPU checks that TOLL_TREE_REMOTE stays a separate, unimplemented algorithm."""

import argparse
import unittest
import uuid
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.speculative.opt_tree_remote.opt_tree_layout import build_tree_from_parents
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.toll_tree_remote.config import TollConfig
from sglang.srt.speculative.toll_tree_remote.protocol import (
    PROTO_VERSION,
    TollAction,
    TollReply,
    TollReplyItem,
    TollReqItem,
    TollRequest,
    TollStatus,
    decode_reply,
    encode_reply,
    encode_request,
    request_ok,
    spec_shape_of,
)
from sglang.srt.speculative.toll_tree_remote.transport import TollDraftServer, TollTargetClient
from sglang.srt.speculative.toll_tree_remote.tree.layout import build_verify_layout
from sglang.srt.speculative.toll_tree_remote.tree.mask import layer_mask
from sglang.srt.speculative.toll_tree_remote.tree.state import TollTree
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-test-cpu")

try:
    from sglang.srt.server_args import ServerArgs
except ImportError:
    ServerArgs = None

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase


class TestTollTreeRemoteRegistration(CustomTestCase):
    def test_enum_is_not_opt_or_standalone_remote(self):
        algo = SpeculativeAlgorithm.from_string("TOLL_TREE_REMOTE")
        self.assertIs(algo, SpeculativeAlgorithm.TOLL_TREE_REMOTE)
        self.assertTrue(algo.is_toll_tree_remote())
        self.assertFalse(algo.is_opt_tree_remote())
        self.assertFalse(algo.is_standalone_remote())
        self.assertFalse(algo.is_standalone())
        self.assertFalse(SpeculativeAlgorithm.OPT_TREE_REMOTE.is_toll_tree_remote())
        self.assertFalse(
            SpeculativeAlgorithm.STANDALONE_REMOTE.is_toll_tree_remote()
        )

    def test_create_worker_is_target_only(self):
        algo = SpeculativeAlgorithm.from_string("TOLL_TREE_REMOTE")
        with self.assertRaises(ValueError) as ctx:
            algo.create_worker(
                SimpleNamespace(disable_overlap_schedule=True, toll_tree_role="draft")
            )
        self.assertIn("TpModelWorker", str(ctx.exception))
        worker = algo.create_worker(
            SimpleNamespace(disable_overlap_schedule=True, toll_tree_role="target")
        )
        self.assertEqual(worker.__name__, "TollTreeRemoteWorker")

    @unittest.skipIf(ServerArgs is None, "sglang.srt.server_args is not importable")
    def test_cli_accepts_algorithm_name(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = parser.parse_args(
            ["--model-path", "dummy", "--speculative-algorithm", "TOLL_TREE_REMOTE"]
        )
        self.assertEqual(parsed.speculative_algorithm, "TOLL_TREE_REMOTE")

    @unittest.skipIf(ServerArgs is None, "sglang.srt.server_args is not importable")
    def test_startup_accepts_fixed_batch_one_without_graphs(self):
        args = ServerArgs(model_path="dummy")
        args.speculative_algorithm = "TOLL_TREE_REMOTE"
        args.toll_tree_role = "target"
        args.toll_tree_policy = "fixed"
        args.toll_tree_fixed_width = 8
        args.toll_tree_fixed_tau = 0.1
        args.disable_cuda_graph = True
        args.disable_piecewise_cuda_graph = True
        args.max_running_requests = 1
        args._handle_speculative_decoding()
        self.assertEqual(args.speculative_num_draft_tokens, 9)
        self.assertEqual(args.speculative_num_steps, 10)
        args.toll_tree_policy = "toll"
        with self.assertRaises(ValueError):
            args._handle_speculative_decoding()
        args.toll_tree_policy = "fixed"
        args.disable_cuda_graph = False
        with self.assertRaises(ValueError):
            args._handle_speculative_decoding()


def _request(cfg, seq=1):
    return TollRequest(
        proto=PROTO_VERSION,
        session="s",
        seq=seq,
        action=TollAction.STEP,
        spec_shape=spec_shape_of(cfg),
        t_target_ns=10,
        reqs=[
            TollReqItem(
                rid="a",
                commit_len=2,
                new_tokens=[7],
                input_ids=[1, 2],
                sampling={"temperature": 0.0},
            )
        ],
    )


class TestTollProtocol(CustomTestCase):
    def test_round_trip_and_bad_frames(self):
        cfg = TollConfig()
        request = _request(cfg)
        self.assertTrue(request_ok(request, cfg))
        reply = TollReply(
            proto=PROTO_VERSION,
            session="s",
            seq=1,
            verify_bucket=8,
            residence_ns=5,
            reqs=[
                TollReplyItem(
                    rid="a",
                    base_len=2,
                    tokens=[4, 5],
                    parents=[-1, 0],
                    status=TollStatus.OK,
                )
            ],
        )
        decoded = decode_reply(encode_reply(reply), cfg)
        self.assertEqual(decoded.reqs[0].tokens, [4, 5])
        broken = TollReply(
            proto=PROTO_VERSION,
            session="s",
            seq=1,
            verify_bucket=8,
            residence_ns=0,
            reqs=[
                TollReplyItem(
                    rid="a",
                    base_len=2,
                    tokens=[4],
                    parents=[0],
                    status=TollStatus.OK,
                )
            ],
        )
        empty = decode_reply(encode_reply(broken), cfg)
        self.assertEqual(empty.reqs[0].tokens, [])
        self.assertEqual(empty.reqs[0].status, TollStatus.EMPTY)
        foreign = _request(cfg)
        foreign.spec_shape = [[2, 4], [8], 3]
        self.assertFalse(request_ok(foreign, cfg))
        foreign.proto = 99
        self.assertFalse(request_ok(foreign, cfg))

    def test_graph_flags_are_refused(self):
        allowed = SimpleNamespace(
            disable_cuda_graph=True,
            disable_piecewise_cuda_graph=True,
            toll_tree_max_depth=6,
            toll_tree_width_buckets="2,4,8",
            toll_tree_verify_buckets="8,16",
        )
        built = TollConfig.from_server_args(allowed, "target")
        self.assertEqual(built.max_depth, 6)
        self.assertEqual(built.width_buckets, (2, 4, 8))
        with self.assertRaises(ValueError):
            TollConfig.from_server_args(
                SimpleNamespace(disable_cuda_graph=False, disable_piecewise_cuda_graph=True),
                "draft",
            )


class TestTollTransport(CustomTestCase):
    def test_inproc_request_reply_stale_and_timeout(self):
        zmq = pytest_zmq()
        if zmq is None:
            self.skipTest("pyzmq is not importable")
        cfg = TollConfig()
        url = f"inproc://toll-{uuid.uuid4().hex}"
        draft = TollDraftServer(url)
        target = TollTargetClient(url, cfg, timeout_ms=200)
        try:
            request = _request(cfg, seq=1)
            target.send_request(request)
            got = draft.recv_request(timeout_ms=1000)
            self.assertIsNotNone(got)
            self.assertEqual(got.seq, 1)
            draft.send_reply(
                TollReply(
                    proto=PROTO_VERSION,
                    session="s",
                    seq=99,
                    verify_bucket=8,
                    residence_ns=1,
                    reqs=[],
                )
            )
            draft.send_reply(
                TollReply(
                    proto=PROTO_VERSION,
                    session="s",
                    seq=1,
                    verify_bucket=8,
                    residence_ns=1,
                    reqs=[
                        TollReplyItem(
                            rid="a",
                            base_len=2,
                            tokens=[3],
                            parents=[-1],
                            status=TollStatus.OK,
                        )
                    ],
                )
            )
            reply = target.recv_reply("s", 1)
            self.assertEqual(reply.reqs[0].tokens, [3])
            self.assertGreaterEqual(target.stale_replies, 1)
            self.assertGreater(target.bytes_recv, 0)
            self.assertIsNone(target.recv_reply("s", 2))
            self.assertGreaterEqual(target.timeouts, 1)
        finally:
            target.close()
            draft.close()


def pytest_zmq():
    try:
        import zmq
    except ImportError:
        return None
    return zmq


class TestTollLayout(CustomTestCase):
    def test_full_row_matches_opt_and_padding_is_unlinked(self):
        parents = [[-1, 0, 0], [-1, 0, 1]]
        seqs = [4, 6]
        ours = build_verify_layout(parents, seqs, num_verify=4)
        opt = build_tree_from_parents(
            torch.tensor(parents, dtype=torch.long),
            torch.tensor(seqs, dtype=torch.long),
        )
        self.assertTrue(np.array_equal(ours.tree_mask, opt.tree_mask.cpu().numpy()))
        self.assertTrue(np.array_equal(ours.positions, opt.positions.cpu().numpy()))
        self.assertTrue(np.array_equal(ours.retrive_next_token, opt.retrive_next_token.cpu().numpy()))
        self.assertTrue(
            np.array_equal(ours.retrive_next_sibling, opt.retrive_next_sibling.cpu().numpy())
        )
        padded = build_verify_layout(parents, seqs, num_verify=6)
        linked = set(int(v) for v in padded.retrive_next_token.reshape(-1) if int(v) >= 0)
        linked.update(int(v) for v in padded.retrive_next_sibling.reshape(-1) if int(v) >= 0)
        self.assertTrue(linked.isdisjoint({4, 5}))

    def test_layer_mask_counts_padding_slots(self):
        tree = TollTree(prefix_len=3)
        first = tree.add_child(-1, token=1, raw_p=0.4, a=0.4)
        tree.add_child(-1, token=2, raw_p=0.2, a=0.2)
        child = tree.add_child(first, token=3, raw_p=0.5, a=0.5)
        # Previous layer occupies slots 0..3. This child is slot 4.
        slots = {first: 0, 1: 1, child: 4}
        mask = layer_mask(3, tree, [child], width=4, slot_of_node=slots)
        self.assertEqual(mask.shape, (4, 3 + 4 + 4))
        self.assertTrue(mask[0, 3])
        self.assertTrue(mask[0, 3 + 4])
        self.assertFalse(mask[0, 3 + 2])
        self.assertTrue(np.all(mask[1, :3]))
        self.assertTrue(mask[1, 3 + 4 + 1])
        self.assertFalse(mask[1, 3])


class TestTollVerifyBounds(CustomTestCase):
    def test_spec_steps_stay_inside_the_verify_slots(self):
        from sglang.srt.speculative.toll_tree_remote.target.worker import walk_accept

        max_depth = 4
        parents = [[-1, 0, 1, 2]]
        layout = build_verify_layout(parents, [3], num_verify=max_depth + 1)
        walk_accept(layout, max_depth)

    def test_tokens_outside_the_vocab_are_dropped(self):
        from sglang.srt.speculative.toll_tree_remote.target.worker import filter_vocab

        tokens, parents = filter_vocab([[4, 1000]], [[-1, 0]], vocab_size=128)
        self.assertEqual(tokens, [[]])
        self.assertEqual(parents, [[]])
        tokens, parents = filter_vocab([[4, 5]], [[-1, 0]], vocab_size=128)
        self.assertEqual(tokens, [[4, 5]])


class TestTollSpecInfo(CustomTestCase):
    def test_toll_role_does_not_leak_into_opt(self):
        from sglang.srt.speculative.spec_info import remote_spec_role

        algo = SpeculativeAlgorithm.from_string("TOLL_TREE_REMOTE")
        draft = SimpleNamespace(
            speculative_algorithm="TOLL_TREE_REMOTE",
            toll_tree_role="draft",
            standalone_remote_role="target",
            opt_tree_role="target",
        )
        self.assertEqual(remote_spec_role(draft), "draft")
        self.assertTrue(algo.uses_draft_paused_reqs(draft))
        self.assertTrue(algo.uses_ordinary_decode_prepare(draft))
        self.assertTrue(algo.reports_remote_spec_metrics())
        target = SimpleNamespace(speculative_algorithm="TOLL_TREE_REMOTE", toll_tree_role="target")
        self.assertFalse(algo.uses_draft_paused_reqs(target))
        self.assertFalse(algo.uses_ordinary_decode_prepare(target))
        opt = SimpleNamespace(
            speculative_algorithm="OPT_TREE_REMOTE",
            opt_tree_role="draft",
            toll_tree_role="target",
        )
        self.assertEqual(remote_spec_role(opt), "draft")
        self.assertTrue(SpeculativeAlgorithm.OPT_TREE_REMOTE.reports_remote_spec_metrics())


class TestTollCommitAndFakeDraft(CustomTestCase):
    def test_target_sends_the_commit_delta_and_keeps_the_worker(self):
        import time

        from sglang.srt.speculative.toll_tree_remote.policy.warmup import WarmupSchedule
        from sglang.srt.speculative.toll_tree_remote.target.scheduler_mixin import (
            TollTargetSchedulerMixin,
            committed_len,
            tail_token,
        )
        from sglang.srt.speculative.toll_tree_remote.target.worker import TollTreeRemoteWorker

        req = SimpleNamespace(origin_input_ids=[1, 2], output_ids=[3])
        self.assertEqual(committed_len(req), 3)
        self.assertEqual(tail_token(req), 3)
        self.assertEqual(tail_token(SimpleNamespace(origin_input_ids=[1, 2], output_ids=[])), 2)
        host = TollTargetSchedulerMixin()
        host.toll_sent_len = {}
        host.toll_need_snapshot = set()
        host.toll_seq = 0
        host.toll_session = "abc"
        host._toll_last_reply_ns = time.perf_counter_ns()
        host.toll_cfg = TollConfig(policy="fixed", fixed_width=8, fixed_tau=0.0, max_depth=2)
        step = SimpleNamespace(rid="a", origin_input_ids=[1, 2, 3], output_ids=[4])
        delta, total, full = TollTargetSchedulerMixin._toll_delta(host, step)
        self.assertEqual((delta, total, full), ([1, 2, 3, 4], 4, [1, 2, 3, 4]))
        host.toll_sent_len["a"] = 3
        delta, total, _full = TollTargetSchedulerMixin._toll_delta(host, step)
        self.assertEqual(delta, [4])
        self.assertEqual(total, 4)
        host.toll_need_snapshot.add("a")
        delta, _total, full = TollTargetSchedulerMixin._toll_delta(host, step)
        self.assertEqual(delta, full)
        batch = SimpleNamespace(reqs=[step])
        request = TollTargetSchedulerMixin._toll_request(host, "STEP", batch)
        self.assertEqual(request.session, "abc")
        self.assertLess(request.t_target_ns, 1_000_000_000)
        self.assertEqual(request.reqs[0].input_ids, [1, 2, 3, 4])
        host.toll_need_snapshot.clear()
        host._toll_last_reply_ns = time.perf_counter_ns()
        incremental = host._toll_request("STEP", batch)
        self.assertEqual(incremental.reqs[0].input_ids, [])
        self.assertEqual(incremental.reqs[0].new_tokens, [4])
        inner = object()
        worker = TollTreeRemoteWorker(
            server_args=SimpleNamespace(page_size=128), target_worker=inner
        )
        self.assertIs(worker.target_worker, inner)
        self.assertEqual(worker.page_size, 128)
        self.assertIsNone(worker.draft_model_runner)

        class _PoolWorker:
            def get_memory_pool(self):
                return "req-pool", "kv-pool"

        pooled = TollTreeRemoteWorker(
            server_args=SimpleNamespace(page_size=128), target_worker=_PoolWorker()
        )
        self.assertEqual(pooled.req_to_token_pool, "req-pool")
        self.assertEqual(pooled.token_to_kv_pool_allocator, "kv-pool")
        for width, bucket in WarmupSchedule(TollConfig()).pairs:
            self.assertLessEqual(width, bucket)

    def test_greedy_health_and_abort(self):
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.toll_tree_remote.protocol import TollAction
        from sglang.srt.speculative.toll_tree_remote.target.scheduler_mixin import (
            TollTargetSchedulerMixin,
            skip_speculative,
        )

        params = SamplingParams(temperature=0.0)
        self.assertEqual(params.temperature, 1.0)
        self.assertEqual(params.top_k, 1)
        greedy = SimpleNamespace(
            is_all_greedy=True, penalizer_orchestrator=None, logit_bias=None
        )
        plain = SimpleNamespace(
            reqs=[SimpleNamespace(rid="a")], sampling_info=greedy, has_grammar=False
        )
        self.assertFalse(skip_speculative(plain))
        health = SimpleNamespace(
            reqs=[SimpleNamespace(rid="HEALTH_CHECK_1")],
            sampling_info=greedy,
            has_grammar=False,
        )
        self.assertTrue(skip_speculative(health))
        biased = SimpleNamespace(
            is_all_greedy=True, penalizer_orchestrator=None, logit_bias=object()
        )
        self.assertTrue(
            skip_speculative(
                SimpleNamespace(
                    reqs=[SimpleNamespace(rid="a")],
                    sampling_info=biased,
                    has_grammar=False,
                )
            )
        )
        sampled = SimpleNamespace(
            is_all_greedy=False, penalizer_orchestrator=None, logit_bias=None
        )
        self.assertTrue(
            skip_speculative(
                SimpleNamespace(
                    reqs=[SimpleNamespace(rid="a")],
                    sampling_info=sampled,
                    has_grammar=False,
                )
            )
        )

        host = TollTargetSchedulerMixin()
        host.spec_algorithm = SpeculativeAlgorithm.TOLL_TREE_REMOTE
        host.server_args = SimpleNamespace(toll_tree_role="target")
        host.tp_rank = 0
        host.toll_seq = 0
        host.toll_session = "abc"
        host._toll_last_reply_ns = 0
        host.toll_cfg = TollConfig(policy="fixed", fixed_width=8, fixed_tau=0.0, max_depth=2)
        host.toll_sent_len = {"a": 3}
        host.toll_need_snapshot = set()
        sent = []
        host.toll_client = SimpleNamespace(send_request=sent.append)
        host.maybe_abort_remote_target()
        self.assertNotEqual(host.toll_session, "abc")
        self.assertEqual(host.toll_sent_len, {})
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].action, TollAction.ABORT)

    def test_inproc_commit_kv_and_snapshot(self):
        import time
        from pathlib import Path

        import importlib.util

        from sglang.srt.speculative.toll_tree_remote.draft.kv import TreeKV
        from sglang.srt.speculative.toll_tree_remote.draft.reqs import layer_positions
        from sglang.srt.speculative.toll_tree_remote.draft.scheduler_mixin import TollDraftService

        self.assertEqual(layer_positions(4, 1, 3), [4, 4, 4])
        self.assertEqual(layer_positions(4, 2, 2), [5, 5])

        path = Path(__file__).resolve().parents[3] / "manual" / "toll_fake_draft.py"
        spec = importlib.util.spec_from_file_location("toll_fake_draft", path)
        fake = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fake)
        baseline = [1, 2, 3, 10, 11, 12]
        gold = fake.draft_window("gold", baseline, 3, 3)
        self.assertEqual(gold, ([10, 11, 12], [-1, 0, 1]))
        wrong = fake.draft_window("wrong", baseline, 3, 3)
        self.assertEqual(wrong[0], [11, 12, 13])
        pad = fake.draft_window("pad", baseline, 3, 3)
        self.assertEqual(pad, ([10], [-1]))
        self.assertIsNone(fake.draft_window("silent", baseline, 3, 3))

        cfg = TollConfig(policy="fixed", fixed_width=8, fixed_tau=0.0, max_depth=4)
        pool = _SlotPool()
        box = {}

        def root(_item):
            scores = np.full(8, 0.1, dtype=np.float64)
            scores[0] = 0.5
            return scores, np.arange(8)

        def layer(trees, frontiers, width):
            tree = trees[0]
            front = [int(node) for node in list(frontiers[0])]
            if front:
                parent = int(tree.parent[front[0]])
                if parent != -1:
                    box["ancestor"] = parent in box["service"].slot_of_node
            time.sleep(0.002)
            scores = np.full((int(width), int(width)), 0.9, dtype=np.float64)
            ids = np.arange(int(width) * int(width)).reshape(int(width), int(width))
            return [(scores, ids, 0.0)]

        def make_req(rid, ids):
            return SimpleNamespace(
                rid=rid,
                origin_input_ids=list(ids),
                output_ids=[],
                fill_ids=list(ids),
                req_pool_idx=None,
                kv_committed_len=0,
            )

        service = TollDraftService(
            cfg,
            forward=lambda item, slots: ([], []),
            alloc=pool.alloc,
            free=pool.free,
            ttl_s=1.0,
            root_topk=root,
            layer_forward=layer,
            req_factory=make_req,
        )
        box["service"] = service
        self.assertEqual(service.round.price.alpha, cfg.price_alpha)
        self.assertEqual(service.round.cost.probe_interval, cfg.probe_interval)
        self.assertEqual(service.round.cost.warmup_rounds, cfg.warmup_rounds)
        zmq = pytest_zmq()
        server = client = None
        if zmq is not None:
            url = f"inproc://toll-{uuid.uuid4().hex}"
            server = TollDraftServer(url)
            client = TollTargetClient(url, cfg, timeout_ms=1000)

        def exchange(request):
            if server is None:
                return service.handle(request)
            client.send_request(request)
            got = server.recv_request(timeout_ms=1000)
            reply = service.handle(got)
            server.send_reply(reply)
            return client.recv_reply(request.session, request.seq)

        try:
            prefill = _toll_request(cfg, TollAction.PREFILL, 1, [1, 2, 3], [1, 2, 3], 3)
            reply = exchange(prefill)
            self.assertEqual(reply.reqs[0].status, TollStatus.OK)
            step = _toll_request(cfg, TollAction.STEP, 2, [4], [1, 2, 3, 4], 4)
            reply = exchange(step)
            self.assertEqual(reply.reqs[0].status, TollStatus.OK)
            self.assertGreater(len(reply.reqs[0].tokens), 0)
            self.assertEqual(list(service.reqs["a"].fill_ids), [1, 2, 3, 4])
            self.assertTrue(box.get("ancestor"))
            self.assertGreaterEqual(service.round.last_result.layers[0].ms, 1.0)
            self.assertEqual(len(pool.live), 4)
            more = _toll_request(cfg, TollAction.STEP, 3, [5, 6], [1, 2, 3, 4, 5, 6], 6)
            reply = exchange(more)
            self.assertEqual(reply.reqs[0].status, TollStatus.OK)
            self.assertEqual(list(service.reqs["a"].fill_ids), [1, 2, 3, 4, 5, 6])
            bad = _toll_request(cfg, TollAction.STEP, 4, [9], [1, 2, 3, 4, 5, 6], 99)
            reply = exchange(bad)
            self.assertEqual(reply.reqs[0].status, TollStatus.NEED_SNAPSHOT)
            self.assertEqual(len(service.reqs["a"].fill_ids), 6)
            snap = _toll_request(cfg, TollAction.STEP, 5, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 5)
            reply = exchange(snap)
            self.assertEqual(reply.reqs[0].status, TollStatus.OK)
            self.assertEqual(list(service.reqs["a"].fill_ids), [1, 2, 3, 4, 5])
            service.prefix_slots["stale"] = pool.alloc(2)
            service.state.upsert("stale", 1, 1)
            service.state.reqs["stale"].last_seen = 0
            service.kv["stale"] = TreeKV(pool.alloc, pool.free)
            service.state.get("a")
            self.assertGreater(service.state.reqs["a"].last_seen, 0)
            done = _toll_request(cfg, TollAction.FINISH, 6, [], [1, 2, 3, 4, 5], 5)
            reply = exchange(done)
            self.assertEqual(reply.reqs[0].status, TollStatus.OK)
            self.assertEqual(pool.live, [])
            self.assertNotIn("a", service.reqs)
            self.assertNotIn("stale", service.prefix_slots)
        finally:
            if client is not None:
                client.close()
            if server is not None:
                server.close()


class _SlotPool:
    def __init__(self) -> None:
        self.live = []
        self._next = 0

    def alloc(self, width: int):
        slots = list(range(self._next, self._next + int(width)))
        self._next += int(width)
        self.live.extend(slots)
        return slots

    def free(self, slots) -> None:
        for slot in slots:
            self.live.remove(slot)


class TestTollDraftPool(CustomTestCase):
    def test_requests_past_the_pool_size_return_their_slots(self):
        from sglang.srt.speculative.toll_tree_remote.draft.scheduler_mixin import TollDraftService

        class IndexPool:
            def __init__(self, size: int) -> None:
                self.free_slots = list(range(size))

            def alloc(self, req) -> None:
                if not self.free_slots:
                    raise RuntimeError("request pool is empty")
                req.req_pool_idx = self.free_slots.pop(0)

            def free(self, req) -> None:
                self.free_slots.append(req.req_pool_idx)
                req.req_pool_idx = None

        indexes = IndexPool(2)
        kv = _SlotPool()

        def make_req(rid, ids):
            req = SimpleNamespace(
                rid=rid,
                origin_input_ids=list(ids),
                output_ids=[],
                fill_ids=list(ids),
                req_pool_idx=None,
                kv_committed_len=0,
            )
            indexes.alloc(req)
            return req

        cfg = TollConfig(policy="fixed", fixed_width=8, fixed_tau=0.0, max_depth=2)
        service = TollDraftService(
            cfg,
            forward=lambda item, slots: ([], []),
            alloc=kv.alloc,
            free=kv.free,
            ttl_s=1.0,
            req_factory=make_req,
            release_req=indexes.free,
        )
        for index in range(3):
            reply = service.handle(
                _toll_request(cfg, TollAction.PREFILL, index + 1, [], [1, 2], 2, rid=f"r{index}")
            )
            self.assertEqual(reply.reqs[0].status, TollStatus.OK)
            done = service.handle(
                _toll_request(cfg, TollAction.FINISH, index + 10, [], [1, 2], 2, rid=f"r{index}")
            )
            self.assertEqual(done.reqs[0].status, TollStatus.OK)
        self.assertEqual(sorted(indexes.free_slots), [0, 1])
        self.assertEqual(kv.live, [])

    def test_idle_expiry_frees_prefix_slots(self):
        from sglang.srt.speculative.toll_tree_remote.draft.scheduler_mixin import TollDraftService

        kv = _SlotPool()
        cfg = TollConfig(policy="fixed", fixed_width=8, fixed_tau=0.0, max_depth=2)
        service = TollDraftService(
            cfg,
            forward=lambda item, slots: ([], []),
            alloc=kv.alloc,
            free=kv.free,
            ttl_s=1.0,
            req_factory=lambda rid, ids: SimpleNamespace(
                rid=rid,
                origin_input_ids=list(ids),
                output_ids=[],
                fill_ids=list(ids),
                req_pool_idx=None,
                kv_committed_len=0,
            ),
        )
        service.state.upsert("stale", 1, 1)
        service.state.reqs["stale"].last_seen = 0
        service.prefix_slots["stale"] = kv.alloc(2)
        service.expire_idle()
        self.assertNotIn("stale", service.prefix_slots)
        self.assertEqual(kv.live, [])

    def test_padding_row_is_not_part_of_the_logit_check(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[3] / "manual" / "toll_draft_logits_check.py"
        spec = importlib.util.spec_from_file_location("toll_draft_logits_check", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        decode = torch.tensor([0.1, 0.7, 0.2])
        layer = torch.tensor([[0.1, 0.7, 0.2], [9.0, 9.0, 9.0]])
        self.assertTrue(module.row_matches_decode(layer, decode))
        layer[0, 0] = 5.0
        self.assertFalse(module.row_matches_decode(layer, decode))


def _toll_request(cfg, action, seq, new_tokens, input_ids, commit_len, rid="a"):
    return TollRequest(
        proto=PROTO_VERSION,
        session="s",
        seq=seq,
        action=action,
        spec_shape=spec_shape_of(cfg),
        t_target_ns=1_000_000,
        reqs=[
            TollReqItem(
                rid=rid,
                commit_len=int(commit_len),
                new_tokens=list(new_tokens),
                input_ids=list(input_ids),
                sampling={"temperature": 0.0},
            )
        ],
    )


if __name__ == "__main__":
    unittest.main()
