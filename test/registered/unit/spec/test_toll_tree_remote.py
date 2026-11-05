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


if __name__ == "__main__":
    unittest.main()
