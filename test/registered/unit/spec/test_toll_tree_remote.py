"""CPU checks that TOLL_TREE_REMOTE stays a separate, unimplemented algorithm."""

import argparse
import unittest
from types import SimpleNamespace

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

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

    def test_create_worker_refuses(self):
        algo = SpeculativeAlgorithm.from_string("TOLL_TREE_REMOTE")
        with self.assertRaises(ValueError) as ctx:
            algo.create_worker(SimpleNamespace(disable_overlap_schedule=True))
        self.assertIn("not implemented", str(ctx.exception))

    @unittest.skipIf(ServerArgs is None, "sglang.srt.server_args is not importable")
    def test_cli_accepts_algorithm_name(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = parser.parse_args(
            ["--model-path", "dummy", "--speculative-algorithm", "TOLL_TREE_REMOTE"]
        )
        self.assertEqual(parsed.speculative_algorithm, "TOLL_TREE_REMOTE")

    @unittest.skipIf(ServerArgs is None, "sglang.srt.server_args is not importable")
    def test_startup_refuses(self):
        args = ServerArgs(model_path="dummy")
        args.speculative_algorithm = "TOLL_TREE_REMOTE"
        with self.assertRaises(ValueError) as ctx:
            args._handle_speculative_decoding()
        self.assertIn("not implemented", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
