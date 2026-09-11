"""[SPECTRE-VL] Unit tests for SPECTRE Qwen3-VL mm sidecar (pad_value, shm, multipart)."""
import os
import pickle
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalInputs,
)
from sglang.srt.speculative.spectre.spectre_mm_transport import (
    SpectreMMPayload,
    SpectreMMReceiver,
    SpectreMMSender,
    clear_spectre_mm_env_cache,
    deserialize_mm_item,
    reset_mm_mrope,
    serialize_mm_item,
    spectre_mm_prewarm_bytes,
    spectre_mm_prewarm_max,
    spectre_mm_stale_s,
    spectre_mm_wait_ms,
)
from sglang.srt.speculative.spectre.spectre_protocol import (
    SpectreAction,
    SpectreRequest,
    SpecType,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


def _make_image_item(pad_value=1_000_042, h=3, w=4):
    feature = torch.arange(h * w, dtype=torch.float32).reshape(h, w)
    grid = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=42,
        pad_value=pad_value,
        offsets=[0, 4],
        format=MultimodalInputFormat.NORMAL,
        feature=feature,
        model_specific_data={"image_grid_thw": grid},
    )
    return item


class TestSpectreProtocolMmRef(CustomTestCase):
    def test_mm_ref_roundtrip(self):
        req = SpectreRequest(
            request_id="abc",
            spec_cnt=0,
            action=SpectreAction.DRAFT,
            spec_type=SpecType.DRAFT_REQUEST,
            input_ids=[1, 2, 3],
            mm_ref="abc",
        )
        restored = SpectreRequest.from_dict(req.to_dict())
        self.assertEqual(restored.mm_ref, "abc")
        self.assertEqual(restored.input_ids, [1, 2, 3])

    def test_mm_ref_optional(self):
        req = SpectreRequest(request_id="t", action=SpectreAction.DRAFT)
        data = req.to_dict()
        self.assertNotIn("mm_ref", data)
        restored = SpectreRequest.from_dict(data)
        self.assertIsNone(restored.mm_ref)

    def test_text_only_request_omits_mm_ref(self):
        req = SpectreRequest(
            request_id="text",
            spec_cnt=0,
            action=SpectreAction.DRAFT,
            spec_type=SpecType.DRAFT_REQUEST,
            input_ids=[10, 11, 12],
        )
        data = req.to_dict()
        self.assertNotIn("mm_ref", data)
        restored = SpectreRequest.from_dict(data)
        self.assertIsNone(restored.mm_ref)
        self.assertEqual(restored.input_ids, [10, 11, 12])


class TestSpectreMMPayload(CustomTestCase):
    def test_pad_value_preserved_through_serialize(self):
        item = _make_image_item(pad_value=1_000_123)
        data = serialize_mm_item(item, use_shm=False)
        restored = deserialize_mm_item(data)
        self.assertEqual(restored.pad_value, 1_000_123)
        self.assertEqual(restored.hash, 42)
        self.assertTrue(torch.equal(restored.feature, item.feature))
        self.assertTrue(
            torch.equal(
                restored.model_specific_data["image_grid_thw"], item.image_grid_thw
            )
        )

    def test_set_pad_value_does_not_overwrite_explicit_pad(self):
        restored = deserialize_mm_item(
            serialize_mm_item(_make_image_item(pad_value=1_000_321))
        )
        restored.set_pad_value()
        self.assertEqual(restored.pad_value, 1_000_321)
        self.assertEqual(restored.hash, 42)

    def test_from_req_keeps_padded_ids_and_pad_value(self):
        item = _make_image_item(pad_value=1_000_888)
        mm = MultimodalInputs(mm_items=[item], im_token_id=151655)
        req = SimpleNamespace(
            rid="req-1",
            origin_input_ids=[151655, 151655, 7, 8],
            multimodal_inputs=mm,
        )
        payload = SpectreMMPayload.from_req(req)
        self.assertEqual(payload.padded_input_ids, [151655, 151655, 7, 8])
        self.assertEqual(payload.mm_items[0]["pad_value"], 1_000_888)
        self.assertEqual(payload.mm_items[0]["hash"], 42)
        restored = payload.to_multimodal_inputs()
        self.assertEqual(restored.mm_items[0].pad_value, 1_000_888)
        self.assertEqual(restored.im_token_id, 151655)

    def test_shm_roundtrip_preserves_pad_value(self):
        item = _make_image_item(pad_value=1_000_777)
        payload = SpectreMMPayload(
            rid="r1",
            padded_input_ids=[7, 7, 7, 7],
            mm_items=[serialize_mm_item(item, use_shm=False)],
            im_token_id=151655,
        )
        wire = pickle.loads(pickle.dumps(payload.to_pickleable(use_shm=True)))
        restored = SpectreMMPayload.from_pickleable(wire)
        mm = restored.to_multimodal_inputs()
        self.assertEqual(mm.mm_items[0].pad_value, 1_000_777)
        self.assertEqual(mm.mm_items[0].hash, 42)
        self.assertTrue(torch.equal(mm.mm_items[0].feature, item.feature))
        self.assertEqual(restored.padded_input_ids, [7, 7, 7, 7])
        self.assertEqual(mm.im_token_id, 151655)

    def test_multipart_roundtrip(self):
        item = _make_image_item(pad_value=1_000_009)
        payload = SpectreMMPayload(
            rid="r2",
            padded_input_ids=[9, 8, 7],
            mm_items=[serialize_mm_item(item, use_shm=False)],
        )
        meta, buffers = payload.to_multipart()
        restored = SpectreMMPayload.from_multipart(meta, buffers)
        mm = restored.to_multimodal_inputs()
        self.assertEqual(mm.mm_items[0].pad_value, 1_000_009)
        self.assertTrue(torch.equal(mm.mm_items[0].feature, item.feature))
        self.assertTrue(
            torch.equal(mm.mm_items[0].image_grid_thw, item.image_grid_thw)
        )
        self.assertEqual(restored.padded_input_ids, [9, 8, 7])

        restored_bytes = SpectreMMPayload.from_multipart(
            meta, [bytes(b) for b in buffers]
        )
        restored_mm = restored_bytes.to_multimodal_inputs()
        self.assertTrue(torch.equal(restored_mm.mm_items[0].feature, item.feature))
        self.assertTrue(
            torch.equal(restored_mm.mm_items[0].image_grid_thw, item.image_grid_thw)
        )

    def test_payload_resident_bytes(self):
        from sglang.srt.speculative.spectre.spectre_mm_transport import (
            payload_resident_bytes,
        )

        item = _make_image_item()
        payload = SpectreMMPayload(
            rid="bytes",
            padded_input_ids=[1, 2],
            mm_items=[serialize_mm_item(item)],
        )
        expected = int(item.feature.numel() * item.feature.element_size())
        self.assertEqual(payload_resident_bytes(payload), expected)

    def test_gpu_bytes_counts_fake_npu_and_cuda_not_cpu(self):
        from sglang.srt.speculative.spectre.spectre_mm_transport import (
            _tensor_gpu_bytes,
            _to_cpu_contiguous_tensor,
        )

        cpu = torch.ones(4)
        self.assertEqual(_tensor_gpu_bytes(cpu), 0)
        out = _to_cpu_contiguous_tensor(cpu)
        self.assertEqual(out.device.type, "cpu")

        fake_npu = MagicMock(spec=torch.Tensor)
        fake_npu.device = SimpleNamespace(type="npu")
        fake_npu.numel.return_value = 16
        fake_npu.element_size.return_value = 2
        self.assertEqual(_tensor_gpu_bytes(fake_npu), 32)

        real = torch.arange(3, dtype=torch.float32)
        fake_dev = MagicMock(spec=torch.Tensor)
        fake_dev.device = SimpleNamespace(type="npu")
        fake_dev.detach.return_value = fake_dev
        fake_dev.cpu.return_value = real
        moved = _to_cpu_contiguous_tensor(fake_dev)
        self.assertTrue(torch.equal(moved, real))

    def test_ipc_sender_receiver_roundtrip(self):
        try:
            import zmq  # noqa: F401
        except ImportError:
            self.skipTest("pyzmq is not installed")
        item = _make_image_item(pad_value=1_000_424)
        payload = SpectreMMPayload(
            rid="ipc-1",
            padded_input_ids=[1, 2, 3, 4],
            mm_items=[serialize_mm_item(item, use_shm=False)],
            im_token_id=151655,
        )
        ipc_path = os.path.join(
            tempfile.gettempdir(), f"spectre_mm_test_{os.getpid()}.ipc"
        )
        addr = f"ipc://{ipc_path}"
        sender = SpectreMMSender(addr, use_shm=True, bind=True)
        receiver = SpectreMMReceiver(addr, use_shm=True, bind=False)
        try:
            import zmq

            self.assertEqual(sender._socket.socket_type, zmq.PUB)
            self.assertEqual(receiver._socket.socket_type, zmq.SUB)
            # PUB/SUB slow joiner：订阅建立后再发，避免第一包被丢。
            time.sleep(0.05)
            status = sender.send(payload)
            self.assertEqual(status, "sent")
            got = []
            deadline = time.time() + 2.0
            while time.time() < deadline and not got:
                got, _n_err = receiver.recv_all()
                if not got:
                    time.sleep(0.02)
            self.assertEqual(len(got), 1)
            mm = got[0].to_multimodal_inputs()
            self.assertEqual(got[0].rid, "ipc-1")
            self.assertEqual(mm.mm_items[0].pad_value, 1_000_424)
            self.assertTrue(torch.equal(mm.mm_items[0].feature, item.feature))
        finally:
            sender.close()
            receiver.close()
            try:
                os.unlink(ipc_path)
            except OSError:
                pass

    def test_send_failure_unlinks_shm(self):
        try:
            import zmq
        except ImportError:
            self.skipTest("pyzmq is not installed")
        from multiprocessing import shared_memory

        from sglang.srt.speculative.spectre.spectre_mm_transport import (
            _collect_shm_pointers,
        )

        payload = SpectreMMPayload(
            rid="shm-fail",
            padded_input_ids=[1, 2, 3],
            mm_items=[serialize_mm_item(_make_image_item())],
        )
        captured = {"names": []}

        class FailingSocket:
            def send_pyobj(self, obj, flags=0):
                captured["names"] = [
                    p.shm_name for p in _collect_shm_pointers(obj) if p.shm_name
                ]
                raise zmq.Again()

        sender = SpectreMMSender.__new__(SpectreMMSender)
        sender.addr = "ipc://unused"
        sender.use_shm = True
        sender._socket = FailingSocket()

        self.assertEqual(sender.send(payload), "queue_full")

        # 段在序列化时就创建了；发送失败后必须已被回收，否则 /dev/shm 永久泄漏。
        self.assertTrue(captured["names"])
        for name in captured["names"]:
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)

    def test_mm_wait_ms_env(self):
        with patch.dict(os.environ, {"SPECTRE_MM_WAIT_MS": "350"}):
            clear_spectre_mm_env_cache()
            self.assertEqual(spectre_mm_wait_ms(), 350.0)
        # 未显式设置时取 Target recv 超时的一半。
        env = dict(os.environ)
        env.pop("SPECTRE_MM_WAIT_MS", None)
        env["SPECTRE_RECV_TIMEOUT_MS"] = "240"
        with patch.dict(os.environ, env, clear=True):
            clear_spectre_mm_env_cache()
            self.assertEqual(spectre_mm_wait_ms(), 120.0)
        clear_spectre_mm_env_cache()

    def test_reset_mm_mrope(self):
        mm = MultimodalInputs(
            mm_items=[_make_image_item()],
            mrope_positions=torch.zeros(3, 4, dtype=torch.int64),
            mrope_position_delta=torch.tensor([1]),
        )
        mm.mrope_position_delta_repeated_cache = torch.zeros(3, 1, dtype=torch.int64)
        reset_mm_mrope(mm)
        self.assertIsNone(mm.mrope_positions)
        self.assertIsNone(mm.mrope_position_delta)
        self.assertIsNone(mm.mrope_position_delta_repeated_cache)

    def test_precomputed_format_keeps_pad_value(self):
        item = _make_image_item(pad_value=1_000_555)
        item.precomputed_embeddings = torch.randn(4, 8)
        item.format = MultimodalInputFormat.PRECOMPUTED_EMBEDDING
        item.feature = None
        restored = deserialize_mm_item(serialize_mm_item(item))
        self.assertEqual(restored.pad_value, 1_000_555)
        self.assertEqual(restored.format, MultimodalInputFormat.PRECOMPUTED_EMBEDDING)
        self.assertIsNone(restored.feature)
        restored.set_pad_value()
        self.assertEqual(restored.pad_value, 1_000_555)

    def test_prewarm_max_env(self):
        old = os.environ.get("SPECTRE_MM_PREWARM_MAX")
        try:
            os.environ["SPECTRE_MM_PREWARM_MAX"] = "3"
            clear_spectre_mm_env_cache()
            self.assertEqual(spectre_mm_prewarm_max(), 3)
            os.environ["SPECTRE_MM_PREWARM_MAX"] = "0"
            self.assertEqual(spectre_mm_prewarm_max(), 3)
            clear_spectre_mm_env_cache()
            self.assertEqual(spectre_mm_prewarm_max(), 0)
        finally:
            if old is None:
                os.environ.pop("SPECTRE_MM_PREWARM_MAX", None)
            else:
                os.environ["SPECTRE_MM_PREWARM_MAX"] = old
            clear_spectre_mm_env_cache()

    def test_prewarm_bytes_and_stale_env(self):
        with patch.dict(os.environ, {"SPECTRE_MM_PREWARM_BYTES": "1024"}):
            clear_spectre_mm_env_cache()
            self.assertEqual(spectre_mm_prewarm_bytes(), 1024)
        with patch.dict(os.environ, {"SPECTRE_MM_PREWARM_BYTES": "0"}):
            clear_spectre_mm_env_cache()
            self.assertEqual(spectre_mm_prewarm_bytes(), 0)
        with patch.dict(os.environ, {"SPECTRE_MM_STALE_S": "2.5"}):
            clear_spectre_mm_env_cache()
            self.assertEqual(spectre_mm_stale_s(), 2.5)
        clear_spectre_mm_env_cache()

    def test_env_cache_requires_clear(self):
        with patch.dict(os.environ, {"SPECTRE_MM_WAIT_MS": "10"}):
            clear_spectre_mm_env_cache()
            self.assertEqual(spectre_mm_wait_ms(), 10.0)
        with patch.dict(os.environ, {"SPECTRE_MM_WAIT_MS": "99"}):
            self.assertEqual(spectre_mm_wait_ms(), 10.0)
            clear_spectre_mm_env_cache()
            self.assertEqual(spectre_mm_wait_ms(), 99.0)
        clear_spectre_mm_env_cache()


class TestSpectreMmAddr(CustomTestCase):
    def test_ipc_and_tcp_mm_addresses(self):
        try:
            from sglang.srt.speculative.spectre.spectre_communication import (
                SpectreConfig,
            )
        except ImportError as e:
            self.skipTest(f"spectre_communication unavailable: {e}")

        ipc = SpectreConfig(zmq_addr="127.0.0.1", zmq_port="30009", zmq_transport="ipc")
        self.assertTrue(ipc.get_mm_addr().startswith("ipc://"))
        self.assertTrue(ipc.get_mm_addr().endswith("_mm"))
        self.assertTrue(ipc.mm_use_shm)

        tcp = SpectreConfig(zmq_addr="10.0.0.2", zmq_port="30009", zmq_transport="tcp")
        self.assertEqual(tcp.get_mm_addr(), "tcp://10.0.0.2:30012")
        self.assertFalse(tcp.mm_use_shm)


class TestSpectreDraftMmLogic(CustomTestCase):
    def _make_mixin(self):
        from collections import OrderedDict

        from sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin import (
            SpectreDraftSchedulerMixin,
        )
        from sglang.srt.speculative.spectre.drafter.spectre_state_manager import (
            SpectreDraftStateManager,
        )

        mixin = SpectreDraftSchedulerMixin.__new__(SpectreDraftSchedulerMixin)
        mixin.tp_rank = 0
        mixin.tp_size = 1
        mixin.tokenizer = MagicMock()
        mixin.model_config = MagicMock()
        mixin.model_config.hf_eos_token_id = 1
        mixin.model_config.vocab_size = 128
        mixin.draft_waiting_queue = []
        mixin.draft_paused_reqs = []
        mixin.draft_batch = MagicMock()
        mixin.draft_batch.is_empty.return_value = True
        mixin.running_batch = MagicMock()
        mixin.running_batch.is_empty.return_value = True
        mixin.pad_input_ids_func = MagicMock()
        mixin._pending_mm = {}
        mixin._draft_reqs_waiting_mm = []
        mixin._finished_mm_rids = OrderedDict()
        mixin._mm_wait_deadline = {}
        mixin._mm_unavailable_rids = {}
        mixin.max_req_input_len = 4096
        mixin.zmq_communicator = MagicMock()
        mixin.draft_state_manager = SpectreDraftStateManager()
        mixin.draft_kv_manager = MagicMock()
        mixin.spectre_mm_receiver = None
        mixin._maybe_compute_mrope_positions = MagicMock()
        mixin._recv_and_store_mm_payloads = MagicMock()
        mixin._remove_draft_req = MagicMock()
        mixin.tp_worker = MagicMock()
        mixin.tp_worker.model_runner.device = "cpu"
        mixin.tp_worker.model_runner.model = None
        mixin.current_scheduler_metrics_enabled = False
        mixin.metrics_collector = None
        return mixin

    def _enable_mm_metrics(self, mixin):
        mixin.current_scheduler_metrics_enabled = True
        mixin.metrics_collector = MagicMock()
        return mixin.metrics_collector

    def _draft_request(self, rid, **kwargs):
        from sglang.srt.speculative.spectre.spectre_protocol import (
            SpectreAction,
            SpectreRequest,
            SpecType,
        )

        kwargs.setdefault("spec_cnt", 0)
        kwargs.setdefault("input_ids", [1, 2, 3])
        return SpectreRequest(
            request_id=rid,
            action=SpectreAction.DRAFT,
            spec_type=SpecType.DRAFT_REQUEST,
            **kwargs,
        )

    def _sent_responses(self, mixin):
        return [
            msg
            for call in mixin.zmq_communicator.send_objs.call_args_list
            for msg in call[0][0]
        ]

    def test_waits_when_mm_ref_missing_payload(self):
        mixin = self._make_mixin()
        draft_req = self._draft_request("r1", mm_ref="r1")
        self.assertFalse(mixin._create_new_draft_req(draft_req))
        self.assertEqual(len(mixin._draft_reqs_waiting_mm), 1)
        mixin.pad_input_ids_func.assert_not_called()
        self.assertNotIn("r1", mixin._mm_unavailable_rids)
        # 旁路收取内含 TP 广播，绝不能在 per-request 路径上触发。
        mixin._recv_and_store_mm_payloads.assert_not_called()

    def test_mm_wait_uses_time_budget(self):
        mixin = self._make_mixin()
        draft_req = self._draft_request("w1", mm_ref="w1")
        module = "sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin"

        with patch.dict(os.environ, {"SPECTRE_MM_WAIT_MS": "100"}):
            clear_spectre_mm_env_cache()
            with patch(f"{module}.time.time", return_value=1000.0):
                # 预算内：停在等待队列里，不降级。
                self.assertFalse(mixin._create_new_draft_req(draft_req))
                self.assertFalse(mixin._create_new_draft_req(draft_req))
                self.assertEqual(len(mixin._draft_reqs_waiting_mm), 1)
                self.assertNotIn("w1", mixin._mm_unavailable_rids)
            with patch(f"{module}.time.time", return_value=1000.2):
                # 超过 100ms：降级并回空响应。
                self.assertTrue(mixin._create_new_draft_req(draft_req))

        self.assertIn("w1", mixin._mm_unavailable_rids)
        self.assertEqual(mixin._draft_reqs_waiting_mm, [])
        self.assertFalse(mixin._exists_draft_state("w1"))
        sent = self._sent_responses(mixin)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].draft_token_ids, [])
        clear_spectre_mm_env_cache()

    def test_unavailable_rid_sends_empty_draft_response(self):
        from sglang.srt.speculative.spectre.spectre_protocol import SpecType

        mixin = self._make_mixin()
        mixin._mm_unavailable_rids["u1"] = time.time()
        draft_req = self._draft_request("u1", spec_cnt=7)

        mixin._process_draft_requests({"u1": draft_req})

        sent = self._sent_responses(mixin)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].request_id, "u1")
        self.assertEqual(sent[0].draft_token_ids, [])
        self.assertEqual(sent[0].draft_logprobs, [])
        # spec_cnt 必须回原值，否则 Target 的 _should_store_draft_message 会丢弃。
        self.assertEqual(sent[0].spec_cnt, 7)
        self.assertEqual(sent[0].spec_type, SpecType.DRAFT_RESPONSE)
        self.assertFalse(mixin._exists_draft_state("u1"))
        self.assertEqual(mixin.draft_waiting_queue, [])

    def test_sticky_degradation_across_spec_cnts(self):
        mixin = self._make_mixin()
        mixin._mm_unavailable_rids["s1"] = time.time()
        # 后续步骤的 DRAFT_REQUEST 不带 input_ids；没有粘性标记就会被静默 continue，
        # Target 每步白等满 SPECTRE_RECV_TIMEOUT_MS。
        for spec_cnt in (1, 2, 3):
            mixin._process_draft_requests(
                {"s1": self._draft_request("s1", spec_cnt=spec_cnt, input_ids=None)}
            )

        sent = self._sent_responses(mixin)
        self.assertEqual([m.spec_cnt for m in sent], [1, 2, 3])
        self.assertTrue(all(m.draft_token_ids == [] for m in sent))

    def test_finish_clears_mm_degradation(self):
        from sglang.srt.speculative.spectre.spectre_protocol import (
            SpectreAction,
            SpectreRequest,
        )

        mixin = self._make_mixin()
        mixin._mm_unavailable_rids["f1"] = time.time()
        mixin._mm_wait_deadline["f1"] = time.time() + 1.0

        mixin._process_control_message(
            [SpectreRequest(request_id="f1", action=SpectreAction.FINISH)]
        )

        self.assertNotIn("f1", mixin._mm_unavailable_rids)
        self.assertNotIn("f1", mixin._mm_wait_deadline)

    def test_padded_ids_mismatch_refuses_attach(self):
        from sglang.srt.speculative.spectre.spectre_mm_transport import SpectreMMPayload

        mixin = self._make_mixin()
        item = _make_image_item(pad_value=1_000_001)
        payload = SpectreMMPayload(
            rid="m1",
            padded_input_ids=[1_000_001, 1_000_001, 9],
            mm_items=[serialize_mm_item(item)],
        )
        mixin._pending_mm["m1"] = payload
        # Target 的 DRAFT_REQUEST 与 payload 快照的序列不一致（少一个占位符）。
        draft_req = self._draft_request(
            "m1", input_ids=[1_000_001, 9, 9], mm_ref="m1"
        )

        with patch(
            "sglang.srt.speculative.spectre.drafter."
            "spectre_draft_scheduler_mixin.Req"
        ) as req_cls:
            self.assertTrue(mixin._create_new_draft_req(draft_req))
            req_cls.assert_not_called()

        self.assertIn("m1", mixin._mm_unavailable_rids)
        self.assertNotIn("m1", mixin._finished_mm_rids)
        self.assertFalse(payload.attached)
        self.assertEqual(mixin.draft_waiting_queue, [])
        self.assertEqual(len(self._sent_responses(mixin)), 1)

    def test_oversized_input_degrades(self):
        mixin = self._make_mixin()
        mixin.max_req_input_len = 8
        draft_req = self._draft_request("o1", input_ids=list(range(20)))

        with patch(
            "sglang.srt.speculative.spectre.drafter."
            "spectre_draft_scheduler_mixin.Req"
        ) as req_cls:
            self.assertTrue(mixin._create_new_draft_req(draft_req))
            req_cls.assert_not_called()

        self.assertIn("o1", mixin._mm_unavailable_rids)
        self.assertEqual(len(self._sent_responses(mixin)), 1)

    def test_first_token_mismatch(self):
        mixin = self._make_mixin()
        self.assertIsNone(mixin._first_token_mismatch([1, 2, 3], [1, 2, 3]))
        self.assertEqual(mixin._first_token_mismatch([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(mixin._first_token_mismatch([1, 2], [1, 2, 3]), 2)
        # 缺失一侧时无法判断，交给上层按 mm_ref 逻辑处理。
        self.assertIsNone(mixin._first_token_mismatch(None, [1]))
        self.assertIsNone(mixin._first_token_mismatch([1], []))

    def test_create_attaches_mm_without_padding(self):
        from sglang.srt.speculative.spectre.spectre_mm_transport import SpectreMMPayload
        from sglang.srt.speculative.spectre.spectre_protocol import (
            SpectreAction,
            SpectreRequest,
            SpecType,
        )

        mixin = self._make_mixin()
        item = _make_image_item(pad_value=1_000_001)
        payload = SpectreMMPayload(
            rid="r2",
            padded_input_ids=[1_000_001, 1_000_001, 9],
            mm_items=[serialize_mm_item(item)],
        )
        mixin._pending_mm["r2"] = payload
        draft_req = SpectreRequest(
            request_id="r2",
            spec_cnt=0,
            action=SpectreAction.DRAFT,
            spec_type=SpecType.DRAFT_REQUEST,
            input_ids=[1_000_001, 1_000_001, 9],
            mm_ref="r2",
        )

        def fake_req(**kwargs):
            obj = MagicMock()
            obj.origin_input_ids = list(kwargs.get("origin_input_ids") or [])
            obj.output_ids = []
            obj.multimodal_inputs = None
            return obj

        with patch(
            "sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin.Req",
            side_effect=fake_req,
        ):
            self.assertTrue(mixin._create_new_draft_req(draft_req))
            mixin.pad_input_ids_func.assert_not_called()
            mixin._recv_and_store_mm_payloads.assert_not_called()
            created = mixin.draft_waiting_queue[0]
            created.extend_image_inputs.assert_called_once()
            mixin._maybe_compute_mrope_positions.assert_called_once_with(created)
            self.assertEqual(
                created.origin_input_ids_unpadded, [1_000_001, 1_000_001, 9]
            )
            self.assertTrue(payload.attached)
            state = mixin._get_draft_state("r2")
            self.assertIsNotNone(state)
            self.assertEqual(state.mm_ref, "r2")

    def test_reprefill_resets_and_recomputes_mrope(self):
        from sglang.srt.speculative.spectre.drafter.spectre_state_manager import (
            SpectreDraftState,
        )
        from sglang.srt.speculative.spectre.spectre_protocol import SpectreRequest

        mixin = self._make_mixin()
        mm = MagicMock()
        req = MagicMock()
        req.rid = "r3"
        req.req_pool_idx = None
        req.multimodal_inputs = mm
        req.origin_input_ids = [1, 2]
        req.output_ids = [3]
        req.fill_ids = [1, 2, 3]
        state = SpectreDraftState(req_id="r3", spec_cnt=0, req_object=req, mm_ref="r3")
        mixin._set_draft_state("r3", state)
        mixin._remove_draft_req = MagicMock()
        mixin._reset_req_logprob_fields = MagicMock()
        draft_req = SpectreRequest(request_id="r3", spec_cnt=1, num_draft_tokens=4)
        mixin._prepare_for_reprefill(req, [1, 2, 3, 4, 5], draft_req, state)
        mixin._maybe_compute_mrope_positions.assert_called_once_with(req)
        self.assertIsNone(mm.mrope_positions)
        self.assertIsNone(mm.mrope_position_delta)
        self.assertIsNone(mm.mrope_position_delta_repeated_cache)
        self.assertEqual(req.origin_input_ids, [1, 2, 3, 4, 5])

    def test_reprefill_degrades_when_mm_tensors_missing(self):
        from sglang.srt.speculative.spectre.drafter.spectre_state_manager import (
            SpectreDraftState,
        )
        from sglang.srt.speculative.spectre.spectre_protocol import SpectreRequest

        mixin = self._make_mixin()
        metrics = self._enable_mm_metrics(mixin)
        item = _make_image_item()
        item.feature = None
        item.precomputed_embeddings = None
        mm = MultimodalInputs(mm_items=[item])
        req = MagicMock()
        req.rid = "r3miss"
        req.req_pool_idx = None
        req.multimodal_inputs = mm
        req.origin_input_ids = [1, 2]
        req.output_ids = [3]
        req.fill_ids = [1, 2, 3]
        req.finished.return_value = True
        state = SpectreDraftState(
            req_id="r3miss", spec_cnt=0, req_object=req, mm_ref="r3miss"
        )
        mixin._set_draft_state("r3miss", state)
        mixin._remove_draft_req = MagicMock()
        mixin._reset_req_logprob_fields = MagicMock()
        draft_req = SpectreRequest(
            request_id="r3miss", spec_cnt=1, num_draft_tokens=4
        )
        mixin._prepare_for_reprefill(req, [1, 2, 3, 4, 5], draft_req, state)
        self.assertIn("r3miss", mixin._mm_unavailable_rids)
        self.assertNotIn(req, mixin.draft_waiting_queue)
        metrics.increment_spectre_mm_degrades.assert_called_with(
            reason="mm_tensors_missing"
        )
        self.assertEqual(len(self._sent_responses(mixin)), 1)

    def test_create_degrades_when_mm_tensors_missing(self):
        from sglang.srt.speculative.spectre.spectre_mm_transport import SpectreMMPayload
        from sglang.srt.speculative.spectre.spectre_protocol import (
            SpectreAction,
            SpectreRequest,
            SpecType,
        )

        mixin = self._make_mixin()
        metrics = self._enable_mm_metrics(mixin)
        item = _make_image_item()
        item.feature = None
        item.precomputed_embeddings = None
        payload = SpectreMMPayload(
            rid="cmiss",
            padded_input_ids=[1_000_001, 1_000_001, 9],
            mm_items=[serialize_mm_item(item)],
        )
        mixin._pending_mm["cmiss"] = payload
        draft_req = SpectreRequest(
            request_id="cmiss",
            spec_cnt=0,
            action=SpectreAction.DRAFT,
            spec_type=SpecType.DRAFT_REQUEST,
            input_ids=[1_000_001, 1_000_001, 9],
            mm_ref="cmiss",
        )

        def fake_req(**kwargs):
            obj = MagicMock()
            obj.origin_input_ids = list(kwargs.get("origin_input_ids") or [])
            obj.output_ids = []
            obj.multimodal_inputs = None

            def extend(mm):
                obj.multimodal_inputs = mm

            obj.extend_image_inputs.side_effect = extend
            return obj

        with patch(
            "sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin.Req",
            side_effect=fake_req,
        ):
            self.assertTrue(mixin._create_new_draft_req(draft_req))

        self.assertIn("cmiss", mixin._mm_unavailable_rids)
        self.assertEqual(mixin.draft_waiting_queue, [])
        metrics.increment_spectre_mm_degrades.assert_called_with(
            reason="mm_tensors_missing"
        )
        self.assertEqual(len(self._sent_responses(mixin)), 1)

    def test_finish_releases_pending_mm(self):
        from sglang.srt.speculative.spectre.drafter.spectre_state_manager import (
            SpectreDraftState,
        )

        mixin = self._make_mixin()
        mm = MagicMock()
        mm.mm_items = []
        req = MagicMock()
        req.rid = "r4"
        req.req_pool_idx = None
        req.multimodal_inputs = mm
        req.finished.return_value = False
        mixin._set_draft_state(
            "r4", SpectreDraftState(req_id="r4", spec_cnt=0, req_object=req, mm_ref="r4")
        )
        mixin._pending_mm["r4"] = MagicMock(mm_inputs=mm)
        mixin._remove_draft_req = MagicMock()
        mixin._finish_draft_request("r4")
        mm.release_features.assert_called()
        self.assertNotIn("r4", mixin._pending_mm)
        self.assertIsNone(req.multimodal_inputs)
        self.assertFalse(mixin._exists_draft_state("r4"))
        self.assertIn("r4", mixin._finished_mm_rids)

    def test_degrade_does_not_mark_finished(self):
        from sglang.srt.speculative.spectre.spectre_mm_transport import SpectreMMPayload

        mixin = self._make_mixin()
        payload = SpectreMMPayload(
            rid="d1",
            padded_input_ids=[1, 2, 3],
            mm_items=[serialize_mm_item(_make_image_item())],
        )
        mixin._pending_mm["d1"] = payload
        mixin._degrade_draft_req(self._draft_request("d1", mm_ref="d1"), "test")
        self.assertNotIn("d1", mixin._pending_mm)
        self.assertNotIn("d1", mixin._finished_mm_rids)
        self.assertIn("d1", mixin._mm_unavailable_rids)

    def test_resend_clears_degradation_and_stores_payload(self):
        from types import MethodType

        from sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin import (
            SpectreDraftSchedulerMixin,
        )
        from sglang.srt.speculative.spectre.spectre_mm_transport import SpectreMMPayload

        mixin = self._make_mixin()
        mixin._recv_and_store_mm_payloads = MethodType(
            SpectreDraftSchedulerMixin._recv_and_store_mm_payloads, mixin
        )
        mixin._degrade_draft_req(self._draft_request("rs1", mm_ref="rs1"), "missing")
        self.assertIn("rs1", mixin._mm_unavailable_rids)

        payload = SpectreMMPayload(
            rid="rs1",
            padded_input_ids=[1, 2, 3],
            mm_items=[serialize_mm_item(_make_image_item())],
        )
        mixin.spectre_mm_receiver = MagicMock()
        mixin.spectre_mm_receiver.recv_all.return_value = [payload]
        mixin._recv_and_store_mm_payloads()

        self.assertNotIn("rs1", mixin._mm_unavailable_rids)
        self.assertIn("rs1", mixin._pending_mm)
        self.assertIs(mixin._pending_mm["rs1"], payload)

    def test_duplicate_attached_payload_keeps_old_tensors(self):
        from types import MethodType

        from sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin import (
            SpectreDraftSchedulerMixin,
        )
        from sglang.srt.speculative.spectre.spectre_mm_transport import SpectreMMPayload

        mixin = self._make_mixin()
        metrics = self._enable_mm_metrics(mixin)
        mixin._recv_and_store_mm_payloads = MethodType(
            SpectreDraftSchedulerMixin._recv_and_store_mm_payloads, mixin
        )
        old = SpectreMMPayload(
            rid="dup1",
            padded_input_ids=[1, 2, 3],
            mm_items=[serialize_mm_item(_make_image_item())],
        )
        mm = old.to_multimodal_inputs()
        old.attached = True
        mixin._pending_mm["dup1"] = old
        live_feature = mm.mm_items[0].feature
        self.assertIsNotNone(live_feature)

        new = SpectreMMPayload(
            rid="dup1",
            padded_input_ids=[1, 2, 3],
            mm_items=[serialize_mm_item(_make_image_item())],
        )
        mixin.spectre_mm_receiver = MagicMock()
        mixin.spectre_mm_receiver.use_shm = False
        mixin.spectre_mm_receiver.recv_all.return_value = ([new], 0)
        mixin._recv_and_store_mm_payloads()

        self.assertIs(mixin._pending_mm["dup1"], old)
        self.assertIs(mm.mm_items[0].feature, live_feature)
        self.assertIsNotNone(mm.mm_items[0].feature)
        metrics.increment_spectre_mm_payloads_dropped.assert_called_with(
            reason="duplicate_attached"
        )
        metrics.increment_spectre_mm_payloads_received.assert_not_called()

    def test_unattached_duplicate_replaces_old_payload(self):
        from types import MethodType

        from sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin import (
            SpectreDraftSchedulerMixin,
        )
        from sglang.srt.speculative.spectre.spectre_mm_transport import SpectreMMPayload

        mixin = self._make_mixin()
        mixin._recv_and_store_mm_payloads = MethodType(
            SpectreDraftSchedulerMixin._recv_and_store_mm_payloads, mixin
        )
        old = SpectreMMPayload(
            rid="dup2",
            padded_input_ids=[1, 2, 3],
            mm_items=[serialize_mm_item(_make_image_item())],
        )
        old.to_multimodal_inputs()
        mixin._pending_mm["dup2"] = old

        new = SpectreMMPayload(
            rid="dup2",
            padded_input_ids=[1, 2, 3],
            mm_items=[serialize_mm_item(_make_image_item())],
        )
        mixin.spectre_mm_receiver = MagicMock()
        mixin.spectre_mm_receiver.use_shm = False
        mixin.spectre_mm_receiver.recv_all.return_value = ([new], 0)
        mixin._recv_and_store_mm_payloads()

        self.assertIs(mixin._pending_mm["dup2"], new)
        self.assertIsNone(old.mm_inputs.mm_items[0].feature)

    def test_mm_embed_error_detects_missing_tensors(self):
        mixin = self._make_mixin()
        item = _make_image_item()
        req = MagicMock()
        req.multimodal_inputs = MultimodalInputs(mm_items=[item])
        self.assertIsNone(mixin._mm_embed_error(req))

        item.feature = None
        item.precomputed_embeddings = torch.randn(4, 8)
        self.assertIsNone(mixin._mm_embed_error(req))

        item.precomputed_embeddings = None
        self.assertIsNotNone(mixin._mm_embed_error(req))

        req.multimodal_inputs = None
        self.assertIsNone(mixin._mm_embed_error(req))

    def _vl_draft_req(self, rid, *, precomputed):
        item = _make_image_item()
        if precomputed:
            item.feature = None
            item.precomputed_embeddings = torch.randn(4, 8)
        req = MagicMock()
        req.rid = rid
        req.multimodal_inputs = MultimodalInputs(mm_items=[item])
        return req

    def test_draft_req_uses_precomputed_mm(self):
        mixin = self._make_mixin()
        self.assertTrue(
            mixin._draft_req_uses_precomputed_mm(self._vl_draft_req("p", precomputed=True))
        )
        self.assertFalse(
            mixin._draft_req_uses_precomputed_mm(
                self._vl_draft_req("v", precomputed=False)
            )
        )
        text = MagicMock()
        text.rid = "t"
        text.multimodal_inputs = None
        self.assertFalse(mixin._draft_req_uses_precomputed_mm(text))

    def test_partition_draft_prefill_queue_splits_mixed(self):
        mixin = self._make_mixin()
        pre_a = self._vl_draft_req("baba", precomputed=True)
        vit_b = self._vl_draft_req("7ede", precomputed=False)
        pre_c = self._vl_draft_req("af9a", precomputed=True)
        groups = mixin._partition_draft_prefill_queue([pre_a, vit_b, pre_c])
        self.assertEqual(len(groups), 2)
        self.assertEqual([r.rid for r in groups[0]], ["baba", "af9a"])
        self.assertEqual([r.rid for r in groups[1]], ["7ede"])

        vit_first = mixin._partition_draft_prefill_queue([vit_b, pre_a])
        self.assertEqual([r.rid for r in vit_first[0]], ["7ede"])
        self.assertEqual([r.rid for r in vit_first[1]], ["baba"])

        only_pre = mixin._partition_draft_prefill_queue([pre_a, pre_c])
        self.assertEqual(len(only_pre), 1)
        self.assertEqual([r.rid for r in only_pre[0]], ["baba", "af9a"])

    def test_prefill_splits_mixed_precomputed_batch(self):
        mixin = self._make_mixin()
        mixin.tree_cache = MagicMock()
        pre = self._vl_draft_req("baba", precomputed=True)
        vit = self._vl_draft_req("7ede", precomputed=False)
        mixin.draft_waiting_queue = [pre, vit]
        seen = []

        def fake_group(group):
            seen.append([r.rid for r in group])
            return list(group)

        mixin._prefill_admitted_group = fake_group
        mixin._prefill_draft_reqs()
        self.assertEqual(seen, [["baba"], ["7ede"]])
        self.assertEqual(mixin.draft_waiting_queue, [])
        pre.init_next_round_input.assert_called_once_with(mixin.tree_cache)
        vit.init_next_round_input.assert_called_once_with(mixin.tree_cache)

    def test_finish_drops_late_payload(self):
        from types import MethodType

        from sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin import (
            SpectreDraftSchedulerMixin,
        )
        from sglang.srt.speculative.spectre.spectre_mm_transport import SpectreMMPayload
        from sglang.srt.speculative.spectre.spectre_protocol import SpectreRequest

        mixin = self._make_mixin()
        mixin._recv_and_store_mm_payloads = MethodType(
            SpectreDraftSchedulerMixin._recv_and_store_mm_payloads, mixin
        )
        mixin._process_control_message(
            [SpectreRequest(request_id="late1", action=SpectreAction.FINISH)]
        )
        self.assertIn("late1", mixin._finished_mm_rids)

        payload = SpectreMMPayload(
            rid="late1",
            padded_input_ids=[1, 2],
            mm_items=[serialize_mm_item(_make_image_item())],
        )
        mixin.spectre_mm_receiver = MagicMock()
        mixin.spectre_mm_receiver.recv_all.return_value = [payload]
        mixin._recv_and_store_mm_payloads()
        self.assertNotIn("late1", mixin._pending_mm)

    def test_prewarm_skips_when_gpu_cap_exceeded(self):
        from sglang.srt.speculative.spectre.spectre_mm_transport import SpectreMMPayload

        mixin = self._make_mixin()
        payload = SpectreMMPayload(
            rid="cap1",
            padded_input_ids=[1, 2],
            mm_items=[serialize_mm_item(_make_image_item())],
        )
        mixin._pending_mm["cap1"] = payload
        with patch.dict(os.environ, {"SPECTRE_MM_PREWARM_BYTES": "8"}):
            clear_spectre_mm_env_cache()
            with patch(
                "sglang.srt.speculative.spectre.drafter."
                "spectre_draft_scheduler_mixin.payload_gpu_bytes",
                return_value=64,
            ):
                self.assertEqual(mixin._decide_prewarm_rids(), [])
        clear_spectre_mm_env_cache()

    def test_stale_orphan_payload_released(self):
        from sglang.srt.speculative.spectre.spectre_mm_transport import SpectreMMPayload

        mixin = self._make_mixin()
        payload = SpectreMMPayload(
            rid="stale1",
            padded_input_ids=[1, 2],
            mm_items=[serialize_mm_item(_make_image_item())],
        )
        payload.created_at = time.time() - 10
        mixin._pending_mm["stale1"] = payload
        with patch.dict(os.environ, {"SPECTRE_MM_STALE_S": "1"}):
            clear_spectre_mm_env_cache()
            mixin._cleanup_stale_draft_states()
        clear_spectre_mm_env_cache()
        self.assertNotIn("stale1", mixin._pending_mm)
        self.assertNotIn("stale1", mixin._finished_mm_rids)

    def test_prewarm_keeps_pad_value(self):
        from sglang.srt.managers.schedule_batch import MultimodalInputFormat
        from sglang.srt.speculative.spectre.spectre_mm_transport import SpectreMMPayload

        mixin = self._make_mixin()
        item = _make_image_item(pad_value=1_000_222)
        payload = SpectreMMPayload(
            rid="r5",
            padded_input_ids=[1, 2],
            mm_items=[serialize_mm_item(item)],
        )
        mixin._pending_mm["r5"] = payload
        emb = torch.randn(4, 8)

        mixin.tp_worker.model_runner.model = MagicMock()
        mixin.tp_worker.model_runner.model.get_image_feature = lambda items: emb

        mixin._maybe_prewarm_pending_mm()

        mm = payload.to_multimodal_inputs()
        self.assertTrue(payload.prewarmed)
        self.assertFalse(payload.prewarm_failed)
        self.assertEqual(mm.mm_items[0].pad_value, 1_000_222)
        self.assertEqual(mm.mm_items[0].hash, 42)
        self.assertEqual(
            mm.mm_items[0].format, MultimodalInputFormat.PRECOMPUTED_EMBEDDING
        )
        self.assertIsNone(mm.mm_items[0].feature)
        self.assertTrue(torch.equal(mm.mm_items[0].precomputed_embeddings, emb))

    def test_prewarmed_payload_not_reselected(self):
        from sglang.srt.speculative.spectre.spectre_mm_transport import SpectreMMPayload

        mixin = self._make_mixin()
        payload = SpectreMMPayload(
            rid="r7",
            padded_input_ids=[1, 2],
            mm_items=[serialize_mm_item(_make_image_item())],
        )
        mixin._pending_mm["r7"] = payload
        calls = {"n": 0}

        def counting_vit(items):
            calls["n"] += 1
            return torch.randn(4, 8)

        mixin.tp_worker.model_runner.model = MagicMock()
        mixin.tp_worker.model_runner.model.get_image_feature = counting_vit

        mixin._maybe_prewarm_pending_mm()
        mixin._maybe_prewarm_pending_mm()
        mixin._maybe_prewarm_pending_mm()

        self.assertEqual(calls["n"], 1)
        self.assertEqual(mixin._decide_prewarm_rids(), [])

    def test_prewarm_failure_does_not_drop_features(self):
        from sglang.srt.speculative.spectre.spectre_mm_transport import SpectreMMPayload

        mixin = self._make_mixin()
        item_a = _make_image_item(pad_value=1_000_301)
        item_b = _make_image_item(pad_value=1_000_302)
        payload = SpectreMMPayload(
            rid="r6",
            padded_input_ids=[1, 2, 3],
            mm_items=[serialize_mm_item(item_a), serialize_mm_item(item_b)],
        )
        mixin._pending_mm["r6"] = payload
        calls = {"n": 0}

        def flaky_vit(items):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("vit oom")
            return torch.zeros(2, 2)

        mixin.tp_worker.model_runner.model = MagicMock()
        mixin.tp_worker.model_runner.model.get_image_feature = flaky_vit

        mixin._maybe_prewarm_pending_mm()

        self.assertTrue(payload.prewarm_failed)
        self.assertFalse(payload.prewarmed)
        mm = payload.to_multimodal_inputs()
        self.assertIsNone(mm.mm_items[0].precomputed_embeddings)
        self.assertIsNone(mm.mm_items[1].precomputed_embeddings)
        self.assertIsNotNone(mm.mm_items[0].feature)
        self.assertIsNotNone(mm.mm_items[1].feature)

    def test_prewarm_batches_items_when_offsets_align(self):
        from sglang.srt.speculative.spectre.spectre_mm_transport import SpectreMMPayload

        mixin = self._make_mixin()
        item_a = _make_image_item(pad_value=1_000_401)
        item_a.offsets = [(0, 3)]  # 4 个占位符
        item_b = _make_image_item(pad_value=1_000_402)
        item_b.offsets = [(4, 9)]  # 6 个占位符
        payload = SpectreMMPayload(
            rid="r8",
            padded_input_ids=list(range(10)),
            mm_items=[serialize_mm_item(item_a), serialize_mm_item(item_b)],
        )
        mixin._pending_mm["r8"] = payload
        emb = torch.arange(10 * 8, dtype=torch.float32).reshape(10, 8)
        calls = {"n": 0, "sizes": []}

        def batched_vit(items):
            calls["n"] += 1
            calls["sizes"].append(len(items))
            return emb

        mixin.tp_worker.model_runner.model = MagicMock()
        mixin.tp_worker.model_runner.model.get_image_feature = batched_vit

        mixin._maybe_prewarm_pending_mm()

        # 一次调用覆盖两个 item，切分按 offsets 的占位符数量。
        self.assertEqual(calls["n"], 1)
        self.assertEqual(calls["sizes"], [2])
        mm = payload.to_multimodal_inputs()
        self.assertTrue(torch.equal(mm.mm_items[0].precomputed_embeddings, emb[0:4]))
        self.assertTrue(torch.equal(mm.mm_items[1].precomputed_embeddings, emb[4:10]))
        self.assertEqual(mm.mm_items[0].pad_value, 1_000_401)
        self.assertEqual(mm.mm_items[1].pad_value, 1_000_402)

    def test_prewarm_batch_row_mismatch_falls_back_per_item(self):
        from sglang.srt.speculative.spectre.spectre_mm_transport import SpectreMMPayload

        mixin = self._make_mixin()
        item_a = _make_image_item(pad_value=1_000_501)
        item_a.offsets = [(0, 3)]
        item_b = _make_image_item(pad_value=1_000_502)
        item_b.offsets = [(4, 9)]
        payload = SpectreMMPayload(
            rid="r9",
            padded_input_ids=list(range(10)),
            mm_items=[serialize_mm_item(item_a), serialize_mm_item(item_b)],
        )
        mixin._pending_mm["r9"] = payload
        calls = {"sizes": []}

        def wrong_rows_vit(items):
            calls["sizes"].append(len(items))
            # 行数与 offsets 声明的 10 对不上，必须退回逐 item。
            return torch.zeros(3, 8)

        mixin.tp_worker.model_runner.model = MagicMock()
        mixin.tp_worker.model_runner.model.get_image_feature = wrong_rows_vit

        mixin._maybe_prewarm_pending_mm()

        self.assertEqual(calls["sizes"], [2, 1, 1])

    def test_prewarm_does_not_move_model_specific_data_to_device(self):
        # grid_thw 在 visual.forward 里就会被拉回 CPU/numpy，搬上 GPU 是无谓的来回拷贝，
        # 也和正常 prefill 路径的 _move_items_to_device 不一致。
        mixin = self._make_mixin()
        item = _make_image_item()
        grid_before = item.model_specific_data["image_grid_thw"]
        mixin._prewarm_item_to_device(item, torch.device("cpu"))
        self.assertIs(item.model_specific_data["image_grid_thw"], grid_before)

    def test_agree_on_prewarm_failures_is_identity_for_tp1(self):
        mixin = self._make_mixin()
        self.assertEqual(mixin._agree_on_prewarm_failures([0, 1, 0]), [0, 1, 0])
        self.assertEqual(mixin._agree_on_prewarm_failures([]), [])

    def _enable_tp(self, mixin, rank=0, size=2):
        mixin.tp_rank = rank
        mixin.tp_size = size
        mixin.tp_group = SimpleNamespace(rank=rank, ranks=list(range(size)))
        mixin.tp_cpu_group = object()

    def test_recv_store_skips_payload_broadcast(self):
        from types import MethodType

        from sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin import (
            SpectreDraftSchedulerMixin,
        )

        mixin = self._make_mixin()
        mixin._recv_and_store_mm_payloads = MethodType(
            SpectreDraftSchedulerMixin._recv_and_store_mm_payloads, mixin
        )
        self._enable_tp(mixin, rank=1)
        payload = SpectreMMPayload(
            rid="tp1",
            padded_input_ids=[1, 2],
            mm_items=[serialize_mm_item(_make_image_item())],
        )
        mixin.spectre_mm_receiver = MagicMock()
        mixin.spectre_mm_receiver.use_shm = False
        mixin.spectre_mm_receiver.recv_all.return_value = [payload]
        module = "sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin"
        with patch(f"{module}.broadcast_pyobj") as bcast:
            mixin._recv_and_store_mm_payloads()
        bcast.assert_not_called()
        mixin.spectre_mm_receiver.recv_all.assert_called()
        self.assertIs(mixin._pending_mm["tp1"], payload)

    def test_recv_shm_barriers_before_materialize(self):
        from types import MethodType

        from sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin import (
            SpectreDraftSchedulerMixin,
        )

        mixin = self._make_mixin()
        mixin._recv_and_store_mm_payloads = MethodType(
            SpectreDraftSchedulerMixin._recv_and_store_mm_payloads, mixin
        )
        self._enable_tp(mixin)
        payload = MagicMock()
        payload.rid = "shm1"
        payload.mm_items = []
        payload.mm_inputs = None
        mixin.spectre_mm_receiver = MagicMock()
        mixin.spectre_mm_receiver.use_shm = True
        mixin.spectre_mm_receiver.recv_all.return_value = [payload]

        def fake_all_reduce(tensor, op=None, group=None):
            tensor.fill_(1)

        with patch("torch.distributed.all_reduce", side_effect=fake_all_reduce):
            with patch("torch.distributed.barrier") as barrier:
                mixin._recv_and_store_mm_payloads()
        barrier.assert_called_once()
        payload.to_multimodal_inputs.assert_called()
        mixin.spectre_mm_receiver.recv_all.assert_called_with(
            defer_shm_materialize=True
        )
        self.assertIs(mixin._pending_mm["shm1"], payload)

    def test_empty_prewarm_rids_skips_object_broadcast(self):
        mixin = self._make_mixin()
        self._enable_tp(mixin)
        module = "sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin"
        with patch(f"{module}.broadcast_pyobj") as bcast:
            with patch("torch.distributed.broadcast") as dist_b:
                rids = mixin._decide_prewarm_rids()
        bcast.assert_not_called()
        dist_b.assert_called_once()
        self.assertEqual(rids, [])

    def test_nonempty_prewarm_rids_broadcasts_body(self):
        mixin = self._make_mixin()
        self._enable_tp(mixin)
        payload = SpectreMMPayload(
            rid="b1",
            padded_input_ids=[1, 2],
            mm_items=[serialize_mm_item(_make_image_item())],
        )
        mixin._pending_mm["b1"] = payload
        module = "sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin"
        with patch(f"{module}.broadcast_pyobj", return_value=["b1"]) as bcast:
            with patch("torch.distributed.broadcast"):
                rids = mixin._decide_prewarm_rids()
        bcast.assert_called_once()
        self.assertEqual(rids, ["b1"])

    def test_empty_draft_recv_skips_object_broadcast(self):
        mixin = self._make_mixin()
        self._enable_tp(mixin)
        mixin._is_self_high_overhead_draft = lambda: False
        mixin._recv_draft_requests = MagicMock(return_value=[])
        module = "sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin"
        with patch(f"{module}.broadcast_pyobj") as bcast:
            with patch("torch.distributed.broadcast") as dist_b:
                mixin.recv_and_process_draft_requests()
        bcast.assert_not_called()
        dist_b.assert_called_once()

    def test_prewarm_batches_across_rids(self):
        mixin = self._make_mixin()
        item_a = _make_image_item(pad_value=1_000_601)
        item_a.offsets = [(0, 3)]
        item_b = _make_image_item(pad_value=1_000_602)
        item_b.offsets = [(0, 5)]
        payload_a = SpectreMMPayload(
            rid="x1",
            padded_input_ids=list(range(4)),
            mm_items=[serialize_mm_item(item_a)],
        )
        payload_b = SpectreMMPayload(
            rid="x2",
            padded_input_ids=list(range(6)),
            mm_items=[serialize_mm_item(item_b)],
        )
        mixin._pending_mm["x1"] = payload_a
        mixin._pending_mm["x2"] = payload_b
        emb = torch.arange(10 * 8, dtype=torch.float32).reshape(10, 8)
        calls = {"n": 0, "sizes": []}

        def batched_vit(items):
            calls["n"] += 1
            calls["sizes"].append(len(items))
            return emb

        mixin.tp_worker.model_runner.model = MagicMock()
        mixin.tp_worker.model_runner.model.get_image_feature = batched_vit

        mixin._maybe_prewarm_pending_mm()

        self.assertEqual(calls["n"], 1)
        self.assertEqual(calls["sizes"], [2])
        self.assertTrue(payload_a.prewarmed)
        self.assertTrue(payload_b.prewarmed)
        mm_a = payload_a.to_multimodal_inputs()
        mm_b = payload_b.to_multimodal_inputs()
        self.assertTrue(torch.equal(mm_a.mm_items[0].precomputed_embeddings, emb[0:4]))
        self.assertTrue(torch.equal(mm_b.mm_items[0].precomputed_embeddings, emb[4:10]))

    def test_finished_mm_rids_evicts_oldest(self):
        mixin = self._make_mixin()
        for i in range(1024):
            mixin._mark_mm_finished(f"old-{i}")
        self.assertIn("old-0", mixin._finished_mm_rids)
        self.assertIn("old-1023", mixin._finished_mm_rids)
        mixin._mark_mm_finished("new")
        self.assertNotIn("old-0", mixin._finished_mm_rids)
        self.assertIn("old-1", mixin._finished_mm_rids)
        self.assertIn("new", mixin._finished_mm_rids)
        self.assertEqual(len(mixin._finished_mm_rids), 1024)

    def test_recv_counts_received_payload(self):
        from types import MethodType

        from sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin import (
            SpectreDraftSchedulerMixin,
        )

        mixin = self._make_mixin()
        metrics = self._enable_mm_metrics(mixin)
        mixin._recv_and_store_mm_payloads = MethodType(
            SpectreDraftSchedulerMixin._recv_and_store_mm_payloads, mixin
        )
        payload = SpectreMMPayload(
            rid="mrecv",
            padded_input_ids=[1, 2],
            mm_items=[serialize_mm_item(_make_image_item())],
        )
        mixin.spectre_mm_receiver = MagicMock()
        mixin.spectre_mm_receiver.use_shm = False
        mixin.spectre_mm_receiver.recv_all.return_value = ([payload], 0)
        mixin._recv_and_store_mm_payloads()
        metrics.increment_spectre_mm_payloads_received.assert_called_once()
        metrics.increment_spectre_mm_payloads_dropped.assert_not_called()
        metrics.set_spectre_mm_pending_bytes.assert_called()
        self.assertGreater(metrics.set_spectre_mm_pending_bytes.call_args[0][0], 0)

    def test_finished_rid_counts_as_drop(self):
        from types import MethodType

        from sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin import (
            SpectreDraftSchedulerMixin,
        )
        from sglang.srt.speculative.spectre.spectre_protocol import SpectreRequest

        mixin = self._make_mixin()
        metrics = self._enable_mm_metrics(mixin)
        mixin._recv_and_store_mm_payloads = MethodType(
            SpectreDraftSchedulerMixin._recv_and_store_mm_payloads, mixin
        )
        mixin._process_control_message(
            [SpectreRequest(request_id="late-m", action=SpectreAction.FINISH)]
        )
        payload = SpectreMMPayload(
            rid="late-m",
            padded_input_ids=[1, 2],
            mm_items=[serialize_mm_item(_make_image_item())],
        )
        mixin.spectre_mm_receiver = MagicMock()
        mixin.spectre_mm_receiver.use_shm = False
        mixin.spectre_mm_receiver.recv_all.return_value = ([payload], 0)
        mixin._recv_and_store_mm_payloads()
        metrics.increment_spectre_mm_payloads_dropped.assert_called_once_with(
            reason="finished_rid"
        )
        metrics.increment_spectre_mm_payloads_received.assert_not_called()
        self.assertNotIn("late-m", mixin._pending_mm)

    def test_prewarm_metrics_success_and_failure(self):
        mixin = self._make_mixin()
        metrics = self._enable_mm_metrics(mixin)
        item = _make_image_item(pad_value=1_000_701)
        payload = SpectreMMPayload(
            rid="pw-ok",
            padded_input_ids=[1, 2],
            mm_items=[serialize_mm_item(item)],
        )
        mixin._pending_mm["pw-ok"] = payload
        mixin.tp_worker.model_runner.model = MagicMock()
        mixin.tp_worker.model_runner.model.get_image_feature = lambda items: torch.randn(
            4, 8
        )
        mixin._maybe_prewarm_pending_mm()
        metrics.increment_spectre_mm_prewarm.assert_called_once_with(result="success")

        mixin2 = self._make_mixin()
        metrics2 = self._enable_mm_metrics(mixin2)
        payload2 = SpectreMMPayload(
            rid="pw-fail",
            padded_input_ids=[1, 2],
            mm_items=[serialize_mm_item(_make_image_item())],
        )
        mixin2._pending_mm["pw-fail"] = payload2
        mixin2.tp_worker.model_runner.model = MagicMock()

        def boom(items):
            raise RuntimeError("vit oom")

        mixin2.tp_worker.model_runner.model.get_image_feature = boom
        mixin2._maybe_prewarm_pending_mm()
        metrics2.increment_spectre_mm_prewarm.assert_called_once_with(result="failure")

    def test_wait_timeout_degrade_metric(self):
        mixin = self._make_mixin()
        metrics = self._enable_mm_metrics(mixin)
        draft_req = self._draft_request("wto", mm_ref="wto")
        module = "sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin"
        with patch.dict(os.environ, {"SPECTRE_MM_WAIT_MS": "100"}):
            clear_spectre_mm_env_cache()
            with patch(f"{module}.time.time", return_value=1000.0):
                self.assertFalse(mixin._create_new_draft_req(draft_req))
                metrics.increment_spectre_mm_degrades.assert_not_called()
            with patch(f"{module}.time.time", return_value=1000.2):
                self.assertTrue(mixin._create_new_draft_req(draft_req))
        metrics.increment_spectre_mm_degrades.assert_called_once_with(
            reason="wait_timeout"
        )
        clear_spectre_mm_env_cache()

    def test_mismatch_is_not_wait_timeout(self):
        mixin = self._make_mixin()
        metrics = self._enable_mm_metrics(mixin)
        payload = SpectreMMPayload(
            rid="mmis",
            padded_input_ids=[1_000_001, 1_000_001, 9],
            mm_items=[serialize_mm_item(_make_image_item(pad_value=1_000_001))],
        )
        mixin._pending_mm["mmis"] = payload
        draft_req = self._draft_request(
            "mmis", input_ids=[1_000_001, 9, 9], mm_ref="mmis"
        )
        with patch(
            "sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin.Req"
        ):
            self.assertTrue(mixin._create_new_draft_req(draft_req))
        metrics.increment_spectre_mm_degrades.assert_called_once_with(
            reason="padded_mismatch"
        )
        wait_calls = [
            c
            for c in metrics.increment_spectre_mm_degrades.call_args_list
            if c.kwargs.get("reason") == "wait_timeout" or (c.args and c.args[0] == "wait_timeout")
        ]
        self.assertEqual(wait_calls, [])

    def test_release_pending_bytes_goes_to_zero(self):
        mixin = self._make_mixin()
        metrics = self._enable_mm_metrics(mixin)
        payload = SpectreMMPayload(
            rid="bytes1",
            padded_input_ids=[1, 2],
            mm_items=[serialize_mm_item(_make_image_item())],
        )
        mixin._pending_mm["bytes1"] = payload
        mixin._refresh_pending_mm_bytes()
        self.assertGreater(metrics.set_spectre_mm_pending_bytes.call_args[0][0], 0)
        mixin._release_mm_for_rid("bytes1", mark_finished=False)
        self.assertEqual(metrics.set_spectre_mm_pending_bytes.call_args[0][0], 0)


class TestSpectreTargetMmSend(CustomTestCase):
    def _make_target(self):
        from sglang.srt.speculative.spectre.verifier.spectre_target_scheduler_mixin import (
            DraftCircuitBreaker,
            SchedulerSpectreTargetMixin,
        )

        target = SchedulerSpectreTargetMixin.__new__(SchedulerSpectreTargetMixin)
        target.spec_algorithm = MagicMock()
        target.spec_algorithm.is_spectre.return_value = True
        target.server_args = MagicMock()
        target.server_args.spectre_role = "target"
        target.server_args.spectre_reject_interval = 0
        target.tp_size = 1
        target.tp_rank = 0
        target.spectre_mm_sender = MagicMock()
        target.spectre_mm_sender.send.return_value = "sent"
        target.draft_circuit_breaker = DraftCircuitBreaker()
        target.is_rejected = False
        target.forward_ct = 1
        target.current_scheduler_metrics_enabled = False
        target.metrics_collector = None
        return target

    def _vl_req(self, *, feature_alive=True, spec_cnt=0, rid="t1"):
        item = _make_image_item()
        if not feature_alive:
            item.feature = None
            item.precomputed_embeddings = None
        req = MagicMock()
        req.rid = rid
        req.spec_cnt = spec_cnt
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = []
        req.cur_drafts = []
        req.sampling_params = None
        req.multimodal_inputs = MultimodalInputs(mm_items=[item])
        return req

    def test_open_skips_send(self):
        from sglang.srt.speculative.spectre.verifier.spectre_target_scheduler_mixin import (
            DraftCircuitBreaker,
        )

        target = self._make_target()
        target.draft_circuit_breaker.state = DraftCircuitBreaker.OPEN
        target.maybe_send_spectre_mm(self._vl_req())
        target.spectre_mm_sender.send.assert_not_called()

    def test_closed_sends_when_features_alive(self):
        target = self._make_target()
        target.maybe_send_spectre_mm(self._vl_req())
        target.spectre_mm_sender.send.assert_called_once()

    def test_no_features_skips_send(self):
        target = self._make_target()
        target.maybe_send_spectre_mm(self._vl_req(feature_alive=False))
        target.spectre_mm_sender.send.assert_not_called()

    def test_send_batch_resends_on_half_open(self):
        from collections import defaultdict

        from sglang.srt.speculative.spectre.verifier.spectre_target_scheduler_mixin import (
            DraftCircuitBreaker,
        )

        target = self._make_target()
        target.draft_circuit_breaker.state = DraftCircuitBreaker.HALF_OPEN
        target.req_to_draft_token = defaultdict(dict)
        target.zmq_communicator = MagicMock()
        target._zmq_send = MagicMock()
        target.maybe_send_spectre_mm = MagicMock()
        req = self._vl_req(spec_cnt=4, rid="half1")
        batch = MagicMock()
        batch.reqs = [req]
        target.send_batch_draft_requests(batch, 5)
        target.maybe_send_spectre_mm.assert_called_once_with(req, force=True)

    def test_send_batch_spec_cnt0_does_not_force(self):
        from collections import defaultdict

        from sglang.srt.speculative.spectre.verifier.spectre_target_scheduler_mixin import (
            DraftCircuitBreaker,
        )

        target = self._make_target()
        target.draft_circuit_breaker.state = DraftCircuitBreaker.CLOSED
        target.req_to_draft_token = defaultdict(dict)
        target.zmq_communicator = MagicMock()
        target._zmq_send = MagicMock()
        target.maybe_send_spectre_mm = MagicMock()
        req = self._vl_req(spec_cnt=0, rid="first")
        batch = MagicMock()
        batch.reqs = [req]
        target.send_batch_draft_requests(batch, 5)
        target.maybe_send_spectre_mm.assert_called_once_with(req, force=False)

    def test_second_send_skipped_without_force(self):
        target = self._make_target()
        req = self._vl_req(rid="once")
        target.maybe_send_spectre_mm(req)
        target.spectre_mm_sender.send.assert_called_once()
        target.spectre_mm_sender.send.reset_mock()
        target.maybe_send_spectre_mm(req)
        target.spectre_mm_sender.send.assert_not_called()

    def test_force_resend_even_if_already_sent(self):
        target = self._make_target()
        req = self._vl_req(rid="force1")
        target.maybe_send_spectre_mm(req)
        target.spectre_mm_sender.send.reset_mock()
        target.maybe_send_spectre_mm(req, force=True)
        target.spectre_mm_sender.send.assert_called_once()

    def test_send_batch_spec_cnt0_skips_if_already_sent(self):
        from collections import defaultdict

        target = self._make_target()
        target.req_to_draft_token = defaultdict(dict)
        target.zmq_communicator = MagicMock()
        target._zmq_send = MagicMock()
        req = self._vl_req(spec_cnt=0, rid="once")
        target.maybe_send_spectre_mm(req)
        target.spectre_mm_sender.send.reset_mock()
        batch = MagicMock()
        batch.reqs = [req]
        target.send_batch_draft_requests(batch, 5)
        target.spectre_mm_sender.send.assert_not_called()

    def test_notify_clears_sent_rid(self):
        from sglang.srt.speculative.spectre.spectre_protocol import SpectreAction

        target = self._make_target()
        target.req_to_draft_token = {"t1": {}}
        target.zmq_communicator = None
        target._spectre_mm_sent_rids = {"t1"}
        target.notify_draft_request_finished_or_aborted(
            self._vl_req(rid="t1"), SpectreAction.FINISH
        )
        self.assertNotIn("t1", target._spectre_mm_sent_rids)

    def test_send_batch_skips_resend_when_not_full_context(self):
        from collections import defaultdict

        from sglang.srt.speculative.spectre.verifier.spectre_target_scheduler_mixin import (
            DraftCircuitBreaker,
        )

        target = self._make_target()
        target.draft_circuit_breaker.state = DraftCircuitBreaker.CLOSED
        target.req_to_draft_token = defaultdict(dict)
        target.zmq_communicator = MagicMock()
        target._zmq_send = MagicMock()
        target.maybe_send_spectre_mm = MagicMock()
        req = self._vl_req(spec_cnt=2, rid="dec1")
        batch = MagicMock()
        batch.reqs = [req]
        target.send_batch_draft_requests(batch, 5)
        target.maybe_send_spectre_mm.assert_not_called()

    def test_send_metrics_sent_and_queue_full(self):
        target = self._make_target()
        target.current_scheduler_metrics_enabled = True
        target.metrics_collector = MagicMock()
        target.maybe_send_spectre_mm(self._vl_req())
        target.metrics_collector.increment_spectre_mm_payloads_sent.assert_called_once()
        target.metrics_collector.increment_spectre_mm_payloads_dropped.assert_not_called()

        target.metrics_collector.reset_mock()
        target.spectre_mm_sender.send.return_value = "queue_full"
        target.maybe_send_spectre_mm(self._vl_req(rid="t2"))
        target.metrics_collector.increment_spectre_mm_payloads_sent.assert_not_called()
        target.metrics_collector.increment_spectre_mm_payloads_dropped.assert_called_once_with(
            reason="queue_full"
        )

    def test_send_success_increments_sent(self):
        target = self._make_target()
        target.current_scheduler_metrics_enabled = True
        target.metrics_collector = MagicMock()
        target.maybe_send_spectre_mm(self._vl_req())
        target.metrics_collector.increment_spectre_mm_payloads_sent.assert_called_once()
        target.metrics_collector.increment_spectre_mm_payloads_dropped.assert_not_called()

    def test_send_queue_full_increments_drop(self):
        target = self._make_target()
        target.current_scheduler_metrics_enabled = True
        target.metrics_collector = MagicMock()
        target.spectre_mm_sender.send.return_value = "queue_full"
        target.maybe_send_spectre_mm(self._vl_req())
        target.metrics_collector.increment_spectre_mm_payloads_sent.assert_not_called()
        target.metrics_collector.increment_spectre_mm_payloads_dropped.assert_called_once_with(
            reason="queue_full"
        )


if __name__ == "__main__":
    unittest.main()
