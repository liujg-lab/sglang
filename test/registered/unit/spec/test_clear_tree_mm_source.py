"""CPU checks for ClearTree's image-source PREFILL path.

The Target ships the original image bytes; the Draft decodes them, resizes to
the Target's grid and runs its own image processor. Other algorithms never
keep the bytes.
"""

import base64
import io
import os
import tempfile
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from sglang.srt.managers.mm_source_bytes import collect_source_image_bytes
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.speculative.clear_tree_remote import clear_tree_transport as transport
from sglang.srt.speculative.clear_tree_remote.clear_tree_mm_source import (
    ClearImageSourcePayload,
    materialize,
    payload_from_pickleable,
    source_payload,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_protocol import (
    CLEAR_PROTOCOL_VERSION,
    ClearAction,
    ClearBatchRequest,
    ClearDraftRequest,
    supported_clear_protocol,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_scheduler import (
    ClearTreeSchedulerMixin,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.standalone_remote.sr_mm_payload import SRMMPayload
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

PATCH = 16


def _jpeg(width: int, height: int, seed: int = 0) -> bytes:
    gen = torch.Generator().manual_seed(seed)
    pixels = torch.randint(0, 256, (height, width, 3), generator=gen, dtype=torch.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels.numpy()).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class _FakeImageProcessor:
    """Qwen-like: grid from the image size, one 3*2*P*P row per patch."""

    patch_size = PATCH

    def __init__(self, grid_bump: int = 0):
        self.grid_bump = grid_bump
        self.calls = []

    def __call__(self, images, do_resize=True, return_tensors=None):
        self.calls.append({"do_resize": do_resize, "sizes": [im.size for im in images]})
        grids, rows = [], []
        for im in images:
            w, h = im.size
            gh, gw = h // PATCH + self.grid_bump, w // PATCH
            grids.append([1, gh, gw])
            rows.append(torch.zeros(gh * gw, 3 * 2 * PATCH * PATCH))
        return {
            "pixel_values": torch.cat(rows, dim=0),
            "image_grid_thw": torch.tensor(grids),
        }


def _target_req(images, grids, rid="r1"):
    feature = torch.randn(sum(g[1] * g[2] for g in grids), 3 * 2 * PATCH * PATCH)
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=12345,
        pad_value=777,
        offsets=[(3, 10)],
        feature=feature,
        model_specific_data={"image_grid_thw": torch.tensor(grids)},
    )
    mm = MultimodalInputs(mm_items=[item])
    mm.im_token_id = 151655
    return SimpleNamespace(rid=rid, source_image_bytes=images, multimodal_inputs=mm)


def _batch(rid="r1"):
    return ClearBatchRequest(
        session_id="s",
        rpc_seq=1,
        action=ClearAction.PREFILL,
        reqs=[ClearDraftRequest(rid=rid, padded_input_ids=[1, 2, 3])],
    )


class TestSourceSwitch(CustomTestCase):
    def test_only_clear_target_keeps_bytes(self):
        clear = SpeculativeAlgorithm.CLEAR_TREE_REMOTE
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_CLEAR_MM_SOURCE", None)
            self.assertTrue(
                clear.keeps_source_image_bytes(SimpleNamespace(clear_tree_role="target"))
            )
            self.assertFalse(
                clear.keeps_source_image_bytes(SimpleNamespace(clear_tree_role="draft"))
            )
            for algo in (
                SpeculativeAlgorithm.OPT_TREE_REMOTE,
                SpeculativeAlgorithm.STANDALONE_REMOTE,
                SpeculativeAlgorithm.SPECTRE,
                SpeculativeAlgorithm.NONE,
            ):
                args = SimpleNamespace(
                    clear_tree_role="target",
                    opt_tree_role="target",
                    standalone_remote_role="target",
                    spectre_role="target",
                )
                self.assertFalse(algo.keeps_source_image_bytes(args), algo)
        with patch.dict(os.environ, {"SGLANG_CLEAR_MM_SOURCE": "0"}):
            self.assertFalse(
                clear.keeps_source_image_bytes(SimpleNamespace(clear_tree_role="target"))
            )

    def test_protocol_version_two(self):
        self.assertEqual(CLEAR_PROTOCOL_VERSION, 2)
        self.assertTrue(supported_clear_protocol(2))
        self.assertFalse(supported_clear_protocol(1))


class TestCollectBytes(CustomTestCase):
    def test_resolvable_inputs(self):
        raw = _jpeg(32, 32)
        b64 = base64.b64encode(raw).decode()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.jpg")
            with open(path, "wb") as f:
                f.write(raw)
            got = collect_source_image_bytes(
                [raw, b64, "data:image/jpeg;base64," + b64, path, "file://" + path]
            )
        self.assertEqual(got, [raw] * 5)

    def test_unresolvable_gives_none(self):
        raw = _jpeg(32, 32)
        self.assertIsNone(collect_source_image_bytes(None))
        self.assertIsNone(collect_source_image_bytes([]))
        self.assertIsNone(collect_source_image_bytes(["http://example.invalid/a.jpg"]))
        self.assertIsNone(collect_source_image_bytes([Image.new("RGB", (4, 4))]))
        self.assertIsNone(collect_source_image_bytes([raw, "https://x.invalid/b.png"]))
        self.assertIsNone(collect_source_image_bytes(["/no/such/file.jpg"]))


class TestSourcePayload(CustomTestCase):
    def test_roundtrip_through_transport_and_draft(self):
        raws = [_jpeg(100, 70, 1), _jpeg(50, 90, 2)]
        grids = [[1, 4, 6], [1, 6, 2]]
        req = _target_req(raws, grids)
        payload = source_payload(req)
        self.assertIsInstance(payload, ClearImageSourcePayload)
        self.assertIsNone(payload.mm_items[0]["feature"])
        # The Target's own feature is untouched.
        self.assertIsNotNone(req.multimodal_inputs.mm_items[0].feature)

        frames = transport._pack_request(_batch(), {"r1": payload})
        sent = transport._multipart_nbytes(frames)
        feature_bytes = req.multimodal_inputs.mm_items[0].feature.numel() * 4
        self.assertLess(sent, sum(len(r) for r in raws) + 8192)
        self.assertLess(sent, feature_bytes)

        batch, mm_by_rid = transport._unpack_request(frames)
        got = mm_by_rid["r1"]
        self.assertIsInstance(got, ClearImageSourcePayload)
        self.assertEqual(got.images, raws)

        proc = _FakeImageProcessor()
        done, reason = materialize(got, SimpleNamespace(image_processor=proc))
        self.assertEqual(reason, "ok")
        self.assertIsInstance(done, SRMMPayload)
        self.assertEqual(proc.calls[0]["do_resize"], False)
        # Resized to the Target's grid: (w, h) = (gw * P, gh * P).
        self.assertEqual(proc.calls[0]["sizes"], [(6 * PATCH, 4 * PATCH), (2 * PATCH, 6 * PATCH)])
        item = done.to_multimodal_inputs().mm_items[0]
        self.assertEqual(tuple(item.feature.shape), (4 * 6 + 6 * 2, 3 * 2 * PATCH * PATCH))
        self.assertIs(done.mm_items[0]["feature"], item.feature)
        self.assertEqual(item.hash, 12345)
        self.assertEqual(item.pad_value, 777)
        self.assertEqual([tuple(o) for o in item.offsets], [(3, 10)])
        self.assertTrue(torch.equal(item.image_grid_thw, torch.tensor(grids)))
        self.assertEqual(done.im_token_id, 151655)

    def test_grid_mismatch_and_errors(self):
        req = _target_req([_jpeg(64, 64)], [[1, 4, 4]])
        payload = source_payload(req)
        done, reason = materialize(
            payload, SimpleNamespace(image_processor=_FakeImageProcessor(grid_bump=1))
        )
        self.assertIsNone(done)
        self.assertEqual(reason, "mismatch")
        broken = ClearImageSourcePayload(meta=payload.meta, images=[b"not an image"])
        done, reason = materialize(
            broken, SimpleNamespace(image_processor=_FakeImageProcessor())
        )
        self.assertIsNone(done)
        self.assertEqual(reason, "error")
        done, reason = materialize(payload, None)
        self.assertEqual((done, reason), (None, "error"))

    def test_tensor_path_when_bytes_do_not_fit(self):
        raw = _jpeg(64, 64)
        self.assertIsNone(source_payload(_target_req(None, [[1, 4, 4]])))
        self.assertIsNone(source_payload(_target_req([raw, raw], [[1, 4, 4]])))
        req = _target_req([raw], [[1, 4, 4]])
        req.multimodal_inputs.mm_items[0].modality = Modality.VIDEO
        self.assertIsNone(source_payload(req))

    def test_pickleable_roundtrip(self):
        raw = _jpeg(64, 64)
        payload = source_payload(_target_req([raw], [[1, 4, 4]]))
        back = payload_from_pickleable(payload.to_pickleable())
        self.assertIsInstance(back, ClearImageSourcePayload)
        self.assertEqual(back.images, [raw])
        plain = SRMMPayload.from_req(_target_req([raw], [[1, 4, 4]]))
        self.assertIsInstance(payload_from_pickleable(plain.to_pickleable()), SRMMPayload)

    def test_plain_payload_unchanged_on_wire(self):
        plain = SRMMPayload.from_req(_target_req(None, [[1, 4, 4]]))
        _, mm_by_rid = transport._unpack_request(
            transport._pack_request(_batch(), {"r1": plain})
        )
        self.assertIsInstance(mm_by_rid["r1"], SRMMPayload)
        self.assertIsNotNone(mm_by_rid["r1"].mm_items[0]["feature"])


class TestDraftMaterializeStep(CustomTestCase):
    def test_failed_entries_are_removed_and_reported(self):
        good = source_payload(_target_req([_jpeg(64, 64)], [[1, 4, 4]], rid="a"))
        bad = ClearImageSourcePayload(
            meta=source_payload(_target_req([_jpeg(64, 64)], [[1, 4, 4]], rid="b")).meta,
            images=[b"junk"],
        )
        plain = SRMMPayload.from_req(_target_req(None, [[1, 4, 4]], rid="c"))
        mm_by_rid = {"a": good, "b": bad, "c": plain}
        metrics = SimpleNamespace(active=False, counts=Counter())
        host = SimpleNamespace(processor=SimpleNamespace(image_processor=_FakeImageProcessor()))
        failed = ClearTreeSchedulerMixin._clear_materialize_sources(host, mm_by_rid, metrics)
        self.assertEqual(failed, {"b": "error"})
        self.assertNotIn("b", mm_by_rid)
        self.assertIsInstance(mm_by_rid["a"], SRMMPayload)
        self.assertIs(mm_by_rid["c"], plain)
        self.assertEqual(metrics.counts["mm_source_ok"], 1)
        self.assertEqual(metrics.counts["mm_source_error"], 1)


if __name__ == "__main__":
    unittest.main()
