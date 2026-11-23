"""Original image bytes of a request, kept next to the processed features.

The tokenizer calls this only when the speculative algorithm asks for it
(``SpeculativeAlgorithm.keeps_source_image_bytes``). Nothing here depends on
a specific algorithm.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Sequence
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)


def _b64decode(text: str) -> bytes:
    try:
        import pybase64

        return pybase64.b64decode(text, validate=True)
    except ImportError:
        import base64

        return base64.b64decode(text, validate=True)


def _read_file(path: str) -> Optional[bytes]:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def _one_image_bytes(item: Any) -> Optional[bytes]:
    from sglang.srt.utils.common import ImageData, get_image_bytes

    if isinstance(item, ImageData):
        item = item.url
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    if not isinstance(item, str) or not item:
        return None
    if item.startswith(("http://", "https://")):
        # Fetching again would download the image twice.
        return None
    if item.startswith("file://"):
        return _read_file(unquote(urlparse(item).path))
    if item.startswith("data:"):
        return get_image_bytes(item)
    if os.path.isfile(item):
        return _read_file(item)
    # Bare base64. A JPEG's base64 starts with "/9j/", so a leading "/" is
    # not a path by itself.
    try:
        return _b64decode(item)
    except Exception:
        return None


def collect_source_image_bytes(image_data: Optional[Sequence[Any]]) -> Optional[List[bytes]]:
    """Encoded bytes of every image, in request order, or None.

    Data URIs, bare base64, raw bytes, local paths and ``file://`` resolve.
    HTTP(S) URLs, decoded images and anything else give None for the whole
    request, so a caller never sees a partial list.
    """
    if not image_data:
        return None
    out: List[bytes] = []
    for item in image_data:
        try:
            raw = _one_image_bytes(item)
        except Exception:
            logger.debug("source image bytes unavailable", exc_info=True)
            return None
        if not raw:
            return None
        out.append(raw)
    return out
