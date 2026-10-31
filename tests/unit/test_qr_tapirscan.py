"""Tapirscan adapter behavior without requiring a native decoder in unit tests."""

import base64
import io
import sys
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from effgen.tools.builtin.qr_read import QRReadTool, _read_qr

Image = pytest.importorskip("PIL.Image", reason="Install effgen[qr-tapirscan] for image tests")


def image_base64():
    stream = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def test_tapirscan_preserves_multiple_results_and_geometry(monkeypatch):
    rect = namedtuple("Rect", "left top width height")
    scanner = MagicMock()
    scanner.__enter__.return_value.scan.return_value.barcodes = [
        SimpleNamespace(text="hello", format="QRCode", rect=rect(1, 2, 3, 4)),
        SimpleNamespace(text="1234", format="ITF", rect=rect(5, 6, 7, 8)),
        SimpleNamespace(text="compact", format="RMQRCode", rect=rect(0, 0, 18, 7)),
    ]
    factory = MagicMock(return_value=scanner)
    monkeypatch.setitem(sys.modules, "tapirscan", SimpleNamespace(Scanner=factory))
    result = _read_qr(image_base64=image_base64(), backend="tapirscan")
    assert result["success"]
    assert result["count"] == result["data"]["count"] == 3
    assert [code["type"] for code in result["codes"]] == ["QRCODE", "I25", "RMQRCODE"]
    assert result["codes"][0]["rect"] == dict(left=1, top=2, width=3, height=4)
    factory.assert_called_once_with(mode="medium", formats="all")
    scanner.__exit__.assert_called_once()


def test_missing_tapirscan_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "tapirscan", None)
    result = _read_qr(image_base64=image_base64(), backend="tapirscan")
    assert not result["success"]
    assert "effgen[qr-tapirscan]" in result["error"]


def test_path_confinement_happens_before_decoding(monkeypatch):
    from effgen.tools.builtin import qr_read

    confined = MagicMock(side_effect=qr_read.PathNotAllowedError("outside allowed roots"))
    decode = MagicMock()
    monkeypatch.setattr(qr_read, "confine_path", confined)
    monkeypatch.setattr(qr_read, "_read_qr_with_tapirscan", decode)
    result = _read_qr("outside.png", allowed_dirs=[], backend="tapirscan")
    assert not result["success"]
    assert result["codes"] == []
    confined.assert_called_once_with("outside.png", [])
    decode.assert_not_called()


def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        QRReadTool(backend="unknown")
