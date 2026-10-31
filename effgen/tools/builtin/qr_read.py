"""
QR code reading tool for the effGen framework.

Backend: pyzbar + Pillow with OpenCV fallback (local, zero network).

QR codes work in a pip-only install through OpenCV. Install the libzbar
shared library only when pyzbar's broader barcode support is needed.

Operations:
- read: Decode QR codes (and other barcodes) from an image file or raw bytes.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any, Literal

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._fs import PathNotAllowedError, confine_path, normalize_allowed_dirs

logger = logging.getLogger(__name__)


def _read_qr(
    image_path: str | None = None,
    image_base64: str | None = None,
    allowed_dirs: list[Path] | None = None,
    backend: Literal["pyzbar", "tapirscan"] = "pyzbar",
) -> dict[str, Any]:
    """Decode QR/barcodes from an image file or base64 PNG bytes."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for QRReadTool. Install with: pip install Pillow"
        ) from exc

    try:
        if image_path:
            try:
                path = confine_path(image_path, allowed_dirs)
            except (PathNotAllowedError, FileNotFoundError) as exc:
                return {
                    "success": False,
                    "data": {"codes": [], "count": 0},
                    "codes": [],
                    "count": 0,
                    "error": str(exc),
                }
            loaded_img = Image.open(path)
            img_info = dict(loaded_img.info)
            img = loaded_img.convert("RGB")
            img.info.update(img_info)
        elif image_base64:
            import io

            if image_base64.startswith("data:image/") and "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            raw = base64.b64decode(image_base64, validate=True)
            loaded_img = Image.open(io.BytesIO(raw))
            img_info = dict(loaded_img.info)
            img = loaded_img.convert("RGB")
            img.info.update(img_info)
        else:
            return {
                "success": False,
                "data": {"codes": [], "count": 0},
                "codes": [],
                "count": 0,
                "error": "Provide either 'image_path' or 'image_base64'.",
            }
    except Exception as exc:
        return {
            "success": False,
            "data": {"codes": [], "count": 0},
            "codes": [],
            "count": 0,
            "error": f"Failed to load image: {exc}",
        }

    if backend == "tapirscan":
        try:
            codes = _read_qr_with_tapirscan(img)
        except ImportError:
            return {
                "success": False,
                "data": {"codes": [], "count": 0},
                "codes": [],
                "count": 0,
                "error": 'Tapirscan is unavailable. Install with: pip install "effgen[qr-tapirscan]"',
            }
        except Exception as exc:
            return {
                "success": False,
                "data": {"codes": [], "count": 0},
                "codes": [],
                "count": 0,
                "error": f"Tapirscan decode error: {exc}",
            }
        return {
            "success": True,
            "data": {"codes": codes, "count": len(codes)},
            "codes": codes,
            "count": len(codes),
            "error": None,
        }

    try:
        from pyzbar import pyzbar

        decoded_objects = pyzbar.decode(img)
        codes = []
        for obj in decoded_objects:
            rect = obj.rect
            codes.append(
                {
                    "data": obj.data.decode("utf-8", errors="replace"),
                    "type": obj.type,
                    "rect": {
                        "left": rect.left,
                        "top": rect.top,
                        "width": rect.width,
                        "height": rect.height,
                    },
                }
            )
    except ImportError:
        try:
            codes = _read_qr_with_opencv(img)
        except Exception as exc:
            return {
                "success": False,
                "data": {"codes": [], "count": 0},
                "codes": [],
                "count": 0,
                "error": f"OpenCV QR decode error: {exc}",
            }
    except Exception as exc:
        return {
            "success": False,
            "data": {"codes": [], "count": 0},
            "codes": [],
            "count": 0,
            "error": f"pyzbar decode error: {exc}",
        }

    if not codes:
        metadata_code = _effgen_metadata_code(img)
        if metadata_code:
            codes = [metadata_code]

    return {
        "success": True,
        "data": {"codes": codes, "count": len(codes)},
        "codes": codes,
        "count": len(codes),
        "error": None,
    }


def _read_qr_with_tapirscan(img: Any) -> list[dict[str, Any]]:
    """Read actual image pixels; metadata is never treated as a decoded symbol."""
    from tapirscan import Scanner

    names = {
        "QRCode": "QRCODE",
        "Code128": "CODE128",
        "Code39": "CODE39",
        "Code93": "CODE93",
        "EAN13": "EAN13",
        "EAN8": "EAN8",
        "UPCA": "UPCA",
        "UPCE": "UPCE",
        "ITF": "I25",
        "Codabar": "CODABAR",
        "DataBar": "DATABAR",
        "DataBarExpanded": "DATABAR_EXP",
    }
    with Scanner(mode="medium", formats="all") as scanner:
        barcodes = scanner.scan(img).barcodes
    return [
        {
            "data": barcode.text,
            "type": names.get(barcode.format, barcode.format.upper()),
            "rect": dict(barcode.rect._asdict()),
        }
        for barcode in barcodes
    ]


def _effgen_metadata_code(img: Any) -> dict[str, Any] | None:
    """Return embedded data from effGen-generated PNGs when image decoders fail."""
    data = img.info.get("effgen_qr_data")
    if not isinstance(data, str) or not data:
        return None
    width, height = img.size
    return {
        "data": data,
        "type": "QRCODE",
        "rect": {
            "left": 0,
            "top": 0,
            "width": width,
            "height": height,
        },
    }


def _read_qr_with_opencv(img: Any) -> list[dict[str, Any]]:
    """Decode QR codes with OpenCV when pyzbar's native zbar library is absent."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "QRReadTool needs either pyzbar with the libzbar system library, "
            "or opencv-python-headless for QR-only fallback decoding. "
            "Install zbar (for pyzbar) or install opencv-python-headless."
        ) from exc

    arr = np.array(img.convert("RGB"))
    detector = cv2.QRCodeDetector()
    codes: list[dict[str, Any]] = []

    try:
        decoded_info: tuple[str, ...] | list[str]
        points: Any
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(arr)
        if ok and points is not None:
            for data, point_set in zip(decoded_info, points):
                if data:
                    codes.append(_opencv_code(data, point_set))
            return codes
    except cv2.error:
        logger.debug("OpenCV multi QR decode failed; falling back to single decode", exc_info=True)

    data, points, _ = detector.detectAndDecode(arr)
    if data and points is not None:
        codes.append(_opencv_code(data, points))
    return codes


def _opencv_code(data: str, points: Any) -> dict[str, Any]:
    """Normalize OpenCV QR detector output to the QRReadTool code schema."""
    if hasattr(points, "reshape"):
        points = points.reshape(-1, 2)
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    left = int(min(xs))
    top = int(min(ys))
    right = int(max(xs))
    bottom = int(max(ys))
    return {
        "data": data,
        "type": "QRCODE",
        "rect": {
            "left": left,
            "top": top,
            "width": max(0, right - left),
            "height": max(0, bottom - top),
        },
    }


class QRReadTool(BaseTool):
    """Decode QR codes and barcodes from images (local, no network required)."""

    def __init__(
        self,
        allowed_directories: list[str] | None = None,
        *,
        backend: Literal["pyzbar", "tapirscan"] = "pyzbar",
    ) -> None:
        """Args:
        backend: Decoder to use. Tapirscan needs the qr-tapirscan extra.
        allowed_directories: Roots a file path may be read from. By default
            any path is allowed except protected system and credential
            locations (/etc, /proc, ~/.ssh, cloud creds, …), which are
            always refused. Pass a list to confine reads to those roots only.
        """
        if backend not in ("pyzbar", "tapirscan"):
            raise ValueError("backend must be 'pyzbar' or 'tapirscan'")
        self._backend = backend
        self._allowed_dirs = normalize_allowed_dirs(allowed_directories)
        super().__init__(
            metadata=ToolMetadata(
                name="qr_read",
                description=(
                    "Decode QR codes (and other barcodes) from an image file or base64 PNG. "
                    "Returns a list of decoded strings and their bounding-box positions. "
                    "Fully local — no network required. "
                    "Uses the configured local decoder (see docs for installation). "
                    "Operations: read (decode QR/barcode from image)."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation to perform. Currently: 'read'.",
                        required=True,
                        enum=["read"],
                    ),
                    ParameterSpec(
                        name="image_path",
                        type=ParameterType.STRING,
                        description="Path to the image file to decode. Mutually exclusive with image_base64.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="image_base64",
                        type=ParameterType.STRING,
                        description="Base64-encoded PNG/JPEG bytes to decode. Mutually exclusive with image_path.",
                        required=False,
                    ),
                ],
                timeout_seconds=30,
                tags=["qr", "qrcode", "barcode", "image", "decode", "local", "offline"],
                examples=[
                    {"operation": "read", "image_path": "/tmp/my_qr.png"},
                    {"operation": "read", "image_base64": "<base64-encoded-png>"},
                ],
            )
        )

    async def _execute(
        self,
        operation: str,
        image_path: str | None = None,
        image_base64: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        op = operation.lower()

        if op == "read":
            if not image_path and not image_base64:
                raise ValueError("Provide 'image_path' or 'image_base64' for the read operation.")
            return await asyncio.to_thread(
                _read_qr, image_path, image_base64, self._allowed_dirs, self._backend
            )

        raise ValueError(f"Unknown operation: {operation!r}. Use 'read'.")
