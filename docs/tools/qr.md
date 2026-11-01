# QR Code Tools

effGen ships two local QR tools that require **no network connection**:

| Tool | Purpose |
|------|---------|
| `QRGenerateTool` | Generate a QR code PNG from any text or URL |
| `QRReadTool` | Decode QR codes (and other barcodes) from an image |

---

## Decoder backends

`QRReadTool` first tries [pyzbar](https://github.com/NaturalHistoryMuseum/pyzbar), which can decode
QR codes and common barcodes when the **zbar** shared library is installed. If zbar is not present,
the tool falls back to OpenCV's QR detector for QR-only decoding. This keeps `pip install "effgen[qr]"`
usable in fresh Python environments without an OS package step.

| Platform | Install command |
|----------|----------------|
| Ubuntu / Debian | `sudo apt-get install libzbar0` |
| Fedora / RHEL / CentOS | `sudo dnf install zbar` |
| macOS (Homebrew) | `brew install zbar` |
| conda (any OS) | `conda install -c conda-forge zbar` |
| Windows | Download the [ZBar Windows installer](https://zbar.sourceforge.net/) |

Install zbar only if you need non-QR barcode formats through pyzbar. `QRGenerateTool` uses only
pure-Python `qrcode` + `Pillow` and has **no system dependency**.

### Tapirscan (optional)

[Tapirscan](https://github.com/kleinicke/tapirscan) reads QR codes and the common 1D and 2D barcode
formats from a pip install, with no system library:

```bash
pip install "effgen[qr-tapirscan]"
```

```python
QRReadTool(backend="tapirscan")
```

Results have the same shape as with pyzbar.

---

## Python dependencies

```bash
pip install "effgen[qr]"
# or manually:
pip install "qrcode[pil]>=7.4" "pyzbar>=0.1.9" "opencv-python-headless>=4.8.0" "Pillow>=12.3.0"
```

---

## QRGenerateTool

**Module:** `effgen.tools.builtin.qr_generate`

### Operations

#### `generate`

Generate a QR code from data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | string | **required** | Text or URL to encode |
| `size` | integer | `200` | Output image size in pixels (square) |
| `error_correction` | string | `"M"` | Error correction level: `L` (7%), `M` (15%), `Q` (25%), `H` (30%) |
| `output_path` | string | `None` | Save PNG to this path (optional) |
| `data_url_return` | boolean | `false` | Also return a `data:image/png;base64,...` URL |
| `include_base64` | boolean | *auto* | Include raw base64 in result. Auto-omitted when `output_path` is set (keeps payload small for short-context LLMs). Set `true` to force inclusion. |

For dense payloads, the tool may increase the final image dimensions above the requested
`size` so scanners still have enough pixels per QR module. The response includes
`requested_size_px`, `size_px`, `size_adjusted`, `qr_version`, and `module_count` so callers
can see exactly what was generated.

**Returns:**

```json
{
  "success": true,
  "qr_base64": "<base64-encoded PNG>",
  "size_px": 200,
  "requested_size_px": 200,
  "size_adjusted": false,
  "qr_version": 2,
  "module_count": 25,
  "error_correction": "M",
  "saved_path": null,
  "error": null
}
```

### Example

```python
import asyncio
from effgen.tools.builtin.qr_generate import QRGenerateTool

tool = QRGenerateTool()
result = asyncio.run(tool._execute(
    operation="generate",
    data="https://effgen.org",
    size=300,
    error_correction="H",
    output_path="/tmp/effgen_qr.png",
    include_base64=True,
))
print(result["saved_path"])   # /tmp/effgen_qr.png
print(result["qr_base64"][:40])  # base64 preview
```

---

## QRReadTool

**Module:** `effgen.tools.builtin.qr_read`

### Operations

#### `read`

Decode QR codes (and other barcodes such as EAN-13, Code-128) from an image.

Provide **one** of:

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_path` | string | Path to a PNG/JPEG/BMP image file |
| `image_base64` | string | Base64-encoded PNG/JPEG bytes |

**Returns:**

```json
{
  "success": true,
  "count": 1,
  "codes": [
    {
      "data": "https://effgen.org",
      "type": "QRCODE",
      "rect": { "left": 32, "top": 32, "width": 236, "height": 236 }
    }
  ],
  "error": null
}
```

If no codes are found `count` is `0` and `codes` is `[]` — the call still returns `success: true`.

### Example

```python
import asyncio
from effgen.tools.builtin.qr_read import QRReadTool

tool = QRReadTool()
result = asyncio.run(tool._execute(
    operation="read",
    image_path="/tmp/effgen_qr.png",
))
for code in result["codes"]:
    print(code["data"], code["type"])
```

---

## Round-trip example

```python
import asyncio, base64
from effgen.tools.builtin.qr_generate import QRGenerateTool
from effgen.tools.builtin.qr_read import QRReadTool

async def demo():
    gen = QRGenerateTool()
    read = QRReadTool()

    gen_result = await gen._execute(operation="generate", data="HELLO_EFFGEN")
    b64 = gen_result["qr_base64"]

    read_result = await read._execute(operation="read", image_base64=b64)
    print(read_result["codes"][0]["data"])  # → HELLO_EFFGEN

asyncio.run(demo())
```

---

## Error handling

| Scenario | Behaviour |
|----------|-----------|
| `data` is empty in `generate` | `ValueError` raised |
| Invalid `error_correction` level | `ValueError` raised |
| `image_path` does not exist | `success: false`, descriptive `error` message |
| Invalid base64 bytes | `success: false`, descriptive `error` message |
| Blank / no-QR image | `success: true`, `count: 0`, `codes: []` |
| `libzbar` not installed | Falls back to OpenCV QR decoding when `opencv-python-headless` is installed |

---

## Preset integration

Both tools are included in the **`general`** preset:

```python
from effgen.presets import create_agent

agent = create_agent("general", model=my_model)
# agent now has qr_generate and qr_read tools available
```
