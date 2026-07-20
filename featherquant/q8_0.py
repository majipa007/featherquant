"""Q8_0 block quantization, byte-compatible with llama.cpp
``quantize_row_q8_0_ref``.

Layout per block of 32 elements: one fp16 scale ``d`` followed by 32 int8
quantized values.  The scale is computed in float32 (``amax / 127``) and
stored as fp16, but the reciprocal ``1/d`` uses the float32 value — exactly
matching the C reference.  Rounding is half-away-from-zero (C ``roundf``),
NOT numpy's default half-to-even.
"""
import numpy as np

# Number of weight elements per Q8_0 block.
BLOCK = 32
# Packed size of one block: 2-byte fp16 scale + 32 int8 values.
TYPE_SIZE = 34


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Convert uint16 bf16 bit patterns to float32.

    bf16 is the top 16 bits of an IEEE float32, so widening to uint32 and
    shifting left by 16 reconstructs the exact float32 value.
    """
    try:
        return (raw.astype(np.uint32) << 16).view(np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bf16_to_f32 expects a uint16 array: {exc}") from exc


def quantize_q8_0(x: np.ndarray) -> bytes:
    """Quantize a 1-D float32 array (length % 32 == 0) to packed Q8_0 bytes."""
    # Validate up front: silent shape/dtype mismatches would corrupt output.
    if x.dtype != np.float32 or x.ndim != 1 or x.size % BLOCK != 0:
        raise ValueError(
            f"quantize_q8_0 expects 1-D float32 with size % {BLOCK} == 0, "
            f"got dtype={x.dtype} ndim={x.ndim} size={x.size}"
        )
    blocks = x.reshape(-1, BLOCK)
    # Per-block scale: amax / 127, computed in float32 like the C reference.
    amax = np.abs(blocks).max(axis=1)
    d = (amax / np.float32(127.0)).astype(np.float32)
    # Reciprocal of the float32 scale; zero blocks get inv = 0 -> all-zero q.
    inv = np.divide(np.float32(1.0), d, out=np.zeros_like(d), where=d > 0)
    v = blocks * inv[:, None]
    # Round half away from zero (C roundf semantics).
    q = np.trunc(v + np.copysign(np.float32(0.5), v)).astype(np.int8)
    # Pack: bytes 0-1 = fp16 scale, bytes 2-33 = int8 values.
    out = np.empty((blocks.shape[0], TYPE_SIZE), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def dequantize_q8_0(raw: bytes) -> np.ndarray:
    """Unpack Q8_0 bytes back to float32 (for tests and validation)."""
    if len(raw) % TYPE_SIZE != 0:
        raise ValueError(f"Q8_0 data must be a multiple of {TYPE_SIZE} bytes")
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, TYPE_SIZE)
    # .copy() gives an owned, aligned buffer so the fp16 view is safe.
    d = b[:, :2].copy().view(np.float16).astype(np.float32)  # shape (n, 1)
    q = b[:, 2:].view(np.int8).astype(np.float32)
    return (q * d).reshape(-1)
