"""Generic single-bit fault tools for any FP16 or FP32 PyTorch tensor.
Same method as the week 1 MLP (common.py), but works on any layer and either precision."""
import hashlib
import torch

INT_FOR = {torch.float16: torch.int16, torch.float32: torch.int32}


def nbits(t):
    return t.element_size() * 8


def bit_field(bit, n):
    """Sign / exponent / mantissa for bit position `bit` in an n-bit float."""
    exp_low = 10 if n == 16 else 23
    return "sign" if bit == n - 1 else ("exponent" if bit >= exp_low else "mantissa")


def fingerprint(t):
    """SHA-256 of the tensor's raw bytes (read as integers, so the bytes are exact)."""
    return hashlib.sha256(t.detach().contiguous().view(INT_FOR[t.dtype]).cpu().numpy().tobytes()).hexdigest()


def flip_bit(t, idx, bit):
    """Flip one bit IN PLACE in the tensor the model uses. Returns the original bit pattern."""
    n = nbits(t)
    flat = t.data.view(INT_FOR[t.dtype]).view(-1)       # shares memory with t
    old = int(flat[idx])
    mask = (1 << bit) if bit < n - 1 else -(1 << (n - 1))  # sign bit as a signed integer
    flat[idx] = old ^ mask
    return old


def restore(t, idx, old):
    """Write the saved pattern back (never flip a second time)."""
    t.data.view(INT_FOR[t.dtype]).view(-1)[idx] = old
