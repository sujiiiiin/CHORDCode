import warp as wp

def from_torch_safe(t, dtype=None, requires_grad=None, grad=None):
    """Warp 1.16-compatible zero-copy wrapper around ``wp.from_torch``."""
    return wp.from_torch(
        t.contiguous(),
        dtype=dtype,
        requires_grad=requires_grad,
        grad=grad,
    )
