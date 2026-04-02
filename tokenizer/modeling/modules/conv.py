import torch
import torch.nn as nn


# ============================
# autopad (same padding)
# ============================
def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# ============================
# YOLO Conv (Conv-BN-Activation)
# ============================
class Conv(nn.Module):
    """Standard convolution: Conv2d + BatchNorm2d + SiLU"""

    default_act = nn.SiLU()  # Default YOLO activation.

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            k (int): Kernel size
            s (int): Stride
            g (int): Groups
            d (int): Dilation
            act (bool | nn.Module): activation
        """
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s,
            autopad(k, p, d),
            groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act if act is True
            else act if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# ============================
# YOLO Bottleneck
# ============================
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# ============================
# LocalBlock: simple local refine (Conv + residual)
# ============================
class LocalBlock(nn.Module):
    """3x3 Conv + BN + SiLU + residual for lightweight local enhancement."""

    def __init__(self, dim):
        super().__init__()
        self.conv = Conv(dim, dim, k=3)

    def forward(self, x):
        return x + self.conv(x)


# ============================
# YOLO-style C2f block.
# Input and output tensors both use [B, C, H, W].
# ============================
class C2fBlock(nn.Module):
    """
    YOLO C2f-like module
    - split channels
    - multiple local blocks
    - concat
    - 1×1 conv fuse
    """

    def __init__(self, dim, n=2):
        """
        Args:
            dim (int): Shared input/output channel width.
            n (int): Number of LocalBlock modules, analogous to YOLO bottlenecks.
        """
        super().__init__()
        hidden = dim // 2              # C2f-style channel split.
        self.cv1 = Conv(dim, hidden * 2, k=1)

        # Stack n LocalBlock modules, all operating on the hidden width.
        self.blocks = nn.ModuleList(LocalBlock(hidden) for _ in range(n))

        # Concatenate to (2 + n) * hidden channels, then fuse back to dim.
        self.cv2 = Conv((2 + n) * hidden, dim, k=1)

    def forward(self, x):
        # Step 1: 1x1 convolution followed by channel split.
        y1, y2 = self.cv1(x).chunk(2, dim=1)

        outs = [y1, y2]

        # Step 2: pass y2 through the LocalBlock stack.
        for block in self.blocks:
            y2 = block(y2)
            outs.append(y2)

        # Step 3: concatenate and fuse.
        return self.cv2(torch.cat(outs, dim=1))
