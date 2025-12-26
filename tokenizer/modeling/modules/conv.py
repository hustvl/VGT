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

    default_act = nn.SiLU()  # YOLO 默认激活

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
    """3×3 Conv + BN + SiLU + residual (轻量、局部增强)"""

    def __init__(self, dim):
        super().__init__()
        self.conv = Conv(dim, dim, k=3)

    def forward(self, x):
        return x + self.conv(x)


# ============================
# YOLO C2f 模块（保持 YOLO 结构哲学）
# 输入输出均为 [B, C, H, W]
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
            dim (int): 输入/输出通道一致
            n (int): LocalBlock 的数量（相当于 YOLO 中的 Bottleneck 数量）
        """
        super().__init__()
        hidden = dim // 2              # C2f 思想：拆分通道
        self.cv1 = Conv(dim, hidden * 2, k=1)

        # n 个 LocalBlock，全部保持 hidden 通道
        self.blocks = nn.ModuleList(LocalBlock(hidden) for _ in range(n))

        # concat 通道为 (2+n)*hidden，再 fuse 回 dim
        self.cv2 = Conv((2 + n) * hidden, dim, k=1)

    def forward(self, x):
        # Step1: 1×1 conv → split channel
        y1, y2 = self.cv1(x).chunk(2, dim=1)

        outs = [y1, y2]

        # Step2: y2 逐层经过 LocalBlock
        for block in self.blocks:
            y2 = block(y2)
            outs.append(y2)

        # Step3: concat → fuse conv
        return self.cv2(torch.cat(outs, dim=1))
