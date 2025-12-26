from .base_model import BaseModel
from .ema_model import EMAModel
from .blocks import TiTokEncoder, TiTokDecoder, TATiTokDecoder, UViTBlock
from .maskgit_vqgan import Decoder as Pixel_Decoder
from .maskgit_vqgan import VectorQuantizer as Pixel_Quantizer

from .losses import (
    VGTAE_ReconstructionLoss
)

loss_map = {
    "VGTAE_ReconstructionLoss": VGTAE_ReconstructionLoss,
}