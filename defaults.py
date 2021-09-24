import torch
from typing import Tuple

DEFAULT_FFT_SAMPLES:int = 256
KNOTS_WITHOUT_LOSS:int = 8
# a peak and a trough are needed at minimum for a legible signal
DEFAULT_KNOT_WAVES:int = int(DEFAULT_FFT_SAMPLES / (KNOTS_WITHOUT_LOSS * 2))
DEFAULT_DTYPE:torch.dtype = torch.float32
DEFAULT_COMPLEX_DTYPE:torch.dtype = torch.complex64
DEFAULT_SPACE_PRIME:int = 11

@torch.jit.script
def isSmear(x:torch.Tensor) -> bool:
    size:torch.Size = len(x.size())
    return (size >= 3) and (size <= 4)

@torch.jit.script
def isOneD(x:torch.Tensor) -> bool:
    return len(x.size()) == 3

@torch.jit.script
def isSmearAll(x:torch.Tensor) -> Tuple[bool]:
    return isSmear(x), isOneD(x)
