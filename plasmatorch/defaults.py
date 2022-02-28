import torch as t
import torch.nn as nn
import torch.nn.functional as nnf
import torch.fft as tfft
from torch.jit import script as ts

from typing import Tuple, List, Dict

DEFAULT_FFT_SAMPLES:int = 196884
KNOTS_WITHOUT_LOSS:int = 8
# a peak and a trough are needed at minimum for a legible signal
DEFAULT_KNOT_WAVES:int = int(DEFAULT_FFT_SAMPLES / (KNOTS_WITHOUT_LOSS * 2))
DEFAULT_DTYPE:t.dtype = t.float32
DEFAULT_COMPLEX_DTYPE:t.dtype = t.complex64
DEFAULT_SPACE_PRIME:int = 11
DEFAULT_PADDING:str = 'circular'

@ts
def isSmear(x:t.Tensor) -> bool:
    size:int = len(x.size())
    return (size >= 3) and (size <= 4)

@ts
def isOneD(x:t.Tensor) -> bool:
    return len(x.size()) == 3

@ts
def isSmearAll(x:t.Tensor) -> Tuple[bool, bool]:
    return isSmear(x), isOneD(x)
