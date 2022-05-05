import torch as t
import torch.nn as nn
import torch.nn.functional as nnf
import torch.fft as tfft
from torch.jit import script as ts

from typing import Tuple, List, Dict

# Ensure some level of default precision
DEFAULT_DTYPE:t.dtype = t.float32
DEFAULT_COMPLEX_DTYPE:t.dtype = t.complex64

# The FFT samples used are done so like this to account for the full expressiveness
#   of the Finite Sporadic Cyclic Groups. Enabling the full causal sample resolution
#   allows for all of the sporadic patterns in reality to be described according to Conway.
DEFAULT_FFT_SAMPLES:int = 196884
DEFAULT_FFT_NORM:str = 'ortho'
# The following numbers are the prime factors of 196884
SMALL_FFT_SAMPLES:int = 1823
SMALL_FFT_BATCH:int = (2*2)*(3*3*3)


# The amount of knot descriptors used determines the ability of the knots to propogate through themselves.
# This parameter is used to define how many times a knot can collide with another knot and still have spectrum space.
KNOTS_WITHOUT_LOSS:int = 9 # Two 3D descriptors superpositioned would be at minimum (3*3)
# Using the previous values, determine how many descriptive waves should be contained in an instantiated
#   know. A peak and a trough are needed at minimum for a legible signal in a knot due to their circular nature.
DEFAULT_KNOT_WAVES:int = int(DEFAULT_FFT_SAMPLES / (KNOTS_WITHOUT_LOSS * 2))
# How many knots should be tracked in parallel. This should be a prime typically speaking so that it can be factored
#   into a set of generic knotted signals.
DEFAULT_SPACE_PRIME:int = 11
# Circular padding is used because knots are, by nature, circular.
DEFAULT_PADDING:str = 'circular'

# The default amount of samples to use in lenses
DEFAULT_SIGNAL_LENS_PADDING:int = 5
DEFAULT_SIGNAL_LENS_SAMPLES:int = DEFAULT_FFT_SAMPLES

# The number of samples to use in the internal lens definition
GREISS_SAMPLES:int = 196884


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
