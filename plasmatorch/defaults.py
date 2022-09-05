import torch as t
import torch.nn as nn
import torch.nn.functional as nnf
import torch.fft as tfft
import torch.cuda as tc
from torch.jit import script as ts

from typing import Tuple, List, Dict, Union, Callable
from .memory import *



# Ensure some level of default precision
DEFAULT_DTYPE:t.dtype = t.float32
DEFAULT_COMPLEX_DTYPE:t.dtype = t.complex64

# Some constants from Greiss Algebra and Monster Group Vertex Algebra
GREISS_SAMPLES:int = 196884
MONSTER_CURVES:int = 196883
SUPERSINGULAR_PRIMES_LH:List[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]
SUPERSINGULAR_PRIMES_HL:List[int] = SUPERSINGULAR_PRIMES_LH[::-1]
SUPERSINGULAR_PRIMES:List[int] = SUPERSINGULAR_PRIMES_HL

# The FFT samples used are done so like this to account for the full expressiveness
#   of the Finite Sporadic Cyclic Groups. Enabling the full causal sample resolution
#   allows for all of the sporadic patterns in reality to be described according to Conway.
DEFAULT_FFT_SAMPLES:int = GREISS_SAMPLES
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
DEFAULT_SIGNAL_LENS_SAMPLES:int = GREISS_SAMPLES

# The normal amount of lenses used in a signal aggregation system
AGGREGATE_LENSES:int = 7

# Figure out the most effective devices to use by default
DEFAULT_FAST_DEV:str = 'cuda' if tc.is_available() else 'cpu'
DEFAULT_MEM_DEV:str = 'cuda' if getCudaMemory() > getSystemMemory() else 'cpu'



@ts
def isSmear(x:t.Tensor) -> bool:
    """Checks the dimensions of a t.Tensor to see if it can be represented as a smear.

    Args:
        x (t.Tensor): The tensor to test.

    Returns:
        bool: If the tensor is a smear.
    """
    size:int = len(x.size())
    return (size >= 3) and (size <= 4)

@ts
def isOneD(x:t.Tensor) -> bool:
    """Checks the dimensions of a t.Tensor to see if it is a one dimension signal.

    Args:
        x (t.Tensor): The tensor to test.

    Returns:
        bool: If the tensor is a one dimension signal.
    """
    return len(x.size()) == 3

@ts
def isSmearAll(x:t.Tensor) -> Tuple[bool, bool]:
    """Run both the `isSmear()` test and `isOneD()` test respectively.

    Args:
        x (t.Tensor): The tensor to test.

    Returns:
        Tuple[bool, bool]: (`isSmear()`, `isOneD()`)
    """
    return isSmear(x), isOneD(x)

@ts
def xbias(n:int, bias:int=0) -> t.Tensor:
    """Creates the torch equivalent of a `range()` call in python, n elements long
    starting at `bias` as a value.

    Args:
        n (int): The number of samples to iterate and save for the function.
        bias (int, optional): The number to start iterating at. Defaults to 0.

    Returns:
        t.Tensor: The unit stepping, summing, iterated tensor.
    """
    composer = t.zeros((n)).add(bias)
    for i in range(n):
        composer[i].add_(i)
    return composer
