import torch as t
import torch.nn as nn
from torch.types import Number

from .defaults import *
from .conversions import *
from .math import *


@torch.jit.script
def __hzetaitr(s:t.Tensor, a:t.Tensor, n:Number) -> t.Tensor:
    """Returns just the value inside that is being infinitely summed
    for the Hurwitz Zeta Function.

    Args:
        s (torch.Tensor): The 's' value of the hzeta function.
        a (torch.Tensor): The 'a' value of the hzeta function.
        n (Number): The current iteration number.

    Returns:
        torch.Tensor: The result of the internal computation only.
    """
    return t.pow(n + a, -s)

@torch.jit.script
def hzeta(s:t.Tensor, a:t.Tensor, eps:t.Tensor, blankSamples:int=0, samples:int=DEFAULT_FFT_SAMPLES) -> torch.Tensor:
    # Parameters for computation
    result:t.Tensor = t.zeros((*s.size(), samples))
    epsig:t.Tensor = isigmoid(eps)
    idx:int = 1

    # Generate the first value without any summation
    result[..., 0] = __hzetaitr(s=s, a=a, n=0)

    # Ignore first set of steps in the system
    for _ in range(blankSamples):
        result[..., 0] = __hzetaitr(s=s, a=a, n=idx) + (epsig * result[..., 0])
        idx += 1
    
    # Calculate each step of the system and store
    for jdx in range(1, samples):
        trueIdx = idx + jdx
        result[..., jdx] = __hzetaitr(s=s, a=a, n=trueIdx) + (epsig * result[..., trueIdx-1])

    return result
