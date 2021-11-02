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
def hzeta(s:t.Tensor, a:t.Tensor, eps:t.Tensor, samples:int=DEFAULT_FFT_SAMPLES) -> torch.Tensor:
    # Generate the starting data to give room for the result of the computation
    result:t.Tensor = t.zeros((*s.size(), samples))
    result[..., 0] = __hzetaitr(s=s, a=a, n=0)
    epsig:t.Tensor = isigmoid(eps)

    # Calculate each step of the system
    for idx in range(1, samples):
        result[..., idx] = __hzetaitr(s=s, a=a, n=idx) + (epsig * result[..., idx-1])
