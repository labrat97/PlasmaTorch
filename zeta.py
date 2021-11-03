import torch as t
import torch.nn as nn
from torch.types import Number

from .defaults import *
from .conversions import *
from .math import *


@t.jit.script
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

@t.jit.script
def hzeta(s:t.Tensor, a:t.Tensor, res:Number, aeps:Number=1e-4, maxiter:int=1024):
    # Make the result the size of the input
    epsig:t.Tensor = isigmoid(t.tensor(res))
    idx:int = 1

    # Generate the first value
    delta:t.Tensor = __hzetaitr(s=s, a=a, n=0)
    result:t.Tensor = t.copy(delta)
    keepGoing:t.Tensor = (delta.abs() >= aeps).type(torch.int64).nonzero()

    # Progress each value forward to convergence or max iteration
    while keepGoing.numel > 0 and idx < maxiter:
        # Find and apply the changes according to the aeps variable
        delta = __hzetaitr(s=s, a=a, n=idx)
        result[keepGoing] = delta + (epsig * result[keepGoing])

        # Keep the reduction iteration going
        keepGoing:t.Tensor = (delta.abs() >= aeps).type(torch.int64).nonzero()
        idx += 1

    return result

@t.jit.script
def hzeta(s:t.Tensor, a:t.Tensor, res:Number, blankSamples:int=0, samples:int=DEFAULT_FFT_SAMPLES) -> torch.Tensor:
    # Make the result the size of the input with the output sample channels
    result:t.Tensor = t.zeros((*s.size(), samples))

    # Set up running parameters
    epsig:t.Tensor = isigmoid(torch.tensor(res))
    idx:int = 1

    # Generate the first value without any summation
    result[..., 0] = __hzetaitr(s=s, a=a, n=0)

    # Ignore first set of steps in the system
    for _ in range(blankSamples):
        result[..., 0] = __hzetaitr(s=s, a=a, n=idx) + (epsig * result[..., 0])
        idx += 1
    
    # Calculate each step of the system and store
    for jdx in range(1, samples):
        result[..., jdx] = __hzetaitr(s=s, a=a, n=idx+jdx) + (epsig * result[..., jdx-1])

    return result

@t.jit.script
def __lerchitr(z:t.Tensor, s:t.Tensor, a:t.Tensor, n:Number) -> t.Tensor:
    # All that's needed is an extra exponentiated multiple from the hzeta iterator
    return t.pow(z, n) * __hzetaitr(s=s, a=a, n=n)

@t.jit.script
def lerch(z:t.Tensor, s:t.Tensor, a:t.Tensor, res:Number, aeps:Number=1e-4, maxiter:int=1024) -> t.Tensor:
    # Set up the running parameters
    epsig:t.Tensor = isigmoid(t.tensor(res))
    idx:int = 1

    # Generate the first value
    delta:t.Tensor = __lerchitr(z=z, s=s, a=a, n=0)
    result:t.Tensor = t.copy(delta)
    keepGoing:t.Tensor = (delta.abs() >= aeps).type(torch.int64).nonzero()

    # Progress each element forward to convergence or max iteration
    while keepGoing.numel() > 0 and idx < maxiter:
        # Find and apply the changes needed according to the aeps variable
        delta = __lerchitr(z=z[keepGoing], s=s[keepGoing], a=a[keepGoing], n=idx)
        result[keepGoing] = delta + (epsig * result[keepGoing])

        # Keep the reducing iteration going
        keepGoing = (delta.abs() >= aeps).type(torch.int64).nonzero()
        idx += 1
    
    return result


@t.jit.script
def lerch(z:t.Tensor, s:t.Tensor, a:t.Tensor, res:Number, blankSamples:int=0, samples:int=DEFAULT_FFT_SAMPLES) -> t.Tensor:
    # Make the result the size of the input with the output samples channels
    result:t.Tensor = t.zeros((*s.size(), samples))

    # Set up running parameters
    epsig:t.Tensor = isigmoid(t.tensor(res))
    idx:int = 1

    # Generate the first sample`
    result[..., 0] = __lerchitr(z=z, s=s, a=a, n=0)

    # Ignore the first blank steps in the system
    for _ in range(blankSamples):
        result[..., 0] = __lerchitr(z=z, s=s, a=a, n=idx) + (epsig * result[..., 0])
        idx += 1
    
    # Calculate each step of the system then store
    for jdx in range(1, samples):
        result[..., jdx] = __lerchitr(z=z, s=s, a=a, n=idx+jdx) + (epsig * result[..., jdx-1])
    
    return result
