import torch as t
from torch.jit import script as ts
import torch.nn as nn
from torch.types import Number
import torch.fft as tfft

from .defaults import *
from .conversions import *
from .math import *


@ts
def __hzetaitr(s:t.Tensor, a:t.Tensor, n:int) -> t.Tensor:
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

@ts
def hzetae(s:t.Tensor, a:t.Tensor, res:t.Tensor=1/phi(), aeps:t.Tensor=t.tensor(1e-4), maxiter:int=1024) -> t.Tensor:
    # Size assertions
    assert s.size() == a.size()
    
    # Set up the parameters for residual evaluation
    epsig:t.Tensor = isigmoid(res)
    idx:int = 1

    # Generate the first value
    delta:t.Tensor = __hzetaitr(s=s, a=a, n=0)
    result:t.Tensor = t.ones_like(delta) * delta
    keepGoing:t.Tensor = (result.abs() >= aeps.abs()).type(t.int32).nonzero()

    # Progress each value forward to convergence or max iteration
    while keepGoing.numel() > 0 and idx < maxiter:
        # Find and apply the changes according to the aeps variable
        delta = __hzetaitr(s=(s[keepGoing]), a=(a[keepGoing]), n=idx)
        result[keepGoing] = delta + (epsig * result[keepGoing])

        # Keep the reduction iteration going
        keepGoing = (result.abs() >= aeps.abs()).type(t.int32).nonzero()
        idx += 1

    return result

@ts
def hzetas(s:t.Tensor, a:t.Tensor, res:t.Tensor=1/phi(), blankSamples:int=0, samples:int=DEFAULT_FFT_SAMPLES, fftformat:bool=True) -> torch.Tensor:
    # Make the result the size of the input with the output sample channels
    result:t.Tensor = toComplex(s.unsqueeze(-1) @ t.zeros((1, samples), dtype=s.dtype))

    # Set up running parameters
    epsig:t.Tensor = isigmoid(res)
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

    # If the signal should be continuous, force it.
    if fftformat:
        return resampleContinuous(result, dim=-1, msi=-1)
    return result

@ts
def __lerchitr(lam:t.Tensor, s:t.Tensor, a:t.Tensor, n:int) -> t.Tensor:
    # Modify the hzeta itr with the provided exponent
    hzetaexp:t.Tensor = 2 * pi() * n * lam

    # If the lambda value is complex, the top of the itr must still be one for convergence
    if t.is_complex(lam):
        hzetaexp = hzetaexp.abs() * (lam / lam.abs())
    
    # Multiply the numberator of the hzeta itr to create the final itr result
    return t.exp(hzetaexp * i()) * __hzetaitr(s=s, a=a, n=n)

@ts
def lerche(lam:t.Tensor, s:t.Tensor, a:t.Tensor, res:t.Tensor=1/phi(), aeps:t.Tensor=t.tensor(1e-4), maxiter:int=1024) -> t.Tensor:
    # Size assertions
    assert lam.size() == s.size() == a.size()
    
    # Set up the running parameters
    epsig:t.Tensor = isigmoid(res)
    idx:int = 1

    # Generate the first value
    delta:t.Tensor = __lerchitr(lam=lam, s=s, a=a, n=0)
    result:t.Tensor = t.ones_like(delta) * delta
    keepGoing:t.Tensor = (result.abs() >= aeps.abs()).type(t.int32).nonzero()

    # Progress each element forward to convergence or max iteration
    while keepGoing.numel() > 0 and idx < maxiter:
        # Find and apply the changes needed according to the aeps variable
        delta = __lerchitr(lam=lam[keepGoing], s=s[keepGoing], a=a[keepGoing], n=idx)
        result[keepGoing] = delta + (epsig * result[keepGoing])

        # Keep the reducing iteration going
        keepGoing = (result.abs() >= aeps.abs()).type(t.int32).nonzero()
        idx += 1
    
    return result

@ts
def lerchs(lam:t.Tensor, s:t.Tensor, a:t.Tensor, res:t.Tensor=1/phi(), blankSamples:int=0, samples:int=DEFAULT_FFT_SAMPLES, fftloss:bool=True) -> t.Tensor:
    # Make the result the size of the input with the output samples channels
    result:t.Tensor = toComplex(s.unsqueeze(-1) @ t.zeros((1, samples), dtype=s.dtype))

    # Set up running parameters
    epsig:t.Tensor = isigmoid(res)
    idx:int = 1

    # Generate the first sample`
    result[..., 0] = __lerchitr(lam=lam, s=s, a=a, n=0)

    # Ignore the first blank steps in the system
    for _ in range(blankSamples):
        result[..., 0] = __lerchitr(lam=lam, s=s, a=a, n=idx) + (epsig * result[..., 0])
        idx += 1
    
    # Calculate each step of the system then store
    for jdx in range(1, samples):
        result[..., jdx] = __lerchitr(lam=lam, s=s, a=a, n=idx+jdx) + (epsig * result[..., jdx-1])

        # If the signal should be continuous, force it.
    if fftloss:
        return resampleContinuous(result, dim=-1, msi=-1)
    return result
