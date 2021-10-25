from .defaults import *
from .conversions import *

import torch

@torch.jit.script
def pi() -> torch.Tensor:
    return (torch.ones((1)) * 3.141592653589793238462643383279502).detach()

@torch.jit.script
def phi() -> torch.Tensor:
    one = torch.ones((1)).detach()
    square = torch.sqrt(one * 5)

    return (one + square) / 2

@torch.jit.script
def latticeParams(dims:int) -> torch.Tensor:
    powers = torch.triu(torch.ones((dims, dims)), diagonal=1).transpose(-1,-2).sum(dim=-1)

    return phi() ** (-powers)

@torch.jit.script
def i() -> torch.Tensor:
    return torch.view_as_complex(torch.stack(
        (torch.zeros(1), torch.ones(1)), \
        dim=-1)).detach()

@torch.jit.script
def imagnitude(x:torch.Tensor) -> torch.Tensor:
    if not x.is_complex():
        return x

    # Main conversion
    return torch.sqrt(x.real.pow(2) + x.imag.pow(2))

@torch.jit.script
def ipolarization(x:torch.Tensor) -> torch.Tensor:
    # Main conversion
    return torch.angle(x)

@torch.jit.script
def isoftmax(x:torch.Tensor, dim:int) -> torch.Tensor:
    # Normal softmax
    if not x.is_complex(): 
        return torch.softmax(x, dim=dim)

    # Imaginary softmax
    angle:torch.Tensor = ipolarization(x)
    magnitude:torch.Tensor = imagnitude(x)
    softMagnitude:torch.Tensor = torch.softmax(magnitude, dim=dim)
    
    # Convert back to imaginary
    newReal:torch.Tensor = softMagnitude * torch.cos(angle)
    newImag:torch.Tensor = softMagnitude * torch.sin(angle)
    
    # Return in proper datatype
    return torch.view_as_complex(torch.stack((newReal, newImag), dim=-1))

@torch.jit.script
def primishvals(n:int, base:torch.Tensor=torch.zeros(0, dtype=torch.int64)) -> torch.Tensor:
    # Not in the 6x -+ 1 domain, or starting domain
    if base.size()[-1] == 0:
        base = torch.ones(3, dtype=base.dtype)
        base[1] += base[0]
        base[2] += base[1]
    if n <= base.size()[-1]:
        return base[:n]
    
    # Construct the output values
    result:torch.Tensor = torch.zeros((n), dtype=torch.int64).detach()
    result[:base.size()[-1]] = base
    
    # Compute every needed 6x -+ 1 value
    itr:int = base.size()[-1]
    pitr:int = int((itr - 3) / 2) + 1
    while itr < n:
        result[itr] = (6 * pitr) - 1
        itr += 1
        pitr = int((itr - 3) / 2) + 1

        if itr >= n: break

        result[itr] = (6 * pitr) + 1
        itr += 1
        pitr  = int((itr - 3) / 2) + 1

    return result

@torch.jit.script
def realprimishdist(x:torch.Tensor, relative:bool=True, gaussApprox:bool=False) -> torch.Tensor:
    assert not torch.is_complex(x)

    # Collect inverse values
    if gaussApprox:
        iprimeGuessTop:torch.Tensor = ((x - 3) / 4).type(torch.int64).unsqueeze(-1)
        iprimeGuessBot:torch.Tensor = ((x + 3) / 4).type(torch.int64).unsqueeze(-1)
    else:
        iprimeGuessTop:torch.Tensor = ((x - 1) / 6).type(torch.int64).unsqueeze(-1)
        iprimeGuessBot:torch.Tensor = ((x + 1) / 6).type(torch.int64).unsqueeze(-1)
    iprimeGuessTop = torch.stack((iprimeGuessTop, iprimeGuessTop+1), dim=-1)
    iprimeGuessBot = torch.stack((iprimeGuessBot, iprimeGuessBot+1), dim=-1)

    # Test nearest values
    if gaussApprox:
        primishTop:torch.Tensor = (iprimeGuessTop * 4) + 3
        primishBot:torch.Tensor = (iprimeGuessBot * 4) - 3
    else:
        primishTop:torch.Tensor = (iprimeGuessTop * 6) + 1
        primishBot:torch.Tensor = (iprimeGuessBot * 6) - 1
    primish:torch.Tensor = torch.stack((primishTop, primishBot), dim=-1).sort(dim=-1)[0]

    # Determine the distance
    lowidx:torch.Tensor = torch.nonzero(x >= primish)[0].max(dim=-1)[0]
    highidx:torch.Tensor = torch.nonzero(x < primish)[0].min(dim=-1)[0]
    low:torch.Tensor = primish[lowidx]
    high:torch.Tensor = primish[highidx]
    totalDistance:torch.Tensor = high - low
    lowDist:torch.Tensor = x - low
    highDist:torch.Tensor = high - x
    highLower:torch.Tensor = highDist < lowDist

    # Turn into one tensor
    result:torch.Tensor = (highLower.type(torch.int64) * highDist) + ((1 - highLower.type(torch.int64)) * lowDist)
    if relative:
        result.div_(totalDistance / 2) # Only ever traversing half of the space
    
    return result

@torch.jit.script
def gaussianprimishdist(x:torch.Tensor, relative:bool=True) -> torch.Tensor:
    if not torch.is_complex(x):
        x = toComplex(x)

    real = x.real
    imag = x.imag
    gauss = (x.real * x.real) + (x.imag * x.imag)

    gaussdist = torch.sqrt(realprimishdist(gauss, relative=relative, gaussApprox=False))
    realdistgauss = realprimishdist(real, relative=relative, gaussApprox=True)
    imagdistgauss = realprimishdist(imag, relative=relative, gaussApprox=True)

    magnitudePerc = realdistgauss.max(other=imagdistgauss)
    return magnitudePerc.max(other=gaussdist)
    
@torch.jit.script
def iprimishdist(x:torch.Tensor, relative:bool=True) -> torch.Tensor:
    if not torch.is_complex(x):
        return realprimishdist(x, relative=relative)
    return gaussianprimishdist(x, relative=relative)

@torch.jit.script
def isigmoid(x:torch.Tensor) -> torch.Tensor:
    # Normal sigmoid
    if not x.is_complex():
        return torch.sigmoid(x)
    
    # Imaginary sigmoid
    angle:torch.Tensor = ipolarization(x)
    magnitude:torch.Tensor = imagnitude(x)
    sigmag:torch.Tensor = nnf.sigmoid(magnitude)

    # Convert back to imaginary
    newReal:torch.Tensor = sigmag * torch.cos(angle)
    newImag:torch.Tensor = sigmag * torch.sin(angle)

    # Return in proper datatype
    return torch.view_as_complex(torch.stack((newReal, newImag), dim=-1))

@torch.jit.script
def icos(x:torch.Tensor) -> torch.Tensor:
    # Normal cos
    if not x.is_complex():
        return torch.cos(x)

    # Main conversion
    I = i()
    return torch.exp(I * x.real) + torch.exp(I * x.imag) - 1

@torch.jit.script
def isin(x:torch.Tensor) -> torch.Tensor:
    # Normal sin
    if not x.is_complex():
        return torch.sin(x)

    # Main conversion
    return i() * icos(x)
