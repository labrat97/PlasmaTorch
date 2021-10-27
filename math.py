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
def xbias(dims:int, bias:int=0):
    composer = torch.triu(torch.ones((dims, dims)), diagonal=1-bias)
    return composer.transpose(-1,-2).sum(dim=-1)

@torch.jit.script
def latticeParams(dims:int) -> torch.Tensor:
    powers = xbias(dims=dims, bias=0)
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
def primishvals(n:int, base:torch.Tensor=torch.zeros(0, dtype=torch.int64), gaussApprox:bool=False) -> torch.Tensor:
    # Not in the 6x -+ 1 domain, or starting domain
    if base.size()[-1] < 3:
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
        if itr & 0x1 != 0:
            if gaussApprox:
                result[itr] = (4 * pitr) + 1
            else:
                result[itr] = (6 * pitr) - 1
        else:
            result[itr] = result[itr - 1] + 2
        itr += 1
        pitr  = int((itr - 3) / 2) + 1

    return result

@torch.jit.script
def realprimishdist(x:torch.Tensor, relative:bool=True, gaussApprox:bool=False) -> torch.Tensor:
    assert not torch.is_complex(x)

    # Collect inverse values
    if gaussApprox:
        iprimeGuessTop:torch.Tensor = ((x - 3.) / 4.).type(torch.int64)
        iprimeGuessBot:torch.Tensor = ((x + 3.) / 4.).type(torch.int64)
    else:
        iprimeGuessTop:torch.Tensor = ((x - 1.) / 6.).type(torch.int64)
        iprimeGuessBot:torch.Tensor = ((x + 1.) / 6.).type(torch.int64)
    iprimeGuessTop = torch.stack((iprimeGuessTop, iprimeGuessTop+1), dim=-1)
    iprimeGuessBot = torch.stack((iprimeGuessBot, iprimeGuessBot+1), dim=-1)

    # Compute nearest values
    if gaussApprox:
        primishTop:torch.Tensor = (iprimeGuessTop * 4) + 3
        primishBot:torch.Tensor = (iprimeGuessBot * 4) - 3
    else:
        primishTop:torch.Tensor = (iprimeGuessTop * 6) + 1
        primishBot:torch.Tensor = (iprimeGuessBot * 6) - 1

    # Collect the primes into one tensor, add the special primes to the tensor, sort
    primish:torch.Tensor = torch.cat((primishTop, primishBot), dim=-1)
    specialPrimes:torch.Tensor = torch.ones_like(primish)[...,:4] * torch.tensor([-1, 1, 2, 3])
    primish = torch.cat((primish, specialPrimes), dim=-1)
    primish = primish.sort(dim=-1, descending=False).values

    # Find the closest prime approximates
    xish:torch.Tensor = torch.ones_like(primish) * x.unsqueeze(-1)
    lowidx:torch.Tensor = (xish >= primish).type(dtype=x.dtype)
    highidx:torch.Tensor = (xish < primish).type(dtype=x.dtype)
    low:torch.Tensor = ((primish * lowidx) + ((1 - lowidx) * primish[...,0].unsqueeze(-1))).max(dim=-1).values
    high:torch.Tensor = ((primish * highidx) + ((1 - highidx) * primish[...,-1].unsqueeze(-1))).min(dim=-1).values
    
    # Calculate approach distance to nearest primes
    lowDist:torch.Tensor = x - low
    highDist:torch.Tensor = high - x
    highLower:torch.Tensor = highDist < lowDist

    # Turn into one tensor
    result:torch.Tensor = (highLower.type(torch.int64) * highDist) + ((1 - highLower.type(torch.int64)) * lowDist)
    if relative:
        # Only ever traversing half of the maximum space
        totalDistance:torch.Tensor = (high - low) / 2
        result.div_(totalDistance / 2) 
    
    return result

@torch.jit.script
def gaussianprimishdist(x:torch.Tensor, relative:bool=True) -> torch.Tensor:
    # Force complex type
    if not torch.is_complex(x):
        x = toComplex(x)

    # Extract values to compute against
    real = x.real
    imag = x.imag
    gauss = (x.real * x.real) + (x.imag * x.imag)

    # Simple first distance calculation for the magnitude
    gaussdist = torch.sqrt(realprimishdist(gauss, relative=relative, gaussApprox=False))

    # Calculate the other distances
    rdgauss = realprimishdist(real, relative=True, gaussApprox=True)
    idgauss = realprimishdist(imag, relative=True, gaussApprox=True)
    # 4k +- 3 leaves only a space of 2 between normal values
    # To normalize, divide by the size of the space (which is 2)
    rdgaussi = imag.abs() / 2
    idgaussi = real.abs() / 2
    # Raw distance function
    rdcomposite = torch.sqrt((rdgauss * rdgauss) + (rdgaussi * rdgaussi))
    idcomposite = torch.sqrt((idgauss * idgauss) + (idgaussi * idgaussi))

    # Take the minimum distance
    magnitudePerc = rdcomposite.min(other=idcomposite)
    return magnitudePerc.min(other=gaussdist)
    
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
