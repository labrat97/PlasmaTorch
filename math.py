from torch.types import List
from .defaults import *
from .conversions import *

import torch

@torch.jit.script
def pi() -> torch.Tensor:
    return (torch.ones((1)) * 3.141592653589793238462643383279502).detach()

@torch.jit.script
def emconst() -> torch.Tensor:
    return (torch.ones((1)) * 0.57721566490153286060651209008240243104215933593992).detach()

@torch.jit.script
def phi() -> torch.Tensor:
    one = torch.ones((1)).detach()
    square = torch.sqrt(one * 5)

    return ((one + square) / 2).detach()

@torch.jit.script
def asigphi() -> torch.Tensor:
    return -torch.log(phi() - 1)

@torch.jit.script
def xbias(n:int, bias:int=0):
    composer = torch.triu(torch.ones((n, n)), diagonal=1-bias)
    return composer.transpose(-1,-2).sum(dim=-1)

@torch.jit.script
def latticeParams(dims:int, basisParam:torch.Tensor=phi()) -> torch.Tensor:
    powers = xbias(n=dims, bias=0)
    return basisParam ** (-powers)

@torch.jit.script
def i() -> torch.Tensor:
    return torch.view_as_complex(torch.stack(
        (torch.zeros(1), torch.ones(1)), \
        dim=-1)).detach()

@torch.jit.script
def isoftmax(x:torch.Tensor, dim:int) -> torch.Tensor:
    # Normal softmax
    if not x.is_complex(): 
        return torch.softmax(x, dim=dim)

    # Imaginary softmax
    angle:torch.Tensor = x.angle()
    magnitude:torch.Tensor = x.abs()
    softMagnitude:torch.Tensor = torch.softmax(magnitude, dim=dim)
    
    # Convert back to imaginary
    newReal:torch.Tensor = softMagnitude * torch.cos(angle)
    newImag:torch.Tensor = softMagnitude * torch.sin(angle)
    
    # Return in proper datatype
    return torch.view_as_complex(torch.stack((newReal, newImag), dim=-1))

@torch.jit.script
def nsoftmax(x:torch.Tensor, dims:List[int]) -> torch.Tensor:
    # Because n^0 == 1, this should be an appropriate initializer
    result = torch.ones_like(x)

    # Creates an n root
    exponent:float = 1. / len(dims)

    # Multiplies each value in the result by the n-root of each isoftmax
    for dim in dims:
        nroot = torch.pow(isoftmax(x, dim=dim), exponent)
        result.mul_(nroot)

    return result

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
    # Can ignore gaussApprox due to the normalized range being equal to one
    if relative and not gaussApprox:
        # Only ever traversing half of the maximum space
        totalDistance:torch.Tensor = (high - low) / 2
        result.div_(totalDistance) 
    
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
    rdgauss = realprimishdist(real, gaussApprox=True)
    idgauss = realprimishdist(imag, gaussApprox=True)
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

    # Extract/calculate required basic parameters
    PI2 = pi() / 2
    ang = x.angle()
    xabs = x.abs()
    
    # Do a sigmoid in the unanimous sign'd quadrants and find the connecting point
    # between the sigmoids if not in the unanimous quadrants.
    posQuad:torch.Tensor = torch.logical_and(x.real >= 0, x.imag >= 0).type(torch.uint8)
    negQuad:torch.Tensor = torch.logical_and(x.real < 0, x.imag < 0).type(torch.uint8)
    examineQuadRight:torch.Tensor = torch.logical_and(x.real >= 0, x.imag < 0)
    examineQuadLeft:torch.Tensor = torch.logical_and(x.imag >= 0, x.real < 0)
    examineQuad:torch.Tensor = torch.logical_and(examineQuadLeft, examineQuadRight).type(torch.uint8)
    

    # The positive and negative quadrants are just the magnitude of the absolute value piped into
    # the evaluation of a normal sigmoid, then bound to the appropriate side of the sign
    posVal:torch.Tensor = posQuad * torch.sigmoid(posQuad * xabs)
    negVal:torch.Tensor = negQuad * torch.sigmoid(negQuad * -xabs)

    # The "examine" quadrants will use a cosine activation to toggle between the signs compounded in the
    # magnitude evaluation for the sigmoid.
    rotScalar:torch.Tensor = (torch.cos(
        (examineQuadLeft.type(torch.uint8) * (ang - (PI2))*2) \
            + (examineQuadRight.type(torch.uint8) * (ang + (PI2))*2)
    ))
    examVal:torch.Tensor = examineQuad * torch.sigmoid(examineQuad * rotScalar * xabs)

    # Add everything together according to the previously applied boolean based scalars
    finalSigmoidMag:torch.Tensor = posVal + negVal + examVal

    # Create the complex value alignment to finally push the signal through. As a note,
    # I really don't like the fact that there are hard non-differentiable absolute
    # values in this evaluation, but I would not like to lose the current sigmoid properties
    xabs_e = torch.view_as_complex(torch.stack((x.real.abs(), x.imag.abs()), dim=-1))
    sigmoidComplexVal:torch.Tensor = xabs_e / xabs

    # Calculate and return
    return finalSigmoidMag * sigmoidComplexVal

@torch.jit.script
def icos(x:torch.Tensor) -> torch.Tensor:
    # Normal cos
    if not x.is_complex():
        return torch.cos(x)

    # Main conversion
    return torch.cos(x.abs()) * torch.exp(i() * 2. * x.angle())

@torch.jit.script
def isin(x:torch.Tensor) -> torch.Tensor:
    # Normal sin
    if not x.is_complex():
        return torch.sin(x)

    # Main conversion
    return torch.sin(x.abs()) * torch.exp(i() * x.angle())

@torch.jit.script
def hmean(x:torch.Tensor, dim:int=-1) -> torch.Tensor:
    """Calculates the harmonic mean according to the given dimension.

    Args:
        x (torch.Tensor): The tensor value to use as the base of calculation.
        dim (int, optional): The dimension to perform the calculation on. Defaults to -1.

    Returns:
        torch.Tensor: The harmonic mean of the input
    """
    # Turn all the values to their -1 power
    invx:torch.Tensor = 1. / x
    # Find the amount of values for the mean
    vals = x.size()[dim]
    
    # Calculate the harmonic mean
    return vals / invx.sum(dim=dim)

@torch.jit.script
def harmonicvals(n:int, nosum:bool=False, addzero:bool=False) -> torch.Tensor:
    # Quick error checking
    assert n >= 1

    # Find all of the 1/n values to be summed
    zeroint = int(addzero)
    factors:torch.Tensor = (1. / xbias(n=n+zeroint, bias=1-zeroint))
    if nosum:
        return factors
    else:
        factors.unsqueeze_(0)

    # Turn the values into a big triangle
    composition:torch.Tensor = torch.triu(torch.ones((n, 1)) @ factors, diagonal=0)

    # Sum the triangle together so that the harmonic values come out
    return composition.transpose(-1, -2).sum(dim=-1)

@torch.jit.script
def harmonicdist(x:torch.Tensor) -> torch.Tensor:
    # Gather constants for evaluation
    em:torch.Tensor = emconst()

    # Take the inverse harmonic index of the input values and flatten them after for indexing
    inverse:torch.Tensor = torch.round(torch.exp(x - em)) + torch.exp(-em)
    finv = inverse.flatten(0, -1)

    # Find the needed harmonics for producing the final value
    maxn:torch.Tensor = finv.max()[0]
    harmonics:torch.Tensor = harmonicvals(n=maxn, addzero=True)
    
    # Find the closest harmonic value, refold the shape, then calculate the result
    closest = harmonics[finv].unflatten(0, inverse.size())
    return x - closest
