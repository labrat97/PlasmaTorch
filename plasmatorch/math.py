from .defaults import *
from .conversions import toComplex



@ts
def i() -> t.Tensor:
    """Constructs the imaginary number.

    Returns:
        t.Tensor: The imaginary number as a tensor of size (1).
    """
    return t.view_as_complex(t.stack(
        (t.zeros(1), t.ones(1)), \
        dim=-1)).detach()



@ts
def pi(dtype:t.dtype=DEFAULT_DTYPE) -> t.Tensor:
    """Gets the value of Pi in the requested datatype.

    Args:
        dtype (t.dtype, optional): The datatype to return Pi in. Defaults to DEFAULT_DTYPE.

    Returns:
        t.Tensor: The value of Pi as a tensor of size (1).
    """
    return t.tensor((3.14159265358979323846264338327950288419716939937510), dtype=dtype).detach()



@ts
def egamma(dtype:t.dtype=DEFAULT_DTYPE) -> t.Tensor:
    """Gets the value of the Euler-Mascheroni constant in the requested datatype.

    Args:
        dtype (t.dtype, optional): The datatype to return the Euler-Mascheroni constant in. Defaults to DEFAULT_DTYPE.

    Returns:
        t.Tensor: The value of the Euler-Mascheroni constant as a tensor of size (1).
    """
    return t.tensor((0.57721566490153286060651209008240243104215933593992), dtype=dtype).detach()



@ts
def phi(dtype:t.dtype=DEFAULT_DTYPE) -> t.Tensor:
    """Calculates the value of Phi in/with the requested datatype.

    Args:
        dtype (t.dtype, optional): The datatype to perform the computation in. Defaults to DEFAULT_DTYPE.

    Returns:
        t.Tensor: The value of Phi as a tensor of size (1).
    """
    one = t.ones((1), dtype=dtype).detach()
    square = t.sqrt(one * 5)

    return ((one + square) / 2).detach()



@ts
def asigphi(dtype:t.dtype=DEFAULT_DTYPE) -> t.Tensor:
    """Computes the inverse of a simoid activation on Phi so that the output of a sigmoid activation
    can come out as the golden ratio.

    Args:
        dtype (t.dtype, optional): The datatype to perform the computation in. Defaults to DEFAULT_DTYPE.

    Returns:
        t.Tensor: The value of the inverse of a sigmoid of the golden ratio.
    """
    return -t.log(phi(dtype=dtype) - 1)



@ts
def latticeParams(n:int, basisParam:t.Tensor=phi()) -> t.Tensor:
    """Creates a set of parameters that decrease their power of the `basisParam` argument
    from 0 to `1-n`. The default value of Phi for the basis parameter makes the lattice parameters
    have a weight that decays with the golden ratio.

    Args:
        n (int): The amount of units to decay (including the zero decay unit).
        basisParam (t.Tensor, optional): The parameter to be put through a series of power. Defaults to phi().

    Returns:
        t.Tensor: The requested, decaying, parameters.
    """
    powers = xbias(n=n, bias=0)
    return basisParam ** (-powers)



@ts
def softunit(x:t.Tensor, dim:int) -> t.Tensor:
    """Performs a softmax on the magnitude of `x`. This allows a system to have an
    equivalent of the signalling functionality of the softmax function, even if it
    uses complex numbers. After computing, the tensor will have no magnitudes over 1,
    but the sign will be preserved.

    Args:
        x (t.Tensor): The tensor to soften the magnitude of.
        dim (int): The dimension to soften and use for averaging.

    Returns:
        t.Tensor: The input tensor with a magnitude no larger than 1.
    """
    # Normal magnitude based softmax, keep signal
    return t.softmax(x.abs(), dim=dim) * x.sgn()



@ts
def nsoftunit(x:t.Tensor, dims:List[int]) -> t.Tensor:
    """Stacks a `softunit()` call onto multiple dimensions (provided in argument
    `dims`) using a complex arethmetic mean.

    Args:
        x (t.Tensor): The tensor to soften the magnitude of.
        dims (List[int]): The dimensions to use for softening and averaging.

    Returns:
        t.Tensor: The input tensor with a magnitude no larger than 1.
    """
    # Performing a mean at the end of the computation
    result = t.ones_like(x)

    # Multiplies each softmax to the resultant accumulator, performing a complex arethmetic mean
    for dim in dims:
        result.mul_(t.softmax(x.abs(), dim=dim))
    
    # Return the complex arethmetic mean (all angles should be preserved through the softunit function)
    return t.pow(result, 1 / len(dims)) * x.sgn()


@ts
def primishvals(n:int, base:Union[t.Tensor, None]=None, gaussApprox:bool=False) -> t.Tensor:
    """Gets a set of `n` values of primish numbers (6k+-1) or Gaussian primish numbers
    (4k+-1) optionally using a base of primish values to build off of. The first three prime numbers,
    ([1, 2, 3]) are given as uncalculated constants.

    Args:
        n (int): The amount of primish values to calculate.
        base (Union[t.Tensor, None], optional): If provided, primish values will build after this sequence at
        the appropriate index. Defaults to None.
        gaussApprox (bool, optional): If True, 4k+-1 prime approximation is used instead of 6k+-1. Defaults to False.

    Returns:
        t.Tensor: The primish numbers of sequence length `n`.
    """
    # Not in the 6x -+ 1 domain, or starting domain
    # Initialize the base of the primish values tensor
    if base is None:
        base = t.tensor([1, 2, 3])
    # Invalid base, rebuild
    elif base.size(-1) < 3:
        base = t.tensor([1, 2, 3], dtype=base.dtype)

    # No need to calculate if the amount of samples is below 
    if n <= base.size(-1):
        return base[:n]
    
    # Construct the output values
    result:t.Tensor = t.zeros((n), dtype=t.int64).detach()
    result[:base.size(-1)] = base

    # Set up main calculation iterators
    itr:int = base.size(-1)
    pitr:int = int((itr - 3) / 2) + 1

    # Compute the requested primish values
    while itr < n:
        # Add either the iterator's specific definition of a calculation to the system,
        #   or just add 2 to the previous value due to 4k+-1 and 6k+-1 having a maximum
        #   error swing from the scalar multiple applied to the index of only 2.
        if itr & 0b1 != 0: # The iterator is odd
            if gaussApprox:
                result[itr] = (4 * pitr) + 1
            else:
                result[itr] = (6 * pitr) - 1
        else: # The iterator is even
            result[itr] = result[itr - 1] + 2
        
        # Increase the iterators accordingly
        itr += 1

        # There are two primes around every power of the scalar
        pitr = int((itr - 3) / 2) + 1

    return result



@ts
def realprimishdist(x:t.Tensor, relative:bool=True, gaussApprox:bool=False) -> t.Tensor:
    """Gets the distance from the nearest prime approximation, optionally binding the
    result to a relative positional value between the primes [0, 1].

    Args:
        x (t.Tensor): The tensor to be evaluated unit-wise.
        relative (bool, optional): If False, the absolute distance is returned. Defaults to True.
        gaussApprox (bool, optional): If True, uses 4k+-1 rather than 6k+-1. Defaults to False.

    Returns:
        t.Tensor: The distances to the prime approximations.
    """
    assert not t.is_complex(x)

    # Collect inverse values
    if gaussApprox:
        iprimeGuessTop:t.Tensor = ((x - 3.) / 4.).type(t.int64)
        iprimeGuessBot:t.Tensor = ((x + 3.) / 4.).type(t.int64)
    else:
        iprimeGuessTop:t.Tensor = ((x - 1.) / 6.).type(t.int64)
        iprimeGuessBot:t.Tensor = ((x + 1.) / 6.).type(t.int64)
    iprimeGuessTop = t.stack((iprimeGuessTop, iprimeGuessTop+1), dim=-1)
    iprimeGuessBot = t.stack((iprimeGuessBot, iprimeGuessBot+1), dim=-1)

    # Compute nearest values
    if gaussApprox:
        primishTop:t.Tensor = (iprimeGuessTop * 4) + 3
        primishBot:t.Tensor = (iprimeGuessBot * 4) - 3
    else:
        primishTop:t.Tensor = (iprimeGuessTop * 6) + 1
        primishBot:t.Tensor = (iprimeGuessBot * 6) - 1

    # Collect the primes into one tensor, add the special primes to the tensor, sort
    primish:t.Tensor = t.cat((primishTop, primishBot), dim=-1)
    specialPrimes:t.Tensor = t.ones_like(primish)[...,:4] * t.tensor([-1, 1, 2, 3])
    primish = t.cat((primish, specialPrimes), dim=-1)
    primish = primish.sort(dim=-1, descending=False).values

    # Find the closest prime approximates
    xish:t.Tensor = t.ones_like(primish) * x.unsqueeze(-1)
    lowidx:t.Tensor = (xish >= primish).type(dtype=x.dtype)
    highidx:t.Tensor = (xish < primish).type(dtype=x.dtype)
    low:t.Tensor = ((primish * lowidx) + ((1 - lowidx) * primish[...,0].unsqueeze(-1))).max(dim=-1).values
    high:t.Tensor = ((primish * highidx) + ((1 - highidx) * primish[...,-1].unsqueeze(-1))).min(dim=-1).values
    
    # Calculate approach distance to nearest primes
    lowDist:t.Tensor = x - low
    highDist:t.Tensor = high - x
    highLower:t.Tensor = highDist < lowDist

    # Turn into one tensor
    result:t.Tensor = (highLower.type(t.int64) * highDist) + ((1 - highLower.type(t.int64)) * lowDist)
    # Can ignore gaussApprox due to the normalized range being equal to one
    if relative and not gaussApprox:
        # Only ever traversing half of the maximum space
        totalDistance:t.Tensor = (high - low) / 2
        result.div_(totalDistance) 
    
    return result



@ts
def gaussianprimishdist(x:t.Tensor, relative:bool=True) -> t.Tensor:
    """Gets the distance from the nearest complex prime approximation, optionally binding
    the result to a relative positional value between the primes [0, 1].

    Args:
        x (t.Tensor): The tensor to be evaluated unit-wise.
        relative (bool, optional): If False, the absolute distance is returned. Defaults to True.

    Returns:
        t.Tensor: The distances (of real value) to the prime approximation.
    """
    # Force complex type
    if not t.is_complex(x):
        x = toComplex(x)

    # Extract values to compute against
    real = x.real
    imag = x.imag
    gauss = (x.real * x.real) + (x.imag * x.imag)

    # Simple first distance calculation for the magnitude
    gaussdist = t.sqrt(realprimishdist(gauss, relative=relative, gaussApprox=False))

    # Calculate the other distances
    rdgauss = realprimishdist(real, gaussApprox=True)
    idgauss = realprimishdist(imag, gaussApprox=True)
    
    # 4k +- 3 leaves only a space of 2 between normal values
    # To normalize, divide by the size of the space (which is 2)
    rdgaussi = imag.abs() / 2
    idgaussi = real.abs() / 2

    # Raw distance function
    # Using multiply calls due to higher potential accuracy and speed in hardware
    rdcomposite = t.sqrt((rdgauss * rdgauss) + (rdgaussi * rdgaussi))
    idcomposite = t.sqrt((idgauss * idgauss) + (idgaussi * idgaussi))

    # Take the minimum distance
    magnitudePerc = rdcomposite.min(other=idcomposite)
    return magnitudePerc.min(other=gaussdist)



@ts
def iprimishdist(x:t.Tensor, relative:bool=True, forceGauss:bool=False) -> t.Tensor:
    """Checks the distance betwwen the input tensor `x` and the nearest primish value
    according to if the type of `x` is complex. If `x` is not complex, the `forceGauss`
    option can be used to ensure 4k+-1 prime value approximation. This function can
    also passthrough both `realprimishdist()` and `gaussianprimishdist()`'s relative
    distance arguments.

    Args:
        x (t.Tensor): The tensor to be evaluated unit-wise.
        relative (bool, optional): If False, the absolute distance is returned. Defaults to True.
        forceGauss (bool, optional): If True, and `x` is a real valued tensor, do 4k+-1 approximation. Defaults to False.

    Returns:
        t.Tensor: The distances to the prime approximations.
    """
    if not t.is_complex(x):
        return realprimishdist(x, relative=relative, gaussApprox=forceGauss)
    return gaussianprimishdist(x, relative=relative)



@ts
def quadcheck(x:t.Tensor, boolChannel:bool=False) -> t.Tensor:
    """Figures out the quadrant that each unit of the provided tensor is in, returning as an int.

    Args:
        x (t.Tensor): The tensor to evaluate unit-wise into the occupied quadrants.
        boolChannel (bool): If True, output the quadrants into a channel wise output (adding a dimension). Defaults to False.

    Returns:
        t.Tensor: The occupied quadrants in t.uint8 datatype.
    """
    # Cache local Pi tensor
    PI:t.Tensor = pi()

    # Gather the angle of the complex numbers unit-wise
    xang:t.Tensor = toComplex(x).angle()

    # If the angle is negative, rotate it around a circle once, else rotate none
    angOffset:t.Tensor = 2. * PI * (xang < 0).type(PI.dtype)

    # Make all the angles positively bound, then label the quadrant from 0 to 3
    quadint:t.Tensor = ((xang + angOffset) * 2. / PI).type(t.uint8)

    # No need to create channel-wise output
    if not boolChannel:
        return quadint
    
    # Turn into channel-wise output
    result:t.Tensor = t.empty(x.size() + [4], dtype=t.uint8)
    for idx in range(4):
        result[..., idx] = (quadint == idx).type(t.uint8)

    return result



@ts
def csigmoid(x:t.Tensor) -> t.Tensor:
    """Calculate the magnitude of a complex number run through a complex sigmoid
    function. This function works like a normal sigmoid in the similarly signed quadrants
    of the complex plane. Outside of these quadrants, interpolation is performed
    with a bounded and scaled sin function.

    Args:
        x (t.Tensor): The tensor to calculate the complex sigmoid magnitude of unit-wise.

    Returns:
        t.Tensor: The complex sigmoid magnitude of `x`.
    """
    # Normal sigmoid
    if not x.is_complex():
        return t.sigmoid(x)

    # Extract/calculate required basic parameters
    PI2 = pi() / 2
    ang = x.angle()
    xabs = x.abs()
    
    # Do a sigmoid in the unanimous sign'd quadrants and find the connecting point
    # between the sigmoids if not in the unanimous quadrants.
    quadresult:t.Tensor = quadcheck(x, boolChannel=True)
    posQuad = quadresult[..., 0]
    examineQuadLeft = quadresult[..., 1]
    negQuad = quadresult[..., 2]
    examineQuadRight = quadresult[..., 3]
    examineQuad:t.Tensor = t.logical_or(examineQuadLeft, examineQuadRight).type(t.uint8)

    # The positive and negative quadrants are just the magnitude of the absolute value piped into
    #   the evaluation of a normal sigmoid, then bound to the appropriate side of the sign
    posVal:t.Tensor = posQuad * t.sigmoid(xabs)
    negVal:t.Tensor = negQuad * t.sigmoid(-xabs)

    # The "examine" quadrants will use a cosine activation to toggle between the signs compounded in the
    #   magnitude evaluation for the sigmoid.
    rotScalar:t.Tensor = t.cos(
        (examineQuadLeft * (ang - (PI2)) * 2) \
            + (examineQuadRight * (ang + (PI2)) * 2)
    )
    examVal:t.Tensor = examineQuad * t.sigmoid(rotScalar * xabs)

    # Add everything together according to the previously applied boolean based scalars
    return posVal + negVal + examVal



@ts
def isigmoid(x:t.Tensor) -> t.Tensor:
    """Calculate the complex number equivalent of a sigmoid function. This function
    internally calls the `csigmoid()` function to get the magnitude of the result, then
    multiplies it by the unit signal unit-wise from `x`.

    Args:
        x (t.Tensor): The tensor to perform the computation on unit-wise.

    Returns:
        t.Tensor: The complex sigmoid of `x`.
    """
    if x.is_complex():
        return csigmoid(x) * x.sgn()
    return csigmoid(x)



@ts
def ctanh(x:t.Tensor) -> t.Tensor:
    """Calculate the magnitude of a complex number run through a tanh function only
    on the magnitude. This function works like a normal tanh call on a real value for the
    magnitude in the similarly signed quadrants. If the quadrants are negative, the output of the 

    Args:
        x (t.Tensor): _description_

    Returns:
        t.Tensor: _description_
    """
    # Normal tanh
    if not x.is_complex():
        return t.tanh(x)

    # Extract/calculate required basic parameters
    PI2 = pi() / 2
    ang = x.angle()
    xabs = x.abs()

    # Check the quadrant that the elements of the provided tensor lie in, output channel-wise rather
    #   than as a basic int
    quadresult:t.Tensor = quadcheck(x, boolChannel=True)
    posQuad = quadresult[..., 0]
    examineQuadLeft = quadresult[..., 1]
    negQuad = quadresult[..., 2]
    examineQuadRight = quadresult[..., 3]
    examineQuad:t.Tensor = t.logical_or(examineQuadLeft, examineQuadRight).type(t.uint8)

    # Get the values that are at each quadrant, zeroing out the quadrants that
    #   should not be in the output of the system. This works because of the return
    #   addition
    posVal:t.Tensor = posQuad * t.tanh(xabs)
    negVal:t.Tensor = negQuad * t.tanh(-xabs)

    # Calculate a scalar value between [-1.0, 1.0] that interpolates the output of the function
    #   over half of the curvature of a cosine function.
    rotScalar:t.Tensor = t.cos(
        (examineQuadLeft * (ang - (PI2)) * 2) \
            + (examineQuadRight * (ang + (PI2)) * 2)
    )

    # Get the values that don't lie at the mutually positive or negative quadrants (1 or 3)
    #   and zero out if not in this quadrant
    examVal:t.Tensor = examineQuad * t.tanh(rotScalar * xabs)

    # Calculate the summation of all of the results and return
    return posVal + negVal + examVal



@ts
def itanh(x:t.Tensor) -> t.Tensor:
    """Calculate the complex number equivalent of the `ctanh()` function defined above.
    The result of this method ends up as a complex number with the magnitude defined by the
    return of the `ctanh()` function called with the passed tensor. The result ends up
    being the absolute value of `ctanh()` due to the negative properties of the `tanh()`
    function, then is multiplied unit-wise to the signal of the passed tensor.

    Args:
        x (t.Tensor): The tensor to run the calculation on unit-wise.

    Returns:
        t.Tensor: The tanh magnitud'd number.
    """
    if x.is_complex():
        return ctanh(x).abs() * x.sgn()
    return ctanh(x)



@ts
def icos(x:t.Tensor) -> t.Tensor:
    """Calculate a direct real cosine equivalent in the complex plane. This is done
    by using the magnitude of the complex tensor provided in a unit-wise fashion, the
    result is the cosine and the winding of the complex number on the cosine doubled.
    The winding must be doubled (the complex angle) in order to keep the differentiability
    of the function.

    Args:
        x (t.Tensor): The tensor to run through the cosine unit-wise.

    Returns:
        t.Tensor: `x` evaluated unit-wise through a complex cosine function.
    """
    # Normal cos
    if not x.is_complex():
        return t.cos(x)

    # Main computation.
    # A multiplier of 2 is needed on the angling system due to the fact that cos()
    #   is secretly actually just sin() squared.
    return t.cos(x.abs()) * t.exp(i() * 2. * x.angle())



@ts
def isin(x:t.Tensor) -> t.Tensor:
    """Calculate a direct real sine equivalent in the complex plane. This is done
    by using the magnitude of the complex tensor provided in a unit-wise fashion, the
    result is the sine function and the signal of the complex number getting carried through.
    The winding of the signal is not doubled here as continuity is reachable due to the
    shape of the sine function.

    Args:
        x (t.Tensor): The tensor to run through the sine unit-wise.

    Returns:
        t.Tensor: `x` evaluated unit-wise through a complex sine function.
    """
    # Normal sin
    if not x.is_complex():
        return t.sin(x)

    # Main computation.
    # The sin() function actually maps perfectly to the complex plane, no weird identities
    #   and fuckery are needed.
    return t.sin(x.abs()) * t.exp(i() * x.angle())



@ts
def hmean(x:t.Tensor, dim:int=-1) -> t.Tensor:
    """Calculates the harmonic mean according to the given dimension.

    Args:
        x (t.Tensor): The tensor to use as the base of calculation.
        dim (int, optional): The dimension to perform the calculation on. Defaults to -1.

    Returns:
        t.Tensor: The harmonic mean of the input.
    """
    # Turn all the values to their -1 power
    invx:t.Tensor = 1. / x
    
    # Find the amount of values for the mean
    vals = x.size(dim)
    
    # Calculate the harmonic mean
    return vals / invx.sum(dim=dim)



@ts
def harmonicvals(n:int, noSum:bool=False, useZero:bool=False) -> t.Tensor:
    """Calculates `n` values of the harmonic series optionally not summing the result of the system.
    If not summing (`nosum` is True) the values will come out in ascending order (to match the order of the summed
    values).

    Args:
        n (int): The amount of hamonic values to generate.
        noSum (bool, optional): If True, turn off the summation that actually builds the harmonic values and reverse the unsummed factors. Defaults to False.
        useZero (bool, optional): If True, zero is added as the first value. Defaults to False.

    Returns:
        t.Tensor: The set of `n` harmonic series values with a size of `n`.
    """
    # Quick error checking
    assert n >= 1

    # Find all of the 1/n values to be summed
    zeroint = int(useZero) # 0 false, 1 true
    factors:t.Tensor = (1. / xbias(n=n-zeroint, bias=1))
    if useZero:
        factors = t.cat([t.zeros((1)), factors], dim=-1)

    # Break early if skipping the final summation
    if noSum:
        # Flip around order of values to keep ascending ordering
        factors[zeroint:] = factors[zeroint:][::-1]
        return factors

    # Avoid squaring the memory requirement by sequencing accelerated summations
    accum:t.Tensor = t.zeros_like(factors) + factors

    # Slide the factors over the accumulator, summing up the results
    for idx in range(1, n):
        accum[...,idx:].add_(factors[...,:-idx])

    # Return the accumulated factors
    return accum



@ts
def harmonicdist(x:t.Tensor) -> t.Tensor:
    """Get the distance of each value from its nearest harmonic series value using
    the Euler-Mascheroni constant.

    Args:
        x (t.Tensor): The tensor to evaluate the distance of.

    Returns:
        t.Tensor: The raw distance of each value from its nearest harmonic series value.
    """
    # Gather constants for evaluation
    em:t.Tensor = egamma()

    # Take the inverse harmonic index of the input values and flatten them after for indexing
    inverse:t.Tensor = t.round(t.exp(x - em)) + t.exp(-em)
    finv = inverse.flatten(0, -1)

    # Find the needed harmonics for producing the final value
    maxn:t.Tensor = finv.max()[0]
    harmonics:t.Tensor = harmonicvals(n=maxn, useZero=True)
    
    # Find the closest harmonic value, refold the shape, then calculate the result
    closest = harmonics[finv].unflatten(0, inverse.size())
    return x - closest



@ts
def realfold(x:t.Tensor, phase:t.Tensor=pi()) -> t.Tensor:
    """Fold the imaginary values of a complex tensor into the real values that are
    also represented by it. Folding, in this case, means rotating the magnitude of
    the imaginary value with a cosine function, and summing it into the real value.

    Args:
        x (t.Tensor): The complex tensor to fold into itself.
        phase (t.Tensor, optional): Rotates the imaginary values through an `icos()` function, so pi() results in a value of -1. Defaults to pi().

    Returns:
        t.Tensor: The folded values from the input tensor `x`.
    """
    if x.is_complex():
        return x.real + (icos(phase) * x.imag)
    return x



@ts
def fft(x:t.Tensor, n:Union[int, List[int], None]=-1, dim:Union[int, List[int], None]=-1) -> t.Tensor:
    """Performs an orthonormal fast fourier transform on `x`, optionally handling
    multiple dimensions.

    Args:
        x (t.Tensor): The tensor to perform the fft on.
        n (Union[int, List[int], None], optional): The amount of samples to use for gathering the
        basis frequencies. Using -1 takes the dim's size from the tensor. If the sample count
        provided is higher than the amount of samples in the dimension, the samples are zero padded. Defaults to -1.
        dim (Union[int, List[int], None], optional): The dimension(s) to perform the FFT on. Defaults to -1.

    Raises:
        ValueError: `n` and `dim` are of unequal length.
        ValueError: `n` and `dim` must have the same type.

    Returns:
        t.Tensor: The fast fourier transformed `x` tensor.
    """
    # Pass values through to a normal function, leave true 1/sqrt(n) definition
    # Optionally do an n dimensional fft if the dim is a List
    if isinstance(dim, List[int]):
        # Make sure that the length of the iterated arguments are the same
        if isinstance(n, int):
            return tfft.fftn(x, s=[n]*len(dim), dim=dim, norm='ortho')
        elif isinstance(n, List[int]) and (len(n) != len(dim)):
            raise ValueError('n and dim are of unequal length')
        
        return tfft.fftn(x, s=n, dim=dim, norm='ortho')

    # Normal single dimension fft
    elif isinstance(n, int) and isinstance(dim, int):
        # Bounds checking for auto sample count
        # Not filtering 0 value as that has handling in the `tfft.ifft()` method
        if n < 0:
            n = x.size(dim)
        
        return tfft.fft(x, n=n, dim=dim, norm='ortho')

    # Invalid arguments were provided
    else:
        raise ValueError('n and dim must have the same type and be either ints or lists of ints')



@ts
def ifft(x:t.Tensor, n:Union[int, List[int], None]=-1, dim:Union[int, List[int], None]=-1) -> t.Tensor:  
    """Performs an orthonormal inverse fast fourier transform on `x`, optionally handling multiple
    dimensions.

    Args:
        x (t.Tensor): The tensor to perform the ifft on.
        n (Union[int, List[int], None], optional): The amount of samples to generate from the
        basis frequencies. Using -1 takes the dim's size from the tensor. If the sample count
        provided is higher than the amount of samples in the dimension, the basis frequencies 
        are zero padded. Defaults to -1.
        dim (Union[int, List[int], None], optional): The dimension(s) to perform the FFT on. Defaults to -1.

    Raises:
        ValueError: 'n' and 'dim' are of unequal length.
        ValueError: 'n' and 'dim' must have the same type.

    Returns:
        t.Tensor: The inverse fast fourier transformed `x` tensor.
    """
    # Pass values through to normal function, leave true 1/sqrt(n) definition
    # Optionally do an n dimensional inverse fft if the dim is a Tuple
    if isinstance(dim, List[int]):
        # Make sure that the length of the iterated arguments are the same
        if isinstance(n, int):
            return tfft.ifftn(x, s=[n]*len(dim), dim=dim, norm='ortho')
        elif isinstance(n, List[int]) and (len(n) != len(dim)):
            raise ValueError('n and dim are of unequal length')

        return tfft.ifftn(x, s=n, dim=dim, norm='ortho')

    # Normal single dimension inverse fft
    elif isinstance(n, int) and isinstance(dim, int):
        # Bounds checking for auto sample count
        # Not filtering 0 value as that has handling in the `tfft.ifft()` method
        if n < 0:
            n = x.size(dim)

        return tfft.ifft(x, n=n, dim=dim, norm='ortho')

    # Invalid arguments were provided
    else:
        raise ValueError('n and dim must have the same type and be either ints or tuples of ints')
