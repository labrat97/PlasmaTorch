from .defaults import *
from .math import asigphi, i, pi, csigmoid
from .conversions import toComplex
from .sizing import resignal


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
    return 1. / t.pow(n + a, s)

@ts
def hzetae(s:t.Tensor, a:t.Tensor, res:t.Tensor=asigphi(), aeps:t.Tensor=t.tensor((1e-8)), maxiter:int=1024) -> t.Tensor:
    """Returns the value that the hzeta function converges to after either the max iterations or
    when the change in the position of the function is unanimously below the arc-epsilon value (aeps).

    Args:
        s (t.Tensor): The `s` value of the Hurwitz Zeta function.
        a (t.Tensor): The `a` value of the Hurwitz Zeta function.
        res (t.Tensor, optional): The amount of residual evaluation used to determine the output value. This
            value is piped through the isigmoid() fuction. A full activation means
            a normal evaluation of the zeta function, as where a 0 activation means
            something closer to just the evaluation of the delta value. Defaults to asigphi().
        aeps (t.Tensor, optional): The arc-epsilon value. If the delta value is less
            than this value, the evaluation is considered complete. Defaults to t.tensor((1e-8)).
        maxiter (int, optional): The maximum amount of evaluation iterations used for the
            finding the convergent values. Defaults to 1024.

    Returns:
        t.Tensor: The element-wise convergent values of the input tensors through
            the Hurwitz Zeta function.
    """
    # Set up the parameters for residual evaluation
    epsig:t.Tensor = csigmoid(res)
    idx:int = 1
    epsigexp:t.Tensor = t.ones_like(epsig)

    # Generate the first value
    delta:t.Tensor = __hzetaitr(s=s, a=a, n=0)
    result:t.Tensor = delta
    keepGoing:t.Tensor = (result.abs() >= aeps.abs()).type(t.int64)

    # Progress each value forward to convergence or max iteration
    while t.any(keepGoing) and idx < maxiter:
        # Find and apply the changes according to the aeps variable
        # Multiplying s by keepGoing allows for a quicker exponential eval potentially
        # on the finished values
        delta = __hzetaitr(s=s, a=a, n=idx)
        epsigexp = t.pow(epsig, float(idx) / (maxiter - 1))
        result.add_(epsigexp * delta)

        # Check to see if the values are still needing iteration
        keepGoing = (result.abs() >= aeps.abs()).type(t.int64)
        idx += 1

    return result

@ts
def hzetas(s:t.Tensor, a:t.Tensor, res:t.Tensor=asigphi()*3, blankSamples:int=0, samples:int=DEFAULT_FFT_SAMPLES, fftformat:bool=True) -> t.Tensor:
    """Returns a set of samples from the Hurwitz Zeta function with an optional continuous, non-singularity occupied, resampling.

    Args:
        s (t.Tensor): The `s` value of the Hurwitz Zeta function.
        a (t.Tensor): The `a` value of the Hurwitz Zeta function.
        res (t.Tensor, optional): The amount of residual evaluation used to determine the output value. This
            value is piped through the isigmoid() fuction. A full activation means
            a normal evaluation of the zeta function, as where a 0 activation means
            something closer to just the evaluation of the delta value. Defaults to asigphi().
        blankSamples (int, optional): The amount of samples to ignore at the start. Defaults to 0.
        samples (int, optional): The total amount of samples per element to output. Defaults to DEFAULT_FFT_SAMPLES.
        fftformat (bool, optional): If enabled, runs the output through the resampleSmear() function, 
            using the final sampled value as the most significant value. Defaults to True.

    Returns:
        torch.Tensor: A tensor of the size of the input with the amount of samples collected
            in a new last dimension through the Hurwitz Zeta function.
    """
    # Make the result the size of the input with the output sample channels
    result:t.Tensor = toComplex(s.unsqueeze(-1) @ t.zeros((1, samples), dtype=s.dtype))

    # Set up running parameters
    epsig:t.Tensor = csigmoid(res)
    idx:int = 1
    epsigexp:t.Tensor = t.ones_like(epsig)
    totsamples:int = blankSamples + samples

    # Generate the first value without any summation
    result[..., 0] = __hzetaitr(s=s, a=a, n=0)

    # Ignore first set of steps in the system
    for _ in range(blankSamples):
        epsigexp = t.pow(epsig, float(idx) / (totsamples - 1))
        result[..., 0] = (epsigexp * __hzetaitr(s=s, a=a, n=idx)) + result[..., 0]
        idx += 1
    
    # Calculate each step of the system and store
    for jdx in range(1, samples):
        epsigexp = t.pow(epsig, float(idx+jdx) / (totsamples - 1))
        result[..., jdx] = (epsigexp * __hzetaitr(s=s, a=a, n=idx+jdx)) + result[..., jdx-1]

    # If the signal should be continuous, force it.
    if fftformat:
        return resignal(result, samples=result.size(-1), dim=-1)
    return result

@ts
def __lerchitr(lam:t.Tensor, s:t.Tensor, a:t.Tensor, n:int) -> t.Tensor:
    """Returns just the value being infinitely summed in the Lerch Zeta function.

    Args:
        lam (t.Tensor): The rotation multiple to used to generate the top of the transcedent.
        s (t.Tensor): The `s` value of the Lerch Zeta function.
        a (t.Tensor): The `a` value of the Lerch Zeta function.
        n (int): The current iteration number.

    Returns:
        t.Tensor: The value to be summed for the provided iteration of the function.
    """
    # Modify the hzeta itr with the provided exponent
    hzetaexp:t.Tensor = 2 * pi() * n

    # If the lambda value is complex, account for the phase angle in the production of z
    if t.is_complex(lam):
        hzetaexp = (hzetaexp * lam.abs()) + lam.angle()
    else:
        hzetaexp = hzetaexp * lam
    
    # Multiply the numerator of the hzeta itr to create the final itr result
    return t.exp(hzetaexp * i()) * __hzetaitr(s=s, a=a, n=n)

@ts
def lerche(lam:t.Tensor, s:t.Tensor, a:t.Tensor, res:t.Tensor=asigphi(), aeps:t.Tensor=t.tensor(1e-8), maxiter:int=1024) -> t.Tensor:
    """Returns the values that the Lerch Zeta function converges to after either the max iterations or
    when the change in the position of the function is unanimously below the arc-epsilon value (aeps).

    Args:
        lam (t.Tensor): The rotation multiple used to generate the top of the transcedent.
        s (t.Tensor): The `s` value of the Lerch Zeta function.
        a (t.Tensor): The `a` value of the Lerch Zeta function.
        res (t.Tensor, optional): The amount of residual evaluation used to determine the output value. This
            value is piped through the isigmoid() fuction. A full activation means
            a normal evaluation of the zeta function, as where a 0 activation means
            something closer to just the evaluation of the delta value. Defaults to asigphi().
        aeps (t.Tensor, optional): The arc-epsilon value. If the delta value is less than
            this value, the evaluation is considered complete. Defaults to t.tensor(1e-8).
        maxiter (int, optional): The maximum amount of evaluation iterations used 
            for finding the convergent values. Defaults to 1024.

    Returns:
        t.Tensor: The element-wise convergent values of the input tensors through
            the Lerch Zeta function.
    """
    # Set up the running parameters
    epsig:t.Tensor = csigmoid(res)
    idx:int = 1
    epsigexp:t.Tensor = t.ones_like(epsig)

    # Generate the first value
    delta:t.Tensor = __lerchitr(lam=lam, s=s, a=a, n=0)
    result:t.Tensor = delta
    keepGoing:t.Tensor = (result.abs() >= aeps.abs()).type(t.int64)

    # Progress each element forward to convergence or max iteration
    while t.any(keepGoing) and idx < maxiter:
        # Find and apply the changes according to the aeps variable
        # Multiplying lam & s by keepGoing allows for a quicker exponential eval potentially
        # on the finished values
        delta = __lerchitr(lam=lam, s=s, a=a, n=idx)
        epsigexp = t.pow(epsig, float(idx) / (maxiter - 1))
        result.add_(epsigexp * delta)

        # Check to see if the values are still needing iteration
        keepGoing = (result.abs() >= aeps.abs()).type(t.int64)
        idx += 1
    
    return result

@ts
def lerchs(lam:t.Tensor, s:t.Tensor, a:t.Tensor, res:t.Tensor=asigphi()*3, blankSamples:int=0, samples:int=DEFAULT_FFT_SAMPLES, fftformat:bool=True) -> t.Tensor:
    """Returns a set of samples from the Lerch Zeta function.

    Args:
        lam (t.Tensor): The rotation multiple used to generate the top of the transcedent.
        s (t.Tensor): The `s` value of the Lerch Zeta function.
        a (t.Tensor): The `a` value of the Lerch Zeta function.
        res (t.Tensor, optional): The amount of residual evaluation used to determine the output value. This
            value is piped through the isigmoid() fuction. A full activation means
            a normal evaluation of the zeta function, as where a 0 activation means
            something closer to just the evaluation of the delta value. Defaults to asigphi()*3.
        blankSamples (int, optional): The amount of samples to ignore at the start. Defaults to 0.
        samples (int, optional): The total amount of samples per element to output. Defaults to DEFAULT_FFT_SAMPLES.
        fftformat (bool, optional): If enabled, runs the output through the resampleSmear() function, 
            using the final sampled value as the most significant value. Defaults to True.

    Returns:
        torch.Tensor: A tensor of the size of the input with the amount of samples collected
            in a new last dimension through the Lerch Zeta function.
    """
    # Make the result the size of the input with the output samples channels
    result:t.Tensor = toComplex(s.unsqueeze(-1) @ t.zeros((1, samples), dtype=s.dtype))

    # Set up running parameters
    epsig:t.Tensor = csigmoid(res)
    idx:int = 1
    epsigexp:t.Tensor = t.ones_like(epsig)
    totsamples:int = blankSamples + samples

    # Generate the first sample
    result[..., 0] = __lerchitr(lam=lam, s=s, a=a, n=0)

    # Ignore the first blank steps in the system
    for _ in range(blankSamples):
        epsigexp = t.pow(epsig, float(idx) / (totsamples - 1))
        result[..., 0] = (epsigexp * __lerchitr(lam=lam, s=s, a=a, n=idx)) + result[..., 0]
        idx += 1
    
    # Calculate each step of the system then store
    for jdx in range(1, samples):
        epsigexp = t.pow(epsig, float(idx+jdx) / (totsamples - 1))
        result[..., jdx] = (epsigexp * __lerchitr(lam=lam, s=s, a=a, n=idx+jdx)) + result[..., jdx-1]

        # If the signal should be continuous, force it.
    if fftformat:
        return resignal(result, samples=result.size(-1), dim=-1)
    return result
