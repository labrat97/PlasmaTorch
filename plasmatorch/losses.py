from .__defimp__ import *
from .activations import *
from .conversions import toComplex
from .sizing import paddim, resignal
from .math import rms



@ts
def energyLoss(x:t.Tensor, y:t.Tensor, dim:Union[int, List[int]]=-1, keepElements:bool=False) -> t.Tensor:
    """Calculate the energy lost between the signals in terms of the difference
    between `y` and `x`  root means squared (being the regular value multiplied times the conjugate) 
    values. The actual calculation ends up being 'yrms - xrms'.

    Args:
        x (t.Tensor): The starting tensor for calculating the loss.
        y (t.Tensor): The resulting tensor for calculating the loss.
        dim (Union[int, List[int]], optional): The dimension(s) to perform the mean on. Defaults to -1.
        keepElements (bool, optional): If True, the root mean ratio squared function \
            (`rmrs()`) is used as opposed to the root mean squared function \
            (`rms()`). Defaults to False.

    Returns:
        t.Tensor: The real valued result of the rms difference calculations.
    """
    if keepElements:
        xrms:t.Tensor = rmrs(x, dim=dim)
        yrms:t.Tensor = rmrs(y, dim=dim)
    else:
        # `keepdim` enabled to maintain resultant dim count, ignoring function selection
        xrms:t.Tensor = rms(x, dim=dim, keepdim=True)
        yrms:t.Tensor = rms(y, dim=dim, keepdim=True)

    return yrms - xrms



@ts
def energyGain(x:t.Tensor, y:t.Tensor, dim:Union[int, List[int]]=-1, keepElements:bool=False) -> t.Tensor:
    """Calculate the energy gain from the `x` signal to the `y` signal as a real
    valued scalar. The resulting value will is filtered from NaN's to 1 for decibel
    like calculations.

    Args:
        x (t.Tensor): The starting tensor to use for calculating the gain.
        y (t.Tensor): The resulting tensor to use for calculating the gain.
        dim (Union[int, List[int]], optional): The dimension(s) to perform the mean on. Defaults to -1.
        keepElements (bool, optional): If True, the root mean ratio squared function \
            (`rmrs()`) is used as opposed to the root mean squared function \
            (`rms()`). Defaults to False.
        
    Returns:
        t.Tensor: The real valued scalar representing the gain on the rms values.
    """
    if keepElements:
        xrms:t.Tensor = rmrs(x, dim=dim)
        yrms:t.Tensor = rmrs(y, dim=dim)
    else:
        # `keepdim` enabled to maintain resultant dim count, ignoring function selection
        xrms:t.Tensor = rms(x, dim=dim, keepdim=True)
        yrms:t.Tensor = rms(y, dim=dim, keepdim=True)

    # Calculate the result, leaving nans for infinite replacement
    result:t.Tensor = yrms / xrms

    # Create the replacement infinities and negative infinities
    infmask:t.Tensor = (yrms.sgn() * t.inf) * (result == t.nan).type(result.type, non_blocking=True)

    # Apply the infinities to the nans
    return nantonum(result) + infmask



@ts
def correlation(x:t.Tensor, y:t.Tensor, dim:int=-1, isbasis:bool=False) -> t.Tensor:
    """Find the standard cross-correlation between the natural signals provided.

    Args:
        x (t.Tensor): One set of signals to use for the final computation.
        y (t.Tensor): Another set of signals to use for the final compuation.
        dim (int, optional): The dimension to apply the computation to. Defaults to -1.
        isbasis (bool, optional): If False, the vectors coming in are preFFT'd. Defaults to False.

    Returns:
        t.Tensor: The cross-correlation of the two input signals.
    """
    # Some size assertions/extractions
    xsize = x.size()
    ysize = y.size()
    assert len(x.size()) == len(y.size())
    samples:int = max(xsize[dim], ysize[dim])

    if not isbasis:
        # Find basis of the signals
        xfft = fft(toComplex(x), n=samples, dim=dim)
        yfft = fft(toComplex(y), n=samples, dim=dim)
    elif xsize[dim] != ysize[dim]:
        # Transfer over the signals to just be computed onto
        xfft:t.Tensor = toComplex(paddim(x, lowpad=0, highpad=samples-xsize[dim], dim=dim))
        yfft:t.Tensor = toComplex(paddim(y, lowpad=0, highpad=samples-ysize[dim], dim=dim))
    else:
        xfft:t.Tensor = toComplex(x)
        yfft:t.Tensor = toComplex(y)

    # Calculate the correlation
    return ifft(xfft * yfft.conj(), n=samples, dim=dim)



@ts
def hypercorrelation(x:t.Tensor, y:t.Tensor, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE, \
    dim:int=-1, fullOutput:bool=False, extraTransform:bool=False) -> t.Tensor:
    """Run the cross-correlation function accross every single frequency-space domain
    superposition that can fall out of the input signals. If specified, perform some
    analysis on the output of the signal, otherwise, give the full square of the domain
    superposition back to the user.

    Args:
        x (t.Tensor): One set of signals to use for the final computation.
        y (t.Tensor): Another set of signals to use for the final computation.
        cdtype (t.dtype, optional): The complex datatype to use. Defaults to DEFAULT_COMPLEX_DTYPE.
        dim (int, optional): The dimension to perform the computation on. Defaults to -1.
        fullOutput (bool, optional): If true, return the raw output of the hypercorrelation function. Defaults to False.
        extraTransform (bool, optional): If true, run through one more layer of fourier transforms to verify that the signals aren't super lossy in a continuous sense. Defaults to False.

    Returns:
        t.Tensor: A set of all of the possible correlation functions attached to the end of the input tensor shape.
    """
    # Some size assertions/extractions
    xsize = x.size()
    ysize = y.size()
    assert len(x.size()) == len(y.size())

    # Because ffts are kinda supposed to reverse themselves, running the signals
    # through two permuatations from both ffts and iffts should give the full spectral
    # evaluation of the input signals
    FFT_LAYERS:int = 2 + int(extraTransform)
    FFT_TAPE_LENGTH:int = (2 * FFT_LAYERS) + 1
    FFT_TAPE_CENTER:int = FFT_TAPE_LENGTH

    # Extract the max samples used for the function and pad with an FFT relay
    if xsize[dim] > ysize[dim]:
        samples:int = xsize[dim]
        y = resignal(y, samples=samples, dim=dim)
    elif ysize[dim] > xsize[dim]:
        samples:int = ysize[dim]
        x = resignal(x, samples=samples, dim=dim)
    else:
        samples:int = xsize[dim]

    # Create the storage needed for the output signals and populate the center
    tapeConstructor:t.Tensor = t.ones((1, FFT_TAPE_LENGTH), dtype=cdtype)
    xfftTape:t.Tensor = t.zeros_like(x).unsqueeze(-1) @ tapeConstructor
    yfftTape:t.Tensor = t.zeros_like(y).unsqueeze(-1) @ tapeConstructor
    assert t.is_complex(xfftTape)
    xfftTape[FFT_TAPE_CENTER] = t.view_as_complex(x)
    yfftTape[FFT_TAPE_CENTER] = t.view_as_complex(y)

    # Deconstruct signals
    for tape in [xfftTape, yfftTape]:
        for idx in range(1, FFT_LAYERS + 1):
            tape[FFT_TAPE_CENTER + idx] = fft(tape[FFT_TAPE_CENTER + idx - 1], n=samples, dim=dim)
            tape[FFT_TAPE_CENTER - idx] = fft(tape[FFT_TAPE_CENTER + idx + 1], n=samples, dim=dim)
    
    # Apply full cross correlation between different representations of the input signals
    if x.numel() > y.numel():
        unfiltered:t.Tensor = t.zeros_like(xfftTape).unsqueeze(-1) @ tapeConstructor
    else:
        unfiltered:t.Tensor = t.zeros_like(yfftTape).unsqueeze(-1) @ tapeConstructor
    for xidx in range(FFT_TAPE_LENGTH):
        for yidx in range(FFT_TAPE_LENGTH):
            unfiltered[..., xidx, yidx] = correlation(xfftTape[..., xidx], yfftTape[..., yidx], dim=dim, isbasis=True)
    
    # Quick exit from the computation
    if fullOutput:
        return unfiltered
    
    # Find mean, min, max, median, and mode
    meanbase:t.Tensor = unfiltered.mean(dim=dim)
    corrmean:t.Tensor = meanbase.mean(dim=dim)
    corrmin:t.Tensor = meanbase.min(dim=dim)[0]
    corrmax:t.Tensor = meanbase.max(dim=dim)[0]
    corrmedian:t.Tensor = meanbase.median(dim=dim)[0]
    corrmode:t.Tensor = meanbase.mode(dim=dim)[0]
    corrmse:t.Tensor = (unfiltered * unfiltered).mean(dim=dim).mean(dim=dim)

    # Return as a single tensor in the previously commented/written order
    return t.stack((corrmean, corrmin, corrmax, corrmedian, corrmode, corrmse), dim=-1)

# Some constants for indexing the hypercorrelation function
@ts
def HYDX_CORRMEAN() -> int:
    return 0
@ts
def HYDX_CORRMIN() -> int:
    return 1
@ts
def HYDX_CORRMAX() -> int:
    return 2
@ts
def HYDX_CORRMEDIAN() -> int:
    return 3
@ts
def HYDX_CORRMODE() -> int:
    return 4
@ts
def HYDX_CORRMSE() -> int:
    return 5



@ts
def entropy(x:t.Tensor, softmax:bool=True, startdim:int=0, countrot:bool=True) -> t.Tensor:
    """Gets the entropy of a matrix from the startdim on using a variation of Shannon Entropy.

    Args:
        x (t.Tensor): The input matrix to calculate the entropy of.
        softmax (bool, optional): If enabled, run the magnitude of the function through a softmax before processing. Defaults to True.
        startdim (int, optional): The dim to start calculating the entropy at. Defaults to 0.
        countrot (bool, optional): If enabled, count the rotation of the numbers in the complex plane
            into the entropy. Defaults to True.

    Returns:
        t.Tensor: The entropy of the matrix at the dim that was started at for calculation.
    """
    # Flatten the matrix to make it so the data can all be operated on at once by the
    #   softmax() function.
    # Doing this is done also to help optimize the idea that every single complex
    #   number is just a frozen wave function's eigenvalues. So, to calculate this,
    #   something similar to Shannon Entropy is used.
    xflat = x.flatten(start_dim=startdim)
    xabs:t.Tensor = xflat.abs()
    if countrot:
        xang:t.Tensor = (xflat.angle() + pi()) / (2. * pi())
    else:
        xang:t.Tensor = t.zeros_like(xabs)

    # Turn into density matrices
    if softmax:
        density:t.Tensor = xabs.softmax(dim=-1)
    else:
        density = xabs
    if countrot:
        density = t.cat((density, xang), dim=-1)
    
    # Calculate the entropy
    nits:t.Tensor = density * t.log2(density)
    return -1. * nits.sum(dim=-1)
    


@ts
def skeeter(teacher:t.Tensor, student:t.Tensor, center:t.Tensor, teacherTemp:float=1., \
    studentTemp:float=1., dim:int=-1, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE) -> t.Tensor:
    """A loss used to try to replicate the neat results of the DINO paper's cross-correlation loss.
    The loss itself is not really mathematically similar at all, however the correlation that gets produced
    from the function is similar to the correlation that would get produced from the loss function in mention.
    To do this, the hypercorrelation function is used, then the losses are added together, much the same as parrallel
    impedance, to produce a final composite loss based around the hypercorrelation mean, square mean, median, and mode.

    Args:
        teacher (t.Tensor): The teacher, or view dominant tensor output to use for computation. This tensor's
        gradient is detached immediately.
        student (t.Tensor): The student, or view dependent tensor output with full gradient attachment.
        center (t.Tensor): The centering tensor, with gradient attachment, used for modifying the teacher's say
        on the situation.
        teacherTemp (float, optional): The temperature of the teacher's evaluation. Defaults to 1.
        studentTemp (float, optional): The temperature of the student's evaluation. Defaults to 1.
        dim (int, optional): The dimension to compute the loss only. Defaults to -1.
        cdtype (t.dtype, optional): The complex datatype to use. Defaults to DEFAULT_COMPLEX_DTYPE.

    Returns:
        t.Tensor: The parallel impedance style inverse hypercorrelation for each value.
    """
    # Find the maximum amount of samples on the loss dim
    tenure:t.Tensor = teacher.detach()
    tsize = tenure.size()
    ssize = student.size()
    if tsize[dim] > ssize[dim]:
        samples:int = tsize[dim]
    else:
        samples:int = ssize[dim]
    
    # Soften and sharpen the inputs as somewhat seen in the DINO loss from (what is likely now formerly known as) Facebook AI
    softten:t.Tensor = softunit((tenure - center) / teacherTemp, dim=dim)
    softstu:t.Tensor = softunit(student / studentTemp, dim=dim)

    # Turning the signal from a frequency domain signal back to a time domain
    # signal can be temporarily ignored as hypercorrelation operates accross all
    # possible occuring orders of the time-frequency superposition
    hypercorr:t.Tensor = hypercorrelation(x=softten, y=softstu, cdtype=cdtype, \
        dim=dim, fullOutput=False, extraTransform=False)
    corrmean = hypercorr[..., HYDX_CORRMEAN()]
    corrmedian = hypercorr[..., HYDX_CORRMEDIAN()]
    corrmode = hypercorr[..., HYDX_CORRMODE()]
    corrmse = hypercorr[..., HYDX_CORRMSE()]

    # Get the harmonic mean of the extracted values, and the higher the mean, the lower the loss
    stackedResult = t.stack((corrmean, corrmedian, corrmode, corrmse), dim=-1)
    return -1 * hmean(stackedResult, dim=-1)



def bloodmuck(teacher:nn.Module, student:nn.Module, sigma:t.Tensor):
    """Update the weights of the networks by "mucking up the blood" or doing, essentially,
    exponential moving average.

    Args:
        teacher (nn.Module): The teacher module for update.
        student (nn.Module): The student module used for updating the parameters of the teacher.
        sigma (t.Tensor): The value of movement towards the student weights.
    """
    # Disable gradient calculation as this just forces over the resultant weights
    #   the student to the teacher.
    with t.no_grad():
        # Lock the momentum to be activated inside of a sigmoid like function
        epsig:t.Tensor = csigmoid(sigma.detach())
        # No need to always re-calc
        aepsig:t.Tensor = 1 - epsig
        
        # Update each parameter in each module
        for tparam, sparam in zip(teacher.parameters(), student.parameters()):
            # Reduce the weights by the momentum in a decaying fashion
            tparam.mul_(epsig)

            # Add the student weights using the momentum in a compounding fashion
            tparam.add_(aepsig * sparam.detach())
