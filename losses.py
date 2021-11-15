from .math import *
from .activations import *
from .distributions import *
from .defaults import *
from .conversions import *

import torch as t
import torch.nn as nn
import torch.nn.functional as nnf
from torch.jit import script as ts
import torch.fft as tfft

@ts
def correlation(x:t.Tensor, y:t.Tensor, dim:int=-1) -> t.Tensor:
    """Find the standard cross-correlation between the natural signals provided.

    Args:
        x (t.Tensor): One set of signals to use for the final computation.
        y (t.Tensor): Another set of signals to use for the final compuation.
        dim (int, optional): The dimension to apply the computation to. Defaults to -1.

    Returns:
        t.Tensor: The cross-correlation of the two input signals.
    """
    # Some size assertions/extractions
    xsize = x.size()
    ysize = y.size()
    assert len(x.size()) == len(y.size())
    samples:int = max(xsize[dim], ysize[dim])

    # Find basis of the signals
    xfft:t.Tensor = tfft.fft(x, n=samples)
    yfft:t.Tensor = tfft.fft(y, n=samples)

    # Calculate the correlation
    corr:t.Tensor = tfft.ifft(xfft * yfft.conj())

    return corr


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
        tempfft:t.Tensor = tfft.fft(y, n=samples, dim=dim)
        y = tfft.ifft(tempfft, n=samples, dim=dim)
    elif ysize[dim] > xsize[dim]:
        samples:int = ysize[dim]
        tempfft:t.Tensor = tfft.fft(x, n=samples, dim=dim)
        x = tfft.ifft(tempfft, n=samples, dim=dim)
    else:
        samples:int = xsize[dim]

    # Create the storage needed for the output signals and populate the center
    tapeConstructor:t.Tensor = t.ones((1, FFT_TAPE_LENGTH), dtype=cdtype)
    xfftTape:t.Tensor = t.zeros_like(x).unsqueeze(-1) @ tapeConstructor
    yfftTape:t.Tensor = t.zeros_like(y).unsqueeze(-1) @ tapeConstructor
    assert torch.is_complex(xfftTape)
    xfftTape[FFT_TAPE_CENTER] = torch.view_as_complex(x)
    yfftTape[FFT_TAPE_CENTER] = torch.view_as_complex(y)

    # Deconstruct signals
    for tape in [xfftTape, yfftTape]:
        for idx in range(1, FFT_LAYERS + 1):
            tape[FFT_TAPE_CENTER + idx] = tfft.fft(tape[FFT_TAPE_CENTER + idx - 1], n=samples, dim=dim)
            tape[FFT_TAPE_CENTER - idx] = tfft.fft(tape[FFT_TAPE_CENTER + idx + 1], n=samples, dim=dim)
    
    # Apply full cross correlation between different representations of the input signals
    if x.numel() > y.numel():
        unfiltered:t.Tensor = t.zeros_like(xfftTape).unsqueeze(-1) @ tapeConstructor
    else:
        unfiltered:t.Tensor = t.zeros_like(yfftTape).unsqueeze(-1) @ tapeConstructor
    for xidx in range(FFT_TAPE_LENGTH):
        for yidx in range(FFT_TAPE_LENGTH):
            unfiltered[..., xidx, yidx] = correlation(xfftTape[..., xidx], yfftTape[..., yidx])
    
    # Quick exit from the computation
    if fullOutput:
        return unfiltered
    
    # Find mean, min, max, median, and mode
    corrmean:t.Tensor = unfiltered.mean(dim=-1).mean(dim=-1)
    corrmin:t.Tensor = unfiltered.min(dim=-1)[0].min(dim=-1)[0]
    corrmax:t.Tensor = unfiltered.max(dim=-1)[0].max(dim=-1)[0]
    corrmedian:t.Tensor = unfiltered.median(dim=-1)[0].median(dim=-1)[0]
    corrmode:t.Tensor = unfiltered.mode(dim=-1)[0].mode(dim=-1)[0]
    corrmse:t.Tensor = (unfiltered * unfiltered).mean(dim=-1)

    # Return as a single tensor in the previously commented/written order
    return t.stack((corrmean, corrmin, corrmax, corrmedian, corrmode, corrmse), dim=-1)

# Some constants for indexing the hypercorrelation function
HYDX_CORRMEAN:int = 0
HYDX_CORRMIN:int = 1
HYDX_CORRMAX:int = 2
HYDX_CORRMEDIAN:int = 3
HYDX_CORRMODE:int = 4
HYDX_CORRMSE:int = 5


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
    
    # Get the basis of the incoming signals
    tenurefft = tfft.fft(tenure, n=samples, dim=dim)
    studentfft = tfft.fft(student, n=samples, dim=dim)
    
    # Soften and sharpen the inputs as somewhat seen in the DINO loss from (what is likely now formerly known as) Facebook AI
    softtenfft:t.Tensor = isoftmax((tenurefft - center) / teacherTemp, dim=dim)
    softstufft:t.Tensor = isoftmax(studentfft / studentTemp, dim=dim)

    # Turning the signal from a frequency domain signal back to a time domain
    # signal can be temporarily ignored as hypercorrelation operates accross all
    # possible occuring orders of the time-frequency superposition
    hypercorr:t.Tensor = hypercorrelation(x=softtenfft, y=softstufft, cdtype=cdtype, \
        dim=dim, fullOutput=False, extraTransform=False)
    corrmean = 1. / hypercorr[..., HYDX_CORRMEAN]
    corrmedian = 1. / hypercorr[..., HYDX_CORRMEDIAN]
    corrmode = 1. / hypercorr[..., HYDX_CORRMODE]
    corrmse = 1. / hypercorr[..., HYDX_CORRMSE]

    # Add together the signals in the style of parallel impedance
    return -1. / (corrmean + corrmedian + corrmode + corrmse)

@ts
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
        epsig:t.Tensor = isigmoid(sigma.detach())
        # No need to always re-calc
        aepsig:t.Tensor = 1 - epsig
        
        # Update each parameter in each module
        for tparam, sparam in zip(teacher.parameters(), student.parameters()):
            # Reduce the weights by the momentum in a decaying fashion
            tparam.mul_(epsig)

            # Add the student weights using the momentum in a compounding fashion
            tparam.add_(aepsig * sparam.detach())
