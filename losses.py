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
def hypercorrelation(x:t.Tensor, y:t.Tensor, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE, dim:int=-1, fullOutput:bool=False, extraTransform:bool=False):
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

    # Return as a single tensor in the previously commented/written order
    return t.stack((corrmean, corrmin, corrmax, corrmedian, corrmode), dim=-1)

@ts
def skeeter(teacher:t.Tensor, student:t.Tensor, center:t.Tensor, teacherTemp:float=1., studentTemp:float=1., dim:int=-1, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
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
    corrmean = hypercorr[...,0]
    corrmedian = hypercorr[..., 3]
    corrmode = hypercorr[..., 4]
