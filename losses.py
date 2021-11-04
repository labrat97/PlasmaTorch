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
    corr:t.Tensor = tfft.ifft(xfft * yfft.conj()).mean(dim=dim).abs()

    return corr

# Unfortunately, the indexing this currently required doesn't support torch script jit compilation
# TODO: Torch script support
def hypercorrelation(x:t.Tensor, y:t.Tensor, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE, dim:int=-1, fullOutput:bool=False):
    # Some size assertions/extractions
    xsize = x.size()
    ysize = y.size()
    assert len(x.size()) == len(y.size())

    # Because ffts are kinda supposed to reverse themselves, running the signals
    # through two permuatations from both ffts and iffts should give the full spectral
    # evaluation of the input signals
    FFT_LAYERS:int = 3
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
    xfftTape:t.Tensor = t.zeros((FFT_TAPE_LENGTH, *xsize[:-1], samples), dtype=cdtype)
    yfftTape:t.Tensor = t.zeros((FFT_TAPE_LENGTH, *ysize[:-1], samples), dtype=cdtype)
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
        unfiltered:t.Tensor = t.zeros((*xfftTape.size()[:-1], FFT_TAPE_LENGTH, FFT_TAPE_LENGTH), dtype=cdtype)
    else:
        unfiltered:t.Tensor = t.zeros((*yfftTape.size()[:-1], FFT_TAPE_LENGTH, FFT_TAPE_LENGTH), dtype=cdtype)
    for xidx in range(FFT_TAPE_LENGTH):
        for yidx in range(FFT_TAPE_LENGTH):
            unfiltered[..., xidx, yidx] = correlation(xfftTape[xidx], yfftTape[idx])
    
    # Quick exit from the computation
    if fullOutput:
        return unfiltered
    
    # Find mean, min, max, median, and mode
    corrmean:t.Tensor = unfiltered.mean(dim=(-1, -2))
    corrmin:t.Tensor = unfiltered.min(dim=(-1, -2))
    corrmax:t.Tensor = unfiltered.max(dim=(-1, -2))
    corrmedian:t.Tensor = unfiltered.median(dim=(-1, -2))
    corrmode:t.Tensor = unfiltered.mode(dim=(-1, -2))

    # Return as a single tensor in the previously commented/written order
    return t.stack((corrmean, corrmin, corrmax, corrmedian, corrmode), dim=-1)
