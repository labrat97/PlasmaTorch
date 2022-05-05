from .defaults import *
from .math import nantonum, xbias


@ts
def paddim(x:t.Tensor, lowpad:int, highpad:int, dim:int, mode:str=DEFAULT_PADDING, value:float=0.0):
    # Transpose the dim of interest to the end of the tensor
    xT:t.Tensor = x.transpose(dim, -1)

    # Pad the dimension with the padding parameters
    xPad:t.Tensor = nnf.pad(xT, (lowpad, highpad), mode=mode, value=value)
    
    # Put the dimension back in the appropriate place
    return xPad.transpose(dim, -1)

@ts
def dimmatch(a:t.Tensor, b:t.Tensor, dim:int, mode:str=DEFAULT_PADDING, value:float=0.0) -> Tuple[t.Tensor, t.Tensor]:
    # Extract sizing parameters
    asize:int = a.size()[dim]
    bsize:int = b.size()[dim]

    # Pad whichever dim needs padding
    if asize < bsize:
        return paddim(a, 0, bsize-asize, dim=dim, mode=mode, value=value), b
    elif bsize < asize:
        return a, paddim(b, 0, asize-bsize, dim=dim, mode=mode, value=value)
    return a, b

@ts
def unflatten(x:t.Tensor, dim:int, size:List[int]):
    # Assert that the tensor unflattens to the appropriate size at the provided dim
    numel:int = 1
    for n in size:
        numel = numel * n
    assert numel == x.size(dim)

    # Accumulate the result
    y:t.Tensor = x

    # Unfold the specified dimension for each descriptive dimension of size
    for idx, n in enumerate(list(size)):
        target:int = dim + idx
        y = y.unfold(target, n, n).movedim(-1, target)

    # Return fully unflattened tensor
    return y

@ts
def resignal(x:t.Tensor, samples:int, dim:int=-1) -> t.Tensor:
    # I know there are redundant `if` calls in this equation. Due to the out of order
    #     size aquisition, this should have minimal performance impact relative to the actual 
    #     computation and improves readability.
    xcomp:bool = x.is_complex()

    # Sample the constructing frequencies and phases, zero padding. Get rid of
    #     inifinite values while evaluating.
    if xcomp:
        xfft:t.Tensor = t.fft.fft(nantonum(x), dim=dim, n=x.size(dim))
    else:
        xfft:t.Tensor = t.fft.rfft(nantonum(x), dim=dim, n=x.size(dim))

    # Put the samples back to smearwise where no zero padding exists
    # This can be done because this is a natural signal
    # No data is lost or obscured in theory during upsampling, downsampling loses higher frequencies
    if xcomp:
        y:t.Tensor = t.fft.ifft(xfft, dim=dim, n=samples)
    else:
        y:t.Tensor = t.fft.irfft(xfft, dim=dim, n=samples)

    return y

@ts 
def weightedResample(x:t.Tensor, lens:t.Tensor, dim:int=-1) -> t.Tensor:
    # Make sure there isn't an imaginary lens vector
    assert not t.is_complex(lens)
    # Make sure the dim can be referenced
    lensSize = lens.size()
    assert not (dim == 0 and len(lensSize) > 1)
    # Make sure the lens is for a 1D system
    assert len(lensSize) <= 2
    slens = lens.squeeze(0)
    lensSize = slens.size()

    # Ensure that there is a batch
    batchOffset:int = int(len(lensSize) == 2)
    # Put the dim in the appropriate place for sampling
    dimout:t.Tensor = x.transpose(dim, -1)
    # Flatten the batch into one dim
    flatbatch:t.Tensor = t.flatten(dimout, start_dim=batchOffset, end_dim=-2) # [..., x] -> [..., b, x]

    # Assert the batch dimension isn't poorly sized. This is needed because the above created
    #   batch is actually just an aggregate of all of the unaccounted for dimensions in the resample
    assert (len(lensSize) == 1) or (lensSize[0] == flatbatch.size(0))

    # Put channels in the appropriate place for complex numbers
    if t.is_complex(x):
        wx:t.Tensor = t.view_as_real(flatbatch).movedim(-1, -2) # [..., b, x] -> [..., b, 2, x]
    else:
        wx:t.Tensor = t.unsqueeze(flatbatch, -2) # [..., b, x] -> [..., b, 1, x]

    # Add dummy dimensions for sampling
    wx = wx.unsqueeze(-2) # [..., b, c, x] -> [..., b, c, 1, x]
    wxsize = wx.size()
    wl = slens.unsqueeze(-2).unsqueeze(-2).unsqueeze(-1) # [..., y] -> [..., 1, 1, y, 1]
    wl = t.cat([wl] * wx.size(-4), dim=-4) # [..., 1, 1, y, 1] -> [..., b, 1, y, 1]
    wl = t.cat((wl, t.zeros_like(wl)), dim=-1) # [..., b, 1, y, 1] -> [..., b, 1, y, [x_iter, 0]]
    wlsize = wl.size()

    # Get ready for resampling
    result:t.Tensor = t.zeros(wxsize[:-1] + [lensSize[-1]], 
        dtype=x.dtype, device=x.device) # [..., b, c, 1, x]
    poslut:t.Tensor = (((2. * xbias(wlsize[-2])) / (wlsize[-2] - 1.)) - 1.).unsqueeze(-1)

    # Resample each batch
    if batchOffset == 0:
        wx = wx.unsqueeze(0) # [..., b, c, 1, x] -> [B, b, c, 1, x]
        wl = wl.unsqueeze(0) # [..., b, 1, y, [x_iter, 0]] -> [B, b, 1, y, [x_iter, 0]]
        result = result.unsqueeze(0) # [..., b, c, 1, x] -> [B, b, c, 1, y]
    for idx in range(wx.size(0)):
        wwx = wx[idx] # [b, c, 1, x]
        wwl = wl[idx] + poslut # [b, 1, x, [x_iter, 0]] centered at 0
        result[idx] = nnf.grid_sample(wwx, wwl, mode='bilinear', padding_mode='reflection', align_corners=True)

    # Format the result
    if batchOffset == 0:
        result = result.squeeze(0) # [B, b, c, 1, y] -> [..., b, c, 1, y]
    result = result.movedim(-3, -1) # [..., b, c, 1, y] -> [..., b, 1, y, c]
    if t.is_complex(x): # [..., b, 1, y, c] -> [..., b, 1, y]
        result = t.view_as_complex(result)
    else:
        result = result.squeeze(-1)
    result = result.squeeze(-2) # [..., b, 1, y] -> [..., b, y]
    result = unflatten(result, batchOffset, dimout.size()[:-1]) # [..., b, y] -> [..., y]

    # Reapply the computed dimension to the appropriate dimension according to the
    #   seeding tensor.
    return result.transpose(-1, dim)