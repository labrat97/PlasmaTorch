from torch import chunk
from .defaults import *
from .math import xbias
from .conversions import nantonum



@ts
def unflatten(x:t.Tensor, dim:int, size:List[int]) -> t.Tensor:
    """Run the equivalent of a functional unflatten on the provided signal.

    Args:
        x (t.Tensor): The signal to unflatten.
        dim (int): The dimension to unflatten in the signal.
        size (List[int]): The new expanded size of the unflattened dimension.

    Returns:
        t.Tensor: The unflattened signal.
    """
    # Assert that the tensor unflattens to the appropriate size at the provided dim
    numel:int = 1
    for n in size:
        numel = numel * n
    assert numel == x.size(dim), f'{numel} != {x.size(dim)}'

    # Accumulate the result
    y:t.Tensor = x
    xdim:int = x.dim()

    # Unfold the specified dimension for each descriptive dimension of size
    # The dimensions must be unflattened in a queue, so flip the order of the list
    #   and reverse the indexing system to be end-relative
    for idx, n in enumerate(size[::-1]):
        target:int = -xdim + dim - idx
        y = y.unfold(target, n, n).movedim(-1, target)

    # Return fully unflattened tensor, squeezing the leftover element dim
    return y.squeeze(dim)


@ts
def resignal(x:t.Tensor, samples:int, dim:int=-1) -> t.Tensor:
    """Takes the input signal, finds the basis frequency responses for the signal, then applies said
    responses to a properly sized dimension.

    Args:
        x (t.Tensor): The signal to resignal.
        samples (int): The amount of samples to use in the selected dimension of the input signal.
        dim (int, optional): The dimension to resignal in x. Defaults to -1.

    Returns:
        t.Tensor: The resignalled input signal.
    """
    # I know there are redundant `if` calls in this equation. Due to the out of order
    #     size aquisition, this should have minimal performance impact relative to the actual 
    #     computation and improves readability.
    xcomp:bool = x.is_complex()

    # Sample the constructing frequencies and phases, zero padding. Get rid of
    #     inifinite values while evaluating.
    if xcomp:
        xfft:t.Tensor = tfft.fft(nantonum(x), dim=dim, n=x.size(dim), norm='ortho')
    else:
        xfft:t.Tensor = tfft.rfft(nantonum(x), dim=dim, n=x.size(dim), norm='ortho')

    # Put the samples back to smearwise where no zero padding exists
    # This can be done because this is a natural signal
    # No data is lost or obscured in theory during upsampling, downsampling loses higher frequencies
    if xcomp:
        y:t.Tensor = tfft.ifft(xfft, dim=dim, n=samples, norm='ortho')
    else:
        y:t.Tensor = tfft.irfft(xfft, dim=dim, n=samples, norm='ortho')

    return y


@ts
def paddim(x:t.Tensor, lowpad:int, highpad:int, dim:int, mode:str=DEFAULT_PADDING) -> t.Tensor:
    """Pad the selected dimension with the torch.nnf pad() method internally.

    Args:
        x (t.Tensor): The signal to pad.
        lowpad (int): The number of samples to pad at the start of the tensor's indices.
        highpad (int): The number of samples to pad at the end of the tensor's indices.
        dim (int): The dimension to pad.
        mode (str, optional): The padding mode to use for the specified dim. Defaults to DEFAULT_PADDING.

    Returns:
        t.Tensor: The padded signal.
    """
    # Transpose the dim of interest to the end of the tensor
    wx:t.Tensor = x.transpose(dim, -1)
    unsqueezes:int = 0

    # Force a certain level of dimensions
    while wx.dim() <= 2:
        wx.unsqueeze_(0)
        unsqueezes += 1
    chunkShape:List[int] = wx.size()[:-2]
    wx = wx.flatten(start_dim=0, end_dim=-3) 

    # Prep the padding description
    padform = (lowpad, highpad)

    # Handle number complexity
    xcomp:bool = wx.is_complex()

    # Pad the dimension with the padding parameters
    if xcomp:
        xpadr:t.Tensor = nnf.pad(wx.real, pad=padform, mode=mode)
        xpadi:t.Tensor = nnf.pad(wx.imag, pad=padform, mode=mode)
        xpad:t.Tensor = t.view_as_complex(t.stack((xpadr, xpadi), dim=-1))
    else:
        xpad:t.Tensor = nnf.pad(wx, pad=padform, mode=mode)

    # Put the dimension structure back in the appropriate place
    xpad = unflatten(xpad, dim=0, size=chunkShape)
    for _ in range(unsqueezes):
        xpad.squeeze_(0)
    return xpad.transpose(dim, -1)


@ts
def dimmatch(a:t.Tensor, b:t.Tensor, dim:int) -> Tuple[t.Tensor, t.Tensor]:
    """Make the selected dimension have the same size between two tensors using
    the `resignal()` function.

    Args:
        a (t.Tensor): The first tensor to match.
        b (t.Tensor): The second tensor to match.
        dim (int): The dimension to perform the matching operation on.
        
    Returns:
        Tuple[t.Tensor, t.Tensor]: Signals a and b, respectively, with the selected dim having matching sample counts.
    """
    # Extract sizing parameters
    asize:int = a.size(dim)
    bsize:int = b.size(dim)

    # Pad whichever dim needs padding
    if asize < bsize:
        return resignal(a, samples=bsize, dim=dim), b
    elif bsize < asize:
        return a, resignal(b, samples=asize, dim=dim)
    # ==
    return a, b


@ts 
def weightedResample(x:t.Tensor, pos:t.Tensor, dim:int=-1, ortho:bool=True) -> t.Tensor:
    """Resample the specified dimension with the position offsets provided.

    Args:
        x (t.Tensor): The signal to perform the resampling on.
        pos (t.Tensor): The positions to sample from per sample.
        dim (int, optional): The dimension to resample. Defaults to -1.
        ortho (bool, optional): If True, 0.0 equates to a perfect per-element passthrough. If False, 0.0 is the default center from the align_corners option in grid_sample(). Defaults to True.

    Returns:
        t.Tensor: The resampled input signal.
    """
    # Make sure there isn't an imaginary lens vector
    assert not t.is_complex(pos)
    xdims:int = x.dim()
    assert xdims > 0

    # Make sure the lens is for a 1D system
    lensSize = pos.size()
    lensDims = pos.dim()
    assert (lensDims <= 2) and (lensDims != 0)
    # [batch, pos]

    # Make sure the dim can be referenced
    slensSize = lensSize if pos.dim() == 1 else pos.squeeze(0).size()
    slensDims:int = len(slensSize)
    assert not (dim == 0 and slensDims > 1)
    
    # Assert the batch dimension isn't poorly sized. This is needed because the above created
    #   batch is actually just an aggregate of all of the unaccounted for dimensions in the resample
    assert (slensDims == 1) or (slensSize[0] == x.size(0)), f'{lensSize}\t->||<-\t{x.size()}'

    # Ensure that there is a batch
    batchOffset:int = int(pos.dim() == 2)
    
    # Put the dim in the appropriate place for sampling
    dimout:t.Tensor = x.movedim(dim, -1)
    extrabatchDim:bool = (batchOffset != 0) and (xdims == 2)
    if extrabatchDim: dimout.unsqueeze_(-2)

    # Flatten the higher dimensional, less significant, batches into one dim
    flatsize = dimout.size()[batchOffset:-1]
    flatbatch:t.Tensor = t.flatten(dimout, start_dim=batchOffset, end_dim=-2) # [..., x] -> [..., F, x]
    if batchOffset == 0: flatbatch.unsqueeze_(0) # [..., F, x] -> [b, F, x]

    # Put channels in the appropriate place for complex numbers
    if x.is_complex():
        wx:t.Tensor = t.view_as_real(flatbatch).transpose(-1, -2) # [b, F, x] -> [b, F, (c)2, x]
    else:
        wx:t.Tensor = flatbatch.unsqueeze(-2) # [b, F, x] -> [b, F, (c)1, x]
    wx.unsqueeze_(-2) # [b, F, c, 1, x]
    wxsize = wx.size() # [batch, flat, channels, units(rows:1), units(cols)]

    # Add dummy dimensions for sampling
    wl:t.Tensor = pos.unsqueeze(0) if lensDims == 1 else pos # [..., p] -> [b, p]
    wl = wl.unsqueeze(-2).unsqueeze(-2) # [b, p] -> [b, 1, 1, p]
    wl = t.cat([wl] * wx.size(1), dim=1) # [b, 1, 1, p] -> [b, F, 1, p]
    wl = t.stack((wl, t.zeros_like(wl)), dim=-1) # [b, F, 1, p] -> [b, F, 1, p, [x, (y)0]]
    # [batch, flat, units(rows:1), positions, sample position[x, (y)0]]

    # Get ready for resampling
    result:t.Tensor = t.zeros(wxsize[:-1] + [lensSize[-1]], 
        dtype=wx.dtype, device=x.device) # [batch, flat, channels, units(rows:1), positions]
    
    # Set up an orthonormal lookup system
    if ortho:
        posCount:int = result.size(-1)
        ortholut:t.Tensor = (((2. * xbias(posCount)) / (posCount - 1.)) - 1.).unsqueeze(0) # [1, positions]
    # Keep the normal [-1.0, 1.0] corner alignment
    else:
        ortholut:t.Tensor = t.zeros(result.size(-1)).unsqueeze(0) # [1, positions]
        assert ortholut.size() == t.Size([1, result.size(-1)])
    ortholut = t.stack([ortholut, t.zeros_like(ortholut)], dim=-1) # [1, p] -> [1, p, [x, (y)0]]

    # Resample each batch
    for idx in range(wx.size(0)):
        wwx = wx[idx] # [F, c, 1, x]
        wwl = wl[idx] + ortholut # [F, 1, p, [x, (y)0]] + [1, p, [x, (y)0]] -> [F, 1, p, [x, (y)0]]
        assert wwl.size() == wl[idx].size()
        result[idx] = nnf.grid_sample(wwx, wwl, mode='bilinear', padding_mode='reflection', align_corners=True)
        # [flat, channels, units(rows:1), positions]

    # Format the result
    result.squeeze_(-2) # [b, F, c, 1, p] -> [b, F, c, p]
    result.transpose_(-1, -2) # [b, F, c, p] -> [b, F, p, c]

    # Reintroduce complexity
    if x.is_complex(): # [b, F, p, c] -> [b, F, p]
        # Had to restack due to poor stride issue
        result = t.view_as_complex(t.stack((result[..., 0], result[..., 1]), dim=-1))
    else:
        result = result.squeeze(-1)

    # Restore original size
    if batchOffset == 0: # [b, F, p] -> [..., F, p]
        result = result.squeeze(0)
    result = unflatten(result, batchOffset, flatsize) # [..., b, y] -> [..., y]

    # Reapply the computed dimension to the appropriate dimension according to the
    #   seeding tensor.
    if extrabatchDim: result.squeeze_(-2)
    return result.movedim(-1, dim)
