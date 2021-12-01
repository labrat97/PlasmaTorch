import torch as t
import torch.nn.functional as nnf
from torch.jit import script as ts
from typing import Tuple

@ts
def paddim(x:t.Tensor, lowpad:int, highpad:int, dim:int, mode:str='constant', value:float=0.0):
    # Transpose the dim of interest to the end of the tensor
    xT:t.Tensor = x.transpose(dim, -1)

    # Pad the dimension with the padding parameters
    xPad:t.Tensor = nnf.pad(xT, (lowpad, highpad), mode=mode, value=value)
    
    # Put the dimension back in the appropriate place
    return xPad.transpose(dim, -1)

@ts
def dimmatch(a:t.Tensor, b:t.Tensor, dim:int, mode:str='constant', value:float=0.0) -> Tuple[t.Tensor, t.Tensor]:
    # Extract sizing parameters
    asize:int = a.size()[dim]
    bsize:int = b.size()[dim]

    # Pad whichever dim needs padding
    if asize < bsize:
        return paddim(a, 0, bsize-asize, dim=dim, mode=mode, value=value), b
    elif bsize < asize:
        return a, paddim(b, 0, asize-bsize, dim=dim, mode=mode, value=value)
    return a, b
