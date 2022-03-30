from .defaults import *
from .sizing import *
from .math import nantonum


class SignalCompressor(nn.Module):
    """
    Compresses a signal with a 1D lens like system into the monster grouping,
    then resignal down to the appropriate size. When signalling back outwards
    use the inverse of the encoding lens and assert some flow of a datatype.
    """
    def __init__(self, supersamples:int=DEFAULT_SIGNAL_COMPRESSION_SUPERSAMPLES, 
        padding:int=DEFAULT_SIGNAL_COMPRESSION_PADDING, dtype:t.dtype=DEFAULT_DTYPE):

