from ..defaults import *

class WeightedAggregator(nn.Module):
    def __init__(self, samplesA:int=DEFAULT_FFT_SAMPLES, dimA:int=-1, samplesB:int=DEFAULT_FFT_SAMPLES, dimB:int=-1):
        super(WeightedAggregator, self).__init__()
