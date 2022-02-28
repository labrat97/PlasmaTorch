import sys
import os
import importlib.util
import torch
from plasmatorch import defaults, Smear

KYLABATCH:int = 23
TEST_DEFAULT_COMPLEX:bool = True
TEST_FFT_SAMPLES:int = defaults.DEFAULT_FFT_SAMPLES + 1

def getsmear(dtype:torch.dtype, ones:bool = False):
    # The smear to test
    smear = Smear(samples=TEST_FFT_SAMPLES, lowerScalar=1/16, upperScalar=1/16, dtype=dtype)
    x = torch.zeros((KYLABATCH,1), dtype=dtype)
    if ones: x = torch.ones_like(x)
    return smear.forward(x), smear
