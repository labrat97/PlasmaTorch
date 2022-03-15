import sys
import os
import importlib.util
import torch
from plasmatorch import defaults, Smear

TBATCH:int = 3
TEST_DEFAULT_COMPLEX:bool = True

TEST_FFT_SAMPLES:int = defaults.DEFAULT_FFT_SAMPLES
TEST_FFT_SMALL_BATCHES:int = defaults.SMALL_FFT_BATCH
TEST_FFT_SMALL_SAMPLES:int = defaults.SMALL_FFT_SAMPLES

def getsmear(dtype:torch.dtype, ones:bool = False):
    # The smear to test
    smear = Smear(samples=TEST_FFT_SMALL_SAMPLES, lowerScalar=1/16, upperScalar=1/16, dtype=dtype)
    x = torch.zeros((TBATCH,1), dtype=dtype)
    if ones: x = torch.ones_like(x)
    return smear.forward(x), smear
