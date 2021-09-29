import sys
import os
import importlib.util

# Force plasmatorch into my heart
PARENT_DIR_NAME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(f'..{os.path.sep}{PARENT_DIR_NAME}')
spec = importlib.util.spec_from_file_location('plasmatorch', PARENT_DIR_NAME+os.path.sep+'__init__.py')
modul = importlib.util.module_from_spec(spec)
sys.modules['plasmatorch'] = modul
spec.loader.exec_module(modul)

from plasmatorch import defaults, Smear

KLYBATCH:int = 23
TEST_DEFAULT_COMPLEX:bool = True
TEST_FFT_SAMPLES:int = defaults.DEFAULT_FFT_SAMPLES + 1

import torch

def getsmear(dtype:torch.dtype, ones:bool = False):
    # The smear to test
    smear = Smear(samples=TEST_FFT_SAMPLES, lowerScalar=1/16, upperScalar=1/16, dtype=dtype)
    x = torch.zeros((KLYBATCH,1), dtype=dtype)
    if ones: x = torch.ones_like(x)
    return smear.forward(x), smear
