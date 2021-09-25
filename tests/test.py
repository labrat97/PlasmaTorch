import sys
import os
import importlib.util

PARENT_DIR_NAME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(f'..{os.path.sep}{PARENT_DIR_NAME}')
spec = importlib.util.spec_from_file_location('plasmatorch', PARENT_DIR_NAME+os.path.sep+'__init__.py')
modul = importlib.util.module_from_spec(spec)
sys.modules['plasmatorch'] = modul
spec.loader.exec_module(modul)

from plasmatorch import defaults

KLYBATCH:int = 23
TEST_DEFAULT_COMPLEX:bool = True
TEST_FFT_SAMPLES:int = defaults.DEFAULT_FFT_SAMPLES + 1
