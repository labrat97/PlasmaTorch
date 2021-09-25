from ..defaults import *

from abc import ABC, abstractmethod


KLYBATCH = 23
DEFAULT_COMPLEX = True
TEST_FFT_SAMPLES = DEFAULT_FFT_SAMPLES + 1



class Test(ABC):
    tests = []
    def __init__(self, hardFail:bool):
        super(Test, self).__init__()
        Test.tests.append(self)
        self.complex:bool = DEFAULT_COMPLEX
        self.hardFail:bool = hardFail

    @abstractmethod
    def test(self):
        pass

    def log(self, msg:str, passed:bool):
        # Print pass fail message
        if passed:
            passMark = 'Y'
            passMsg = 'PASSED'
        else:
            passMark = 'N'
            passMsg = 'FAILED'
        print(f'[{passMark}]   {msg}\t->\t << {passMsg} >>')
        
        # Stop execution of the program
        if self.hardFail and not passed:
            assert passed
