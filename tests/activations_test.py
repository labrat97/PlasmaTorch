from defaults import DEFAULT_DTYPE, DEFAULT_COMPLEX_DTYPE, DEFAULT_FFT_SAMPLES, DEFAULT_SPACE_PRIME
import unittest
import test

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from plasmatorch import *

class LissajousTest(unittest.TestCase):
    def testSizing(self):
        # Default conversions and logits
        x = torch.ones((1, test.KLYBATCH, 1), dtype=DEFAULT_DTYPE)
        xc = torch.ones((1, test.KLYBATCH, 1), dtype=DEFAULT_COMPLEX_DTYPE)
        
        one = torch.ones((1, 1, 1), dtype=DEFAULT_DTYPE)
        onec = torch.ones((1, 1, 1), dtype=DEFAULT_COMPLEX_DTYPE)

        s, _ = test.getsmear(DEFAULT_DTYPE, ones=True)
        sc, _ = test.getsmear(DEFAULT_COMPLEX_DTYPE, ones=True)
        s.unsqueeze_(0)
        sc.unsqueeze_(0)

        # The curves to test
        lisa = Lissajous(size=test.KLYBATCH, dtype=DEFAULT_DTYPE)
        lisac = Lissajous(size=test.KLYBATCH, dtype=DEFAULT_COMPLEX_DTYPE)

        # Test the 23 curve pass
        lx = lisa.forward(x, oneD=True)
        lxl = lisa.forward(x, oneD=False)
        self.assertTrue(lx.size() == (1, test.KLYBATCH, test.KLYBATCH, 1), msg=f'size: {lx.size()}')
        self.assertTrue(lxl.size() == (1, test.KLYBATCH, 1), msg=f'size: {lxl.size()}')

        lxc = lisac.forward(xc, oneD=True)
        lxcl = lisac.forward(xc, oneD=False)
        self.assertTrue(lxc.size() == (1, test.KLYBATCH, test.KLYBATCH, 1), msg=f'size: {lxc.size()}')
        self.assertTrue(lxcl.size() == (1, test.KLYBATCH, 1), msg=f'size: {lxcl.size()}')

        # Test the signle logit pass
        lone = lisa.forward(one, oneD=True)
        lonec = lisac.forward(onec, oneD=True)
        self.assertTrue(lone.size() == (1, 1, test.KLYBATCH, 1), msg=f'size: {lone.size()}')
        self.assertTrue(lonec.size() == (1, 1, test.KLYBATCH, 1), msg=f'size: {lonec.size()}')

        # Test the 23 smear-curve pass
        ls = lisa.forward(s, oneD=True)
        lsl = lisa.forward(s, oneD=False)
        self.assertTrue(ls.size() == (1, test.KLYBATCH, test.KLYBATCH, s.size()[-1]), msg=f'size: {ls.size()}')
        self.assertTrue(lsl.size() == (1, test.KLYBATCH, s.size()[-1]), msg=f'size: {lsl.size()}')

        lsc = lisac.forward(sc, oneD=True)
        lscl = lisac.forward(sc, oneD=False)
        self.assertTrue(lsc.size() == (1, test.KLYBATCH, test.KLYBATCH, sc.size()[-1]), msg=f'size: {lsc.size()}')
        self.assertTrue(lscl.size() == (1, test.KLYBATCH, sc.size()[-1]), msg=f'size: {lscl.size()}')

    def testValues(self):
        x = torch.exp(torch.randn((test.KLYBATCH, DEFAULT_SPACE_PRIME, test.TEST_FFT_SAMPLES), dtype=DEFAULT_DTYPE))
        xc = torch.exp(torch.randn((test.KLYBATCH, DEFAULT_SPACE_PRIME, test.TEST_FFT_SAMPLES), dtype=DEFAULT_COMPLEX_DTYPE))

        # Assuming zero value for initial training
        lisa = Lissajous(size=DEFAULT_SPACE_PRIME, dtype=DEFAULT_DTYPE)
        lisac = Lissajous(size=DEFAULT_SPACE_PRIME, dtype=DEFAULT_COMPLEX_DTYPE)

        # Should have no delta in any case
        l00 = lisa.forward(torch.zeros_like(x), oneD=True)
        l01 = lisa.forward(x, oneD=True)
        self.assertTrue(torch.all(l00 == l01), msg="Non-zero initialization.")
        ll00 = lisa.forward(torch.zeros_like(x), oneD=False)
        ll01 = lisa.forward(x, oneD=False)
        self.assertTrue(torch.all(ll00 == ll01), msg="Non-zero initialization.")
        lc00 = lisac.forward(torch.zeros_like(xc), oneD=True)
        lc01 = lisac.forward(xc, oneD=True)
        self.assertTrue(torch.all(lc00 == lc01), msg="Non-zero initialization.")
        lcl00 = lisac.forward(torch.zeros_like(xc), oneD=False)
        lcl01 = lisac.forward(xc, oneD=False)
        self.assertTrue(torch.all(lcl00 == lcl01), msg="Non-zero initialization.")

        # Add an irregular angular frequency of 1
        lisa.frequency = nn.Parameter(lisa.frequency + 1)
        lisac.frequency = nn.Parameter(lisac.frequency + 1)

        # Should have delta 
        l10 = lisa.forward(torch.zeros_like(x), oneD=True)
        l11 = lisa.forward(x, oneD=True)
        self.assertTrue(torch.all(l10 != l11), msg="Frequency delta not working (oneD, real).")
        ll10 = lisa.forward(torch.zeros_like(x), oneD=False)
        ll11 = lisa.forward(x, oneD=False)
        self.assertTrue(torch.all(ll10 != ll11), msg="Frequency delta not working (!oneD, real).")
        self.assertTrue(torch.all(ll11 == torch.cos(x)), msg="Cos values don't check out for real values.")
        lc10 = lisac.forward(torch.zeros_like(xc), oneD=True)
        lc11 = lisac.forward(xc, oneD=True)
        self.assertTrue(torch.all(lc10 != lc11), msg="Frequency delta not working (oneD, complex).")
        lcl10 = lisac.forward(torch.zeros_like(xc), oneD=False)
        lcl11 = lisac.forward(xc, oneD=False)
        self.assertTrue(torch.all(lcl10 != lcl11), msg="Frequency delta not working (!oneD, complex).")
        self.assertTrue(torch.all(lcl11 == torch.cos(xc)), msg="Cos values don't check out for complex values.")

        # Assuming the other things work, this should complete testing
        lisa.frequency = nn.Parameter(lisa.frequency * 0)
        lisac.frequency = nn.Parameter(lisac.frequency * 0)

        # Should have all of the same delta due to phasing
        phi0 = lisa.forward(torch.zeros_like(x), oneD=True)
        self.assertTrue(torch.all(phi0[:,:,:,:-1] == phi0[:,:,:,1:]), msg='Phi not consistent (oneD, real).')
        phil0 = lisa.forward(torch.zeros_like(x), oneD=False)
        self.assertTrue(torch.all(phil0[:,:,:-1] == phil0[:,:,1:]), msg='Phi not consistent (!oneD, real).')
        phic0 = lisac.forward(torch.zeros_like(xc), oneD=True)
        self.assertTrue(torch.all(phic0[:,:,:,:-1] == phic0[:,:,:,1:]), msg='Phi not consistent (oneD, complex).')
        phicl0 = lisac.forward(torch.zeros_like(xc), oneD=False)
        self.assertTrue(torch.all(phicl0[:,:,:-1] == phicl0[:,:,1:]), msg='Phi not consistent (!oneD, complex).')
