from defaults import DEFAULT_DTYPE, DEFAULT_COMPLEX_DTYPE, DEFAULT_FFT_SAMPLES, DEFAULT_SPACE_PRIME
import unittest
import test

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from plasmatorch import *
from random import randint

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
        x = torch.randn((test.KLYBATCH, DEFAULT_SPACE_PRIME, test.TEST_FFT_SAMPLES), dtype=DEFAULT_DTYPE)
        xc = torch.randn((test.KLYBATCH, DEFAULT_SPACE_PRIME, test.TEST_FFT_SAMPLES), dtype=DEFAULT_COMPLEX_DTYPE)

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
        self.assertFalse(torch.all(l10 == l11), msg="Frequency delta not working (oneD, real).")
        ll10 = lisa.forward(torch.zeros_like(x), oneD=False)
        ll11 = lisa.forward(x, oneD=False)
        self.assertFalse(torch.all(ll10 == ll11), msg="Frequency delta not working (!oneD, real).")
        self.assertTrue(torch.all(ll11 == torch.cos(x)), msg="Cos values don't check out for real values.")
        lc10 = lisac.forward(torch.zeros_like(xc), oneD=True)
        lc11 = lisac.forward(xc, oneD=True)
        self.assertFalse(torch.all(lc10 == lc11), msg="Frequency delta not working (oneD, complex).")
        lcl10 = lisac.forward(torch.zeros_like(xc), oneD=False)
        lcl11 = lisac.forward(xc, oneD=False)
        self.assertFalse(torch.all(lcl10 == lcl11), msg="Frequency delta not working (!oneD, complex).")
        self.assertTrue(torch.all(lcl11 == icos(xc)), \
            msg="Cos values don't check out for complex values.")

        # Phase testing
        lisa.frequency = nn.Parameter(lisa.frequency * 0)
        lisa.phase = nn.Parameter(lisa.phase + 1)
        lisac.frequency = nn.Parameter(lisac.frequency * 0)
        lisac.phase = nn.Parameter(lisac.phase + 1)

        # Should have all of the same delta due to phasing
        phi0 = lisa.forward(torch.zeros_like(x), oneD=True)
        self.assertTrue(torch.all(phi0[:,:,:,:-1] == phi0[:,:,:,1:]), msg='Phi not consistent (oneD, real).')
        phil0 = lisa.forward(torch.zeros_like(x), oneD=False)
        self.assertTrue(torch.all(phil0[:,:,:-1] == phil0[:,:,1:]), msg='Phi not consistent (!oneD, real).')
        self.assertTrue(torch.all(phil0 == torch.cos(torch.ones_like(phil0))), msg="Phi values don't check out for real values.")
        phic0 = lisac.forward(torch.zeros_like(xc), oneD=True)
        self.assertTrue(torch.all(phic0[:,:,:,:-1] == phic0[:,:,:,1:]), msg='Phi not consistent (oneD, complex).')
        phicl0 = lisac.forward(torch.zeros_like(xc), oneD=False)
        self.assertTrue(torch.all(phicl0[:,:,:-1] == phicl0[:,:,1:]), msg='Phi not consistent (!oneD, complex).')
        self.assertTrue(torch.all(phicl0 == icos(torch.ones_like(xc))), msg="Phi values don't check out for complex values.")

        # Final value testing, both phase and frequency
        lisa.frequency = nn.Parameter(lisa.frequency + 1)
        lisac.frequency = nn.Parameter(lisac.frequency + 1)

        final0 = lisa.forward(x, oneD=False)
        finalc0 = lisac.forward(xc, oneD=False)
        self.assertTrue(torch.all(final0 == torch.cos(x+1)), msg="Composite values don't check out for real values.")
        self.assertTrue(torch.all(finalc0 == icos(xc+1)), \
            msg="Composite values don't check out for complex values.")


class KnotTest(unittest.TestCase):
    def testSizing(self):
        # Generate all testing datatypes
        x = torch.ones((test.KLYBATCH, 1), dtype=DEFAULT_DTYPE)
        xl = torch.ones((test.KLYBATCH, DEFAULT_SPACE_PRIME, 1), dtype=DEFAULT_DTYPE)
        xc = torch.ones((test.KLYBATCH, 1), dtype=DEFAULT_COMPLEX_DTYPE)
        xcl = torch.ones((test.KLYBATCH, DEFAULT_SPACE_PRIME, 1), dtype=DEFAULT_COMPLEX_DTYPE)

        s, smear = test.getsmear(DEFAULT_DTYPE)
        sc, smearc = test.getsmear(DEFAULT_COMPLEX_DTYPE)

        xSmear = smear.forward(x)
        xlSmear = smear.forward(xl)
        xcSmear = smearc.forward(xc)
        xclSmear = smearc.forward(xcl)

        # Hold the testing knots
        knot = Knot(knotSize=DEFAULT_SPACE_PRIME, knotDepth=test.KLYBATCH, dtype=DEFAULT_DTYPE)
        knotc = Knot(knotSize=DEFAULT_SPACE_PRIME, knotDepth=test.KLYBATCH, dtype=DEFAULT_COMPLEX_DTYPE)

        # Test sizing for testing chunk 0
        kx = knot.forward(x, oneD=True)
        self.assertTrue(kx.size() == (test.KLYBATCH, DEFAULT_SPACE_PRIME, 1), msg=f'size: {kx.size()}')
        kxll = knot.forward(xl, oneD=True)
        self.assertTrue(kxll.size() == (test.KLYBATCH, DEFAULT_SPACE_PRIME, DEFAULT_SPACE_PRIME, 1), msg=f'size: {kxll.size()}')
        kxl = knot.forward(xl, oneD=False)
        self.assertTrue(kxl.size() == (test.KLYBATCH, DEFAULT_SPACE_PRIME, 1), msg=f'size: {kxl.size()}')
        kxc = knotc.forward(xc, oneD=True)
        self.assertTrue(kxc.size() == (test.KLYBATCH, DEFAULT_SPACE_PRIME, 1), msg=f'size: {kxc.size()}')
        kxcll = knotc.forward(xcl, oneD=True)
        self.assertTrue(kxcll.size() == (test.KLYBATCH, DEFAULT_SPACE_PRIME, DEFAULT_SPACE_PRIME, 1), msg=f'size: {kxcll.size()}')
        kxcl = knotc.forward(xcl, oneD=False)
        self.assertTrue(kxcl.size() == (test.KLYBATCH, DEFAULT_SPACE_PRIME, 1), msg=f'size: {kxcl.size()}')

        # Test sizing for testing chunk 1
        ks = knot.forward(s, oneD=True)
        self.assertTrue(ks.size() == (test.KLYBATCH, DEFAULT_SPACE_PRIME, s.size()[-1]), msg=f'size: {ks.size()}')
        ksc = knotc.forward(sc, oneD=True)
        self.assertTrue(ksc.size() == (test.KLYBATCH, DEFAULT_SPACE_PRIME, sc.size()[-1]), msg=f'size: {ksc.size()}')

        # Test sizing for testing chunk 2
        kxs = knot.forward(xSmear, oneD=True)
        self.assertTrue(kxs.size() == (test.KLYBATCH, DEFAULT_SPACE_PRIME, xSmear.size()[-1]), msg=f'size: {kxs.size()}')
        kxls = knot.forward(xlSmear, oneD=True)
        self.assertTrue(kxls.size() == (test.KLYBATCH, DEFAULT_SPACE_PRIME, DEFAULT_SPACE_PRIME, xlSmear.size()[-1]), msg=f'size: {kxls.size()}')
        kxlsl = knot.forward(xlSmear, oneD=False)
        self.assertTrue(kxlsl.size() == (test.KLYBATCH, DEFAULT_SPACE_PRIME, xlSmear.size()[-1]), msg=f'size: {kxlsl.size()}')
        kxcs = knotc.forward(xcSmear, oneD=True)
        self.assertTrue(kxcs.size() == (test.KLYBATCH, DEFAULT_SPACE_PRIME, xcSmear.size()[-1]), msg=f'size: {kxcs.size()}')
        kxcls = knotc.forward(xclSmear, oneD=True)
        self.assertTrue(kxcls.size() == (test.KLYBATCH, DEFAULT_SPACE_PRIME, DEFAULT_SPACE_PRIME, xclSmear.size()[-1]), msg=f'size: {kxcls.size()}')
        kxclsl = knotc.forward(xclSmear, oneD=False)
        self.assertTrue(kxclsl.size() == (test.KLYBATCH, DEFAULT_SPACE_PRIME, xclSmear.size()[-1]), msg=f'size: {kxlsl.size()}')

    def testValues(self):
        # Generate all testing datatypes
        x = torch.randn((test.KLYBATCH, 1), dtype=DEFAULT_DTYPE)
        xl = torch.randn((test.KLYBATCH, DEFAULT_SPACE_PRIME, 1), dtype=DEFAULT_DTYPE)
        xc = torch.randn((test.KLYBATCH, 1), dtype=DEFAULT_COMPLEX_DTYPE)
        xcl = torch.randn((test.KLYBATCH, DEFAULT_SPACE_PRIME, 1), dtype=DEFAULT_COMPLEX_DTYPE)

        s, smear = test.getsmear(DEFAULT_DTYPE)
        sc, smearc = test.getsmear(DEFAULT_COMPLEX_DTYPE)

        xSmear = smear.forward(x)
        xlSmear = smear.forward(xl)
        xcSmear = smearc.forward(xc)
        xclSmear = smearc.forward(xcl)

        # Construct some knots to test, make sure values come out at 1.13
        knot = Knot(knotSize=DEFAULT_SPACE_PRIME, knotDepth=test.KLYBATCH, dtype=DEFAULT_DTYPE)
        knot.regWeights = nn.Parameter(knot.regWeights+(1/knot.knotSize))
        knot.knotRadii = nn.Parameter(knot.knotRadii+(test.KLYBATCH/100))
        knotc = Knot(knotSize=DEFAULT_SPACE_PRIME, knotDepth=test.KLYBATCH, dtype=DEFAULT_COMPLEX_DTYPE)
        knotc.regWeights = nn.Parameter(torch.randn_like(knot.regWeights))
        knotc.knotRadii = nn.Parameter(torch.randn_like(knot.knotRadii))
        
        # No change from these values should occur according to the lissajous tests
        dummy = torch.zeros((1), dtype=DEFAULT_DTYPE)
        dummyc = torch.zeros((1), dtype=DEFAULT_COMPLEX_DTYPE)
        d = knot.forward(dummy, oneD=True)
        dc = knotc.forward(dummyc, oneD=True)


        # I don't even know how to begin to handle these fucking values...
        # Like, I just dealt with testing lissajous shit, now I have to do the
        # same thing over again with a little more weights. God damn, this is the
        # non-reproducability of programming. This is what really "grinds my gears."
        # Fuck boilerplate code, if you can be automated by an AI you weren't a very
        # good one.
        kx = knot.forward(x, oneD=True)
        self.assertTrue(torch.all(kx == d))
        kxll = knot.forward(xl, oneD=True)
        self.assertTrue(torch.all(kxll == d))
        kxl = knot.forward(xl, oneD=False)
        self.assertTrue(torch.all(kxl == d))
        kxc = knotc.forward(xc, oneD=True)
        self.assertTrue(torch.all(kxc == dc))
        kxcll = knotc.forward(xcl, oneD=True)
        self.assertTrue(torch.all(kxcll == dc))
        kxcl = knotc.forward(xcl, oneD=False)
        self.assertTrue(torch.all(kxcl == dc))

        ks = knot.forward(s, oneD=True)
        self.assertTrue(torch.all(ks == d))
        ksc = knotc.forward(sc, oneD=True)
        self.assertTrue(torch.all(ksc == dc))

        kxs = knot.forward(xSmear, oneD=True)
        self.assertTrue(torch.all(kxs == d))
        kxls = knot.forward(xlSmear, oneD=True)
        self.assertTrue(torch.all(kxls == d))
        kxlsl = knot.forward(xlSmear, oneD=False)
        self.assertTrue(torch.all(kxlsl == d))
        kxcs = knotc.forward(xcSmear, oneD=True)
        self.assertTrue(torch.all(kxcs == dc))
        kxcls = knotc.forward(xclSmear, oneD=True)
        self.assertTrue(torch.all(kxcls == dc))
        kxclsl = knotc.forward(xclSmear, oneD=False)
        self.assertTrue(torch.all(kxclsl == dc))

    def testHarmonicPhaseStacking(self):
        # Generate all testing datatypes
        x = torch.ones((test.KLYBATCH, 1), dtype=DEFAULT_DTYPE)
        xl = torch.ones((test.KLYBATCH, DEFAULT_SPACE_PRIME, 1), dtype=DEFAULT_DTYPE)
        xc = torch.ones((test.KLYBATCH, 1), dtype=DEFAULT_COMPLEX_DTYPE)
        xcl = torch.ones((test.KLYBATCH, DEFAULT_SPACE_PRIME, 1), dtype=DEFAULT_COMPLEX_DTYPE)
        r = torch.randn_like(x)
        rl = torch.randn_like(xl)
        rc = torch.randn_like(xc)
        rcl = torch.randn_like(xcl)

        _, smear = test.getsmear(DEFAULT_DTYPE)
        _, smearc = test.getsmear(DEFAULT_COMPLEX_DTYPE)

        xSmear = smear.forward(x)
        rSmear = smear.forward(r)
        xlSmear = smear.forward(xl)
        rlSmear = smear.forward(rl)
        xcSmear = smearc.forward(xc)
        rcSmear = smearc.forward(rc)
        xclSmear = smearc.forward(xcl)
        rclSmear = smearc.forward(rcl)

        CONSTANTS = [x, xl, xc, xcl, xSmear, xlSmear, xcSmear, xclSmear]
        RANDOMS =   [r, rl, rc, rcl, rSmear, rlSmear, rcSmear, rclSmear]

        # Construct knots for testing
        knot = Knot(knotSize=DEFAULT_SPACE_PRIME, knotDepth=test.KLYBATCH, dtype=DEFAULT_DTYPE)
        phaseProto = torch.zeros_like(knot.phases)
        phaseProto[:,0] = -2
        knot.phases = nn.Parameter(phaseProto)
        self.assertTrue(torch.all(knot.phases[:,0] == -2))
        knotc = Knot(knotSize=DEFAULT_SPACE_PRIME, knotDepth=test.KLYBATCH, dtype=DEFAULT_COMPLEX_DTYPE)
        phasecProto = torch.zeros_like(knotc.phases)
        phasecProto[:,0] = -2
        knotc.phases = nn.Parameter(phasecProto)
        self.assertTrue(torch.all(knotc.phases[:,0] == toComplex(torch.ones((1)) * -2)))

        # Test phase stacking with only the most significant phase seeded
        cout = [knotc.forward(c) if torch.is_complex(c) else knot.forward(c) for c in CONSTANTS]
        rout = [knotc.forward(c) if torch.is_complex(c) else knot.forward(c) for c in RANDOMS]

        # Because of frequency zeroing, all values should be equal
        for idx in range(len(cout)):
            self.assertTrue(torch.all(cout[idx] == rout[idx]))

            if torch.is_complex(cout[idx]):
                self.assertTrue(torch.all(cout[idx].real - icos(torch.ones((1)) * -2) < 0.0001))
            else:
                self.assertTrue(torch.all(cout[idx] - icos(torch.ones((1)) * -2) < 0.0001))

        # Make sure that the phasing of the signal is stacking at a rate of phi
        phaseProto = torch.zeros_like(knot.phases)
        phaseProto[:,0] = torch.ones((1))
        phaseProto[:,1] = phi()
        phaseProto[:,2] = phi() * phi()
        knot.phases = nn.Parameter(phaseProto)

        phasecProto = torch.zeros_like(knotc.phases)
        phasecProto[:,0].real = torch.ones((1))
        phasecProto[:,1].real = phi()
        phasecProto[:,2].real = phi() * phi()
        knotc.phases = nn.Parameter(phasecProto)

        # Test phase with three phases seeded according to phi
        cout = [knotc.forward(c) if torch.is_complex(c) else knot.forward(c) for c in CONSTANTS]
        rout = [knotc.forward(c) if torch.is_complex(c) else knot.forward(c) for c in RANDOMS]
        stackedVal = (icos(torch.ones((1))) + icos(torch.ones((1)) * 2) + ((test.KLYBATCH - 2) * icos(torch.ones((1)) * 3))) / test.KLYBATCH

        # Because of frequency zeroing, all values should be equal
        for idx in range(len(cout)):
            self.assertTrue(torch.all(cout[idx] == rout[idx]))

            if torch.is_complex(cout[idx]):
                self.assertTrue(torch.all(cout[idx].real - stackedVal < 0.0001))
            else:
                self.assertTrue(torch.all(cout[idx] - stackedVal < 0.0001))

    def testHarmonicFrequencyStacking(self):
        # Generate all testing datatypes
        x = torch.ones((test.KLYBATCH, 1), dtype=DEFAULT_DTYPE)
        xl = torch.ones((test.KLYBATCH, DEFAULT_SPACE_PRIME, 1), dtype=DEFAULT_DTYPE)
        xc = torch.ones((test.KLYBATCH, 1), dtype=DEFAULT_COMPLEX_DTYPE)
        xcl = torch.ones((test.KLYBATCH, DEFAULT_SPACE_PRIME, 1), dtype=DEFAULT_COMPLEX_DTYPE)
        r = torch.randn_like(x)
        rl = torch.randn_like(xl)
        rc = torch.randn_like(xc)
        rcl = torch.randn_like(xcl)

        CONSTANTS = [x, xl, xc, xcl]
        RANDOMS =   [r, rl, rc, rcl]

        # Construct knots for testing
        knot = Knot(knotSize=DEFAULT_SPACE_PRIME, knotDepth=test.KLYBATCH, dtype=DEFAULT_DTYPE)
        freqProto = torch.zeros_like(knot.frequencies)
        freqProto[:,0] = 1
        knot.frequencies = nn.Parameter(freqProto)
        knotc = Knot(knotSize=DEFAULT_SPACE_PRIME, knotDepth=test.KLYBATCH, dtype=DEFAULT_COMPLEX_DTYPE)
        freqcProto = torch.zeros_like(knotc.frequencies)
        freqcProto[:,0] = 1
        knotc.frequencies = nn.Parameter(freqcProto)

        # Verify that the most significant frequency of the signal is the one present
        cout = [knotc.forward(c) if torch.is_complex(c) else knot.forward(c) for c in CONSTANTS]
        rout = [knotc.forward(c) if torch.is_complex(c) else knot.forward(c) for c in RANDOMS]

        for idx in range(len(cout)):
            self.assertFalse(torch.all(cout[idx] == rout[idx]))

            if torch.is_complex(cout[idx]):
                self.assertTrue(torch.all(cout[idx].real - icos(toComplex(torch.ones((1)))).real < 0.0001))
                self.assertTrue(torch.all(cout[idx].imag - icos(toComplex(torch.ones((1)))).imag < 0.0001))
            else:
                self.assertTrue(torch.all(cout[idx] - icos(torch.ones((1))) < 0.0001))
        
        # Add stacked frequency definition
        freqProto = torch.zeros_like(knot.frequencies)
        freqProto[:,0] = 1
        freqProto[:,1] = phi()
        freqProto[:,2] = phi() * phi()
        knot.frequencies = nn.Parameter(freqProto)

        freqcProto = torch.zeros_like(knotc.frequencies)
        freqcProto[:,0] = 1
        freqcProto[:,1] = phi()
        freqcProto[:,2] = phi() * phi()
        knotc.frequencies = nn.Parameter(freqcProto)

        # Verify frequency stacking property
        cout = [knotc.forward(c) if torch.is_complex(c) else knot.forward(c) for c in CONSTANTS]
        rout = [knotc.forward(c) if torch.is_complex(c) else knot.forward(c) for c in RANDOMS]
        stackedVal = (icos(torch.ones((1))) + icos(torch.ones((1)) * 2) + ((test.KLYBATCH - 2) * icos(torch.ones((1)) * 3))) / test.KLYBATCH
        
        for idx in range(len(cout)):
            self.assertFalse(torch.all(cout[idx] == rout[idx]))

            if torch.is_complex(cout[idx]):
                self.assertTrue(torch.all(cout[idx].real - stackedVal < 0.0001))
            else:
                self.assertTrue(torch.all(cout[idx] - stackedVal < 0.0001))


class RingingTest(unittest.TestCase):
    def testForwardSizing(self):
        # Generate random sizing
        SIZELEN:int = randint(6, 10)
        FORK_DISP:int = randint(0, 5)
        FORKS:int = SIZELEN - FORK_DISP
        SIZE = torch.Size((torch.randn((SIZELEN), dtype=DEFAULT_DTYPE) * SIZELEN).type(dtype=torch.int64).abs() + 1)
        
        # Generate the control tensors to test against
        x = torch.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Construct the required classes for Ringing
        ring = Ringing(forks=FORKS, dtype=DEFAULT_DTYPE)
        ringc = Ringing(forks=FORKS, dtype=DEFAULT_COMPLEX_DTYPE)

        # Compute the ringing results
        xr = ring.forward(x, irfft=False, stopTime=False)
        xc = ringc.forward(x, irfft=False, stopTime=False)
        xrr = ring.forward(x, irfft=True, stopTime=False)
        xcr = ringc.forward(x, irfft=True, stopTime=False)
        sxr = ring.forward(x, irfft=False, stopTime=True)
        sxc = ringc.forward(x, irfft=False, stopTime=True)
        sxrr = ring.forward(x, irfft=True, stopTime=True)
        sxcr = ringc.forward(x, irfft=True, stopTime=True)

        # Make sure the sizes translated through properly
        self.assertEqual(x.size(), xr.size())
        self.assertEqual(x.size(), xc.size())
        self.assertEqual(x.size(), xrr.size())
        self.assertEqual(x.size(), xcr.size())
        self.assertEqual(x.size(), sxr.size())
        self.assertEqual(x.size(), sxc.size())
        self.assertEqual(x.size(), sxrr.size())
        self.assertEqual(x.size(), sxcr.size())

    def testViewSizing(self):
        # FFT Sample Generation from the view function should also be consist
        SAMPLES = randint(1, 1024)
        vrc = ring.view(samples=SAMPLES, irfft=False)
        vrr = ring.view(samples=SAMPLES, irfft=True)
        vcc = ringc.view(samples=SAMPLES, irfft=False)
        vcr = ringc.view(samples=SAMPLES, irfft=True)

        self.assertTrue(vrc.size() == vrr.size() == vcc.size() == vcr.size())
        self.assertEqual(len(vrc.size()), 1)
