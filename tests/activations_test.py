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
        x = torch.ones((1, test.TBATCH, 1), dtype=DEFAULT_DTYPE)
        xc = torch.ones((1, test.TBATCH, 1), dtype=DEFAULT_COMPLEX_DTYPE)
        
        one = torch.ones((1, 1, 1), dtype=DEFAULT_DTYPE)
        onec = torch.ones((1, 1, 1), dtype=DEFAULT_COMPLEX_DTYPE)

        s, _ = test.getsmear(DEFAULT_DTYPE, ones=True)
        sc, _ = test.getsmear(DEFAULT_COMPLEX_DTYPE, ones=True)
        s.unsqueeze_(0)
        sc.unsqueeze_(0)

        # The curves to test
        lisa = Lissajous(size=test.TBATCH, dtype=DEFAULT_DTYPE)
        lisac = Lissajous(size=test.TBATCH, dtype=DEFAULT_COMPLEX_DTYPE)

        # Test the 23 curve pass
        lx = lisa.forward(x, oneD=True)
        lxl = lisa.forward(x, oneD=False)
        self.assertTrue(lx.size() == (1, test.TBATCH, 1, test.TBATCH), msg=f'size: {lx.size()}')
        self.assertTrue(lxl.size() == (1, test.TBATCH, 1), msg=f'size: {lxl.size()}')

        lxc = lisac.forward(xc, oneD=True)
        lxcl = lisac.forward(xc, oneD=False)
        self.assertTrue(lxc.size() == (1, test.TBATCH, 1, test.TBATCH), msg=f'size: {lxc.size()}')
        self.assertTrue(lxcl.size() == (1, test.TBATCH, 1), msg=f'size: {lxcl.size()}')

        # Test the signle logit pass
        lone = lisa.forward(one, oneD=True)
        lonec = lisac.forward(onec, oneD=True)
        self.assertTrue(lone.size() == (1, 1, 1, test.TBATCH), msg=f'size: {lone.size()}')
        self.assertTrue(lonec.size() == (1, 1, 1, test.TBATCH), msg=f'size: {lonec.size()}')

        # Test the 23 smear-curve pass
        ls = lisa.forward(s, oneD=True)
        lsl = lisa.forward(s, oneD=False)
        self.assertTrue(ls.size() == (1, test.TBATCH, s.size()[-1], test.TBATCH), msg=f'size: {ls.size()}')
        self.assertTrue(lsl.size() == (1, test.TBATCH, s.size()[-1]), msg=f'size: {lsl.size()}')

        lsc = lisac.forward(sc, oneD=True)
        lscl = lisac.forward(sc, oneD=False)
        self.assertTrue(lsc.size() == (1, test.TBATCH, sc.size()[-1], test.TBATCH), msg=f'size: {lsc.size()}')
        self.assertTrue(lscl.size() == (1, test.TBATCH, sc.size()[-1]), msg=f'size: {lscl.size()}')


    def testValues(self):
        x = torch.randn((test.TBATCH, DEFAULT_SPACE_PRIME, test.TEST_FFT_SMALL_SAMPLES), dtype=DEFAULT_DTYPE)
        xc = torch.randn((test.TBATCH, DEFAULT_SPACE_PRIME, test.TEST_FFT_SMALL_SAMPLES), dtype=DEFAULT_COMPLEX_DTYPE)

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
        self.assertTrue(torch.all((ll11 - torch.sin(x)).abs() < 1e-4), msg="Sin values don't check out for real values.")
        lc10 = lisac.forward(torch.zeros_like(xc), oneD=True)
        lc11 = lisac.forward(xc, oneD=True)
        self.assertFalse(torch.all(lc10 == lc11), msg="Frequency delta not working (oneD, complex).")
        lcl10 = lisac.forward(torch.zeros_like(xc), oneD=False)
        lcl11 = lisac.forward(xc, oneD=False)
        self.assertFalse(torch.all(lcl10 == lcl11), msg="Frequency delta not working (!oneD, complex).")
        self.assertTrue(torch.all((lcl11 - csin(xc)).abs() < 1e-4), \
            msg="Sin values don't check out for complex values.")

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
        self.assertTrue(torch.all(phil0 == torch.sin(torch.ones_like(phil0))), msg="Phi values don't check out for real values.")
        phic0 = lisac.forward(torch.zeros_like(xc), oneD=True)
        self.assertTrue(torch.all(phic0[:,:,:,:-1] == phic0[:,:,:,1:]), msg='Phi not consistent (oneD, complex).')
        phicl0 = lisac.forward(torch.zeros_like(xc), oneD=False)
        self.assertTrue(torch.all(phicl0[:,:,:-1] == phicl0[:,:,1:]), msg='Phi not consistent (!oneD, complex).')
        self.assertTrue(torch.all(phicl0 == csin(torch.ones_like(xc))), msg="Phi values don't check out for complex values.")

        # Final value testing, both phase and frequency
        lisa.frequency = nn.Parameter(lisa.frequency + 1)
        lisac.frequency = nn.Parameter(lisac.frequency + 1)

        final0 = lisa.forward(x, oneD=False)
        finalc0 = lisac.forward(xc, oneD=False)
        self.assertTrue(torch.all((final0 - torch.sin(x+1)).abs() < 1e-4), msg="Composite values don't check out for real values.")
        self.assertTrue(torch.all((finalc0 - csin(xc+1)).abs() < 1e-4), \
            msg="Composite values don't check out for complex values.")



class KnotTest(unittest.TestCase):
    def testSizing(self):
        # Generate all testing datatypes
        x = torch.ones((test.TBATCH, 1), dtype=DEFAULT_DTYPE)
        xl = torch.ones((test.TBATCH, DEFAULT_SPACE_PRIME, 1), dtype=DEFAULT_DTYPE)
        xc = torch.ones((test.TBATCH, 1), dtype=DEFAULT_COMPLEX_DTYPE)
        xcl = torch.ones((test.TBATCH, DEFAULT_SPACE_PRIME, 1), dtype=DEFAULT_COMPLEX_DTYPE)

        s, smear = test.getsmear(DEFAULT_DTYPE)
        sc, smearc = test.getsmear(DEFAULT_COMPLEX_DTYPE)

        xSmear = smear.forward(x)
        xlSmear = smear.forward(xl)
        xcSmear = smearc.forward(xc)
        xclSmear = smearc.forward(xcl)

        # Hold the testing knots
        knot = Knot(knotSize=DEFAULT_SPACE_PRIME, knotDepth=test.TBATCH, dtype=DEFAULT_DTYPE)
        knotc = Knot(knotSize=DEFAULT_SPACE_PRIME, knotDepth=test.TBATCH, dtype=DEFAULT_COMPLEX_DTYPE)

        # Test sizing for testing chunk 0
        kx = knot.forward(x, oneD=True)
        self.assertTrue(kx.size() == (test.TBATCH, 1, DEFAULT_SPACE_PRIME), msg=f'size: {kx.size()}')
        kxll = knot.forward(xl, oneD=True)
        self.assertTrue(kxll.size() == (test.TBATCH, DEFAULT_SPACE_PRIME, 1, DEFAULT_SPACE_PRIME), msg=f'size: {kxll.size()}')
        kxl = knot.forward(xl, oneD=False)
        self.assertTrue(kxl.size() == (test.TBATCH, DEFAULT_SPACE_PRIME, 1), msg=f'size: {kxl.size()}')
        kxc = knotc.forward(xc, oneD=True)
        self.assertTrue(kxc.size() == (test.TBATCH, 1, DEFAULT_SPACE_PRIME), msg=f'size: {kxc.size()}')
        kxcll = knotc.forward(xcl, oneD=True)
        self.assertTrue(kxcll.size() == (test.TBATCH, DEFAULT_SPACE_PRIME, 1, DEFAULT_SPACE_PRIME), msg=f'size: {kxcll.size()}')
        kxcl = knotc.forward(xcl, oneD=False)
        self.assertTrue(kxcl.size() == (test.TBATCH, DEFAULT_SPACE_PRIME, 1), msg=f'size: {kxcl.size()}')

        # Test sizing for testing chunk 1
        ks = knot.forward(s, oneD=True)
        self.assertTrue(ks.size() == (test.TBATCH, s.size()[-1], DEFAULT_SPACE_PRIME), msg=f'size: {ks.size()}')
        ksc = knotc.forward(sc, oneD=True)
        self.assertTrue(ksc.size() == (test.TBATCH, sc.size()[-1], DEFAULT_SPACE_PRIME), msg=f'size: {ksc.size()}')

        # Test sizing for testing chunk 2
        kxs = knot.forward(xSmear, oneD=True)
        self.assertTrue(kxs.size() == (test.TBATCH, xSmear.size()[-1], DEFAULT_SPACE_PRIME), msg=f'size: {kxs.size()}')
        kxls = knot.forward(xlSmear, oneD=True)
        self.assertTrue(kxls.size() == (test.TBATCH, DEFAULT_SPACE_PRIME, xlSmear.size()[-1], DEFAULT_SPACE_PRIME), msg=f'size: {kxls.size()}')
        kxlsl = knot.forward(xlSmear, oneD=False)
        self.assertTrue(kxlsl.size() == (test.TBATCH, DEFAULT_SPACE_PRIME, xlSmear.size()[-1]), msg=f'size: {kxlsl.size()}')
        kxcs = knotc.forward(xcSmear, oneD=True)
        self.assertTrue(kxcs.size() == (test.TBATCH, xcSmear.size()[-1], DEFAULT_SPACE_PRIME), msg=f'size: {kxcs.size()}')
        kxcls = knotc.forward(xclSmear, oneD=True)
        self.assertTrue(kxcls.size() == (test.TBATCH, DEFAULT_SPACE_PRIME, xclSmear.size()[-1], DEFAULT_SPACE_PRIME), msg=f'size: {kxcls.size()}')
        kxclsl = knotc.forward(xclSmear, oneD=False)
        self.assertTrue(kxclsl.size() == (test.TBATCH, DEFAULT_SPACE_PRIME, xclSmear.size()[-1]), msg=f'size: {kxlsl.size()}')


    def testValues(self):
        # Generate all testing datatypes
        x = torch.randn((test.TBATCH, 1), dtype=DEFAULT_DTYPE)
        xl = torch.randn((test.TBATCH, DEFAULT_SPACE_PRIME, 1), dtype=DEFAULT_DTYPE)
        xc = torch.randn((test.TBATCH, 1), dtype=DEFAULT_COMPLEX_DTYPE)
        xcl = torch.randn((test.TBATCH, DEFAULT_SPACE_PRIME, 1), dtype=DEFAULT_COMPLEX_DTYPE)

        s, smear = test.getsmear(DEFAULT_DTYPE)
        sc, smearc = test.getsmear(DEFAULT_COMPLEX_DTYPE)

        xSmear = smear.forward(x)
        xlSmear = smear.forward(xl)
        xcSmear = smearc.forward(xc)
        xclSmear = smearc.forward(xcl)

        # Construct some knots to test, make sure values come out at 1.13
        knot = Knot(knotSize=DEFAULT_SPACE_PRIME, knotDepth=test.TBATCH, dtype=DEFAULT_DTYPE)
        knot.regWeights = nn.Parameter(knot.regWeights+(1/knot.knotSize))
        knot.knotRadii = nn.Parameter(knot.knotRadii+(test.TBATCH/100))
        knotc = Knot(knotSize=DEFAULT_SPACE_PRIME, knotDepth=test.TBATCH, dtype=DEFAULT_COMPLEX_DTYPE)
        knotc.regWeights = nn.Parameter(torch.randn_like(knot.regWeights))
        knotc.knotRadii = nn.Parameter(torch.randn_like(knot.knotRadii))
        
        # No change from these values should occur according to the lissajous tests
        d = knot.forward(t.zeros_like(x[0]), oneD=True)
        dc = knotc.forward(t.zeros_like(xc[0]), oneD=True)
        dl = knot.forward(t.zeros_like(xl[0]), oneD=False)
        dcl = knotc.forward(t.zeros_like(xcl[0]).unsqueeze(0), oneD=False)

        # I don't even know how to begin to handle these fucking values...
        # Like, I just dealt with testing lissajous shit, now I have to do the
        # same thing over again with a little more weights. God damn, this is the
        # non-reproducability of programming. This is what really "grinds my gears."
        # Fuck boilerplate code, if you can be automated by an AI you weren't a very
        # good one.
        kx = knot.forward(x, oneD=True)
        self.assertTrue(torch.all((kx - d).abs() <= 1e-4))
        kxll = knot.forward(xl, oneD=True)
        self.assertTrue(torch.all((kxll - d).abs() <= 1e-4))
        kxl = knot.forward(xl, oneD=False)
        self.assertTrue(torch.all((kxl - dl).abs() <= 1e-4))
        kxc = knotc.forward(xc, oneD=True)
        self.assertTrue(torch.all((kxc - dc).abs() <= 1e-4))
        kxcll = knotc.forward(xcl, oneD=True)
        self.assertTrue(torch.all((kxcll - dc).abs() <= 1e-4))
        kxcl = knotc.forward(xcl, oneD=False)
        self.assertTrue(torch.all((kxcl - dcl).abs() <= 1e-4))

        ks = knot.forward(s, oneD=True)
        self.assertTrue(torch.all((ks - d).abs() <= 1e-4))
        ksc = knotc.forward(sc, oneD=True)
        self.assertTrue(torch.all((ksc - dc).abs() <= 1e-4))

        kxs = knot.forward(xSmear, oneD=True)
        self.assertTrue(torch.all((kxs - d).abs() <= 1e-4))
        kxls = knot.forward(xlSmear, oneD=True)
        self.assertTrue(torch.all((kxls - d).abs() <= 1e-4))
        kxlsl = knot.forward(xlSmear, oneD=False)
        self.assertTrue(torch.all((kxlsl - dl).abs() <= 1e-4))
        kxcs = knotc.forward(xcSmear, oneD=True)
        self.assertTrue(torch.all((kxcs - dc).abs() <= 1e-4))
        kxcls = knotc.forward(xclSmear, oneD=True)
        self.assertTrue(torch.all((kxcls - dc).abs() <= 1e-4))
        kxclsl = knotc.forward(xclSmear, oneD=False)
        self.assertTrue(torch.all((kxclsl - dcl).abs() <= 1e-4))


    def testHarmonicPhaseStacking(self):
        # Generate all testing datatypes
        x = torch.ones((test.TBATCH, 1), dtype=DEFAULT_DTYPE)
        xl = torch.ones((test.TBATCH, DEFAULT_SPACE_PRIME, 1), dtype=DEFAULT_DTYPE)
        xc = torch.ones((test.TBATCH, 1), dtype=DEFAULT_COMPLEX_DTYPE)
        xcl = torch.ones((test.TBATCH, DEFAULT_SPACE_PRIME, 1), dtype=DEFAULT_COMPLEX_DTYPE)
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
        knot = Knot(knotSize=DEFAULT_SPACE_PRIME, knotDepth=test.TBATCH, dtype=DEFAULT_DTYPE)
        phaseProto = torch.zeros_like(knot.phases)
        phaseProto[:,0] = -2
        knot.phases = nn.Parameter(phaseProto)
        self.assertTrue(torch.all(knot.phases[:,0] == -2))
        knotc = Knot(knotSize=DEFAULT_SPACE_PRIME, knotDepth=test.TBATCH, dtype=DEFAULT_COMPLEX_DTYPE)
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
                self.assertTrue(torch.all(cout[idx].real - csin(torch.ones((1)) * -2) < 0.0001))
            else:
                self.assertTrue(torch.all(cout[idx] - csin(torch.ones((1)) * -2) < 0.0001))

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
        stackedVal = (csin(torch.ones((1))) + csin(torch.ones((1)) * 2) + ((test.TBATCH - 2) * csin(torch.ones((1)) * 3))) / test.TBATCH

        # Because of frequency zeroing, all values should be equal
        for idx in range(len(cout)):
            self.assertTrue(torch.all(cout[idx] == rout[idx]))

            if torch.is_complex(cout[idx]):
                self.assertTrue(torch.all(cout[idx].real - stackedVal < 0.0001))
            else:
                self.assertTrue(torch.all(cout[idx] - stackedVal < 0.0001))


    def testHarmonicFrequencyStacking(self):
        # Generate all testing datatypes
        x = torch.ones((test.TBATCH, 1), dtype=DEFAULT_DTYPE)
        xl = torch.ones((test.TBATCH, DEFAULT_SPACE_PRIME, 1), dtype=DEFAULT_DTYPE)
        xc = torch.ones((test.TBATCH, 1), dtype=DEFAULT_COMPLEX_DTYPE)
        xcl = torch.ones((test.TBATCH, DEFAULT_SPACE_PRIME, 1), dtype=DEFAULT_COMPLEX_DTYPE)
        r = torch.randn_like(x)
        rl = torch.randn_like(xl)
        rc = torch.randn_like(xc)
        rcl = torch.randn_like(xcl)

        CONSTANTS = [x, xl, xc, xcl]
        RANDOMS =   [r, rl, rc, rcl]

        # Construct knots for testing
        knot = Knot(knotSize=DEFAULT_SPACE_PRIME, knotDepth=test.TBATCH, dtype=DEFAULT_DTYPE)
        freqProto = torch.zeros_like(knot.frequencies)
        freqProto[:,0] = 1
        knot.frequencies = nn.Parameter(freqProto)
        knotc = Knot(knotSize=DEFAULT_SPACE_PRIME, knotDepth=test.TBATCH, dtype=DEFAULT_COMPLEX_DTYPE)
        freqcProto = torch.zeros_like(knotc.frequencies)
        freqcProto[:,0] = 1
        knotc.frequencies = nn.Parameter(freqcProto)

        # Verify that the most significant frequency of the signal is the one present
        cout = [knotc.forward(c) if torch.is_complex(c) else knot.forward(c) for c in CONSTANTS]
        rout = [knotc.forward(c) if torch.is_complex(c) else knot.forward(c) for c in RANDOMS]

        for idx in range(len(cout)):
            self.assertFalse(torch.all(cout[idx] == rout[idx]))

            if torch.is_complex(cout[idx]):
                self.assertTrue(torch.all(cout[idx].real - csin(toComplex(torch.ones((1)))).real < 0.0001))
                self.assertTrue(torch.all(cout[idx].imag - csin(toComplex(torch.ones((1)))).imag < 0.0001))
            else:
                self.assertTrue(torch.all(cout[idx] - csin(torch.ones((1))) < 0.0001))
        
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
        stackedVal = (csin(torch.ones((1))) + csin(torch.ones((1)) * 2) + ((test.TBATCH - 2) * csin(torch.ones((1)) * 3))) / test.TBATCH
        
        for idx in range(len(cout)):
            self.assertFalse(torch.all(cout[idx] == rout[idx]))

            if torch.is_complex(cout[idx]):
                self.assertTrue(torch.all(cout[idx].real - stackedVal < 0.0001))
            else:
                self.assertTrue(torch.all(cout[idx] - stackedVal < 0.0001))



class RingingTest(unittest.TestCase):
    def testForwardSizing(self):
        # Generate random sizing
        SIZELEN:int = randint(1, 5)
        SIZESCALAR:int = randint(6, 10)
        FORK_DISP:int = randint(0, 5)
        SIZE = torch.Size(((torch.randn((SIZELEN), dtype=DEFAULT_DTYPE)).type(dtype=torch.int64).abs() + 1) * SIZESCALAR)
        FORKS:int = SIZE[-1] - FORK_DISP

        # Generate the control tensors to test against
        x = torch.randn(SIZE, dtype=DEFAULT_DTYPE)
        xc = torch.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Construct the required classes for Ringing
        ring = Ringing(forks=FORKS, dtype=DEFAULT_DTYPE)
        ringc = Ringing(forks=FORKS, dtype=DEFAULT_COMPLEX_DTYPE)

        # Compute the ringing results
        xrr = ring.forward(x)
        xrc = ringc.forward(x)
        xcr = ring.forward(xc)
        xcc = ringc.forward(xc)

        # Make sure the sizes translated through properly
        self.assertEqual(x.size(), xrr.size())
        self.assertEqual(x.size(), xrc.size())
        self.assertEqual(x.size(), xcr.size())
        self.assertEqual(x.size(), xcc.size())


    def testViewSizing(self):
        # Generate random sizing
        SAMPLES = randint(10, 1024)
        FORKS = (randint(1, 10))
        
        # Generate the control tensor to test against
        x = torch.randn((SAMPLES), dtype=DEFAULT_COMPLEX_DTYPE)

        # Construct the required classes for Ringing and testing
        ring = Ringing(forks=FORKS, dtype=DEFAULT_DTYPE)
        ringc = Ringing(forks=FORKS, dtype=DEFAULT_COMPLEX_DTYPE)
        _ = ring.forward(x)
        _ = ringc.forward(x)

        # FFT Sample Generation from the view function should also be consist
        vr = ring.view(samples=SAMPLES)
        vc = ringc.view(samples=SAMPLES)

        # Make sure that all of the lengths that come out have the appropriate samples and dims
        self.assertTrue(vr.size() == vc.size())
        self.assertEqual(len(vc.size()), 1)


    def testSmallSizing(self):
        # Are these next tests useful to output? Not really from what I can see, however
        # they are quite good for stability reasons
        SAMPLES:int = 1
        FORKS:int = randint(1, 3)

        # Generate the control tensor to test against
        x = torch.randn((SAMPLES), dtype=DEFAULT_COMPLEX_DTYPE)

        # Construct the required classes for Ringing
        ring = Ringing(forks=FORKS, dtype=DEFAULT_DTYPE)
        ringc = Ringing(forks=FORKS, dtype=DEFAULT_COMPLEX_DTYPE)
        
        # Push values through
        xr = ring.forward(x)
        xc = ringc.forward(x)

        # View the system
        vr = ring.view(samples=SAMPLES)
        vc = ringc.view(samples=SAMPLES)

        # Assert that the sizes that come out are all (1)
        self.assertTrue(xr.size() == xc.size() == vr.size() == vc.size())
        self.assertTrue(x.size() == xr.size())
    

    def testViewValues(self):
        # Generate random sizing
        SIZELEN:int = randint(1, 5)
        SIZE:int = randint(5, 32)
        FORK_DISP:int = 2
        FORKS:int = SIZE - FORK_DISP
        DECAYSCALAR:t.Tensor = phi().pow(-1)
        GAIN:t.Tensor = t.zeros_like(DECAYSCALAR)
        for n in range(0, SIZELEN+1):
            GAIN.add_(DECAYSCALAR.pow(n))
        DECAY:t.Tensor = DECAYSCALAR.pow(SIZELEN)

        # Generate the control tensors to test against
        x = csigmoid(t.stack([torch.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)]*SIZELEN, dim=0))

        # Construct the required classes for Ringing
        ring = Ringing(forks=FORKS, dtype=DEFAULT_DTYPE)
        ringc = Ringing(forks=FORKS, dtype=DEFAULT_COMPLEX_DTYPE)
        v0r = ring.view(samples=x.size()[-1])
        v0c = ringc.view(samples=x.size()[-1])

        # Make sure there is no default ringing in the forks
        self.assertTrue(torch.all(v0r.abs() < 1e-4), msg=f'Latent ringing with real initialization.\t{v0r}')
        self.assertTrue(torch.all(v0c.abs() < 1e-4), msg=f'Latent ringing with complex initalization.\t{v0c}')

        # Compute the ringing results
        _ = ring.forward(x[0]) # Don't compound the decay
        _ = ringc.forward(x[0]) # Don't compound the decay
        vr = ring.view(samples=x.size()[-1])
        vc = ringc.view(samples=x.size()[-1])
        _ = ring.forward(x)
        _ = ringc.forward(x)
        vr2 = ring.view(samples=x.size()[-1])
        vc2 = ringc.view(samples=x.size()[-1])
        _ = ring.forward(torch.zeros_like(x))
        _ = ringc.forward(torch.zeros_like(x))
        vr3 = ring.view(samples=x.size()[-1])
        vc3 = ringc.view(samples=x.size()[-1])

        # Check for proper signal degredations on forks
        self.assertTrue(torch.all((vr2 - (vr * GAIN)).abs() < 1e-4), \
            msg=f'build degredation: vr2/vr ({(vr2/vr).abs()}) != GAIN ({GAIN})')
        self.assertTrue(torch.all((vc2 - (vc * GAIN)).abs() < 1e-4), \
            msg=f'build degredation: vc2/vc ({(vc2/vc).abs()}) != GAIN ({GAIN})')
        self.assertTrue(torch.all((vr3 - (vr2 * DECAY)).abs() < 1e-4), \
            msg=f'decay degredation: vr3/vr2 ({(vr3/vr2).abs()}) != DECAY ({DECAY})')
        self.assertTrue(torch.all((vc3 - (vc2 * DECAY)).abs() < 1e-4), \
            msg=f'decay degredation: vc3/vc2 ({(vc3/vc2).abs()}) != DECAY ({DECAY})')


    def testForwardValues(self):
        # Generate random sizing
        SIZELEN:int = randint(1, 4)
        SIZESCALAR:int = randint(3, 8)
        FORK_DISP:int = 2
        SIZE = torch.Size(((torch.randn((SIZELEN), dtype=DEFAULT_DTYPE)).type(dtype=torch.int64).abs() + 1) * SIZESCALAR)
        FORKS:int = SIZE[-1] - FORK_DISP

        # Generate the control tensors
        z = torch.zeros(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        o = torch.ones(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        r = torch.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Construct the required classes for Ringing
        ring = Ringing(forks=FORKS, dtype=DEFAULT_DTYPE)
        ringc = Ringing(forks=FORKS, dtype=DEFAULT_COMPLEX_DTYPE)

        # Do a latent oscillation test
        z0r = ring.forward(z)
        z0c = ringc.forward(z)
        v0r = ring.view(samples=SIZE[-1])
        v0c = ringc.view(samples=SIZE[-1])

        # Make sure when applying no signal, no signals begin to seep
        self.assertTrue(torch.all(z0r.abs() < 1e-4))
        self.assertTrue(torch.all(z0c.abs() < 1e-4))
        self.assertTrue(torch.all(v0r.abs() < 1e-4))
        self.assertTrue(torch.all(v0c.abs() < 1e-4))

        # Compute the ringing results
        controlTensors = [z, o, r]
        results = torch.zeros((len(controlTensors), 2, *SIZE), dtype=DEFAULT_COMPLEX_DTYPE)
        for idx, control in enumerate(controlTensors):
            results[idx, 0] = ring.forward(control)
            results[idx, 1] = ringc.forward(control)
        
        # Assert that the forward signal decay is appropriate (assuming view testing at different
        #   times is working).
        # If zeros start putting out any sort of signalling, something is really wrong
        self.assertTrue(torch.all(results[0, :].abs() < 1e-4), \
            msg=f'Zeros carrying noise for some reason.')
        
        # The output signal of the ringing should look something along the lines of
        #   the input signal multiplied by some decay between (0., 1.), added to a
        #   a set of tuning forks linearly positionined and mixed into an appropriately
        #   sized/sampled output tensor. The lower bound of the multiple from the input
        #   should be roughly 1/phi, with the top bound being roughly phi (1 + (1/phi)).
        sumControl = []
        for idx, control in enumerate(controlTensors):
            if len(control.size()) <= 1:
                sumControl.append(control)
                continue

            # Create copies for output
            # I know the max doesn't necessarily match the normal type of output calculated using mean,
            # however, creating the output signal this way can garuntee atleast the lack of a total runaway
            tempSum = (torch.fft.fft(control, dim=-1).transpose(-1, 0)).abs()

            # Regularize for testing
            for _ in range(len(tempSum.size()) - 1):
                tempSum = torch.sum(tempSum, dim=1)
            tempSum = torch.max(tempSum, dim=-1)[0] * torch.ones_like(tempSum)

            # Add to test sets
            sumControl.append(torch.max(torch.fft.ifft(tempSum, dim=-1).abs(), dim=-1)[0])

        # Run tests for other prior tensors accordingly
        for idx in range(1, len(controlTensors)):
            sControl = sumControl[idx].unsqueeze(0)

            normalDiff = (results[idx, :] - (phi() * (controlTensors[idx]))).abs()
            normalResult = torch.all(normalDiff.abs() <= torch.max(phi() * sControl.abs()) + 1e-4)
            self.assertTrue(normalResult, \
                msg=f'[idx:{idx}] A value higher than a non-regularized value added to the forks has appeared.\n|{normalDiff}| <= {phi() * sControl.abs() + 1e-4}')
