import unittest
import test
import math

import torch
from plasmatorch import *

from random import randint

class ConstantsTest(unittest.TestCase):
    def testPhi(self):
        self.assertTrue(torch.all((phi() - 1.61803398875).abs() < 1e-4))

    def testAsigphi(self):
        self.assertTrue(torch.all((isigmoid(asigphi()) - (1/phi())).abs() < 1e-4))
        self.assertTrue(torch.all((isigmoid(toComplex(asigphi())) - (1/phi())).abs() < 1e-4))
    
    def testLattice(self):
        paramControl = latticeParams(10)
        paramSub = latticeParams(7)
        # Going over roughly this number will cause float innacuracy with a 32-bit float
        paramLong = latticeParams(192)

        self.assertTrue(torch.all(paramSub == paramControl[:7]))
        self.assertEqual(paramControl[0], 1.)
        self.assertTrue(paramControl[1] - (1./phi()) < 0.0001)
        self.assertTrue(paramControl[9] - (1./(phi() ** 9)) < 0.0001)
        self.assertEqual(paramSub[0], 1.)
        self.assertTrue(paramSub[1] - (1./phi()) < 0.0001)
        self.assertTrue(paramSub[6] - (1./(phi() ** 6)) < 0.0001)

        self.assertTrue(torch.all((paramLong[1:]/paramLong[:-1]) - (1/phi()) < 0.0001))

    def testPi(self):
        self.assertTrue(torch.all(pi() - 3.1415926535 < 0.0001))
    
    def testI(self):
        built = i()
        homebrew = torch.sqrt(-1 * torch.ones((1), 
            dtype=DEFAULT_COMPLEX_DTYPE))

        self.assertTrue(torch.all(built.real - homebrew.real < 0.0001))
        self.assertTrue(torch.all(built.imag - homebrew.imag < 0.0001))

class ComplexQualiaTest(unittest.TestCase):
    SIZE = (97, 23, 256)

    def testSizing(self):
        # Seeding tensors
        x = torch.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = torch.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Calculate results
        mag = torch.abs(x)
        magc = torch.abs(xc)
        pol = torch.angle(x)
        polc = torch.angle(xc)

        # Test that the values have the same size
        self.assertEqual(x.size(), mag.size(), msg=f'{x.size()} != {mag.size()}')
        self.assertEqual(xc.size(), magc.size(), msg=f'{xc.size()} != {magc.size()}')
        self.assertEqual(x.size(), pol.size(), msg=f'{x.size()} != {pol.size()}')
        self.assertEqual(xc.size(), polc.size(), msg=f'{x.size()} != {polc.size()}')
    
    def testMagnitude(self):
        # Seeding tensors
        zeros = torch.zeros(self.SIZE, dtype=DEFAULT_DTYPE)
        zerosc = torch.zeros(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        ones = zeros + 1
        onesc = zerosc + 1
        imag = torch.view_as_complex(torch.stack((zeros, ones), dim=-1))
        root2 = torch.view_as_complex(torch.stack((ones, ones), dim=-1))

        # Run the calculations
        zmag = torch.abs(zeros)
        zmagc = torch.abs(zerosc)
        omag = torch.abs(ones)
        omagc = torch.abs(onesc)
        imagc = torch.abs(imag)
        rmag = torch.abs(root2)

        # Check the values
        self.assertTrue(torch.all(zmag == 0))
        self.assertTrue(torch.all(zmagc == 0))
        self.assertTrue(torch.all(omag == 1))
        self.assertTrue(torch.all(omagc == 1))
        self.assertTrue(torch.all(imagc == 1))
        self.assertTrue(torch.all(rmag == torch.sqrt(ones * 2)))

    def testPolarization(self):
        # Seeding tensors
        zeros = torch.zeros(self.SIZE, dtype=DEFAULT_DTYPE)
        zerosc = torch.zeros(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        ones = zeros + 1
        onesc = zerosc + 1
        imag = torch.view_as_complex(torch.stack((zeros, ones), dim=-1))
        root2 = torch.view_as_complex(torch.stack((ones, ones), dim=-1))

        # Run the calculations
        zpol = torch.angle(zeros)
        zpolc = torch.angle(zerosc)
        opol = torch.angle(ones)
        opolc = torch.angle(onesc)
        ipol = torch.angle(imag)
        iroot2 = torch.angle(root2)

        # Check the values
        self.assertTrue(torch.all(zpol == zeros))
        self.assertTrue(torch.all(zpolc == zeros))
        self.assertTrue(torch.all(opol == zeros))
        self.assertTrue(torch.all(opolc == zeros))
        self.assertTrue(torch.all(ipol - pi()/2 < 0.0001))
        self.assertTrue(torch.all(iroot2 - pi()/4 < 0.0001))

class SoftmaxTest(unittest.TestCase):
    SIZE = (97, 11, 13, 128)

    def testSizing(self):
        # Seeding tensors
        x = torch.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = torch.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Calculate
        y = isoftmax(x, dim=-1)
        y0 = isoftmax(x, dim=0)
        yc = isoftmax(xc, dim=-1)
        yc0 = isoftmax(xc, dim=0)

        # Test the sizing through the function
        self.assertEqual(x.size(), y.size(), msg=f'{x.size()} != {y.size()}')
        self.assertEqual(x.size(), y0.size(), msg=f'{x.size()} != {y0.size()}')
        self.assertEqual(x.size(), yc.size(), msg=f'{x.size()} != {yc.size()}')
        self.assertEqual(x.size(), yc0.size(), msg=f'{x.size()} != {yc0.size()}')

    def testValues(self):
        # Seeding tensors
        x = torch.randn(self.SIZE, dtype=DEFAULT_DTYPE) * torch.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = torch.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE) * torch.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Calculate
        y = isoftmax(x, dim=-1)
        y0 = isoftmax(x, dim=0)
        yc = isoftmax(xc, dim=-1)
        yc0 = isoftmax(xc, dim=0)

        # Test that the values are actually softmax'd at least normally
        self.assertTrue(torch.all(y == torch.softmax(x, dim=-1)))
        self.assertTrue(torch.all(y0 == torch.softmax(x, dim=0)))

        # Test to make sure that the magnitudes are softmax'd
        self.assertTrue(torch.all(torch.angle(xc) - torch.angle(yc) < 0.0001))
        self.assertTrue(torch.all(torch.angle(xc) - torch.angle(yc0) < 0.0001))
        self.assertTrue(torch.all(torch.softmax(torch.abs(xc), dim=-1) - torch.abs(yc) < 0.0001))
        self.assertTrue(torch.all(torch.softmax(torch.abs(xc), dim=0) - torch.abs(yc0) < 0.0001))

class TrigTest(unittest.TestCase):
    SIZE = (11, 23, 1024, 3)

    def testSizing(self):
        # Seeding tensors
        x = torch.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = torch.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Calculate
        cosx = icos(x)
        cosxc = icos(xc)
        sinx = isin(x)
        sinxc = isin(xc)

        # Test that the values go in and come out the same
        self.assertEqual(cosx.size(), x.size(), msg=f'{cosx.size()} != {x.size()}')
        self.assertEqual(cosxc.size(), xc.size(), msg=f'{cosxc.size()} != {xc.size()}')
        self.assertEqual(sinx.size(), x.size(), msg=f'{sinx.size()} != {x.size()}')
        self.assertEqual(sinxc.size(), xc.size(), msg=f'{sinxc.size()} != {xc.size()}')

    def testCos(self):
        # Seeding tensors
        x = torch.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = torch.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Calculate
        cosx = icos(x)
        cosxc = icos(xc)

        # Test the values and assert lack of runaway
        self.assertTrue(torch.all(cosx == torch.cos(x)))
        self.assertFalse(torch.all(cosxc == torch.cos(xc)))

        # Test the values of the exp construction to assert some cos() equivalence
        self.assertTrue(torch.all(
            (cosxc.abs() - (icos(xc.abs()) * torch.exp(i() * pi() / 4.)).abs()) < 1e-4
        ))
        self.assertTrue(torch.all(icos(torch.zeros_like(xc)) == torch.ones_like(xc)))
        self.assertTrue(torch.all(toComplex(icos(torch.zeros_like(x))) == icos(torch.zeros_like(xc))))
    
    def testSin(self):
        # Seeding tensors
        x = torch.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = torch.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Calculate
        sinx = isin(x)
        sinxc = isin(xc)

        # Test the values and assert lack of runaway
        self.assertTrue(torch.all(sinx == torch.sin(x)))
        self.assertFalse(torch.all(sinxc == torch.sin(xc)))

        # Test that the regular sin function is present
        self.assertTrue(torch.all((sinx - isin(toComplex(x)).real).abs() < 1e-4))
        self.assertTrue(torch.all(isin(toComplex(x)).imag.abs() < 1e-4))

        # Double check by asserting that the real value of the function is 0
        self.assertTrue(torch.all(isin(torch.zeros_like(xc)).abs() < 1e-4))

class PrimishDistTest(unittest.TestCase):
    def testSizing(self):
        # Generate random sizing
        SIZELEN = randint(1, 5)
        SIZE = torch.Size((torch.randn((SIZELEN), dtype=DEFAULT_DTYPE) * SIZELEN).type(dtype=torch.int64).abs() + 1)
        
        # Generate the control tensors
        x = torch.randn(SIZE, dtype=DEFAULT_DTYPE)
        xc = torch.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Run the control tensors through the target functions
        xpr = realprimishdist(x, relative=True, gaussApprox=False)
        xpa = realprimishdist(x, relative=False, gaussApprox=False)
        xgr = realprimishdist(x, relative=True, gaussApprox=True)
        xga = realprimishdist(x, relative=False, gaussApprox=True)
        xcgr = gaussianprimishdist(xc, relative=True)
        xcga = gaussianprimishdist(xc, relative=False)
        xfr = iprimishdist(x, relative=True)
        xfa = iprimishdist(x, relative=False)
        xcr = iprimishdist(xc, relative=True)
        xca = iprimishdist(xc, relative=False)

        # Assert size equivalences
        self.assertEqual(x.size(), xpr.size())
        self.assertEqual(x.size(), xpa.size())
        self.assertEqual(x.size(), xgr.size())
        self.assertEqual(x.size(), xga.size())
        self.assertEqual(xc.size(), xcgr.size())
        self.assertEqual(xc.size(), xcga.size())
        self.assertEqual(x.size(), xfr.size())
        self.assertEqual(x.size(), xfa.size())
        self.assertEqual(xc.size(), xcr.size())
        self.assertEqual(xc.size(), xca.size())
    
    def testValuesReal(self):
        # Generate the control tensors
        tprimes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).type(dtype=DEFAULT_DTYPE)
        tapres =   torch.tensor([1, 0, 0, 0, 1, 0, 1, 0, 1, 2]).type(dtype=DEFAULT_DTYPE)
        trpres =   torch.tensor([1, 0, 0, 0, 1, 0, 1, 0, 1/2., 1]).type(dtype=DEFAULT_DTYPE)
        tgrimes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).type(dtype=DEFAULT_DTYPE)
        tagres =   torch.tensor([1, 0, 0, 0, 1, 0, 1, 0, 1, 0]).type(dtype=DEFAULT_DTYPE)
        trgres =   torch.tensor([1, 0, 0, 0, 1, 0, 1, 0, 1, 0]).type(dtype=DEFAULT_DTYPE)

        # Compute
        apres = realprimishdist(tprimes, relative=False, gaussApprox=False)
        rpres = realprimishdist(tprimes, relative=True, gaussApprox=False)
        agres = realprimishdist(tgrimes, relative=False, gaussApprox=True)
        rgres = realprimishdist(tgrimes, relative=True, gaussApprox=True)
        aires = iprimishdist(tprimes, relative=False)
        rires = iprimishdist(tprimes, relative=True)

        # First assert that the iprimishdist function is switching properly
        self.assertTrue(torch.all((aires - apres).abs() < 1e-4), msg=f'{aires} != {apres}')
        self.assertTrue(torch.all((rires - rpres).abs() < 1e-4), msg=f'{rires} != {rpres}')

        # Assert that the values of the computed output are the same as the ones
        # provided. If not, display the differences.
        self.assertTrue(torch.all((tapres - apres).abs() < 1e-4), msg=f'{tapres} != {apres}')
        self.assertTrue(torch.all((trpres - rpres).abs() < 1e-4), msg=f'{trpres} != {rpres}')
        self.assertTrue(torch.all((tagres - agres).abs() < 1e-4), msg=f'{tagres} != {agres}')
        self.assertTrue(torch.all((trgres - rgres).abs() < 1e-4), msg=f'{trgres} != {rgres}')

    def testConsistencyGauss(self):
        # Generate random sizing
        SIZELEN = randint(1, 5)
        SIZESCALAR = randint(1, 5)
        SIZE = torch.Size(isigmoid(torch.randn((SIZELEN), dtype=DEFAULT_DTYPE)).type(dtype=torch.int64).abs() + 1) * SIZESCALAR
        
        # Generate the randomized control tensors
        trandr = torch.randn(SIZE, dtype=DEFAULT_DTYPE)
        trandrc = toComplex(trandr)
        trandrci = trandrc * i()

        # Compute the gaussian distances for both formats of the randomized input
        agres = gaussianprimishdist(trandr, relative=False)
        agces = gaussianprimishdist(trandrc, relative=False)
        agcesi = gaussianprimishdist(trandrci, relative=False)
        agies = iprimishdist(trandrc, relative=False)
        rgres = gaussianprimishdist(trandr, relative=True)
        rgces = gaussianprimishdist(trandrc, relative=True)
        rgies = iprimishdist(trandrc, relative=True)
        rgcesi = gaussianprimishdist(trandrci, relative=True)

        # Assert conversion equivalencies
        self.assertTrue(torch.all((agres - agces).abs() < 1e-4), msg=f'{agres} != {agces}')
        self.assertTrue(torch.all((agces - agies).abs() < 1e-4), msg=f'{agces} != {agies}')
        self.assertTrue(torch.all((agces - agcesi).abs() < 1e-4), msg=f'{agces} != {agcesi}')
        self.assertTrue(torch.all((rgres - rgces).abs() < 1e-4), msg=f'{rgres} != {rgces}')
        self.assertTrue(torch.all((rgces - rgies).abs() < 1e-4), msg=f'{rgces} != {rgies}')
        self.assertTrue(torch.all((rgces - rgcesi).abs() < 1e-4), msg=f'{rgces} != {rgcesi}')
        self.assertTrue(torch.all((agres - rgres).abs() < 1e-4), msg=f'{agres} != {rgres}')

    def testValuesGauss(self):
        # Generate the control tensors
        tprimes = xbias(n=10, bias=0).type(dtype=DEFAULT_COMPLEX_DTYPE)
        tpres = torch.tensor([1, 0, 0, 0, 1, 0, 1, 0, 1, 0]).type(dtype=DEFAULT_DTYPE)
        tgaussians = torch.tensor([2+(3*i()), 2-(3*i()), -2+(3*i()), -2-(3*i()), 6+(3*i())])
        tgres = torch.tensor([0, 0, 0, 0, torch.sqrt(torch.ones(1) * 2)]).type(dtype=DEFAULT_DTYPE)
        trgres = torch.tensor([0, 0, 0, 0, 1])

        # Compute the result of the test vectors
        pres = gaussianprimishdist(tprimes, relative=False)
        gres = gaussianprimishdist(tgaussians, relative=False)
        rgres = gaussianprimishdist(tgaussians, relative=True)

        # Assert the precomputed values
        self.assertTrue(torch.all((tpres - pres).abs() < 1e-4), msg=f'{tpres} != {pres}')
        self.assertTrue(torch.all((tgres - gres).abs() < 1e-4), msg=f'{tgres} != {gres}')
        self.assertTrue(torch.all((trgres - rgres).abs() < 1e-4), msg=f'{trgres} != {rgres}')

class PrimishValsTest(unittest.TestCase):
    def testSizing(self):
        # Generate two random sizes, one being dependent
        SIZE = randint(1, 100)
        SIZE2 = randint(1, 100) + SIZE

        # Generate two seperate tapes of primish values, each dependent on the last
        x = primishvals(n=SIZE)
        y = primishvals(n=SIZE2, base=x)
        z = primishvals(n=SIZE, base=y)

        # Assert that the amount of dimensions are the same no matter what
        self.assertTrue(len(x.size()) == len(y.size()) == len(z.size()) == 1)

        # Assert that the sizing of the tapes extended and contracted properly
        self.assertTrue(x.size()[0] == SIZE)
        self.assertTrue(y.size()[0] == SIZE2)
        self.assertTrue(z.size()[0] == SIZE)

    def testSmallValues(self):
        # No more randomness needed
        SIZE = 10

        # Generate the primishes to test
        x = primishvals(n=1)
        y = primishvals(n=2, base=x)
        z = primishvals(n=3, base=y)
        w = primishvals(n=10, base=z)

        # Check to make sure the right values are present
        self.assertTrue(x[-1] == 1)
        self.assertTrue(y[-1] == 2, msg=f'{y[-1]}')
        self.assertTrue(z[-1] == 3, msg=f'{z[-1]}')
        self.assertTrue(torch.all(z == w[:3]))
        # Ascending order
        self.assertTrue(torch.all(w[:-1] < w[1:]))
        # 6k +- 1 values past first three
        self.assertTrue(torch.all(((w[3:] % 6) - 5) * ((w[3:] % 6) - 1) == 0))

    def testLargeValues(self):
        # Random size extension for testing large generation
        SIZE = randint(100, 1000)
        SIZE2 = randint(100, 1000) + SIZE

        # Generate primishes to test
        x = primishvals(n=SIZE)
        y = primishvals(n=SIZE2, base=x)

        # Check to make sure that the values are consistent
        self.assertTrue(torch.all(x == y[:x.size()[-1]]))
        
        # Check to make sure that all values are of 6k +- 1 and ascending
        self.assertTrue(torch.all(y[:-1] < y[1:]))
        self.assertTrue(torch.all(((y[3:] % 6) - 1) * ((y[3:] % 6) - 5) == 0))
    
    def testGaussSwap(self):
        # Random size extension for testing large generation
        SIZE = randint(100, 1000)
        SIZE2 = randint(100, 1000) + SIZE

        # Generate primishes to test
        x = primishvals(n=SIZE, gaussApprox=True)
        y = primishvals(n=SIZE2, base=x, gaussApprox=True)

        # Check to make sure that the values are consistent
        self.assertTrue(torch.all(x == y[:x.size()[-1]]))

        # Check to make sure that all values are of 4k +- 3 and ascending
        self.assertTrue(torch.all(y[:-1] < y[1:]))
        self.assertTrue(torch.all(((y[3:] % 4) - 1) * ((y[3:] % 4) - 3) == 0))
