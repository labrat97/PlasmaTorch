import unittest

import torch as t
from plasmatorch import *

from random import randint



class ConstantsTest(unittest.TestCase):
    def testPhi(self):
        self.assertTrue(t.all((phi() - 1.61803398875).abs() < 1e-4))

    def testAsigphi(self):
        self.assertTrue(t.all((isigmoid(asigphi()) - (1/phi())).abs() < 1e-4))
        self.assertTrue(t.all((isigmoid(toComplex(asigphi())) - (1/phi())).abs() < 1e-4))
    
    def testLattice(self):
        paramControl = latticeParams(10)
        paramSub = latticeParams(7)
        # Going over roughly this number will cause float innacuracy with a 32-bit float
        paramLong = latticeParams(192)

        self.assertTrue(t.all(paramSub == paramControl[:7]))
        self.assertEqual(paramControl[0], 1.)
        self.assertTrue(paramControl[1] - (1./phi()) < 0.0001)
        self.assertTrue(paramControl[9] - (1./(phi() ** 9)) < 0.0001)
        self.assertEqual(paramSub[0], 1.)
        self.assertTrue(paramSub[1] - (1./phi()) < 0.0001)
        self.assertTrue(paramSub[6] - (1./(phi() ** 6)) < 0.0001)

        self.assertTrue(t.all((paramLong[1:]/paramLong[:-1]) - (1/phi()) < 0.0001))

    def testPi(self):
        self.assertTrue(t.all(pi() - 3.1415926535 < 0.0001))
    
    def testI(self):
        built = i()
        homebrew = t.sqrt(-1 * t.ones((1), 
            dtype=DEFAULT_COMPLEX_DTYPE))

        self.assertTrue(t.all(built.real - homebrew.real < 0.0001))
        self.assertTrue(t.all(built.imag - homebrew.imag < 0.0001))

    def testEulerMascheroni(self):
        self.assertTrue((egamma() - 0.57721566490153286060651209008240243104215933593992).abs() < 1e-8)



class LatticeParamsTest(unittest.TestCase):
    def testSizingTyping(self):
        # Generate the testing tensors
        x = t.randn((1), dtype=DEFAULT_DTYPE)
        xc = t.randn_like(x, dtype=DEFAULT_COMPLEX_DTYPE)
        params = randint(1, 1024)

        # Generate the parameters to test
        y = latticeParams(n=params, basisParam=x)
        yc = latticeParams(n=params, basisParam=xc)

        # Check the sizing of the output
        self.assertEqual(y.size(-1), params)
        self.assertEqual(yc.size(-1), params)

        # Check the type of the output
        self.assertFalse(y.is_complex())
        self.assertTrue(yc.is_complex())


    def testValues(self):
        # Generate the testing tensors
        x = t.randn((1), dtype=DEFAULT_DTYPE).abs()
        params = randint(2, 64)

        # Generate the parameters to test
        ctrl = -t.log(latticeParams(n=params)) / t.log(phi())
        y = -t.log(latticeParams(n=params, basisParam=x)) / t.log(x)

        # Generate log deltas
        dctrl = ctrl[1:] - ctrl[:-1]
        dy = y[1:] - y[:-1]

        # Check to make sure that all of the values are within a single scalar of each other
        self.assertTrue((t.max(dctrl.abs()) - 1).abs() < 1e-4, msg=f'{dctrl}')
        self.assertTrue((t.max(dy.abs()) - 1).abs() < 1e-4, msg=f'{dy}')



class SoftunitTest(unittest.TestCase):
    SIZE = SUPERSINGULAR_PRIMES_HL[:4]

    def testSizingTyping(self):
        # Seeding tensors
        x = t.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Calculate
        y = softunit(x, dim=-1)
        y0 = softunit(x, dim=0)
        yc = softunit(xc, dim=-1)
        yc0 = softunit(xc, dim=0)

        # Test the sizing through the function
        self.assertEqual(x.size(), y.size(), msg=f'{x.size()} != {y.size()}')
        self.assertEqual(x.size(), y0.size(), msg=f'{x.size()} != {y0.size()}')
        self.assertEqual(x.size(), yc.size(), msg=f'{x.size()} != {yc.size()}')
        self.assertEqual(x.size(), yc0.size(), msg=f'{x.size()} != {yc0.size()}')

        # Test that only things that need to be complex are
        self.assertFalse(y.is_complex())
        self.assertFalse(y0.is_complex())
        self.assertTrue(yc.is_complex())
        self.assertTrue(yc0.is_complex())


    def testValues(self):
        # Seeding tensors
        x = t.randn(self.SIZE, dtype=DEFAULT_DTYPE) * t.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE) * t.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Calculate
        y = softunit(x, dim=-1)
        y0 = softunit(x, dim=0)
        yc = softunit(xc, dim=-1)
        yc0 = softunit(xc, dim=0)

        # Test that the values are actually softunited'd at least normally in a real value operating mode
        self.assertTrue(t.all(y == x.sign() * t.softmax(x.abs(), dim=-1)))
        self.assertTrue(t.all(y0 == x.sign() * t.softmax(x.abs(), dim=0)))

        # Test to make sure that the magnitudes are softmax'd
        self.assertTrue(t.all(t.angle(xc) - t.angle(yc) < 0.0001))
        self.assertTrue(t.all(t.angle(xc) - t.angle(yc0) < 0.0001))
        self.assertTrue(t.all(t.softmax(t.abs(xc), dim=-1) - t.abs(yc) < 0.0001))
        self.assertTrue(t.all(t.softmax(t.abs(xc), dim=0) - t.abs(yc0) < 0.0001))



class NSoftunitTest(unittest.TestCase):
    SIZE = SoftunitTest.SIZE

    def testSizingTyping(self):
        # Seeding tensors
        x = t.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Calculate the n-dimensional softmax
        y = nsoftunit(x, dims=[-1,-2])
        yc = nsoftunit(xc, dims=[-1,-2])
        z = nsoftunit(x, dims=range(len(x.size())))
        zc = nsoftunit(xc, dims=range(len(xc.size())))

        # Check to make sure that sizes came through appropriately
        self.assertEqual(x.size(), y.size())
        self.assertEqual(x.size(), z.size())
        self.assertEqual(xc.size(), yc.size())
        self.assertEqual(xc.size(), zc.size())

        # Check to make sure that only the things that should be complex, are
        self.assertFalse(y.is_complex())
        self.assertFalse(z.is_complex())
        self.assertTrue(yc.is_complex())
        self.assertTrue(zc.is_complex())


    def testRanges(self):
        # Seeding tensors
        x = t.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Calculate the n-dimensional softmax
        y = nsoftunit(x, dims=[-1,-2])
        yc = nsoftunit(xc, dims=[-1,-2])
        z = nsoftunit(x, dims=range(len(x.size())))
        zc = nsoftunit(xc, dims=range(len(xc.size())))

        # Test the ranges of the output values to verify that they do not go over
        self.assertTrue(t.min(y.abs()) >= 0)
        self.assertTrue(t.max(y.abs()) <= 1)
        self.assertTrue(t.min(yc.abs()) >= 0)
        self.assertTrue(t.max(yc.abs()) <= 1)
        self.assertTrue(t.min(z.abs()) >= 0)
        self.assertTrue(t.max(z.abs()) <= 1)
        self.assertTrue(t.min(zc.abs()) >= 0)
        self.assertTrue(t.max(zc.abs()) <= 1)



class TrigTest(unittest.TestCase):
    SIZE = SUPERSINGULAR_PRIMES_HL[:4]

    def testSizing(self):
        # Seeding tensors
        x = t.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

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
        x = t.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Calculate
        cosx = icos(x)
        cosxc = icos(xc)

        # Test the values and assert lack of runaway
        self.assertTrue(t.all(cosx == t.cos(x)))
        self.assertTrue(t.all(cosxc.abs() - t.cos(xc.abs()).abs() < 1e-4))

        # Test the values of the exp construction to assert some cos() equivalence
        self.assertTrue(t.all(
            (cosxc.abs() - (icos(xc.abs()) * t.exp(i() * pi() / 4.)).abs()) < 1e-4
        ))
        self.assertTrue(t.all(icos(t.zeros_like(xc)) == t.ones_like(xc)))
        self.assertTrue(t.all(toComplex(icos(t.zeros_like(x))) == icos(t.zeros_like(xc))))
    
    def testSin(self):
        # Seeding tensors
        x = t.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Calculate
        sinx = isin(x)
        sinxc = isin(xc)

        # Test the values and assert lack of runaway
        self.assertTrue(t.all(sinx == t.sin(x)))
        self.assertTrue(t.all(sinxc.abs() - t.sin(xc.abs()).abs() < 1e-4))

        # Test that the regular sin function is present
        self.assertTrue(t.all((sinx - isin(toComplex(x)).real).abs() < 1e-4))
        self.assertTrue(t.all(isin(toComplex(x)).imag.abs() < 1e-4))

        # Double check by asserting that the real value of the function is 0
        self.assertTrue(t.all(isin(t.zeros_like(xc)).abs() < 1e-4))
    
    def testTanh(self):
        # Seeding tensors
        x = t.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        zeros = t.zeros_like(x)
        zerosc = t.zeros_like(xc)

        # Calculate zero pass
        ctanhz = ctanh(zeros)
        ctanhzc = ctanh(zerosc)

        # Test zero passthroughs
        self.assertTrue(t.all(ctanhz == zeros.abs()))
        self.assertTrue(t.all(ctanhzc == zerosc.abs()))

        # Calculate
        ctanhx = ctanh(x)
        ctanhxc = ctanh(xc)

        # Test random passthroughs
        self.assertTrue(t.all((ctanhx - t.tanh(x)).abs() < 1e-4))
        self.assertTrue(t.all(ctanhxc.abs() < 1))



class PrimishDistTest(unittest.TestCase):
    def testSizing(self):
        # Generate random sizing
        SIZELEN = randint(1, 5)
        SIZE = t.Size((t.randn((SIZELEN), dtype=DEFAULT_DTYPE) * SIZELEN).type(dtype=t.int64).abs() + 1)
        
        # Generate the control tensors
        x = t.randn(SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

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
        tprimes = t.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).type(dtype=DEFAULT_DTYPE)
        tapres =   t.tensor([1, 0, 0, 0, 1, 0, 1, 0, 1, 2]).type(dtype=DEFAULT_DTYPE)
        trpres =   t.tensor([1, 0, 0, 0, 1, 0, 1, 0, 1/2., 1]).type(dtype=DEFAULT_DTYPE)
        tgrimes = t.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).type(dtype=DEFAULT_DTYPE)
        tagres =   t.tensor([1, 0, 0, 0, 1, 0, 1, 0, 1, 0]).type(dtype=DEFAULT_DTYPE)
        trgres =   t.tensor([1, 0, 0, 0, 1, 0, 1, 0, 1, 0]).type(dtype=DEFAULT_DTYPE)

        # Compute
        apres = realprimishdist(tprimes, relative=False, gaussApprox=False)
        rpres = realprimishdist(tprimes, relative=True, gaussApprox=False)
        agres = realprimishdist(tgrimes, relative=False, gaussApprox=True)
        rgres = realprimishdist(tgrimes, relative=True, gaussApprox=True)
        aires = iprimishdist(tprimes, relative=False)
        rires = iprimishdist(tprimes, relative=True)

        # First assert that the iprimishdist function is switching properly
        self.assertTrue(t.all((aires - apres).abs() < 1e-4), msg=f'{aires} != {apres}')
        self.assertTrue(t.all((rires - rpres).abs() < 1e-4), msg=f'{rires} != {rpres}')

        # Assert that the values of the computed output are the same as the ones
        # provided. If not, display the differences.
        self.assertTrue(t.all((tapres - apres).abs() < 1e-4), msg=f'{tapres} != {apres}')
        self.assertTrue(t.all((trpres - rpres).abs() < 1e-4), msg=f'{trpres} != {rpres}')
        self.assertTrue(t.all((tagres - agres).abs() < 1e-4), msg=f'{tagres} != {agres}')
        self.assertTrue(t.all((trgres - rgres).abs() < 1e-4), msg=f'{trgres} != {rgres}')

    def testConsistencyGauss(self):
        # Generate random sizing
        SIZELEN = randint(1, 5)
        SIZESCALAR = randint(1, 5)
        SIZE = t.Size(isigmoid(t.randn((SIZELEN), dtype=DEFAULT_DTYPE)).type(dtype=t.int64).abs() + 1) * SIZESCALAR
        
        # Generate the randomized control tensors
        trandr = t.randn(SIZE, dtype=DEFAULT_DTYPE)
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
        self.assertTrue(t.all((agres - agces).abs() < 1e-4), msg=f'{agres} != {agces}')
        self.assertTrue(t.all((agces - agies).abs() < 1e-4), msg=f'{agces} != {agies}')
        self.assertTrue(t.all((agces - agcesi).abs() < 1e-4), msg=f'{agces} != {agcesi}')
        self.assertTrue(t.all((rgres - rgces).abs() < 1e-4), msg=f'{rgres} != {rgces}')
        self.assertTrue(t.all((rgces - rgies).abs() < 1e-4), msg=f'{rgces} != {rgies}')
        self.assertTrue(t.all((rgces - rgcesi).abs() < 1e-4), msg=f'{rgces} != {rgcesi}')

    def testValuesGauss(self):
        # Generate the control tensors
        tprimes = xbias(n=10, bias=0).type(dtype=DEFAULT_COMPLEX_DTYPE)
        tpres = t.tensor([1, 0, 0, 0, 1, 0, 1, 0, 1, 0]).type(dtype=DEFAULT_DTYPE)
        tgaussians = t.tensor([2+(3*i()), 2-(3*i()), -2+(3*i()), -2-(3*i()), 6+(3*i())])
        tgres = t.tensor([0, 0, 0, 0, t.sqrt(t.ones(1) * 2)]).type(dtype=DEFAULT_DTYPE)
        trgres = t.tensor([0, 0, 0, 0, 1])

        # Compute the result of the test vectors
        pres = gaussianprimishdist(tprimes, relative=False)
        gres = gaussianprimishdist(tgaussians, relative=False)
        rgres = gaussianprimishdist(tgaussians, relative=True)

        # Assert the precomputed values
        self.assertTrue(t.all((tpres - pres).abs() < 1e-4), msg=f'{tpres} != {pres}')
        self.assertTrue(t.all((tgres - gres).abs() < 1e-4), msg=f'{tgres} != {gres}')
        self.assertTrue(t.all((trgres - rgres).abs() < 1e-4), msg=f'{trgres} != {rgres}')



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
        # Generate the primishes to test
        x = primishvals(n=1)
        y = primishvals(n=2, base=x)
        z = primishvals(n=3, base=y)
        w = primishvals(n=10, base=z)

        # Check to make sure the right values are present
        self.assertTrue(x[-1] == 1)
        self.assertTrue(y[-1] == 2, msg=f'{y[-1]}')
        self.assertTrue(z[-1] == 3, msg=f'{z[-1]}')
        self.assertTrue(t.all(z == w[:3]))
        # Ascending order
        self.assertTrue(t.all(w[:-1] < w[1:]))
        # 6k +- 1 values past first three
        self.assertTrue(t.all(((w[3:] % 6) - 5) * ((w[3:] % 6) - 1) == 0))

    def testLargeValues(self):
        # Random size extension for testing large generation
        SIZE = randint(100, 1000)
        SIZE2 = randint(100, 1000) + SIZE

        # Generate primishes to test
        x = primishvals(n=SIZE)
        y = primishvals(n=SIZE2, base=x)

        # Check to make sure that the values are consistent
        self.assertTrue(t.all(x == y[:x.size()[-1]]))
        
        # Check to make sure that all values are of 6k +- 1 and ascending
        self.assertTrue(t.all(y[:-1] < y[1:]))
        self.assertTrue(t.all(((y[3:] % 6) - 1) * ((y[3:] % 6) - 5) == 0))
    
    def testGaussSwap(self):
        # Random size extension for testing large generation
        SIZE = randint(100, 1000)
        SIZE2 = randint(100, 1000) + SIZE

        # Generate primishes to test
        x = primishvals(n=SIZE, gaussApprox=True)
        y = primishvals(n=SIZE2, base=x, gaussApprox=True)

        # Check to make sure that the values are consistent
        self.assertTrue(t.all(x == y[:x.size()[-1]]))

        # Check to make sure that all values are of 4k +- 3 and ascending
        self.assertTrue(t.all(y[:-1] < y[1:]))
        self.assertTrue(t.all(((y[3:] % 4) - 1) * ((y[3:] % 4) - 3) == 0))



class QuadcheckTest(unittest.TestCase):
    def testSizingTyping(self):
        # Generate testing tensors
        SIZELEN = randint(1, 4)
        SIZE = [randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0]) for _ in range(SIZELEN)]
        x = t.randn(SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Check the quadrants of each tensor
        q = quadcheck(x, boolChannel=False)
        qc = quadcheck(xc, boolChannel=False)
        qb = quadcheck(x, boolChannel=True)
        qbc = quadcheck(xc, boolChannel=True)

        # Check the resultant sizing
        self.assertEqual(x.size(), q.size())
        self.assertEqual(xc.size(), qc.size())
        self.assertEqual(len(x.size()) + 1, len(qb.size()))
        self.assertEqual(len(xc.size()) + 1, len(qbc.size()))
        self.assertEqual(qb.size(-1), 4)
        self.assertEqual(qbc.size(-1), 4)

        # Check the resultant types
        self.assertEqual(q.dtype, t.uint8)
        self.assertEqual(qc.dtype, t.uint8)
        self.assertEqual(qb.dtype, t.uint8)
        self.assertEqual(qbc.dtype, t.uint8)


    def testValues(self):
        # Generate testing tensors
        SIZELEN = randint(1, 4)
        SIZE = [randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0]) for _ in range(SIZELEN)]
        x = t.randn(SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        planned = t.tensor([
            1+0j, 1+1j, 0+1j, -1+1j, -1+0j, -1-1j, 0-1j, 1-1j, 0+0j
            ], dtype=DEFAULT_COMPLEX_DTYPE)
        plannedCtrl = t.tensor([
            0,    0,    1,    1,     2,     2,     3,    3,    0
            ], dtype=t.uint8)
        plannedCtrlB = t.tensor([
            [1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1],[1,0,0,0]
            ], dtype=t.uint8)

        # Check the quadrants of each tensor
        q = quadcheck(x, boolChannel=False)
        qc = quadcheck(xc, boolChannel=False)
        qb = quadcheck(x, boolChannel=True)
        qbc = quadcheck(xc, boolChannel=True)
        pq = quadcheck(planned, boolChannel=False)
        pqb = quadcheck(planned, boolChannel=True)

        # Check the range of the output values
        self.assertLessEqual(t.max(q), 3)
        self.assertLessEqual(t.max(qc), 3)
        self.assertLessEqual(t.max(qb), 1)
        self.assertLessEqual(t.max(qbc), 1)
        self.assertLessEqual(t.max(pq), 3)
        self.assertLessEqual(t.max(pqb), 1)

        # Check the pre-determined results
        self.assertTrue(t.all(pq == plannedCtrl))
        self.assertTrue(t.all(pqb == plannedCtrlB))



# TODO: class ComplexSigmoidTest(unittest.TestCase):
# TODO: class HarmonicMeanTest(unittest.TestCase):
# TODO: class HarmonicSeriesTest(unittest.TestCase):
# TODO: class RealfoldTest(unittest.TestCase):
# TODO: class OrthoFFTsTest(unittest.TestCase):