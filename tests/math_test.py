import unittest

import torch as t
from plasmatorch import *

from random import randint



class ConstantsTest(unittest.TestCase):
    def testPhi(self):
        self.assertTrue(t.all((phi() - 1.61803398875).abs() < 1e-4))

    def testAsigphi(self):
        self.assertTrue(t.all((csigmoid(asigphi()).abs() - (1/phi())).abs() < 1e-4))
        self.assertTrue(t.all((csigmoid(toComplex(asigphi())).abs() - (1/phi())).abs() < 1e-4))
    
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
        params = randint(2, 24)

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
        self.assertTrue(t.all(y == sgn(x) * t.softmax(x.abs(), dim=-1)))
        self.assertTrue(t.all(y0 == sgn(x) * t.softmax(x.abs(), dim=0)))

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
        cosx = ccos(x)
        cosxc = ccos(xc)
        sinx = csin(x)
        sinxc = csin(xc)
        tanhx = ctanh(x)
        tanhxc = ctanh(xc)

        # Test that the values go in and come out the same
        self.assertEqual(cosx.size(), x.size(), msg=f'{cosx.size()} != {x.size()}')
        self.assertEqual(cosxc.size(), xc.size(), msg=f'{cosxc.size()} != {xc.size()}')
        self.assertEqual(sinx.size(), x.size(), msg=f'{sinx.size()} != {x.size()}')
        self.assertEqual(sinxc.size(), xc.size(), msg=f'{sinxc.size()} != {xc.size()}')
        self.assertEqual(tanhx.size(), x.size(), msg=f'{tanhx.size()} != {x.size()}')
        self.assertEqual(tanhxc.size(), xc.size(), msg=f'{tanhxc.size()} != {xc.size()}')

    def testCos(self):
        # Seeding tensors
        x = t.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Calculate
        cosx = ccos(x)
        cosxc = ccos(xc)

        # Test the values and assert lack of runaway
        self.assertTrue(t.all(cosx == t.cos(x)))
        self.assertTrue(t.all(cosxc.abs() - t.cos(xc.abs()).abs() < 1e-4))

        # Test the values of the exp construction to assert some cos() equivalence
        self.assertTrue(t.all(
            (cosxc.abs() - (ccos(xc.abs()) * t.exp((pi() * 1j) / 4.)).abs()) < 1e-4
        ))
        self.assertTrue(t.all(ccos(t.zeros_like(xc)) == t.ones_like(xc)))
        self.assertTrue(t.all(toComplex(ccos(t.zeros_like(x))) == ccos(t.zeros_like(xc))))
    
    def testSin(self):
        # Seeding tensors
        x = t.randn(self.SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(self.SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Calculate
        sinx = csin(x)
        sinxc = csin(xc)

        # Test the values and assert lack of runaway
        self.assertTrue(t.all(sinx == t.sin(x)))
        self.assertTrue(t.all(sinxc.abs() - t.sin(xc.abs()).abs() < 1e-4))

        # Test that the regular sin function is present
        self.assertTrue(t.all((sinx - csin(toComplex(x)).real).abs() < 1e-4))
        self.assertTrue(t.all(csin(toComplex(x)).imag.abs() < 1e-4))

        # Double check by asserting that the real value of the function is 0
        self.assertTrue(t.all(csin(t.zeros_like(xc)).abs() < 1e-4))
    
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
        xfr = cprimishdist(x, relative=True)
        xfa = cprimishdist(x, relative=False)
        xcr = cprimishdist(xc, relative=True)
        xca = cprimishdist(xc, relative=False)

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
        aires = cprimishdist(tprimes, relative=False)
        rires = cprimishdist(tprimes, relative=True)

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
        SIZE = t.Size(csigmoid(t.randn((SIZELEN), dtype=DEFAULT_DTYPE)).type(dtype=t.int64).abs() + 1) * SIZESCALAR
        
        # Generate the randomized control tensors
        trandr = t.randn(SIZE, dtype=DEFAULT_DTYPE)
        trandrc = toComplex(trandr)
        trandrci = trandrc * 1j

        # Compute the gaussian distances for both formats of the randomized input
        agres = gaussianprimishdist(trandr, relative=False)
        agces = gaussianprimishdist(trandrc, relative=False)
        agcesi = gaussianprimishdist(trandrci, relative=False)
        agies = cprimishdist(trandrc, relative=False)
        rgres = gaussianprimishdist(trandr, relative=True)
        rgces = gaussianprimishdist(trandrc, relative=True)
        rgies = cprimishdist(trandrc, relative=True)
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
        tgaussians = t.tensor([2+(3j), 2-(3j), -2+(3j), -2-(3j), 6+(3j)])
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



class ComplexSigmoidTest(unittest.TestCase):
    def testSizingTyping(self):
        # Generate testing tensors
        SIZELEN = randint(1, 4)
        SIZE = [randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0]) for _ in range(SIZELEN)]
        x = t.randn(SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Run the testing tensors through the function needing to be tested
        sx = csigmoid(x)
        sxc = csigmoid(xc)

        # Test the result sizes and types
        self.assertEqual(sx.size(), x.size())
        self.assertEqual(sxc.size(), xc.size())
        self.assertFalse(sx.is_complex())
        self.assertTrue(sxc.is_complex())

    
    def testValues(self):
        # Generate testing tensors
        SIZELEN = randint(1, 4)
        SIZE = [randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0]) for _ in range(SIZELEN)]
        x = t.randn(SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        known = nantonum(t.tensor([t.inf+0j, t.inf*(1+1j), t.inf*1j, t.inf*(-1+1j), -t.inf, t.inf*(-1-1j), t.inf*(0-1j), t.inf*(1-1j), 0+0j], dtype=DEFAULT_COMPLEX_DTYPE), posinf=t.inf, neginf=-t.inf)

        # Run the testing tensors through the function needing to be tested
        sx = csigmoid(x)
        sxc = csigmoid(xc)
        sk = csigmoid(known)

        # Consistency check
        self.assertTrue(t.all(sx - t.sigmoid(x)) <= 1e-4)

        # Test the ranges of the values. Because the values of a sigmoid function are always
        #   between [0, 1], just test the magnitude
        self.assertLessEqual(t.max(sx.abs()), 1)
        self.assertLessEqual(t.max(sxc.abs()), 1)
        self.assertGreaterEqual(t.min(sx.abs()), 0)
        self.assertGreaterEqual(t.min(sxc.abs()), 0)

        # Test that the values come out with the appropriate magnitudes
        zeroval = sk[-1].abs()
        oppositeQuads = sk[[3, -2]].abs()
        self.assertTrue(t.sigmoid(t.zeros(1)) - zeroval <= 1e-4, msg=f'{zeroval}')
        self.assertTrue(t.all(oppositeQuads - zeroval <= 1e-4), msg=f'{known[[3, -2]]}->{oppositeQuads}')
        self.assertTrue(t.all(sk[:3].abs() - 1 <= 1e-4), msg=f'{known[:3]}->{sk[:3]}')
        self.assertTrue(t.all(sk[4:-2].abs() <= 1e-4), msg=f'{known[4:-2]}->{sk[4:-2]}')



class SgnTest(unittest.TestCase):
    def testSizingTyping(self):
        # Generate testing tensors
        SIZELEN = randint(1, 4)
        SIZE = [randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0]) for _ in range(SIZELEN)]
        x = t.randn(SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Run the testing tensors through the `sgn()` function
        sgnx = sgn(x)
        sgnxc = sgn(xc)

        # Assert that the sizing is appropriate
        self.assertEqual(x.size(), sgnx.size())
        self.assertEqual(xc.size(), sgnxc.size())

        # Assert that the typing is appropriate
        self.assertEqual(x.is_complex(), sgnx.is_complex())
        self.assertEqual(xc.is_complex(), sgnxc.is_complex())


    def testValues(self):
        # Generate testing tensors
        SIZELEN = randint(1, 4)
        SIZE = [randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0]) for _ in range(SIZELEN)]
        x = t.randn(SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Run the testing tensors through the `sgn()` function
        sgnx = sgn(x)
        sgnxc = sgn(xc)

        # For the cases where x and xc are zero, calculate if they come out as one
        zeros = t.logical_and(x == 0., sgnx == 1.)
        zerosc = t.logical_and(x == 0.+0.j, sgnxc == 1.+0.j)

        # Calculate equivalency
        validsgn = (sgnx - x.sgn()).abs() <= 1e-4
        validsgnc = (sgnxc - xc.sgn()).abs() <= 1e-4
        valid = t.logical_or(validsgn, zeros)
        validc = t.logical_or(validsgnc, zerosc)

        # Evaluate the test results
        self.assertTrue(t.all(valid))
        self.assertTrue(t.all(validc))



class HarmonicMeanTest(unittest.TestCase):
    def testSizingTyping(self):
        # Generate testing tensors
        SIZELEN = randint(2, 4)
        SIZE = [randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0]) for _ in range(SIZELEN)]
        x = t.randn(SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Store the outputs of the means for later sizing evaluation
        hx = []
        hxc = []

        # Iterate through each dim to check sizing in testing
        for idx in range(SIZELEN):
            hx.append(hmean(x, dim=idx))
            hxc.append(hmean(xc, dim=idx))

        # Test the typing results
        for idx in range(SIZELEN):
            self.assertEqual(hx[idx].dtype, x.dtype)
            self.assertFalse(hx[idx].is_complex())
            self.assertEqual(hxc[idx].dtype, xc.dtype)
            self.assertTrue(hxc[idx].is_complex())

        # Test the sizing results
        for idx in range(SIZELEN):
            tsize = SIZE[:idx] + SIZE[idx+1:]
            self.assertEqual(list(hx[idx].size()), tsize)
            self.assertEqual(list(hxc[idx].size()), tsize)
    

    def testValues(self):
        # Generate testing tensors
        SIZELEN = randint(1, 4)
        SIZE = [randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0]) for _ in range(SIZELEN)]
        x = t.randn(SIZE, dtype=DEFAULT_DTYPE)
        xc = x * sgn(t.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE))
        ones = t.ones_like(x)

        # Store the outputs of the means to be tested against later
        hx = hmean(x, dim=-1)
        hxc = hmean(xc, dim=-1)
        ho = hmean(ones, dim=-1)

        # Test to make sure that only the magnitudes are affected
        self.assertTrue(t.all((hx.abs() - hxc.abs()) <= 1e-4))
        
        # Test to make sure it is not a normal mean
        self.assertTrue(t.any(hx != t.mean(x, dim=-1)))
        self.assertTrue(t.any(hxc != t.mean(xc, dim=-1)))

        # Test to make sure that the mean never exceeds the max value of the system
        self.assertTrue(t.all(hx.abs() <= t.max(x.abs())))
        self.assertTrue(t.all(hxc.abs() <= t.max(xc.abs())))



class HarmonicValuesTest(unittest.TestCase):
    def testSizingTyping(self):
        # Set starting parameters
        SIZE = randint(SUPERSINGULAR_PRIMES_HL[0], GREISS_SAMPLES)
        
        # Run the function
        hv = harmonicvals(SIZE, noSum=False, useZero=False)
        hvz = harmonicvals(SIZE, noSum=False, useZero=True)
        hvs = harmonicvals(SIZE, noSum=True, useZero=False)
        hvsz = harmonicvals(SIZE, noSum=True, useZero=True)

        # Check that the size of the resultant vector is appropriate
        self.assertEqual(hv.numel(), SIZE)
        self.assertEqual(hvz.numel(), SIZE)
        self.assertEqual(hvs.numel(), SIZE)
        self.assertEqual(hvsz.numel(), SIZE)

        # Check that all of the values are not complex
        self.assertFalse(hv.is_complex())
        self.assertFalse(hvz.is_complex())
        self.assertFalse(hvs.is_complex())
        self.assertFalse(hvsz.is_complex())

    
    def testValues(self):
        # The control for the test
        ctrl = t.Tensor([1, 3/2, 11/6, 25/12, 137/60, 49/20, 363/140, 761/280])
        ctrls = t.Tensor([1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7]).flip(-1)

        # Run the function
        hv = harmonicvals(ctrl.numel(), noSum=False, useZero=False)
        hvz = harmonicvals(ctrl.numel()+1, noSum=False, useZero=True)
        hvs = harmonicvals(ctrls.numel(), noSum=True, useZero=False)
        hvsz = harmonicvals(ctrls.numel()+1, noSum=True, useZero=True)

        # Check consistency accross sum-similar results
        self.assertTrue(t.all((hv - ctrl) <= 1e-4))
        self.assertTrue(t.all((hvz[1:] - ctrl <= 1e-4)))
        self.assertTrue(t.all((hvs - ctrls) <= 1e-4))
        self.assertTrue(t.all((hvsz[1:] - ctrls) <= 1e-4))
        self.assertTrue(hvz[0] == 0)
        self.assertTrue(hvsz[0] == 0)



class HarmonicDistanceTest(unittest.TestCase):
    def testSizingTyping(self):
        # Generate testing tensors
        SIZELEN = randint(1, 4)
        SIZE = [randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0]) for _ in range(SIZELEN)]
        TSIZE = t.Size(SIZE)
        x = t.randn(SIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Run the function
        hx = harmonicdist(x)
        hxc = harmonicdist(xc)

        # Test that the sizing hasn't changed after passing through the function
        self.assertTrue(hx.size() == TSIZE, msg=f'{hx.size()}\t!=\t{TSIZE}')
        self.assertTrue(hxc.size() == TSIZE, msg=f'{hxc.size()}\t!=\t{TSIZE}')

        # Test to make sure that complexity of the number-space hasn't changed
        self.assertFalse(hx.is_complex())
        self.assertTrue(hxc.is_complex())


    def testValues(self):
        # The tensors to test with
        SIZELEN = randint(1, 4)
        SIZE = [randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0]) for _ in range(SIZELEN)]
        TSIZE = t.Size(SIZE)
        x = t.randn(TSIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(TSIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        ctrl = t.Tensor([1, 3/2, 11/6, 25/12, 137/60, 49/20, 363/140, 761/280])
        ctrlc = t.view_as_complex(t.stack([ctrl, ctrl.flip(-1)], dim=-1))
        harm = harmonicvals(n=randint(SUPERSINGULAR_PRIMES_HL[0], GREISS_SAMPLES), useZero=True)

        # Run the testing tensors through the function to find the distances from each unit
        hx = harmonicdist(x)
        hxc = harmonicdist(xc)
        hctrl = harmonicdist(ctrl)
        hctrlc = harmonicdist(ctrlc)
        hharm = harmonicdist(harm)

        # Test the resultant values to make sure none of them are over half of the
        #   harmonic series' maximum value of 1 (so 0.5)
        self.assertTrue(t.all(hx.abs().max() <= 0.5), msg=f'{x}->{hx}')
        self.assertTrue(t.all(hxc.abs().max() <= t.sqrt(t.tensor([0.5]))), msg=f'{xc}->{hxc}')
        self.assertTrue(t.all(hctrl.abs().max() <= 0.5), msg=f'{ctrl}->{hctrl}')
        self.assertTrue(t.all(hctrlc.abs().max() <= t.sqrt(t.tensor([0.5]))), msg=f'{ctrlc}->{hctrlc}')
        self.assertTrue(t.all(hharm.abs() <= 1e-4), msg=f'{hharm}')



class RealfoldTest(unittest.TestCase):
    def testSizingTyping(self):
        # Generate testing tensors
        SIZELEN = randint(1, 4)
        SIZE = [randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0]) for _ in range(SIZELEN)]
        TSIZE = t.Size(SIZE)
        x = t.randn(TSIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(TSIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        xo = x.to(DEFAULT_COMPLEX_DTYPE)

        # Run the tensors through the function
        rx = realfold(x)
        rxc = realfold(xc)
        rxo = realfold(xo)

        # Test to make sure no complex values propogate through
        self.assertFalse(rx.is_complex())
        self.assertFalse(rxc.is_complex())
        self.assertFalse(rxo.is_complex())

        # Test to make sure that the sizing is consistent
        self.assertEqual(rx.size(), TSIZE)
        self.assertEqual(rxc.size(), TSIZE)
        self.assertEqual(rxo.size(), TSIZE)


    def testValues(self):
        # Generate testing tensors
        SIZELEN = randint(1, 4)
        SIZE = [randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0]) for _ in range(SIZELEN)]
        TSIZE = t.Size(SIZE)
        x = t.randn(TSIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(TSIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        xo = x.to(DEFAULT_COMPLEX_DTYPE)

        # Run the tensors through the function
        rx = realfold(x)
        rxc = realfold(xc)
        rxo = realfold(xo)
        rxc2 = realfold(xc, phase=t.zeros(1))

        # Test to make sure that real values are not operated on
        self.assertTrue(t.all(rx == x))
        self.assertTrue(t.all(rxo == x))
        
        # Test to make sure that the imaginary values are just subtracted
        self.assertTrue(t.all((rxc - (xc.real - xc.imag)) <= 1e-4))

        # Test to make sure that the 0 phase option has the values added
        self.assertTrue(t.all((rxc2 - (xc.real + xc.imag)) <= 1e-4))



class OrthoFFTsTest(unittest.TestCase):
    # Use the Plancherel theorem to ensure energy conservation, maybe
    # https://en.wikipedia.org/wiki/Plancherel_theorem
    def testSizingTyping(self):
        # Generate testing tensors
        SIZELEN = randint(2, 4)
        SIZE = [randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0]) for _ in range(SIZELEN)]
        TSIZE = t.Size(SIZE)
        x = t.randn(TSIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(TSIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        ns = [-1] * SIZELEN
        dims = [element for element in range(SIZELEN)]

        # Run the vectors through the fft functions
        fx = fft(x, n=-1, dim=-1)
        ffx = fft(x, n=ns, dim=dims)
        ifx = ifft(x, n=-1, dim=-1)
        iffx = ifft(x, n=ns, dim=dims)
        fxc = fft(xc, n=-1, dim=-1)
        ffxc = fft(xc, n=ns, dim=dims)
        ifxc = ifft(xc, n=-1, dim=-1)
        iffxc = ifft(xc, n=ns, dim=dims)

        # Assert that the sizing has not changed
        self.assertEqual(fx.size(), TSIZE)
        self.assertEqual(ffx.size(), TSIZE)
        self.assertEqual(ifx.size(), TSIZE)
        self.assertEqual(iffx.size(), TSIZE)
        self.assertEqual(fxc.size(), TSIZE)
        self.assertEqual(ffxc.size(), TSIZE)
        self.assertEqual(ifxc.size(), TSIZE)
        self.assertEqual(iffxc.size(), TSIZE)

        # Assert that all results are complex, ignoring the input type
        self.assertTrue(fx.is_complex())
        self.assertTrue(ffx.is_complex())
        self.assertTrue(ifx.is_complex())
        self.assertTrue(iffx.is_complex())
        self.assertTrue(fxc.is_complex())
        self.assertTrue(ffxc.is_complex())
        self.assertTrue(ifxc.is_complex())
        self.assertTrue(iffxc.is_complex())


    def testConsistency(self):
        # Generate testing tensors
        SIZELEN = randint(2, 4)
        SIZE = [randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0]) for _ in range(SIZELEN)]
        TSIZE = t.Size(SIZE)
        x = t.randn(TSIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(TSIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        ns = [-1] * SIZELEN
        dims = [element for element in range(SIZELEN)]

        # Run the vectors through the fft functions
        fx = fft(x, n=-1, dim=-1)
        ffx = fft(x, n=ns, dim=dims)
        ifx = ifft(x, n=-1, dim=-1)
        iffx = ifft(x, n=ns, dim=dims)
        fxc = fft(xc, n=-1, dim=-1)
        ffxc = fft(xc, n=ns, dim=dims)
        ifxc = ifft(xc, n=-1, dim=-1)
        iffxc = ifft(xc, n=ns, dim=dims)

        # Check to make sure that the single dimension ffts are checking out to
        #   the standard torch library
        cfx = tfft.fft(x, n=x.size(-1), dim=-1, norm='ortho')
        cifx = tfft.ifft(x, n=x.size(-1), dim=-1, norm='ortho')
        cfxc = tfft.fft(xc, n=xc.size(-1), dim=-1, norm='ortho')
        cifxc = tfft.ifft(xc, n=xc.size(-1), dim=-1, norm='ortho')
        self.assertTrue(t.all((cfx - fx).abs() <= 1e-4))
        self.assertTrue(t.all((cifx - ifx).abs() <= 1e-4))
        self.assertTrue(t.all((cfxc - fxc).abs() <= 1e-4))
        self.assertTrue(t.all((cifxc - ifxc).abs() <= 1e-4))

        # Check to make sure that the multiple dimension ffts are checking out also
        cffx = tfft.fftn(x, s=ns, dim=dims, norm='ortho')
        ciffx = tfft.ifftn(x, s=ns, dim=dims, norm='ortho')
        cffxc = tfft.fftn(xc, s=ns, dim=dims, norm='ortho')
        ciffxc = tfft.ifftn(xc, s=ns, dim=dims, norm='ortho')
        self.assertTrue(t.all((cffx - ffx).abs() <= 1e-4))
        self.assertTrue(t.all((ciffx - iffx).abs() <= 1e-4))
        self.assertTrue(t.all((cffxc - ffxc).abs() <= 1e-4))
        self.assertTrue(t.all((ciffxc - iffxc).abs() <= 1e-4))


    def testPlancherel(self):
        # Generate testing tensors
        SIZELEN = randint(2, 4)
        SIZE = [randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0]) for _ in range(SIZELEN)]
        TSIZE = t.Size(SIZE)
        x = t.randn(TSIZE, dtype=DEFAULT_DTYPE)
        xc = t.randn(TSIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Run the vectors through the fft functions
        fx =    (fft(x,   n=-1, dim=-1  ).abs() ** 2).sum(-1)
        ifx =   (ifft(x,  n=-1, dim=-1  ).abs() ** 2).sum(-1)
        fxc =   (fft(xc,  n=-1, dim=-1  ).abs() ** 2).sum(-1)
        ifxc =  (ifft(xc, n=-1, dim=-1  ).abs() ** 2).sum(-1)

        # Get the magnitude squared of x and xc
        plan =  (x.abs()  ** 2).sum(-1)
        planc = (xc.abs() ** 2).sum(-1)
        
        # Compare the values of the functions according to the Plancherel theorem
        #   ensuring conservation of energy
        # Single dims
        self.assertTrue(t.all((fx - plan).abs() <= 1e-4))
        self.assertTrue(t.all((fxc - planc).abs() <= 1e-4))
        self.assertTrue(t.all((ifx - plan).abs() <= 1e-4))
        self.assertTrue(t.all((ifxc - planc).abs() <= 1e-4))
        