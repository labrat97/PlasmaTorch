import unittest

import torch as t
from plasmatorch import *



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
        self.assertTrue(t.all((pi() - 3.1415926535).abs() <= 1e-4))

    def testTau(self):
        self.assertTrue(t.all((tau() - (2*3.1415926535)).abs() <= 1e-4))

    def testEulerMascheroni(self):
        self.assertTrue((egamma() - 0.57721566490153286060651209008240243104215933593992).abs() < 1e-8)
