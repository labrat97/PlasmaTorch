import unittest
import test

import torch
from plasmatorch import *
from plasmatorch.defaults import DEFAULT_DTYPE
import math
from random import randint



class EntangleTest(unittest.TestCase):
    def testParameters(self):
        # Create the modules required to test the enclosed parameters for consistency
        signals:int = 3
        channels:int = 3
        subject = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SMALL_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_DTYPE)
        subjectc = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SMALL_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_COMPLEX_DTYPE)
        

        # Assert that both structures have the same settings after construction
        self.assertEqual(subject.signalCount, subjectc.signalCount)
        self.assertEqual(subject.curveChannels, subjectc.curveChannels)
        self.assertEqual(subject.samples, subjectc.samples)
        self.assertEqual(subject.outputMode, subjectc.outputMode)
        self.assertEqual(len(subject.entangleActivation), len(subjectc.entangleActivation))
        self.assertEqual(subject.entanglePolarization.size(), subjectc.entanglePolarization.size())
        self.assertEqual(subject.knowledgeMask.size(), subject.knowledgeMask.size())

        # Make sure that the weights aren't the wrong size resulting in less effecient computation
        self.assertEqual(subject.signalCount, signals)
        self.assertEqual(subject.curveChannels, channels)
        self.assertEqual(subject.samples, test.TEST_FFT_SMALL_SAMPLES)
        self.assertEqual(subject.outputMode, EntangleOutputMode.BOTH)
        self.assertEqual(len(subject.entangleActivation), signals)
        self.assertEqual(len(subject.entanglePolarization), signals)
        self.assertEqual(subject.knowledgeMask.size(), (signals, channels, test.TEST_FFT_SMALL_SAMPLES, test.TEST_FFT_SMALL_SAMPLES))

        # Assert that the identity matrix is properly being transfered over to the
        #   real numbers only to initialize the knowledge mask weights
        self.assertTrue(torch.all(subject.knowledgeMask == toComplex(torch.eye(test.TEST_FFT_SMALL_SAMPLES))))
        self.assertTrue(torch.all(subjectc.knowledgeMask == toComplex(torch.eye(test.TEST_FFT_SMALL_SAMPLES))))


    def testSizing(self):
        # Parameter initialization
        batches:int = 2
        signals:int = 2
        channels:int = 2

        # Create the modules required to test the enclosed parameters for consistency
        subject = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SMALL_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_DTYPE)
        subjectc = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SMALL_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_COMPLEX_DTYPE)

        # Tensors for evaluation
        x = torch.randn((batches, signals, channels, test.TEST_FFT_SMALL_SAMPLES), dtype=DEFAULT_DTYPE)
        xc = torch.randn((batches, signals, channels, test.TEST_FFT_SMALL_SAMPLES), dtype=DEFAULT_COMPLEX_DTYPE)

        # Push data through entanglers
        col, sup = subject.forward(x)
        colc, supc = subjectc.forward(xc)

        # Assert equivalence between real and complex data sizes
        self.assertEqual(col.size(), colc.size())
        self.assertEqual(sup.size(), supc.size())
        
        # Assert equivalence between input and output sizes
        self.assertEqual(x.size(), col.size())
        self.assertEqual(xc.size(), colc.size())

        # Assert equivalence between the knowledge mask sizing and the
        #   the superposition sizing
        self.assertEqual(sup.size()[1:], subject.knowledgeMask.size())
        self.assertEqual(supc.size()[1:], subjectc.knowledgeMask.size())


    def testValues(self):
        # Parameter initialization
        batches:int = 2
        signals:int = 2
        channels:int = 2

        # Create the modules required to test the enclosed parameters for consistency
        subject = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SMALL_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_DTYPE)
        subjectc = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SMALL_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_COMPLEX_DTYPE)

        # Tensors for evaluation
        x = torch.zeros((batches, signals, channels, test.TEST_FFT_SMALL_SAMPLES), dtype=DEFAULT_DTYPE)
        xc = torch.zeros((batches, signals, channels, test.TEST_FFT_SMALL_SAMPLES), dtype=DEFAULT_COMPLEX_DTYPE)

        # Push data through entanglers
        col, sup = subject.forward(x)
        colc, supc = subjectc.forward(xc)

        # Make sure that the values recieved on the other end come through as zero
        self.assertTrue(torch.all(col == 0))
        self.assertTrue(torch.all(sup == 0))
        self.assertTrue(torch.all(colc == 0))
        self.assertTrue(torch.all(supc == 0))

        # More tensors for evaluation, but for more value checking
        x = torch.randn_like(x)
        xc = torch.randn_like(xc)

        # Push data through the entanglers
        colrr, suprr = subject.forward(x)
        colrc, suprc = subject.forward(xc)
        colcr, supcr = subjectc.forward(x)
        colcc, supcc = subjectc.forward(xc)
        _, suprc0 = subject.forward(toComplex(x))
        _, supcc0 = subjectc.forward(toComplex(x))

        # Make sure the values that come through on the other side consistent with themselves
        self.assertTrue(torch.all(colrr == colcr))
        self.assertTrue(torch.all(suprr == supcr))
        self.assertTrue(torch.all(colrc == colcc))
        self.assertTrue(torch.all(suprc == supcc))

        # Test for self similarity
        self.assertTrue(torch.all(suprr.real - suprc0.real < 1e-5), msg=f'{t.max(suprr.real - suprc0.real)}')
        self.assertTrue(torch.all(suprr.imag - suprc0.imag < 1e-5))
        self.assertTrue(torch.all(supcr.real - supcc0.real < 1e-5))
        self.assertTrue(torch.all(supcr.imag - supcc0.imag < 1e-5))


    def testDifference(self):
        # Parameter initialization
        batches:int = 2
        signals:int = 2
        channels:int = 2

        # Create the modules required to test the enclosed parameters for inconsistency
        subject = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SMALL_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_DTYPE)
        subjectc = Entangle(inputSignals=signals, curveChannels=channels, \
            samples=test.TEST_FFT_SMALL_SAMPLES, useKnowledgeMask=True, \
            outputMode=EntangleOutputMode.BOTH, dtype=DEFAULT_COMPLEX_DTYPE)

        # Replace parameters to generate inconsistency
        subject.knowledgeMask = nn.Parameter(torch.randn_like(subject.knowledgeMask))
        subjectc.knowledgeMask = nn.Parameter(torch.randn_like(subjectc.knowledgeMask))

        # Tensors for evaluation
        x = torch.randn((batches, signals, channels, test.TEST_FFT_SMALL_SAMPLES), dtype=DEFAULT_DTYPE)
        xc = torch.view_as_complex(torch.stack((x, torch.randn_like(x)), dim=-1))

        # Push data through the entanglers
        colrr, suprr = subject.forward(x)
        colrc, suprc = subject.forward(xc)
        colcr, supcr = subjectc.forward(x)
        colcc, supcc = subjectc.forward(xc)

        # All of these should not be the same
        self.assertTrue(torch.all(colrr != colrc))
        self.assertTrue(torch.all(suprr != suprc))
        self.assertTrue(torch.all(colcr != colcc))
        self.assertTrue(torch.all(supcr != supcc))



class SuperPositionTest(unittest.TestCase):
    def testSizingTyping(self):
        # Create the testing tensors
        x = t.randn((randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0])), dtype=DEFAULT_DTYPE)
        xc = t.randn_like(x, dtype=DEFAULT_COMPLEX_DTYPE)

        # Run through
        y = superposition(x, x)
        yc = superposition(xc, xc)

        # Test sizing
        self.assertTrue(2*x.size() == y.size())
        self.assertTrue(2*xc.size() == yc.size())
        
        # Test typing
        self.assertTrue(y.dtype == x.dtype)
        self.assertTrue(yc.dtype == xc.dtype)
        self.assertTrue(yc.dtype != y.dtype)
        self.assertTrue(y.dtype != xc.dtype)
    

    def testRanges(self):
        # Create the testing tensors
        x = t.randn((randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0])), dtype=DEFAULT_DTYPE)
        xc = t.randn_like(x, dtype=DEFAULT_COMPLEX_DTYPE)

        # Run through
        y = superposition(x, x)
        yc = superposition(xc, xc)

        # Test the ranges of the values coming out of the function
        self.assertTrue(t.max(y.abs()) < 1)
        self.assertTrue(t.min(y.abs()) > 0)
        self.assertTrue(t.max(yc.abs()) < 1)
        self.assertTrue(t.min(yc.abs()) > 0)



class CollapseTest(unittest.TestCase):
    def testSizingTyping(self):
        # Create the testing tensors
        x = t.randn(2 * [randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0])], dtype=DEFAULT_DTYPE)
        xc = t.randn_like(x, dtype=DEFAULT_COMPLEX_DTYPE)
        pol = t.randn_like(x[0])
        polc = t.randn_like(xc[0])

        # Collapse the testing tensors
        y = collapse(x, pol)
        yc = collapse(xc, polc)

        # Test sizing
        self.assertTrue(y.size(0) == x.size(0))
        self.assertTrue(yc.size(0) == xc.size(0))
        self.assertTrue(len(y.size()) == 1, msg=f'{x.size()} -> {y.size()}')
        self.assertTrue(len(yc.size()) == 1, msg=f'{x.size()} -> {yc.size()}')

        # Test typing
        self.assertTrue(y.is_complex())
        self.assertTrue(yc.is_complex())


    def testRanges(self):
        # Create the testing tensors
        x = nsoftunit(t.randn(2 * [randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0])], dtype=DEFAULT_DTYPE), dims=[-1,-2])
        xc = nsoftunit(t.randn_like(x, dtype=DEFAULT_COMPLEX_DTYPE), dims=[-1,-2])
        pol = t.randn_like(x[0])
        polc = t.randn_like(xc[0])

        # Collapse the testing tensors
        y = collapse(x, pol)
        yc = collapse(xc, polc)

        # Test
        self.assertTrue(t.max(y.abs()) < 1)
        self.assertTrue(t.min(y.abs()) > 0)
        self.assertTrue(t.max(yc.abs()) < 1)
        self.assertTrue(t.min(yc.abs()) > 0)



class EntangleFunctionTest(unittest.TestCase):
    def testSizingTyping(self):
        # Create the testing tensors
        x = t.randn((randint(SUPERSINGULAR_PRIMES_HL[1], SUPERSINGULAR_PRIMES_HL[0])), dtype=DEFAULT_DTYPE)
        xc = t.randn_like(x, dtype=DEFAULT_COMPLEX_DTYPE)
        pol = t.randn_like(x)
        polc = t.randn_like(xc)
        mask = t.randn(2*x.size(), dtype=x.dtype)
        maskc = t.randn_like(mask, dtype=xc.dtype)

        # Run through
        y = entangle(x, x, mask, pol)
        yc = entangle(xc, xc, maskc, polc)

        # Test sizing
        self.assertTrue(x.size() == y.size())
        self.assertTrue(xc.size() == yc.size())
        
        # Test typing
        self.assertTrue(y.is_complex())
        self.assertTrue(yc.is_complex())
