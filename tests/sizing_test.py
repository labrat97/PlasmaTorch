import unittest

import torch as t
from plasmatorch import *

from random import randint



class UnflattenTest(unittest.TestCase):
    def testConsistency(self):
        # Generate the starting tensors
        SIZELEN:int = randint(2, 4)
        SIZE:List[int] = [randint(1, SUPERSINGULAR_PRIMES_LH[7]) for _ in range(SIZELEN)]
        TSIZE:t.Size = t.Size(SIZE)
        x:t.Tensor = t.randn(TSIZE, dtype=DEFAULT_DTYPE)
        xc:t.Tensor = t.randn(TSIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        START_IDX:int = randint(0, x.dim()-2)
        END_IDX:int = randint(START_IDX, x.dim()-1)
        FLATTENED_SIZE:List[int] = SIZE[START_IDX:END_IDX+1]

        # Flatten the tensors with the working function included from torch
        flat:t.Tensor = t.flatten(x, start_dim=START_IDX, end_dim=END_IDX)
        flatc:t.Tensor = t.flatten(xc, start_dim=START_IDX, end_dim=END_IDX)

        # Unflatten the tensors at the start index
        ux:t.Tensor = unflatten(flat, dim=START_IDX, size=FLATTENED_SIZE)
        uxc:t.Tensor = unflatten(flatc, dim=START_IDX, size=FLATTENED_SIZE)

        # Test sizing
        self.assertEqual(x.size(), ux.size())
        self.assertEqual(xc.size(), uxc.size())

        # Test the values
        self.assertTrue(t.all((ux - x).abs() <= 1e-4), msg=f'{ux}\n!=\n{x}')
        self.assertTrue(t.all((uxc - xc).abs() <= 1e-4), msg=f'{uxc}\n!=\n{xc}')



class ResignalTest(unittest.TestCase):
    def testSizingByDim(self):
        # Generate the starting tensors
        SIZELEN:int = randint(1, 4)
        SIZE:List[int] = [randint(1, SUPERSINGULAR_PRIMES_LH[7]) for _ in range(SIZELEN)]
        TSIZE:t.Size = t.Size(SIZE)
        x:t.Tensor = t.randn(TSIZE, dtype=DEFAULT_DTYPE)
        xc:t.Tensor = t.randn(TSIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Iterate through each possible dim in the tensors for testing
        for idx in range(SIZELEN):
            # Generate the number of samples for the dimension
            samples:int = randint(1, GREISS_SAMPLES)
            
            # Run the tensors through the function
            rx:t.Tensor = resignal(x, samples=samples, dim=idx)
            rxc:t.Tensor = resignal(xc, samples=samples, dim=idx)

            # Generate the size that should come out of the system
            newSize:List[int] = SIZE.copy()
            newSize[idx] = samples

            # Test the sizing
            self.assertEqual(rx.size(), t.Size(newSize))
            self.assertEqual(rxc.size(), t.Size(newSize))


    def testReversability(self):
        # Generate the starting tensors
        SIZELEN:int = randint(1, 4)
        SIZE:List[int] = [randint(1, SUPERSINGULAR_PRIMES_LH[7]) for _ in range(SIZELEN)]
        TSIZE:t.Size = t.Size(SIZE)
        x:t.Tensor = t.randn(TSIZE, dtype=DEFAULT_DTYPE)
        xc:t.Tensor = t.randn(TSIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Iterate through each possible dim in the tensors for testing
        for idx in range(SIZELEN):
            # Generate the number of samples for the dimension
            samples:int = randint(SIZE[idx], GREISS_SAMPLES)

            # Run the tensors through the function to increase the samples
            rx:t.Tensor = resignal(x, samples=samples, dim=idx)
            rxc:t.Tensor = resignal(xc, samples=samples, dim=idx)

            # Run the tensors back through the function to reduce the samples
            rrx:t.Tensor = resignal(rx, samples=SIZE[idx], dim=idx)
            rrxc:t.Tensor = resignal(rxc, samples=SIZE[idx], dim=idx)

            # Check for value consistency
            self.assertTrue(t.all((rrx - x).abs() <= 1e-4))
            self.assertTrue(t.all((rrxc - xc).abs() <= 1e-4))



class PaddimTest(unittest.TestCase):
    PADOPTIONS:List[str] = ['reflect', 'replicate', 'circular']


    def testSizingByDim(self):
        # Generate the starting tensors
        SIZELEN:int = randint(2, 4)
        SIZE:List[int] = [randint(1, SUPERSINGULAR_PRIMES_LH[7]) for _ in range(SIZELEN)]
        TSIZE:t.Size = t.Size(SIZE)
        x:t.Tensor = t.randn(TSIZE, dtype=DEFAULT_DTYPE)
        xc:t.Tensor = t.randn(TSIZE, dtype=DEFAULT_COMPLEX_DTYPE)

        # Iterate through each possible dim in the testing tensors and test the padding
        #   capabilities
        for idx in range(SIZELEN):
            # Generate the random values for padding the tensor on either side
            lowpad:int = randint(0, int((SIZE[idx]/2)-1))
            highpad:int = randint(0, int((SIZE[idx]/2)-1))
            padmode:str = self.PADOPTIONS[randint(0, len(self.PADOPTIONS)-1)]

            # Run the testing tensors through the padding function
            px:t.Tensor = paddim(x, lowpad=lowpad, highpad=highpad, dim=idx, mode=padmode)
            pxc:t.Tensor = paddim(xc, lowpad=lowpad, highpad=highpad, dim=idx, mode=padmode)

            # Assert the sizing for the test
            sizeDummy:List[int] = SIZE.copy()
            sizeDummy[idx] += lowpad + highpad
            self.assertEqual(px.size(), t.Size(sizeDummy), msg=f'{px.size()} != {sizeDummy} with a low and high {padmode}-padding of {(lowpad, highpad)} on dim {idx}')
            self.assertEqual(pxc.size(), t.Size(sizeDummy), msg=f'{pxc.size()} != {sizeDummy} with a low and high {padmode}-padding of {(lowpad, highpad)} on dim {idx}')
    

    def testConsistency(self):
        # Generate the starting tensors
        SIZELEN:int = 3
        SIZE:List[int] = [randint(11, SUPERSINGULAR_PRIMES_LH[7]) for _ in range(SIZELEN)]
        TSIZE:t.Size = t.Size(SIZE)
        x:t.Tensor = t.randn(TSIZE, dtype=DEFAULT_DTYPE)
        xc:t.Tensor = t.randn(TSIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        
        # Test the values of each pad mode to make sure they are consistent with
        #   the return values of `nnf.pad()`
        for padmode in self.PADOPTIONS:
            # Generate the random values for padding the tensor on either side
            lowpad:int = randint(0, 5)
            highpad:int = randint(0, 5)

            # Run the testing tensors through the `paddim()` function
            px:t.Tensor = paddim(x, lowpad=lowpad, highpad=highpad, dim=-1, mode=padmode)
            pxc:t.Tensor = paddim(xc, lowpad=lowpad, highpad=highpad, dim=-1, mode=padmode)

            # Run the testing tensors through the control `nnf.pad()` function
            xcont:t.Tensor = nnf.pad(x, pad=[lowpad, highpad], mode=padmode)
            xccont:t.Tensor = nnf.pad(xc, pad=[lowpad, highpad], mode=padmode)

            # Check that the values of the control functions and the `paddim()` functions are within
            #   a reasonable epsilon value from each other
            self.assertTrue(t.all((px - xcont).abs() <= 1e-4))
            self.assertTrue(t.all((pxc - xccont).abs() <= 1e-4))



class DimmatchTest(unittest.TestCase):
    def testSizingByDim(self):
        # Generate the starting tensors
        SIZELEN:int = randint(2, 4)
        SIZEA:List[int] = [randint(1, SUPERSINGULAR_PRIMES_LH[7]) for _ in range(SIZELEN)]
        SIZEB:List[int] = [randint(1, SUPERSINGULAR_PRIMES_LH[7]) for _ in range(SIZELEN)]
        a:t.Tensor = t.randn(SIZEA, dtype=DEFAULT_DTYPE)
        ac:t.Tensor = t.randn(SIZEA, dtype=DEFAULT_COMPLEX_DTYPE)
        b:t.Tensor = t.randn(SIZEB, dtype=DEFAULT_DTYPE)
        bc:t.Tensor = t.randn(SIZEB, dtype=DEFAULT_COMPLEX_DTYPE)
        testingTensors:List[List[t.Tensor]] = [[a, b], [ac, bc]]

        # Iterate through each possible dim to make sure that the testing tensors
        #   work in any perspective through the dim matching function
        for idx in range(SIZELEN):
            # Check each pair of tensors
            for ta, tb in testingTensors:
                # Run each pair of tensors through the dimension matching function
                da, db = dimmatch(ta, tb, dim=idx)

                # Assert that the sizes are what they should be
                self.assertEqual(da.size(idx), db.size(idx))
                self.assertEqual(max(ta.size(idx), tb.size(idx)), da.size(idx))



class WeightedResampleTest(unittest.TestCase):
    def __testBase__(self, posgen:Callable[[int, int], t.Tensor], test:Callable[[t.Tensor, t.Tensor, t.Tensor, int, bool], str]):
        # Generate the tensors for testing
        SIZELEN:int = randint(2, 4)
        SIZE:List[int] = [randint(1, SUPERSINGULAR_PRIMES_LH[7]) for _ in range(SIZELEN)]
        x:t.Tensor = t.randn(SIZE, dtype=DEFAULT_DTYPE)
        xc:t.Tensor = t.randn(SIZE, dtype=DEFAULT_COMPLEX_DTYPE)
        posWeights:List[t.Tensor] = [posgen(idx, dimlen) for idx, dimlen in enumerate(SIZE)]

        # Randomize batch dimension
        for idx in range(1, len(posWeights)):
            addDim:bool = bool(randint(0, 1))
            if addDim: posWeights[idx] = t.stack([posWeights[idx]] * SIZE[0], dim=0)
        for weight in posWeights: assert not weight.is_complex()

        # Iterate through each possible dim to make sure that the weighted resample is 
        #   performing appropriately. Specifically check the sizing here.
        for idx, posWeight in enumerate(posWeights):
            # Run each dim of the testing tensors through the function
            wx:t.Tensor = weightedResample(x, pos=posWeight, dim=idx, ortho=False)
            wxc:t.Tensor = weightedResample(xc, pos=posWeight, dim=idx, ortho=False)
            wxo:t.Tensor = weightedResample(x, pos=posWeight, dim=idx, ortho=True)
            wxco:t.Tensor = weightedResample(xc, pos=posWeight, dim=idx, ortho=True)

            # Run the perscribed test
            runResults:List[str] = [test(wx, x, posWeight, idx, False),
            test(wxc, xc, posWeight, idx, False),
            test(wxo, x, posWeight, idx, True),
            test(wxco, xc, posWeight, idx, True)]
            for idx, result in enumerate(runResults):
                self.assertTrue(result == None, msg=f'[{idx}]->{result}')


    def __sizingPos__(idx:int, dimlen:int) -> t.Tensor:
        # The actual selected positions shouldn't matter here
        return t.randn(randint(1, dimlen*2), dtype=DEFAULT_DTYPE)

    def __sizingTest__(wx:t.Tensor, x:t.Tensor, pos:t.Tensor, dim:int, ortho:bool) -> str:
        # Iterate only the value on the dim
        size:List[int] = list(x.size())
        size[dim] = pos.size(-1)

        # Check to make sure the size iterated in the expected way
        if wx.size() == t.Size(size):
            return None
        return f'[{dim}][{x.size()}||{pos.size()}]\n{wx.size()} != {t.Size(size)}'

    def testSizingByDim(self):
        # Test the sizing on each dim
        self.__testBase__(posgen=WeightedResampleTest.__sizingPos__, test=WeightedResampleTest.__sizingTest__)


    def __valueOrthoPos__(idx:int, dimlen:int) -> t.Tensor:
        # This is the most predictable and most contrasting position selector when ortho is togglable
        return t.zeros(dimlen, dtype=DEFAULT_DTYPE)

    def __valueOrthoTest__(wx:t.Tensor, x:t.Tensor, pos:t.Tensor, dim:int, ortho:bool) -> str:
        # Get the values readily available
        normx:t.Tensor = wx.transpose(dim,-1)
        
        # Test based on if ortho is enabled
        retmsg = None
        if ortho:
            # If ortho, should be a straight passthrough the system
            retval:bool = t.all((wx - x).abs() <= 5e-3)
            if not retval: retmsg = f'{x} !=-> {wx}'
        else:
            # If not ortho, should all be the same value
            retval:bool = t.all((normx[...,:-1] - normx[...,1:]).abs() <= 1e-4)
            if not retval: retmsg = f'wx.transpose({dim},-1)->({normx}) not self-similar'
        return retmsg
    
    def testValuesOrtho(self):
        # Test the values based around ortho toggling
        self.__testBase__(posgen=WeightedResampleTest.__valueOrthoPos__, test=WeightedResampleTest.__valueOrthoTest__)
    

    def __valueRandnPos__(idx:int, dimlen:int) -> t.Tensor:
        # Random values should have random positions
        return t.randn(randint(1, dimlen*2), dtype=DEFAULT_DTYPE)
    
    def __valueRandnTest__(wx:t.Tensor, x:t.Tensor, pos:t.Tensor, dim:int, ortho:bool) -> str:
        # Precompute re-used values
        wxabs:t.Tensor = wx.abs()
        xabs:t.Tensor = x.abs()

        # Check to make sure that nothing shoots over the original values provided
        ret:bool = wxabs.max() <= (xabs.max() + 1e-4)
        # Can't really test minimum value here due to interpolation mixed with sample positions

        if ret:
            return None
        return f'{(xabs.min(), xabs.max())} -<>> {(wxabs.min(), wxabs.max())}'

    def testValuesRandn(self):
        # Test the values based around random sizing and values
        self.__testBase__(posgen=WeightedResampleTest.__valueRandnPos__, test=WeightedResampleTest.__valueRandnTest__)
