from ..defaults import *
from ..activations import *
from ..entanglement import *
from ..distributions import *
from ..conversions import *
from ..attention import *

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from enum import Enum
from abc import ABC, abstractmethod

class KnowledgeOutputMode(int, Enum):
    """
    The output mode of the KnowledgeFilter class.
    """
    # NONE type and LOGIT type are equivalent because by default torch uses logits
    NONE = 0
    LOGITS = NONE

    # Already time(ish) domain data
    SMEAR = 1 << 0

class KnowledgeFilterProto(nn.Module, ABC):
    """
    A simple abstract class that defines a standard forward call that isn't normally guaranteed.
    """
    @abstractmethod
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Runs a tensor through the contained knowledge graph.

        Args:
            x (torch.Tensor): The tensor to run through the knowledge graph.

        Returns:
            torch.Tensor: The knowledge graph output.
        """
        pass

class KnowledgeFilter(nn.Module):
    """
    Contains all of the addressing(ish) data needed to locate and infer through the 
    internal knowledge graph.
    """
    def __init__(self, child:KnowledgeFilterProto, childInputSize:torch.Size, childOutMode:KnowledgeOutputMode=KnowledgeOutputMode.LOGITS, \
        needsEncoding:bool=True, encoderDimensions:int=DEFAULT_SPACE_PRIME, encoderFullOutput:bool=True, \
        attemptReshape:bool=True, alignCorners:bool=True, outSmearSamples:int=DEFAULT_FFT_SAMPLES, \
        outputKnotCurves:int=None, dtype:torch.dtype=DEFAULT_DTYPE):
        """Builds a knowledge filter with the specified parameters.

        Args:
            child (KnowledgeFilterProto): The internal knowledge filter to be used.
            childInputSize (torch.Size): The input size of the contained knowledge graph.
            childOutMode (KnowledgeOutputMode, optional): The mode of output used by the contained knowledge graph. Defaults to KnowledgeOutputMode.LOGITS.
            needsEncoding (bool, optional): If encoding is needed for the input signal from the provided continuous, sampled, one. Defaults to True.
            encoderDimensions (int, optional): The amount of curves used by the encoder. Defaults to DEFAULT_SPACE_PRIME.
            encoderFullOutput (bool, optional): Use the full output of the signal encoder to the knowledge graph and not a collapsed one. Defaults to True.
            attemptReshape (bool, optional): Attempt to reshape the output of the knowledge graph to fit the following functions better. Defaults to True.
            alignCorners (bool, optional): If using the full output, align the corners of the signal during remapping to attempt to force more continuous subsignals. Defaults to True.
            outSmearSamples (int, optional): The amount of samples to be smeared out of the output logits if in logit mode. Defaults to DEFAULT_FFT_SAMPLES.
            outputKnotCurves (int, optional): The amount of curves to use in the output knot. Defaults to None (meaning no output knot).
            dtype (torch.dtype, optional): The datatype for the internal datastructures and operations. Defaults to DEFAULT_DTYPE.
        """
        super(KnowledgeFilter, self).__init__()

        # Build a turbulence encoder to figure out what's being looked at
        if needsEncoding:
            self.encoder:Turbulence = Turbulence(internalWaves=encoderDimensions, \
                sameDimOut=encoderFullOutput, dtype=dtype)
        else:
            self.encoder:Turbulence = None
        
        # Copy child interfacing parameters
        self.inputReshape:bool = attemptReshape
        self.alignCorners:bool = alignCorners
        self.childInputSize:torch.Size = childInputSize
        self.childOutMode:KnowledgeOutputMode = childOutMode
        self.child:KnowledgeFilterProto = child

        # Convert the childInputSize variable to the flattened sense of itself
        self.flatChild:int = 1
        for n in self.childInputSize:
            self.flatChild = self.flatChild * n
        
        # Add a knot activation to the last layer of the module
        if outputKnotCurves is None:
            self.outKnot:Knot = None
        else:
            complexType:torch.dtype = toComplex(torch.ones((1), dtype=dtype, requires_grad=False)).dtype
            self.outKnot:Knot = Knot(knotSize=outputKnotCurves, knotDepth=int(outSmearSamples/outputKnotCurves), dtype=complexType)

    def forward(self, x:torch.Tensor, oneD:bool=False, batchDims:int=1) -> torch.Tensor:
        # Extract subportions of the original signal through embedding
        if self.encoder is None:
            xEmbed:torch.Tensor = x
        else:
            xEmbed:torch.Tensor = self.encoder.forward(queries=x, states=x, oneD=oneD).flatten(-2, -1)

        # Attempt a reshaping of the input signal
        if self.inputReshape:
            xEmbed = xEmbed.unsqueeze(-2)
            knowledgeSize = xEmbed.size()[:-2]
            for n in self.childInputSize:
                knowledgeSize.append(n)
            
            knowledgeInputs:torch.Tensor = nnf.interpolate(xEmbed, size=self.flatChild, \
                mode='linear', align_corners=self.alignCorners)
            knowledgeInputs = torch.reshape(knowledgeInputs, shape=knowledgeSize)
        else:
            knowledgeInputs:torch.Tensor = xEmbed

        # Run knowledge through the child
        logits:torch.Tensor = self.child.forward(knowledgeInputs)

        # Convert knowledge to a smear if needed
        if self.childOutMode == KnowledgeOutputMode.LOGITS:
            # Reshape the logits to be smear like
            flatLogits:torch.Tensor = torch.flatten(logits, start_dim=batchDims)

            # Pad the last dimension with 0s
            flatLogitsPadded:torch.Tensor = nnf.pad(flatLogits, pad=(0, self.outSmearSamples-flatLogits.size()), mode='constant', value=0.)
            
            # Interprit each activation as a 
            smear:torch.Tensor = torch.fft.ifft(flatLogitsPadded, dim=-1)
        else:
            smear:torch.Tensor = logits

        # Add a final activation if provided
        if self.outKnot is not None:
            return self.outKnot.forward(smear, oneD=True)
        return smear
