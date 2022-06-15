from ..defaults import *
from .scaffold import ScaffoldFilter
from .routing import *
from ..entanglement import *


class PipelineFilter(KnowledgeCollider):
    def __init__(self, pipes:nn.ModuleList, scaleCoeff:t.Tensor=phi(), keySamples:int=DEFAULT_FFT_SAMPLES, ioSamples:int=DEFAULT_FFT_SAMPLES, cdtype:t.dtype=DEFAULT_COMPLEX_DTYPE):
        super(KnowledgeCollider, self).__init__(keySamples=keySamples, inputSamples=ioSamples, outputSamples=ioSamples, cdtype=cdtype)

        # To handle importing sets of Scaffold Filters, a scalar divisor of Phi is recommended
        # for each layer of filter. The idea behind this is the preservation of presented data.
        # If the prior layer's basis frequencies are regressed as not being resonant with later
        # filter frequencies, all of the data should come out and be recoverable by the main agent.
        self.pipeModules:nn.ModuleList = nn.ModuleList(pipes)
        for module in self.pipeModules:
            assert isinstance(module, KnowledgeCollider)
        
        # Each filter must also collapse its superpositional output due to the standard
        # for KnowledgeFilters. To accomplish this, a list of parameters is needed
        # for each pipeModule defining the collapse polarization.
        self.pipePols:nn.ParameterList = nn.ParameterList()
        for module in self.pipeModules:
            pipeParam = nn.Parameter(toComplex(t.zeros((2), dtype=self.cdtype)).real)
            pipeParam[1].add_(pi())
            self.pipePols.append(pipeParam)
        
        # The parameter to use for the power series dividing the lattice
        self.scaleCoeff:nn.Parameter = nn.Parameter(toComplex(scaleCoeff))

        # Run the input and output signals through this mask in order to figure
        # out where to exit early.
        self.pipeMask:nn.Parameter = nn.Parameter(t.ones((2, ioSamples, ioSamples), 
            dtype=self.scaleCoeff.dtype) * t.eye(ioSamples))
        self.pipePol:nn.Parameter = nn.Parameter(t.zeros((1), dtype=self.pipeMask.real.dtype))
     
    def __forward__(self, a:t.Tensor, b:t.Tensor) -> t.Tensor:
        # Check to make sure the input signals are of appropriate size
        assert a.size(-1) == b.size(-1)
        flata = a.flatten(start_dim=0, end_dim=-2)
        flatb = b.flatten(start_dim=0, end_dim=-2)
        assert flata.size(0) == flatb.size(0)
        
        # End early from the pipeline if empty, providing a raw superposition
        if len(self.pipeModules) == 0:
            return superposition(a, b)
        
        # Find the correlation of signal a onto b and turn it into a diagonal
        corr:t.Tensor = correlation(flata, flatb, dim=-1).unsqueeze(-1) @ t.eye(flata.size(-1))
        
        # Take the correlation diagonal previously calculated and matmul it with the
        # stored knowledge mask.
        target:t.Tensor = (corr @ nsoftunit(self.pipeMask[1], dims=[-1, -2])).conj()

        # Store the output of the modules
        result:t.Tensor = toComplex(t.zeros((flata.size(0), flata.size(-1), flatb.size(-1)), dtype=self.cdtype))
        
        # Compute all of the steps in the pipeline per flattened batch
        # Route each batch optimally through the pipeline
        for batch in range(flata.size(0)):
            # Grab the relevant next tensor by batch
            nexta:t.Tensor = flata[batch]
            nextb:t.Tensor = flatb[batch]
            accum:List[t.Tensor] = []
            lattice:List[t.Tensor] = []
            
            # Pipe each batch individually through the pipeline
            for pidx in range(len(self.pipeModules)):
                for filter in lattice:
                    filter.div_(nnf.softplus(self.scaleCoeff))
                # Push the stepped data into the selected filter
                accum.append(self.pipeModules[pidx].forward(a=nexta, b=nextb))
                lattice.append(t.ones((1), dtype=self.scaleCoeff.dtype))
                
                # Check to see if this is the implicit signal that is being looked for
                exitsuper:t.Tensor = nsoftunit(accum[-1], [-1, -2]) * nsoftunit(target, [-1, -2])
                if not t.all((t.flatten(exitsuper).sum(-1)) < 0.5):
                    break
                
                # Calculate the nexta and nextb values through collapse(). Using
                # the parameters stored in self.pipePols, a trainable, autograddable
                # function should emerge.
                polarization = self.pipePols[pidx]
                nexta = collapse(accum[-1], polarization=polarization[0])
                nextb = collapse(accum[-1], polarization=polarization[1])

            # Accumulate the batch accum (multiplied by the calculated lattice parameters)
            # into the result tensor for the function.
            for idx in range(len(accum)):
                result[batch].add(accum[idx] * lattice[idx])
        
        # Unflatten and return the accumulated result
        return nn.Unflatten(dim=0, unflattened_size=a.size()[:-1])(result)    

    def addPipe(self, pipe:KnowledgeCollider):
        # Add the pipe, simple enough. Do a little error checking while you're at it
        assert isinstance(pipe, KnowledgeCollider)
        self.pipeModules.append(pipe)

        # Create a new respective polarization parameter
        newPol:t.Tensor = t.zeros((2), dtype=self.cdtype)
        self.pipePols.append(nn.Parameter(newPol))

    def delPipe(self, idx:int=-1) -> Tuple[KnowledgeCollider, t.Tensor]:
        filter:KnowledgeCollider = self.pipeModules[idx]
        self.pipeModules = self.pipeModules[:idx].extend(self.pipeModules[idx+1:])
        polarization:t.Tensor = self.pipePols[idx].data
        self.pipePols = self.pipePols[:idx].extend(self.pipePols[idx+1:])

        return (filter, polarization)
