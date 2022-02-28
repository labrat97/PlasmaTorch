from argparse import ArgumentError
import subprocess as sp
from io import TextIOBase, SEEK_SET, SEEK_CUR, SEEK_END

import cid

DEFAULT_IPFS_BUFFER_SIZE:int = int(2e20)
DEFAULT_IPFS_COMMAND:str = 'ipfs'
SUCCESS:int = 0
FAILURE:int = 1

class IPFile(TextIOBase):
    def __init__(self, multihash:str, ipns:bool=False, command:str=DEFAULT_IPFS_COMMAND):
        super(IPFile, self).__init__()

        # Make sure the provided cid is valid
        assert cid.is_cid(multihash)
        self.multihash:str = multihash

        # Basic running parameters
        self.command:str = command
        self.ipns:bool = ipns
        self.ippath:str = f"/{'ipns' if self.ipns else 'ipfs'}/{self.multihash}"
        self.curpos:int = 0
        self.eof:bool = False
        self.seekbuf:int = DEFAULT_IPFS_BUFFER_SIZE

        # Hold the process that runs the shell command
        self.lastproc:sp.Popen = None

    def __waitForLast(self):
        # Wait for the self contianed process to end
        if self.lastproc is None: return
        assert self.lastproc.wait() == SUCCESS
        self.lastproc = None

    def read(self, size:int=-1) -> bytes:
        # Quick return, at end of the file
        if self.eof: return None

        # Wait for the last process call to finish if it was non-blocking
        self.__waitForLast()

        # Start the process to pull the data from ipfs, then pull the data into python
        args = [self.command, 'cat', '-o', str(self.curpos)]
        if size > 0:
            args.extend(['-i', str(size)])
        args.append(self.ippath)
        self.lastproc = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE)
        stdout, stderr = self.lastproc.communicate()
        outlen = len(stdout)
        self.lastproc = None

        # Make sure an error didn't happen
        if len(stderr) > 0:
            raise IOError(stderr.decode())

        # Increment cursor
        self.curpos += outlen
        if outlen < size and size > 0:
            self.eof = True

        return stdout
    
    def readline(self, size:int=-1) -> bytes:
        # Quick return, at the end of the file
        if self.eof: return None

        # Wait for the last process call to finish if it was non-blocking
        self.__waitForLast()

        # Some running data
        aggregator = []
        searching:bool = True
        newlineIdx:int = -1

        # Filter the size argument for the method
        if size < 0 or size > self.seekbuf:
            batchsize:int = self.seekbuf
        else:
            batchsize:int = size

        # Scan the read for newlines, ending early if a newline is found
        while searching:
            buffer:bytes = self.read(batchsize)
            aggregator.append(buffer)
            newlineIdx = buffer.find(b'\n')
            searching = (newlineIdx < 0) and not self.eof

        # Trim off the unneeded tail of the buffer
        if newlineIdx > 0:
            self.curpos = (self.curpos - len(aggregator[-1])) + newlineIdx
            aggregator[-1] = aggregator[-1][:newlineIdx]
        elif newlineIdx == 0:
            aggregator = aggregator[:-1]
        
        return b''.join(aggregator)

    def tell(self) -> int:
        if self.eof: return None
        return self.curpos
    
    def seek(self, offset:int, whence:int=SEEK_CUR):
        if whence == SEEK_CUR:
            return self.curpos
        elif whence == SEEK_END:
            self.eof = True
            self.curpos = -1
        elif whence == SEEK_SET:
            if self.curpos == -1 or (self.curpos != -1 and offset < self.curpos):
                self.eof = False
                self.curpos = offset
        else:
            raise ArgumentError(whence, f'Unknown literal provided: \"{whence}\"')
        
        return self.curpos
