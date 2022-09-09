import os
import gc
import torch as t



def getSystemMemory() -> int:
    """Get the total bytes of system memory available.

    Returns:
        int: The number of bytes of system memory available.
    """
    return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')



def getCudaMemory(cudaIdx:int=0) -> int:
    """Get the total bytes of GPU memory available.

    Returns:
        int: The number of bytes of GPU memory available.
    """
    if t.cuda.is_available():
        # Comes back as (free GPU memory, total GPU memory)
        return t.cuda.mem_get_info(device=cudaIdx)[1]
    return 0



def collect() -> int:
    """Run various garbage collection systems in Python and other libraries (if needed
    and available) while hiding the output from the terminal as this is meant to be
    called potentially quite frequently. No device is needed to be set here,
    as the various other devices referenced by the current Python instance should
    also be garbage collected.

    Returns:
        int: The number of objects collected.
    """
    # Set up the lack of printing
    debug = gc.get_debug()
    gc.set_debug(0)

    # Collection code
    result:int = gc.collect(generation=2)

    # Return the garbage collector to the original state
    gc.set_debug(debug)
    return result
