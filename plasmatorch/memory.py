import os
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
