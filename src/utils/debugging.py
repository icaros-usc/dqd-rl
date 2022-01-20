"""Tools for debugging."""
import resource


def mem_usage():
    """Memory usage of current process.

    Can be used to track down memory leaks.

    Example:
        before = mem_usage()

        ... # Some code that may be leaking memory.

        logger.info("Memory | Before: %s, After: %s", before, mem_usage())
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
