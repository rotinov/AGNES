from agnes.runners.single import Single

try:
    from agnes.runners.distributed_mpi import DistributedMPI
except ImportError:
    import warnings
    warnings.warn("DistributedMPI runner is not available due to the lack of mpi4py package. "
                  "You can install it by executing "
                  "'pip install mpi4py' or '"
                  "pip install agnes[distributed]'")

    Distributed = None
