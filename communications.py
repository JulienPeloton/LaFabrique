import os
import sys

verbose = False

if all(os.environ.has_key(k) for k in ['SLURM_SUBMIT_DIR']):
    # job on compute nodes for NERSC
    try:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        barrier = MPI.COMM_WORLD.Barrier
        finalize = MPI.Finalize
        if verbose:
            print 'setup OK, rank %s in %s' % (rank, size)
    except:
        if verbose: sys.stderr.write('pbs.py: unable to setup mpi4py\n')

elif not os.environ.has_key('NERSC_HOST'):
    if verbose:
        print 'Not working at NERSC'
    from mpi4py import MPI
    rank     =  MPI.COMM_WORLD.Get_rank()
    size     =  MPI.COMM_WORLD.Get_size()
    barrier  =  MPI.COMM_WORLD.Barrier
    finalize =  MPI.Finalize
    if verbose:
        print 'setup OK, rank %s in %s' % (rank, size)

else:
    # job on login node
    rank = 0
    size = 1
    barrier = lambda: -1
    finalize = lambda: -1
    if verbose:
        print 'This looks like invocation on login nodes at NERSC'
