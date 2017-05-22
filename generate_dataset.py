import os
import sys
import argparse
import ConfigParser

from LaFabrique import scanning_strategy
from LaFabrique import noise
from LaFabrique import util_CMB
from LaFabrique import communication as comm

try:
    from LaFabrique import foreground
except:
    print 'PySM not found - no foreground generation possible...'

def addargs(parser):
    ''' Parse command line arguments '''
    parser.add_argument(
        '-setup_env', dest='setup_env',
        required=True,
        help='Configuration file for the environment.')
    parser.add_argument(
        '-setup_scanning', dest='setup_scanning',
        required=False, default=None,
        help='Configuration file for the scanning strategy.')
    parser.add_argument(
        '-setup_instrument', dest='setup_instrument',
        required=False, default=None,
        help='Configuration file for the instrument.')
    parser.add_argument(
        '-setup_foregrounds', dest='setup_foregrounds',
        required=False, default=None,
        help='Configuration file for the foregrounds (PySM).')

def grabargs(args_param=None):
    ''' Parse command line arguments '''
    parser = argparse.ArgumentParser(
        description='Package to generate simulated CMB datasets.')
    addargs(parser)
    args = parser.parse_args(args_param)

    if comm.rank == 0:
        list_of_sims = ''
        if args.setup_scanning is not None:
            list_of_sims += ' scans '
        if args.setup_instrument is not None:
            list_of_sims += ' noise '
        if args.setup_foregrounds is not None:
            list_of_sims += ' foregrounds'
        if len(list_of_sims) == 0:
            print 'You need to select at least one ini file!\n'
            print ' * scans (see setup_scanning.ini)\n'
            print ' * instrument (see setup_instrument.ini)\n'
            print ' * foregrounds (see setup_foregrounds.ini)\n'
            sys.exit()
        else:
            print 'Simulations of', list_of_sims

    Config = ConfigParser.ConfigParser()
    Config.read(args.setup_env)
    environment = util_CMB.normalise_env_parser(
        Config._sections['Environment'])

    ## Initialise paths
    environment.outpath_noise = os.path.join(
        environment.out_path, 'noise')
    environment.outpath_masks = os.path.join(
        environment.out_path, 'masks')
    environment.outpath_foregrounds = os.path.join(
        environment.out_path, 'foregrounds')

    ## Create folders if necessary
    if comm.rank == 0:
        ## Create root folder
        if not os.path.exists(environment.out_path):
            os.makedirs(environment.out_path)

        ## Create folders for noise and masks
        if not os.path.exists(environment.outpath_noise):
            os.makedirs(environment.outpath_noise)
        if not os.path.exists(environment.outpath_masks):
            os.makedirs(environment.outpath_masks)
        if not os.path.exists(environment.outpath_foregrounds):
            os.makedirs(environment.outpath_foregrounds)

    return args, environment

if __name__ == '__main__':
    args_param = None
    args, environment = grabargs(args_param)

    if args.setup_scanning is not None and comm.rank == 0:
        scanning_strategy.generate_scans(args.setup_scanning, environment)
    comm.barrier()

    ## Generate noise
    if args.setup_instrument is not None:
        noise.generate_noise_sims(args.setup_instrument, comm, environment)
    comm.barrier()

    ## Generate foregrounds
    if args.setup_foregrounds is not None and comm.rank == 0:
        foreground.generate_foregrounds(args.setup_foregrounds, environment)
    comm.barrier()
