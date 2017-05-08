import os
import sys
import argparse
import ConfigParser

import noise
import foregrounds
import util_CMB
import communications as comm

def addargs(parser):
    ''' Parse command line arguments '''
    parser.add_argument(
        '-setup_env', dest='setup_env',
        required=True,
        help='Configuration file for the environment.')
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
        if args.setup_instrument is not None:
            list_of_sims += ' noise '
        if args.setup_foregrounds is not None:
            list_of_sims += ' foregrounds'
        if len(list_of_sims) == 0:
            print 'You need to select at least one ini file!\n'
            print ' * instrument (see setup_instrument.ini)\n'
            print ' * foregrounds (see setup_foregrounds.ini)\n'
            sys.exit()
        else:
            print 'Simulations of', list_of_sims

    Config = ConfigParser.ConfigParser()
    Config.read(args.setup_env)
    environment = util_CMB.normalise_env_parser(
        Config._sections['Environment'])

    ## Create folders if necessary
    if comm.rank == 0:
        if not os.path.exists(environment.out_name):
            os.makedirs(environment.out_name)
        if not os.path.exists(environment.outpath_noise):
            os.makedirs(environment.outpath_noise)
        if not os.path.exists(environment.outpath_masks):
            os.makedirs(environment.outpath_masks)

    return args, environment

if __name__ == '__main__':
    args_param = None
    args, environment = grabargs(args_param)

    ## Generate noise
    if args.setup_instrument is not None:
        noise.generate_noise_sims(args.setup_instrument, comm, environment)
    comm.barrier()

    ## Generate foregrounds
    if args.setup_foregrounds is not None and comm.rank == 0:
        foregrounds.generate_foregrounds(args.setup_foregrounds, environment)
    comm.barrier()
