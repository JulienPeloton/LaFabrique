import os
import copy
import argparse
import ConfigParser

import covariances
import noise
import foregrounds
import util_CMB
import communications as comm

def addargs(parser):
    ''' Parse command line arguments '''
    parser.add_argument(
        '-setup_instrument', dest='setup_instrument',
        required=True,
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

    Config = ConfigParser.ConfigParser()
    Config.read(args.setup_instrument)
    out_instrument = util_CMB.normalise_instrument_parser(
        Config._sections['InstrumentParameters'])

    if out_instrument.verbose and comm.rank == 0:
        print '############ CONFIG ############'
        print 'Frequency channels (GHz) ------:', out_instrument.frequencies
        print 'Noise per array [uK.sqrt(s)] --:', out_instrument.net_per_arrays
        print 'Focal plane tubes -------------:', out_instrument.tubes
        print 'Number of year of obs ---------:', out_instrument.calendar_time
        print 'Efficiency --------------------:', out_instrument.efficiency
        print 'Output resoution --------------:', out_instrument.nside_out
        print '################################'

    ## Create folders if necessary
    if comm.rank == 0:
        if not os.path.exists(out_instrument.name):
            os.makedirs(out_instrument.name)
        if not os.path.exists(out_instrument.outpath_noise):
            os.makedirs(out_instrument.outpath_noise)
        if not os.path.exists(out_instrument.outpath_masks):
            os.makedirs(out_instrument.outpath_masks)

    ## Save ini file for later comparison
    if comm.rank == 0:
        path = os.path.join(out_instrument.name, 'setup_instrument.ini')
        with open(path, 'w') as configfile:
            Config.write(configfile)

    return args, out_instrument

if __name__ == '__main__':
    args_param = None
    args, instrument = grabargs(args_param)

    ## Load input observations
    ## TODO to be replaced by a call to the scan strategy module
    m1_input = util_CMB.load_hdf5_data(instrument.input_observations)
    m1_input = util_CMB.change_resolution(m1_input, instrument.nside_out)

    center = util_CMB.load_center(m1_input.mapinfo.source)

    for freq in instrument.frequencies:
        if instrument.verbose:
            print 'Processing ', freq, ' GHz'
        instrument.frequency = freq
        instrument.net_per_array = instrument.net_per_arrays[freq]
        instrument.tube_factor = instrument.tubes[freq.split('_')[1]]
        instrument.seed_noise = instrument.seeds[freq]

        ## Modify input maps
        m1_output = copy.copy(m1_input)
        m1_output, sigma_t_theo, sigma_p_theo = noise.modify_input(
            m1_output, instrument)

        ## Generate covariances
        if comm.rank == 0:
            covariances.generate_covariances(m1_output, instrument)
        comm.barrier()

        ## Generate noise simulations
        noise.generate_noise_sims(
            m1_output, instrument, center=center, comm=comm)

    comm.barrier()

    ## Generate foregrounds
    if args.setup_foregrounds is True and comm.rank == 0:
        foregrounds.generate_foregrounds(args.setup_foregrounds)
    comm.barrier()
