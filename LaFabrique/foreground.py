import healpy as hp
import numpy as np
import argparse
import ConfigParser
import os

import pysm_synchrotron
import pysm_thermaldust
import pysm_cmb
import pysm_spinningdust
import pysm_noise
import pysm_freefree

import util_CMB

from pysm import config2list

def generate_foregrounds(config_file, env=None):
    """
    Adapted from the main of the PySM
    """
    ## TODO: transfer things from instrument to foregrounds!!

    ## Get the default parameters
    Config = ConfigParser.ConfigParser()
    Config.read(config_file)
    foregrounds = util_CMB.normalise_foreground_parser(
        Config._sections['GlobalParameters'])

    ## Share few parameters between objects
    if env is not None:
        ## NOTE: the trailing '/' is required.
        foregrounds.output_dir = env.outpath_foregrounds + '/'
        foregrounds.output_prefix = env.out_name + '_'
        Config.set(
            'GlobalParameters', 'output_dir', foregrounds.output_dir)
        Config.set(
            'GlobalParameters', 'output_prefix', foregrounds.output_prefix)

    path = os.path.join(
        foregrounds.output_dir,
        foregrounds.output_prefix + 'main_config.ini')

    ## Save the modified ini file, and reload it
    with open(path, 'w') as configfile:
        Config.write(configfile)

    ## Re-open the file with modified options if necessary
    config_file = path
    Config.read(config_file)
    foregrounds = util_CMB.normalise_foreground_parser(
        Config._sections['GlobalParameters'])

    if foregrounds.debug is True:
        ## Print information about the run:
        print '----------------------------------------------------- \n'
        print ''.join("%s: %s \n" % item for item in vars(foregrounds).items())
        print '-----------------------------------------------------'

    sky = np.zeros(
        [3, len(foregrounds.output_frequency), hp.nside2npix(
            foregrounds.nside)])
    print '----------------------------------------------------- \n'
    ## Create synchrotron, dust, AME,
    ## and cmb maps at output frequencies then add noise.
    if 'synchrotron' in foregrounds.components:
        sky = sky + pysm_synchrotron.main(config_file)

    if 'thermaldust' in foregrounds.components:
        sky = sky + pysm_thermaldust.main(config_file)

    if 'spinningdust' in foregrounds.components:
        sky = sky + pysm_spinningdust.main(config_file)

    if 'freefree' in foregrounds.components:
        sky = sky + pysm_freefree.main(config_file)

    if 'cmb' in foregrounds.components:
        sky = sky + pysm_cmb.main(config_file)

    if foregrounds.smoothing:
        print 'Smoothing output maps.'
        print '----------------------------------------------------- \n'
        for i in xrange(len(foregrounds.output_frequency)):
            sky[:, i, :] = hp.smoothing(
                sky[:, i, :],
                fwhm=(np.pi / 180. / 60.) * foregrounds.fwhm[i],
                verbose=False)

    if foregrounds.instrument_noise is True:
        sky = sky + pysm_noise.instrument_noise(config_file)

    comps = str()

    if foregrounds.instrument_noise:
        foregrounds.components.append('noise')

    sky = np.swapaxes(sky, 0, 1)
    for i in xrange(len(foregrounds.output_frequency)):
        if foregrounds.output_coordinate_system:
            sky[i, ...] = util_CMB.rot_sky_map(
                sky[i, ...],
                coord=['G', foregrounds.output_coordinate_system])
        extra_header = config2list(Config, foregrounds, i)
        util_CMB.write_output_single(
            sky[i, ...],
            foregrounds,
            Config,
            i,
            extra_header=extra_header)

    print '-----------------------------------------------------\n'
    print 'PySM completed successfully. \n'
    print '-----------------------------------------------------'
