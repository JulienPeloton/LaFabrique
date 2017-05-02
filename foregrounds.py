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
from pysm import write_output_single

def generate_foregrounds(config_file):
    """
    Adapted from the main of the PySM
    """
    ## Get the output directory and save the configuration file.
    Config = ConfigParser.ConfigParser()
    Config.read(config_file)
    out = util_CMB.normalise_foreground_parser(
        Config._sections['GlobalParameters'])
    if not os.path.exists(out.output_dir):
        os.makedirs(out.output_dir)
    path = os.path.join(out.output_dir, out.output_prefix + 'main_config.ini')
    with open(path, 'w') as configfile:
        Config.write(configfile)

    if out.debug is True:
        ## Print information about the run:
        print '----------------------------------------------------- \n'
        print ''.join("%s: %s \n" % item for item in vars(out).items())
        print '-----------------------------------------------------'

    sky = np.zeros([3, len(out.output_frequency), hp.nside2npix(out.nside)])
    print '----------------------------------------------------- \n'
    ## Create synchrotron, dust, AME,
    ## and cmb maps at output frequencies then add noise.
    if 'synchrotron' in out.components:
        sky = sky + pysm_synchrotron.main(config_file)

    if 'thermaldust' in out.components:
        sky = sky + pysm_thermaldust.main(config_file)

    if 'spinningdust' in out.components:
        sky = sky + pysm_spinningdust.main(config_file)

    if 'freefree' in out.components:
        sky = sky + pysm_freefree.main(config_file)

    if 'cmb' in out.components:
        sky = sky + pysm_cmb.main(config_file)

    if out.smoothing:
        print 'Smoothing output maps.'
        print '----------------------------------------------------- \n'
        for i in xrange(len(out.output_frequency)):
            sky[:, i, :] = hp.smoothing(
                sky[:, i, :],
                fwhm=(np.pi / 180. / 60.) * out.fwhm[i],
                verbose=False)

    if out.instrument_noise is True:
        sky = sky + pysm_noise.instrument_noise(config_file)

    comps = str()

    if out.instrument_noise:
        out.components.append('noise')

    sky = np.swapaxes(sky, 0, 1)
    for i in xrange(len(out.output_frequency)):
        if out.output_coordinate_system:
            sky[i, ...] = util_CMB.rot_planck_map(
                sky[i, ...],
                coord=['G', out.output_coordinate_system])
        extra_header = config2list(Config, out, i)
        util_CMB.write_output_single(
            sky[i, ...],
            out,
            Config,
            i,
            extra_header=extra_header)

    print '-----------------------------------------------------\n'
    print 'PySM completed successfully. \n'
    print '-----------------------------------------------------'
