import util_CMB
import healpy as hp
import numpy as np
import os

def generate_covariances(m1, inst):
    """
    Create a weight map using the smaller eigenvalue of the polarization matrix
    The resulting covariances are saved on the disk.

    Parameters
    ----------
    * m1: object, contain the observations
    * inst: object, contain the input parameters from the ini file

    """
    nside = m1.mapinfo.nside
    obspix = m1.mapinfo.obspix

    Pw = util_CMB.partial2full(
        util_CMB.qu_weight_mineig(
            m1.cc, m1.cs, m1.ss, epsilon=0.), obspix, nside)
    Iw = util_CMB.partial2full(m1.w, obspix, nside)

    path = os.path.join(
        inst.outpath_masks,
        'IQU_nside%d_%s_weights_freq%s.fits' % (
            nside, inst.out_name, inst.frequency))

    util_CMB.write_map(
            path,
            [Iw, Pw, Pw],
            fits_IDL=False,
            coord='C',
            column_names=['I_WEIGHT', 'P_WEIGHT', 'PP_WEIGHT'],
            column_units=['uK2_CMB', 'uK2_CMB', 'uK2_CMB'],
            partial=True,
            extra_header=[
                ('name', 'SO weight maps'),
                ('sigma_p [uK.arcmin]', m1.sigma_p)])
