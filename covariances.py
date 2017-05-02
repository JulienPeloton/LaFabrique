import util_CMB
import healpy as hp
import numpy as np
import os

def qu_weight_mineig(cc, cs, ss, epsilon=0., verbose=False):
    '''
    Create a weight map using the smaller eigenvalue of the polarization matrix

    Parameters
    ----------
        * cc: 1D array of float, QQ noise weight
        * cs: 1D array of float, QU noise weight
        * ss: 1D array of float, UU noise weight
        * epsilon: float, threshold for the determinant of the inverse weight.
    It has to be 0 <= epsilon < 1/4. Default is 0.

    Outputs
    ----------
    * weight: 1D array of float, vector of (inverse noise) weight per pixel

    '''
    ## Compute trace and determinant of the inverse covariance matrix
    tr = cc + ss
    tr2 = tr*tr
    det = (cc * ss - cs * cs)

    ## Basic maths
    val2 = tr * tr - 4. * det
    valid = (val2 > 0.0)
    val = np.zeros_like(val2)
    val[valid] = np.sqrt(val2[valid])

    ## Apply criterion to reject bad pixels
    weight = np.zeros_like(tr)
    lambda_minus = (tr - val) / 2.
    valid = (lambda_minus > (tr-np.sqrt(tr2 - 4. * epsilon * tr2)) / 2.)
    valid3 = [x for x in valid if x is True]

    if verbose:
        print 'criterion is', epsilon, '< det < 1/4 (epsilon= 0. by default)'
        print 'number of pixels kept:', len(valid3), '/', np.sum(tr > 0)
        print 'Percentage cut: %.3f %%' % (
                    (1. - float(len(valid3)) / np.sum(tr > 0)) * 100.)

    weight[valid] = lambda_minus[valid]

    return weight

def generate_covariances(m1, out, sigma_p_theo):
    """
    Create a weight map using the smaller eigenvalue of the polarization matrix
    The resulting covariances are saved on the disk.

    Parameters
    ----------
    * m1: object, contain the observations
    * out: object, contain the input parameters from the ini file
    * sigma_p_theo: float, level of noise in map domain [uk.arcmin]

    """
    nside = m1.mapinfo.nside
    obspix = m1.mapinfo.obspix

    Pw = util_CMB.partial2full(
        qu_weight_mineig(m1.cc, m1.cs, m1.ss, epsilon=0.), obspix, nside)
    Iw = util_CMB.partial2full(m1.w, obspix, nside)

    path = os.path.join(
        out.outpath_masks,
        'IQU_nside%d_%s_weights_freq%s.fits'%(nside, out.name, out.frequency))
    hp.write_map(
        path, [Iw, Pw, Pw],
        fits_IDL=False, coord='C',
        column_names=['I_WEIGHT', 'P_WEIGHT', 'PP_WEIGHT'],
        column_units=['uK2_CMB', 'uK2_CMB', 'uK2_CMB'],
        partial=False,
        extra_header=[
            ('name', 'SO weight maps'), ('sigma_p [uK.arcmin]', sigma_p_theo)])
