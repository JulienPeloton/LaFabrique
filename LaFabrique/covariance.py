from . import util_CMB
import healpy as hp
import numpy as np
import os
import glob

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
            m1.cc,
            m1.cs,
            m1.ss,
            epsilon=inst.epsilon,
            verbose=inst.verbose),
        obspix,
        nside)
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
            column_names=['I', 'P', 'P'],
            column_units=['1/uK2_CMB', '1/uK2_CMB', '1/uK2_CMB'],
            partial=True,
            extra_header=[
                ('name', 'SO weight maps'),
                ('sigma_p', m1.sigma_p, 'uK.arcmin')])

def inverse_noise_weighted_coaddition(
    m1,
    inst,
    folder_of_covs=None,
    list_of_covs=None,
    temp_only=False,
    save_on_disk=True):
    """
    Combine covariances into one single one.
    Particularly useful to mimick post-component separation analysis.

    Parameters
    ----------
        * inst: object, contain the input parameters from the ini file
        * folder_of_covs: string, folder on disk containing the covariances
            that you want to combine. The code assumes that the files contain
            either 1 or 3 fields.
        * list_of_covs: list of 1D or 3D arrays, the covariances that you want
            to combine. The code assumes that each element of the list
            has 1 (temp only) or 3 fields (temp + polarisation).

    Output:
    ----------
        * cov_combined: 1D or 3D array, contain the combined covariance(s).

    """
    assert (folder_of_covs is None or list_of_covs is None), 'Either you give \
    a folder where covariance maps are stored, \
    or you give a list of covariance maps, but not both!'

    if temp_only:
        fields = 0
    else:
        fields = [0, 1, 2]

    if folder_of_covs is not None:
        fns = glob.glob(os.path.join(folder_of_covs, '*.fits'))
        for pos, fn in enumerate(fns):
            cov_tmp = hp.read_map(fn, fields)
            if pos == 0:
                cov_combined = cov_tmp
                continue
            cov_combined += cov_tmp

            #### TEST
            # m1.w = cov_combined[m1.mapinfo.obspix]
            # from . import noise
            # center = util_CMB.load_center(m1.mapinfo.source)
            # noise.compute_noiselevel(
            #     m1=m1,
            #     pixel_size=hp.nside2resol(m1.mapinfo.nside) * 180. / np.pi * 60,
            #     center=center,
            #     plot=inst.plot)
            #### END TEST

    elif list_of_covs is not None:
        cov_combined = np.sum(list_of_covs, axis=0)


    if save_on_disk is True:
        path = os.path.join(
            inst.outpath_masks,
            'IQU_nside%d_%s_weights_freq_combined.fits' % (
                inst.nside_out, inst.out_name))

        util_CMB.write_map(
                path,
                cov_combined,
                fits_IDL=False,
                coord='C',
                column_names=['I', 'P', 'P'],
                column_units=['1/uK2_CMB', '1/uK2_CMB', '1/uK2_CMB'],
                partial=True,
                extra_header=[
                    ('name', 'SO combined weight maps')])

    return cov_combined
