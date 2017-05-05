import numpy as np
import healpy as hp
import util_CMB
import pylab as pl
import os
from util_CMB import benchmark
from scipy import weave

def modify_input(m1, out):
    """
    Modify the input observations to match the desired output noise level.

    Parameters
    ----------
        * m1: object, contain the observations
        * out: object, contain the input parameters from the ini file

    Outputs
    ----------
        * m1: oject, modified input map
        * sigma_t_theo: float, output level of noise [uk.sqrt(s)]
        * sigma_p_theo: float, output level of noise [uk.arcmin]

    """
    sigma_t_theo, sigma_p_theo = theoretical_noise_level_time_domain(
        m1=m1,
        pixel_size=hp.nside2resol(m1.mapinfo.nside),
        net_per_array=out.net_per_array, cut=0.0,
        calendar_time_in=out.input_calendar_time,
        calendar_time_out=out.calendar_time,
        efficiency_in=out.input_efficiency,
        efficiency_out=out.efficiency,
        freq_in=out.input_sampling_freq,
        freq_out=out.sampling_freq,
        tube_factor=out.tube_factor,
        verbose=out.verbose)

    prepare_map(m1, sigma_t_theo)

    return m1, sigma_t_theo, sigma_p_theo

@benchmark
def generate_noise_sims(m1, out, center=[0, 0]):
    """
    Noise simulations based on the covariance matrix.
    Noise is white but inhomogeneous (specified by nhit pattern in m1).

    Parameters
    ----------
        * m1: object, contain the observations
        * out: object, contain the input parameters from the ini file
        * center: list of float, center coordinates of the patch. Optional.

    """

    if out.verbose is True:
        compute_noiselevel(
            m1=m1,
            pixel_size=hp.nside2resol(m1.mapinfo.nside) * 180. / np.pi * 60,
            center=center,
            plot=out.plot)

    masktot = (m1.w > 0) * (m1.cc > 0.0) * (m1.ss > 0.0)

    cc, ss, cs = compute_weights_fullmap(m1, out, masktot)

    noise_per_pixel_I = np.sqrt(1.0 / (m1.w[masktot])) * 1e6
    noise_per_pixel_Q = cc * 1e6
    noise_per_pixel_U = ss * 1e6
    noise_per_pixel_QU = cs * 1e6

    seed_list_I, seed_list_Q, seed_list_U = util_CMB.init_seeds(
        out.seed_noise, nmc=out.nmc, verbose=out.verbose)

    npix_I = len(noise_per_pixel_I)
    npix_P = len(noise_per_pixel_Q)

    for i in range(int(out.nmc)):
        state_I_MC = np.random.RandomState(seed_list_I[i])
        state_Q_MC = np.random.RandomState(seed_list_Q[i])
        state_U_MC = np.random.RandomState(seed_list_U[i])

        if out.verbose is True:
            print 'MC #', i
            print 'seed I, Q, U', seed_list_I[i], \
                seed_list_Q[i], seed_list_U[i]

        err_pixel_I = state_I_MC.normal(0, 1, npix_I) * noise_per_pixel_I

        pol_array = [
            state_Q_MC.normal(0, 1, npix_P),
            state_U_MC.normal(0, 1, npix_P)]
        err_pixel_Q = pol_array[0] * noise_per_pixel_Q + \
            pol_array[1] * noise_per_pixel_QU
        err_pixel_U = pol_array[1] * noise_per_pixel_U

        SEEN = m1.mapinfo.obspix[masktot]

        Inoise_sim = util_CMB.partial2full(
            err_pixel_I, SEEN, nside=m1.mapinfo.nside)
        Qnoise_sim = util_CMB.partial2full(
            err_pixel_Q, SEEN, nside=m1.mapinfo.nside)
        Unoise_sim = util_CMB.partial2full(
            err_pixel_U, SEEN, nside=m1.mapinfo.nside)

        path = os.path.join(
            out.outpath_noise,
            'IQU_nside%d_%s_freq%s_white_noise_sim%03d.fits' % (
                m1.mapinfo.nside, out.name, out.frequency, i))
        hp.write_map(
            path,
            [Inoise_sim, Qnoise_sim, Unoise_sim],
            fits_IDL=False,
            coord='C',
            column_names=['I_STOKES', 'Q_STOKES', 'U_STOKES'],
            column_units=['uK_CMB', 'uK_CMB', 'uK_CMB'],
            partial=True,
            extra_header=[
                ('name', 'SO noise maps'),
                ('sigma_p [uK.arcmin]', m1.sigma_p)])

    if out.plot is True:
        min_ = -20
        max_ = 20
        Inoise_sim[Inoise_sim == 0] = np.nan
        Qnoise_sim[Qnoise_sim == 0] = np.nan
        Unoise_sim[Unoise_sim == 0] = np.nan
        hp.gnomview(
            Inoise_sim, rot=center, reso=6.8, xsize=800,
            min=min_, max=max_, title='', notext=True,
            unit='$\mu K$', cmap=pl.cm.jet)
        hp.gnomview(
            Qnoise_sim, rot=center, reso=6.8, xsize=800,
            min=min_, max=max_, title='', notext=True,
            unit='$\mu K$', cmap=pl.cm.jet)
        hp.gnomview(
            Unoise_sim, rot=center, reso=6.8, xsize=800,
            min=min_, max=max_, title='', notext=True,
            unit='$\mu K$', cmap=pl.cm.jet)
        pl.show()

@benchmark
def compute_weights_fullmap(map1, out, masktot):
    """
    Perform Cholesky decomposition of the covariance matrice

    Parameters
    ----------
        * map1: hdf5 file, contain the observations
        * masktot: list of boolean, observed pixels

    Outputs
    ----------
        * cct: 1D array, QQ part of the inverse covariance matrix
        * sst: 1D array, UU part of the inverse covariance matrix
        * cst: 1D array, QU part of the inverse covariance matrix

    """
    npix = len(map1.mapinfo.obspix[masktot])

    ## Inversion per block
    det = map1.cc[masktot] * map1.ss[masktot] - map1.cs[masktot]**2

    a00 = 1. / det * (map1.ss[masktot])
    a01 = 1. / det * (-map1.cs[masktot])
    a10 = 1. / det * (-map1.cs[masktot])
    a11 = 1. / det * (map1.cc[masktot])

    def eigenvalue_approximation(mat):
        """
        Approximation in the sense we are neglecting QU correlations inside a
        pixel. But much faster than any other methods.

        Parameters
        ----------
            * mat: 2x2 array, the inverse covariance matrix for a sky pixel
                (2x2 polarisation block).

        Output
        ----------
            * mat: 2x2 array, inverse coupling matrix.
        """
        # trace = np.sum(np.diag(mat))
        # det = np.linalg.det(mat)
        #
        # eigenvalue_min = trace * (1. - np.sqrt(1. - 4.*det)) / 2.
        # e = np.sqrt(eigenvalue_min)
        e = np.sqrt(mat[0][0])

        return np.array( [[e, 0.], [0., e]] )


    def unsafe_cholesky_C(mat, lapack='"mkl_lapack.h"'):
        """
        Unsafe in the sense we are not checking the positive-definitiveness
        of the matrix. But much faster than the python one.

        Parameters
        ----------
            * mat: 2x2 array, the inverse covariance matrix for a sky pixel
                (2x2 polarisation block). Must be positive-definite.

        Output
        ----------
            * mat: 2x2 array, the cholesky factor
                (lower triangle, /!\ 01 block is not used)
        """
        ## TODO check that the matrix is positive-definite before!
        c_code = r"""
            int INFO=1;
            char U='U';
            int N = 2;
            // DPOTRF( &U, &N, mat, &N, &INFO );
            dpotrf_( &U, &N, mat, &N, &INFO );
            """
        weave.inline(c_code, ['mat'], headers=[lapack])
        return mat

    def safe_cholesky_python(mat):
        """
        Safe in the sense we are not checking the positive-definitiveness
        of the matrix. But much slower than the C one.

        Parameters
        ----------
            * mat: 2x2 array, the inverse covariance matrix for a sky pixel
                (2x2 polarisation block). Must be positive-definite.

        Output
        ----------
            * mat: 2x2 array, the cholesky factor (lower triangle)
        """
        try:
            cho = np.linalg.cholesky(mat)
        except:
            cho = np.zeros_like(mat)
        return cho

    ## Cholesky decomposition of each blocks
    mat_full = [
        np.array(
            [[a00[i], a01[i]], [a10[i], a11[i]]]) for i in range(npix)]

    if out.inversion_method == 2:
        coupling = np.array(
            [unsafe_cholesky_C(mat, out.lapack) for mat in mat_full])
    elif out.inversion_method == 1:
        coupling = np.array([safe_cholesky_python(mat) for mat in mat_full])
    elif out.inversion_method == 0:
        coupling = np.array(
            [eigenvalue_approximation(mat) for mat in mat_full])

    return coupling[:,0,0], coupling[:,1,1], coupling[:,1,0]

def prepare_map(map1, sigma_t_theo):
    """
    Given a nhit pattern and level of noise,
    construct a simple covariance map (AN^{-1}A)
    New noise level in temperature is given by sigma_t_theo,
    and that in polarisation by sigma_t_theo * sqrt(2).

    Parameters
    ----------
        * map1: object, contain the observations
        * sigma_t_theo: float, level of noise in time-domain [uK.sqrt(s)]

    """
    sigma_t_theo /= 1e6  ## muk to K

    ## Temperature
    w = map1.nhit / (sigma_t_theo)**2

    ## Polarisation
    boost = map1.nhit / map1.w / (sigma_t_theo * np.sqrt(2))**2
    cc = map1.cc * boost
    ss = map1.ss * boost
    cs = map1.cs * boost

    ## Update the hdf5 field
    map1.w = w
    map1.cc = cc
    map1.ss = ss
    map1.cs = cs

def compute_noiselevel(m1, pixel_size, center=[0, 0], plot=True):
    '''
    Dumb routine to check the noise level in the sense of Knox formula.
    Noise in time domain -> noise in map level

    Parameters
    ----------
        * m1: object, contain the observations
        * pixel_size: float, size of a map pixel [radian]
        * center: list of float, center coordinates of the patch. Optional.
        * plot: boolean, plot stuff if True. Optional.
    '''
    print '#############'
    print 'CHECK THE MAP'
    mask_nhit = m1.nhit > 0.
    npix = len(m1.nhit[mask_nhit]) ## Number of observed pixels
    print "npix = ", npix

    ## find the average noise in time domain : sigma^2_t = <N>_p
    sigma_t = np.sqrt(
        np.mean(m1.nhit[mask_nhit] / m1.w[mask_nhit])) * 1e6 ## in muK
    print 'sigma_t = ', sigma_t, 'muK'

    ## Noise per pixel, In the sense of Knox formula
    sigma_p2 = 4 * np.pi * \
        sigma_t**2 / float(npix) * \
         (float(npix) / hp.nside2npix(m1.mapinfo.nside))
    print 'sigma_p = ', np.sqrt(sigma_p2), 'muK.arcmin (homogeneous)'

    sigma_p2 = 4 * np.pi * \
        sigma_t**2 / npix**2 * \
        np.sum(np.max(m1.nhit[mask_nhit]) / m1.nhit[mask_nhit]) * \
        npix / hp.nside2npix(m1.mapinfo.nside)
    print 'sigma_p = ', np.sqrt(sigma_p2), 'muK.arcmin (inhomogeneous)'

    if plot is 'True':
        ## Build N ~ sqrt(AA/AN-1A)
        map_ = np.zeros(12*m1.mapinfo.nside**2)
        map_[m1.mapinfo.obspix[mask_nhit]] = np.sqrt(
            m1.nhit[mask_nhit] / m1.cc[mask_nhit])
        hp.gnomview(
            map_, rot=center, reso=6.8, xsize=800,
            max=5e-3, title='', notext=True)
        pl.show()
    print '#############'

def ukam(net_per_array, npix, tobs, pixel_size):
    """
    Compute the level of noise in uk.arcmin
    assuming instrument and observation parameters.

    Parameters
    ----------
        * net_per_array: float or list of, Noise Equivalent Temperature of
             the array [uk.sqrt(s)]. (1702.07467)
        * npix: int, number of observed pixels
        * tobs: float, observing time [year]
        * pixel_size: float, size of a map pixel [radian]

    Outputs
    ----------
        * noise_level: float, level of noise in map domain [uk.arcmin]

    """
    noise_level = np.sqrt(
        (net_per_array)**2 * npix / (tobs)) * pixel_size * (180 / np.pi * 60)
    return noise_level

def theoretical_noise_level_time_domain(
    m1, pixel_size, net_per_array=331.,
    cut=0.0, calendar_time_in=12./365, calendar_time_out=2.,
    efficiency_in=1./6., efficiency_out=0.2,
    freq_in=15., freq_out=30., tube_factor=1, verbose=False):
    '''
    Given some instrument configuration, compute the noise level in time-domain
    This noise level corresponds to noise for temperature. Noise level in
    polarisation is sqrt(2) higher.

    Parameters
    ----------
        * m1: object, file containing observations
        * pixel_size: float, size of a map pixel [radian]
        * net_per_array: float or list of, Noise Equivalent Temperature of
             the array [uk.sqrt(s)]. (1702.07467)
        * cut: float, Remove pixel according to the best observed one.
        * calendar_time_in: float, total input time of observation [year]
        * calendar_time_out: float, total time of observation desired [year]
        * efficiency_in: float, input efficiency of observation
        * efficiency_out: float, output efficiency of observation
        * freq_in: float, sampling frequency used in input observation [Hz]
        * freq_out: float, sampling frequency used for output observation [Hz]
        * tube_factor: int, number of optics tube of the instrument
        * verbose: boolean, print-out million of things if True

    '''
    ## Rough estimate of time spent per map pixel
    mask = m1.cc > cut * np.max(m1.cc)
    Npix = len(m1.nhit[mask])

    ## Correlations between tubes (See SO all hands telecon - v2)
    NET_tmp = 1e9
    net_per_array = 1./np.sqrt(
        tube_factor * net_per_array**-2. + NET_tmp**-2.)

    ## Level of noise in the map (rough estimate)
    sigma_p_in = ukam(net_per_array, Npix, util_CMB.year2sec(
        calendar_time_in * efficiency_in), pixel_size)

    sigma_p_out = ukam(net_per_array, Npix, util_CMB.year2sec(
        calendar_time_out * efficiency_out), pixel_size)

    ## RMS per pixel
    boost = (calendar_time_out * efficiency_out) / \
        (calendar_time_in * efficiency_in)

    m1.nhit = m1.nhit * boost
    sigma_t_out = sigma_p_out * np.sqrt(
        hp.nside2npix(m1.mapinfo.nside) / 4. / np.pi)

    if verbose:
        print 'Boost factor (out/bin) = ', boost
        print 'sigma_p (out) = ', sigma_p_out, 'muK.arcmin'
        print 'sigma_t (out) = ', sigma_t_out, 'muK'

    m1.sigma_p = sigma_p_out
    m1.sigma_t = sigma_t_out

    return sigma_t_out, sigma_p_out
