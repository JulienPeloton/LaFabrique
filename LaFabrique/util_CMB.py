from . import InputScan
from InputScan import InputScan
from InputScan import InputScan_full
import numpy as np
import healpy as hp
import os
from time import time
try:
    from scipy import weave
except:
    ## weave has been removed from scipy version > 0.18
    import weave

try:
    from InsideMe import profiler
except:
    ## InsideMe will be released soon ;-)
    ## Stay tuned for a new great package
    class profiler():
        @staticmethod
        def benchmark(field=''):
            def outer_wrapper(func):
                def inner_wrapper(*args, **kwargs):
                    return func(*args, **kwargs)
                return inner_wrapper
            return outer_wrapper


DEBUG = False

def date_to_greg(date):
    date_ = str(date)
    date_ = str(date.datetime())
    return date_.split('.')[0].replace('-','').replace(':','').replace(' ','_')

def rad2am(rad):
    """
    Conversion factor from radian to arcmin
    """
    return rad * 180. / np.pi * 60.

def year2sec(year):
    """
    Conversion factor from year to second
    """
    return year * 365 * 24 * 3600

def check_coord_syst(h):
    """
    Check the coordinate system in a given header.
    """
    coord_sys = np.where(np.transpose(h)[0] == 'COORDSYS')[0]
    print 'Input coordinate system is ', np.transpose(h)[1][coord_sys]
    return np.transpose(h)[1][coord_sys][0]

def change_coord_sys(h, coord_out='C'):
    """
    Change the coordinate system in a header (data are not changed though)
    """
    coord_sys = np.where(np.transpose(h)[0] == 'COORDSYS')[0][0]
    h[coord_sys] = ('COORDSYS', coord_out)

def file_path(o, j):
    comps = str()
    for k in sorted(o.components):
        comps = ''.join([comps, k[0:5], '_'])
    fname = ''.join([
        o.output_prefix,
        comps,
        str(o.output_frequency[j]).replace('.', 'p'),
        '_',
        str(o.nside),
        '.fits'])
    path = os.path.join(o.output_dir, fname)
    return path

def add_hierarch(lis):
    """
    Convert in correct format for fits header.

    Parameters
    ------------
        * lis: list, contains tuples (keyword, value [, comment])

    """
    for i, item in enumerate(lis):
        if len(item) == 3:
            lis[i]= ('HIERARCH '+item[0],item[1],item[2])
        else:
            lis[i]= ('HIERARCH '+item[0],item[1])
    return lis

def write_output_single(sky_freq, o, Config, i, extra_header):
    path = file_path(o, i)
    hp.write_map(
        path, hp.ud_grade(sky_freq, nside_out=o.nside),
        coord=o.output_coordinate_system,
        column_units=''.join(o.output_units),
        column_names=None, extra_header=extra_header)

def init_seeds(seed, nmc, verbose=False):
    """
    Initialise seeds for noise Monte Carlo.
    From the initial seed, it generates 3 lists (I, Q, U)
    of NMC random integers.

    Parameters
    ----------
        * seed: int, the initial seed
        * nmc: nmc, the number of Monte Carlo

    Output
    ----------
        * seed_list_X: 1D array, list of NMC random integers.
        * header: dic, modified header. Optional (if provided in input).
    """
    ## Initialize seeds
    state_initial = np.random.RandomState(seed)
    seed_list_IQU = state_initial.randint(1, 1e6, size=3)
    seed_I = seed_list_IQU[0]
    seed_Q = seed_list_IQU[1]
    seed_U = seed_list_IQU[2]

    state_I = np.random.RandomState(seed_I)
    state_Q = np.random.RandomState(seed_Q)
    state_U = np.random.RandomState(seed_U)

    seed_list_I = state_I.randint(1, 1e6, size=nmc)
    seed_list_Q = state_Q.randint(1, 1e6, size=nmc)
    seed_list_U = state_U.randint(1, 1e6, size=nmc)

    return seed_list_I, seed_list_Q, seed_list_U

def shrink_full_hdf5(fn):
    map_full = InputScan_full.load(fn)
    mapinfo = map_full.mapinfo

    map_light = InputScan(mapinfo)
    map_light.w = map_full.w
    map_light.cc = map_full.cc
    map_light.ss = map_full.ss
    map_light.cs = map_full.cs
    map_light.nhit = map_full.nhit

    map_light.save('additional_files/SO.hdf5')

def load_center(patch):
    """
    Center of the observations (lon,lat)

    Parameters
    ----------
        * patch: string, name of the patch

    """
    if patch == 'RA23':
        center = [345.5, -32.8]
    elif patch == 'RA12':
        center = [178.3, -0.5]
    elif patch == 'LST':
        center = [70, -45]
    elif patch == 'SO_deep':
        center = [0., -57.5]
    elif patch == 'SO_shallow':
        center = [0., 0.]
    elif patch == 'center':
        center = [0, 0]
    else:
        center = None
    return center

## Benchmarked functions

def benchmark(func):
    """
    Print the seconds that a function takes to execute.
    """
    if DEBUG is True:
        def wrapper(*args, **kwargs):
            t0 = time()
            res = func(*args, **kwargs)
            print("function @{0} took {1:0.3f} seconds".format(
                func.__name__, time() - t0))
            return res
    else:
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
    return wrapper

@profiler.benchmark(field='Map manipulation')
def rot_sky_map(data, header=None, coord=['G', 'C']):
    """
    Rotate a sky map from one coordinate system to another.

    Parameters
    ----------
        * data: 1 or N x 1D-array, input map(s)
        * header: dic, the header of the data (from fits file)
        * coord: list of string, the input and output coordinate systems

    Output
    ----------
        * data: 1 or N x 1D-array, rotated map(s)
        * header: dic, modified header. Optional (if provided in input).

    """
    print 'Rotate coordinate from %s to %s' % (coord[0], coord[1])

    nmap = data.shape[0]
    ndim = data.ndim

    #### List of input pixels
    if ndim > 1:
        nside = hp.npix2nside(len(data[0]))
    else:
        nside = hp.npix2nside(len(data))

    obspix_pb = range(12 * nside**2)

    ### Rotation operator
    r_C2G = hp.Rotator(coord=coord, inv=True)

    ### pix2ang
    theta_pb, phi_pb = hp.pix2ang(nside, obspix_pb)

    ### Rotation
    theta_pk, phi_pk = r_C2G(theta_pb, phi_pb)

    ### ang2pix
    obspix_pk = hp.ang2pix(nside, theta_pk, phi_pk)

    ### Fill the map with new locations
    if ndim > 1:
        for i in range(nmap):
            data[i][obspix_pb] = data[i][obspix_pk]
    else:
        data[obspix_pb] = data[obspix_pk]

    if header is not None:
        change_coord_sys(header, coord[1])
        return data, header
    else:
        return data

@profiler.benchmark(field='Map manipulation')
def ud_grade(m_in, nside_out):
    """
    Wrapper around healpy function
    """
    return hp.ud_grade(m_in, nside_out)

@profiler.benchmark(field='IO')
def write_map(
        path, data, fits_IDL=False, coord='C',
        column_names=[''], column_units=[''], partial=True, extra_header=[]):
    """
    Wrapper around heapy write_map function
    """
    for c in column_names:
        extra_header.append(('column_names', c))
    extra_header = add_hierarch(extra_header)
    hp.write_map(
        path,
        data,
        fits_IDL=fits_IDL,
        coord=coord,
        column_names=None,
        column_units=column_units,
        partial=partial,
        extra_header=extra_header)

@profiler.benchmark(field='IO')
def load_hdf5_data(fn):
    """
    Load hdf5 file

    Parameters
    ----------
       * fn: string, name of the file to read.

    Output
    ----------
        * map1: object, contain the observations at resolution nside_input.

    """
    return InputScan.load(fn)

@profiler.benchmark(field='Map manipulation')
def change_resolution(map1, nside_out):
    """
    Change the resolution of the input map

    Parameters
    ----------
       * map1: object, contain the observations
       * nside_out: int, the desired resolution for the output map.

    Output
    ----------
        * map1: object, contain the observations at resolution nside_out.

    """
    if nside_out != map1.mapinfo.nside:
        map1 = InputScan.change_resolution(map1, nside_out)
        print 'NSIDE_OUT', map1.mapinfo.nside
    return map1

@profiler.benchmark(field='Map manipulation')
def partial2full(partial, obspix, nside, fill_with_nan=True):
    """
    Convert partial map into full sky map

    Parameters
    ----------
        * partial: 1D array, the observed data
        * obspix: 1D array, the label of observed pixels
        * nside: int, nside of the partial data
        * fill_with_nan: boolean, if True it fills with nan unobserved
            pixels (it allows compression when saving on the disk)

    Output
    ----------
        * full: 1D array, full sky map

    """
    if fill_with_nan is True:
        full = np.zeros(12 * nside**2) * np.nan
    else:
        full = np.zeros(12 * nside**2)
    full[obspix] = partial
    return full

@profiler.benchmark(field='Computation')
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
    det[det == 0.0] = 1e-30

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

@profiler.benchmark(field='Computation')
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
    valid3 = [x for x in valid if x == True]

    if verbose:
        print 'criterion is', epsilon, '< det < 1/4 (epsilon= 0. by default)'
        print 'number of pixels kept:', len(valid3), '/', np.sum(tr > 0)
        print 'Percentage cut: %.3f %%' % (
                    (1. - float(len(valid3)) / np.sum(tr > 0)) * 100.)

    weight[valid] = lambda_minus[valid]

    return weight

class normalise_parser(object):
    """
    Generic class to handle parsered file.
    Mostly contain basic static functions.
    """
    def __init__(self, config_dict):
        """
        Parameters
        ----------
            * config_dict: dic, dictionary coming from the ini file.
        """
        self.config_dict = config_dict

    @staticmethod
    def floatise_it(entry):
        return float(entry)

    @staticmethod
    def intise_it(entry):
        return int(entry)

    @staticmethod
    def boolise_it(dic, entry):
        if entry in dic:
            out = 'True' in dic[entry]
        else:
            out = False
        return out

    @staticmethod
    def normalise_array(array, func):
        array_out = np.array([func(i) for i in array.split()])
        if len(array_out) > 0:
            return array_out
        else:
            return None

    @staticmethod
    def make_dic(keys, array, func):
        values = np.array([func(i) for i in array.split()])
        return {i: j for i, j in zip(keys, values)}

class normalise_foreground_parser(normalise_parser):
    """
    Class to handle foreground parser.
    It converts initial dictionary into an object.
    """
    def __init__(self, config_dict):
        """
        Parameters
        ----------
            * config_dict: dic, dictionary coming from the ini file.
        """
        normalise_parser.__init__(self, config_dict)

        ## Names
        self.output_prefix = config_dict['output_prefix']
        self.output_dir = config_dict['output_dir']
        self.output_coordinate_system = config_dict['output_coordinate_system']

        ## Booleans
        self.bandpass = self.boolise_it(config_dict, 'bandpass')
        self.debug = self.boolise_it(config_dict, 'debug')
        self.instrument_noise = self.boolise_it(
            config_dict, 'instrument_noise')
        self.smoothing = self.boolise_it(config_dict, 'smoothing')

        ## Integers
        self.nside = self.intise_it(config_dict['nside'])

        ## Arrays
        self.components = self.normalise_array(
            config_dict['components'],
            lambda x: x)
        self.output_frequency = self.normalise_array(
            config_dict['output_frequency'],
            self.floatise_it)
        self.bandpass_widths = self.normalise_array(
            config_dict['bandpass_widths'],
            self.floatise_it)
        self.instrument_noise_i = self.normalise_array(
            config_dict['instrument_noise_i'],
            self.floatise_it)
        self.instrument_noise_pol = self.normalise_array(
            config_dict['instrument_noise_pol'],
            self.floatise_it)
        self.fwhm = self.normalise_array(
            config_dict['fwhm'],
            self.floatise_it)

        self.output_units = np.array([
            config_dict['output_units'][0],
            config_dict['output_units'][1:]])

        if config_dict['instrument_noise_seed'] == 'None':
            self.instrument_noise_seed = None
        else:
            self.instrument_noise_seed = self.intise_it(
                config_dict['instrument_noise_seed'])

class normalise_instrument_parser(normalise_parser):
    """
    Class to handle instrument parser.
    It converts initial dictionary into an object.
    """
    def __init__(self, config_dict):
        """
        Parameters
        ----------
            * config_dict: dic, dictionary coming from the ini file.
        """
        normalise_parser.__init__(self, config_dict)

        ## Paths and names
        self.lapack = config_dict['lapack']

        ## Booleans
        self.only_covariance = self.boolise_it(config_dict, 'only_covariance')

        ## Floats
        self.sampling_freq = self.floatise_it(config_dict['sampling_freq'])
        self.calendar_time = self.floatise_it(config_dict['calendar_time'])
        self.efficiency = self.floatise_it(config_dict['efficiency'])
        self.epsilon = self.floatise_it(config_dict['epsilon'])

        ## Integers
        self.nside_out = self.intise_it(config_dict['nside_out'])
        self.initial_seed = self.intise_it(config_dict['initial_seed'])
        self.nmc = self.intise_it(config_dict['nmc'])
        self.inversion_method = self.intise_it(config_dict['inversion_method'])

        ## Arrays
        self.frequencies = self.normalise_array(
            config_dict['frequencies'],
            lambda x: x)
        state_initial = np.random.RandomState(self.initial_seed)
        self.seeds = {str(i): j for i, j in zip(
            self.frequencies, state_initial.randint(
                1, 1e6, size=len(self.frequencies)))}
        self.net_per_arrays = self.make_dic(
            self.frequencies,
            config_dict['net_per_arrays'],
            self.floatise_it)

        self.tube_numbers, self.tube_names = np.transpose(
            [i.split('_') for i in config_dict['tubes'].split()])
        self.tubes = {str(i): int(j) for i, j in zip(
            self.tube_names, self.tube_numbers)}

        self.tubes_to_freq = {i: [] for i in self.tubes.keys()}

        for freq in self.frequencies:
            val, key = freq.split('_')
            self.tubes_to_freq[key].append(int(val))

class normalise_scanning_parser(normalise_parser):
    """
    Class to handle scanning parser.
    It converts initial dictionary into an object.
    """
    def __init__(self, config_dict):
        """
        Parameters
        ----------
            * config_dict: dic, dictionary coming from the ini file.
        """
        normalise_parser.__init__(self, config_dict)

        ## Paths and names
        self.name_instrument = config_dict['name_instrument']
        self.starting_date = config_dict['starting_date']

        ## Booleans
        self.full_circle = self.boolise_it(config_dict, 'full_circle')

        ## Floats
        self.sky_speed = self.floatise_it(config_dict['sky_speed'])
        self.sampling_freq = self.floatise_it(config_dict['sampling_freq'])
        self.fp_radius_amin = self.floatise_it(config_dict['fp_radius_amin'])
        self.HWP_frequency = self.floatise_it(config_dict['hwp_frequency'])

        ## Integers
        self.number_of_days = self.intise_it(config_dict['number_of_days'])
        self.length_of_cycle = self.intise_it(config_dict['length_of_cycle'])
        self.nside = self.intise_it(config_dict['nside'])
        self.nbolos = self.intise_it(config_dict['nbolos'])

        ## Arrays
        self.el = self.normalise_array(
            config_dict['el'], self.floatise_it)
        self.az_min = self.normalise_array(
            config_dict['az_min'], self.floatise_it)
        self.az_max = self.normalise_array(
            config_dict['az_max'], self.floatise_it)
        self.begin_LST = self.normalise_array(
            config_dict['begin_lst'], lambda x: x)
        self.end_LST = self.normalise_array(
            config_dict['end_lst'], lambda x: x)

        self.el = self.normalise_array(
            config_dict['el'], self.floatise_it)
        self.dec_min = self.normalise_array(
            config_dict['dec_min'], lambda x: x)
        self.dec_max = self.normalise_array(
            config_dict['dec_max'], lambda x: x)
        self.begin_RA = self.normalise_array(
            config_dict['begin_ra'], lambda x: x)
        self.end_RA = self.normalise_array(
            config_dict['end_ra'], lambda x: x)
        self.orientation = self.normalise_array(
            config_dict['orientation'], lambda x: x)

class normalise_env_parser(normalise_parser):
    """
    Class to handle environment parser.
    It converts initial dictionary into an object.
    """
    def __init__(self, config_dict):
        """
        Parameters
        ----------
            * config_dict: dic, dictionary coming from the ini file.
        """
        normalise_parser.__init__(self, config_dict)

        ## Paths and names
        self.out_name = config_dict['out_name']
        self.out_path = os.path.join(
            config_dict['out_path'], self.out_name)

        ## Booleans
        self.plot = self.boolise_it(config_dict, 'plot')
        self.verbose = self.boolise_it(config_dict, 'verbose')
