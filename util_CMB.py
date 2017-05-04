from inputs import InputScan
from inputs import InputScan_full
import numpy as np
import healpy as hp
import os
from time import time

DEBUG = True

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


def write_output_single(sky_freq, o, Config, i, extra_header):
    path = file_path(o, i)
    hp.write_map(
        path, hp.ud_grade(sky_freq, nside_out=o.nside),
        coord=o.output_coordinate_system,
        column_units=''.join(o.output_units),
        column_names=None, extra_header=extra_header)

def rot_planck_map(data, header=None, coord=['G', 'C']):
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

    if verbose:
        print 'seed I', seed_list_I
        print 'seed Q', seed_list_Q
        print 'seed U', seed_list_U
    return seed_list_I, seed_list_Q, seed_list_U

@benchmark
def load_hdf5_data(fn, nside_out):
    """
    Load hdf5 file

    Parameters
    ----------
       * fn: string, name of the file to read.
       * nside_out: int, the desired resolution for the output map.

    Output
    ----------
        * map1: object, contain the observations at resolution nside_out.

    """
    map1 = InputScan.load(fn)
    if nside_out != map1.mapinfo.nside:
        map1 = InputScan.change_resolution(map1, nside_out)
        print 'NSIDE_OUT', map1.mapinfo.nside
    return map1

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
    elif patch == 'BICEP':
        center = [0., -57.5]
    elif patch == 'center':
        center = [0, 0]
    else:
        center = None
    return center

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
        return np.array([func(i) for i in array.split()])

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
        self.input_observations = config_dict['input_observations']
        self.name = config_dict['name']
        self.outpath_noise = config_dict['outpath_noise']
        self.outpath_masks = config_dict['outpath_masks']

        ## Booleans
        self.plot = self.boolise_it(config_dict, 'plot')
        self.verbose = self.boolise_it(config_dict, 'verbose')
        self.do_foregrounds = self.boolise_it(config_dict, 'do_foregrounds')

        ## Floats
        self.input_sampling_freq = self.floatise_it(
            config_dict['input_sampling_freq'])
        self.input_calendar_time = self.floatise_it(
            config_dict['input_calendar_time'])
        self.input_efficiency = self.floatise_it(
            config_dict['input_efficiency'])
        self.sampling_freq = self.floatise_it(config_dict['sampling_freq'])
        self.calendar_time = self.floatise_it(config_dict['calendar_time'])
        self.efficiency = self.floatise_it(config_dict['efficiency'])

        ## Integers
        self.nside_out = self.intise_it(config_dict['nside_out'])
        self.initial_seed = self.intise_it(config_dict['initial_seed'])
        self.nmc = self.intise_it(config_dict['nmc'])

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
