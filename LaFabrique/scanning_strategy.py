#!/usr/bin/python
###################################
# Author: Julien Peloton, j.peloton@sussex.ac.uk
# Pointing module: thanks Neil Goeckner-Wald!
###################################
import ConfigParser
import sys
import ephem
import copy
import os

import numpy as np
import healpy as hp
from . import util_CMB
from . import InputScan

from util_CMB import benchmark
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

# ## numerical constants
radToDeg = 180 / np.pi
sidDayToSec = 86164.0905

def generate_scans(config_file, env=None):
    """
    Main script to generate the scanning strategy.

    Parameters
    ----------
        * config_file: ini file, parser containing scanning parameters
            (see the setup_scanning.ini provided)
        * env: object, contain paths and debugging tools [Optional]
    """
    if env is None:
        env = lambda: -1
        env.verbose = False
        env.plot = False
        env.out_name = 'outputs'
        env.out_path = os.path.join('./', env.out_name)
        env.outpath_noise = os.path.join('./', env.out_name, 'noise')
        env.outpath_masks = os.path.join('./', env.out_name, 'masks')

    ## Load the parameters from the parser
    Config = ConfigParser.ConfigParser()
    Config.read(config_file)
    scanning = util_CMB.normalise_scanning_parser(
        Config._sections['ScanningParameters'])
    checks_param(scanning)

    ## Save ini file for later comparison
    path = os.path.join(env.outpath_masks, 'setup_scanning.ini')
    with open(path, 'w') as configfile:
        Config.write(configfile)

    ## Load experiment technical details
    experience = instrument(scanning.name_instrument)

    ## Initialise hdf5 file
    scan_info = InputScan.HealpixMapInfo(
        npix=hp.nside2npix(scanning.nside),
        obspix=np.zeros(hp.nside2npix(scanning.nside)),
        nside=scanning.nside,
        source=scanning.name_instrument)
    scan = InputScan.InputScan(scan_info)

    ## Loop over CESes and generate scans
    for n in range(scanning.length_of_cycle):
        ## Initialise the starting date of observation
        ## It will be updated then automatically for the run
        ## Do not run CES longer than 24h though!
        experience.inst.date = scanning.starting_date
        experience.inst.date += n * 24 * ephem.second * 3600
        position_in_the_cycle = n % scanning.length_of_cycle

        ## Grab the details of this CES only
        myargs = copy.copy(scanning)
        for myarg in myargs.__dict__.keys():
            if type(myargs.__dict__[myarg]) == np.ndarray:
                myargs.__dict__[myarg] = \
                    myargs.__dict__[myarg][position_in_the_cycle]

        ## Create the scan strategy
        create_scan_strategy(myargs, experience, scan)

    ## Convolve with the focal plane and extrapolate
    ## to obtain the full dataset (total number of days)
    scan.nhit, scan.cc, scan.ss, scan.cs = convolve_focalplane(
        scan.nhit, scan.cc, scan.ss, scan.cs,
        nbolos=myargs.nbolos,
        fp_radius_amin=myargs.fp_radius_amin,
        boost=myargs.number_of_days/float(myargs.length_of_cycle))

    if env.plot is True:
        import pylab as pl
        hp.mollview(scan.nhit, sub=221)
        hp.mollview(scan.cc, sub=222)
        hp.mollview(scan.ss, sub=223)
        hp.mollview(scan.cs, sub=224)
        pl.show()

    scan.mapinfo.obspix = np.where(scan.nhit != 0)[0]
    scan.mapinfo.npix = len(scan.mapinfo.obspix)
    scan.nhit = scan.nhit[scan.mapinfo.obspix]
    scan.w = np.array(scan.nhit, dtype=int)
    scan.cc = scan.cc[scan.mapinfo.obspix]
    scan.ss = scan.ss[scan.mapinfo.obspix]
    scan.cs = scan.cs[scan.mapinfo.obspix]
    scan.mapinfo.source = scanning.name_instrument

    scan.save(os.path.join(env.outpath_masks,env.out_name+'.hdf5'))

class instrument(object):
    """ Class to handle different instruments and their specificities """
    def __init__(self, instrument='so_deep'):
        """ Update here with new informations from TWGs. """

        self.instrument_ok = ['so_deep', 'so_shallow', 'polarbear', 'actpol']

        if instrument.lower() in self.instrument_ok:
            self.inst = self.chili_atacama()
        else:
            print 'Instrument not yet implemented\n'
            print 'Choose among ', self.instrument_ok
            sys.exit()

    def chili_atacama(self):
        """
        Put here instrument constants
        """
        ## Polarbear observatory location
        pb = ephem.Observer()
        pb.long = '-67:46.816'
        pb.lat = '-22:56.396'
        pb.elevation = 5200

        return pb

@profiler.benchmark(field='Convolve focal plane')
def convolve_focalplane(
        bore_nHits, bore_cos, bore_sin, bore_cs,
        nbolos, fp_radius_amin, boost):
    """
    Given a nHits and bore_cos and bore_sin map,
    perform the focal plane convolution.
    Original author: Neil Goeckner-Wald.
    Modifications by Julien Peloton.

    Parameters
    ----------
        * bore_nHits: 1D array, number of hits for the reference detector
        * bore_cos: 1D array, cumulative cos**2 for the reference detector
        * bore_sin: 1D array, cumulative sin**2 for the reference detector
        * bore_cs: 1D array, cumulative cos*sin for the reference detector
        * nbolos: int, total number of bolometers desired
        * fp_radius_amin: float, radius of the focal plane in arcmin
        * boost: float, boost factor to artificially
            increase the number of hits.
            It doesn't change the shape of the survey (just the amplitude)

    Outputs
    ----------
        * focalplane_nHits: 1D array, number of hits
            for the all the detectors
        * focalplane_cos: 1D array, cumulative cos**2
            for the all the detectors
        * focalplane_sin: 1D array, cumulative sin**2
            for the all the detectors
        * focalplane_cossin: 1D array, cumulative cos*sin
            for the all the detectors

    """
    #Now we want to make the focalplane maps
    focalplane_nHits = np.zeros(bore_nHits.shape)
    focalplane_sin = np.zeros(bore_sin.shape)
    focalplane_cos = np.zeros(bore_cos.shape)
    focalplane_cossin = np.zeros(bore_cs.shape)

    #Resolution of our healpix map
    nside = hp.npix2nside(focalplane_nHits.shape[0])
    resol_amin = hp.nside2resol(nside, arcmin=True)
    fp_rad_bins = int(fp_radius_amin * 2. / resol_amin)
    fp_diam_bins = (fp_rad_bins * 2) + 1

    #Build the focal plane model and a list of offsets
    (x_fp, y_fp) = np.array(
        np.unravel_index(
            range(0, fp_diam_bins**2),
            (fp_diam_bins, fp_diam_bins))).reshape(
                2, fp_diam_bins, fp_diam_bins) - (fp_rad_bins)
    fp_map = ((x_fp**2 + y_fp**2) < (fp_rad_bins)**2)

    bolo_per_pix = nbolos / float(np.sum(fp_map))

    dRA = np.ndarray.flatten(
        (x_fp[fp_map].astype(float) * fp_radius_amin) / (
            fp_rad_bins * 60. * (180. / (np.pi))))
    dDec = np.ndarray.flatten(
        (y_fp[fp_map].astype(float) * fp_radius_amin) / (
            fp_rad_bins * 60. * (180. / (np.pi))))

    pixels_global = np.array(np.where(bore_nHits != 0)[0], dtype=int)
    for n in pixels_global:
        n = int(n)

        #compute pointing offsets
        (theta_bore, phi_bore) = hp.pix2ang(nside, n)
        phi = phi_bore + dRA * np.sin(theta_bore)
        theta = theta_bore + dDec

        pixels = hp.ang2pix(nside, theta, phi)

        ## Necessary because the values in pixels aren't necessarily unique!
        ## This is a poor design choice and should probably be fixed
        c_code = r"""
        int i, pix;
        for (i=0;i<npix_loc;i++)
        {
            pix = pixels[i];
            focalplane_nHits[pix] += bore_nHits[n] * bolo_per_pix * boost;
            focalplane_cos[pix] += bore_cos[n] * bolo_per_pix * boost;
            focalplane_sin[pix] += bore_sin[n] * bolo_per_pix * boost;
            focalplane_cossin[pix] += bore_cs[n] * bolo_per_pix * boost;
        }
        """

        npix_loc = len(pixels)
        weave.inline(c_code, [
            'bore_nHits', 'bore_cos', 'bore_sin', 'bore_cs',
            'focalplane_nHits', 'focalplane_cos',
            'focalplane_sin', 'focalplane_cossin',
            'pixels', 'bolo_per_pix', 'npix_loc', 'n', 'boost'])

    return focalplane_nHits, focalplane_cos, focalplane_sin, focalplane_cossin

@profiler.benchmark(field='Generate pointing')
def compute_RA_Dec_PA(
        experience, num_pts,
        pb_az, upper_az, lower_az, pb_az_dir, az_speed,
        el, sampling_freq, HWP_frequency):
    """
    Compute RA, Dec and parallactic angle for the reference detector

    Parameters
    ----------
        * experience: object, contain experience specs
        * num_pts: int, number of time samples
        * pb_az: float, starting azimuth [deg]
        * upper_az: float, maximum azimuth for the survey [deg]
        * lower_az: float, minimum azimuth for the survey [deg]
        * pb_az_dir: int, direction of scan - east or west (+1 or -1). It is
            updated automatically when the scan reaches the boundaries
        * az_speed: float, velocity of scan projected on sky
            (i.e. constant az velocity of the telescope divided [deg/s]
            by the current elevation)
        * el: float, elevation of the scan [deg]
        * sampling_freq: float, sampling frequency at
            which we record the data [Hz]
        * HWP_frequency: float, fudge factor to randomize
            a bit the phase (to be fixed!)
    Outputs
    ----------
        * pb_ra_array: 1D array, RA for all time samples of the scan [rad]
        * pb_dec_array: 1D array, Dec for all time samples of the scan [rad]
        * parallactic_angle: 1D array, pointing for
            all time samples of the scan [rad]
    """
    ## Initialize arrays
    pb_ra_array = np.zeros(num_pts)
    pb_dec_array = np.zeros(num_pts)
    parallactic_angle = np.zeros(num_pts)

    ## Hum hum...
    if HWP_frequency != 0.:
        HWP_pos = np.array(
            [i % (sampling_freq / HWP_frequency) * 180. /
                sampling_freq * HWP_frequency for i in range(num_pts)])
    else:
        HWP_pos = np.zeros(num_pts)

    for t in range(0, num_pts):
        ## Compute Ra, Dec and convert to degrees
        pb_ra_array[t], pb_dec_array[t] = experience.inst.radec_of(
            pb_az / radToDeg, el / radToDeg)

        if t == 0:
            pb_az += az_speed * pb_az_dir / sampling_freq
            experience.inst.date += ephem.second / sampling_freq
            continue

        delta_ra = pb_ra_array[t] - pb_ra_array[t-1]
        delta_dec = pb_dec_array[t] - pb_dec_array[t-1]

        ## Fix coordinate singularity issues
        if(delta_ra < -np.pi):
            delta_ra += 2.*np.pi
        if(delta_ra > np.pi):
            delta_ra -= 2.*np.pi
        ## Put in parallactic angle things
        parallactic_angle[t] = np.arctan(
            (delta_dec) / (delta_ra * np.cos(pb_dec_array[t])))

        ## Add HWP-like effect
        parallactic_angle[t] += 4 * HWP_pos[t]

        ## Case to change the direction of the scan
        if(pb_az > upper_az):
            pb_az_dir = -1.
        elif(pb_az < lower_az):
            pb_az_dir = 1.

        ## Increment the azimuth
        pb_az += az_speed * pb_az_dir / sampling_freq

        ## Increment the time by one second / sampling rate
        experience.inst.date += ephem.second / sampling_freq

    return pb_ra_array, pb_dec_array, parallactic_angle

@profiler.benchmark(field='Mapmaking')
def project_data_on_sky(nside, pb_ra_array, pb_dec_array, parallactic_angle):
    """
    Simple map-making: project time ordered data into sky maps.

    Parameters
    ----------
        * nside: int, resolution of the output map
        * pb_ra_array: 1D array, RA of the scan
        * pb_dec_array: 1D array, Dec of the scan
        * parallactic_angle: 1D array, pointing of the scan

    Outputs
    ----------
        * nhit_loc: 1D array, sky map with cumulative hit counts
        * cos2_alpha: 1D array, sky map with cumulative cos**2
        * sin2_alpha: 1D array, sky map with cumulative sin**2
        * cossin_alpha: 1D array, sky map with cumulative cos*sin
    """
    num_pts = len(pb_dec_array)
    pix_global = hp.pixelfunc.ang2pix(
        nside, (np.pi/2.) - pb_dec_array, pb_ra_array)

    c_code = r"""
    int i, pix;
    double c, s;
    for (i=0;i<num_pts;i++)
    {
        pix = pix_global[i];

        // Number of hits per pixel
        nhit_loc[pix] += 1;

        // Polarised part of the scan (cosine and sine)
        c = cos(2 * parallactic_angle[i]);
        s = sin(2 * parallactic_angle[i]);
        cos2_alpha[pix] += c * c;
        sin2_alpha[pix] += s * s;
        cossin_alpha[pix] += c * s;
    }
    """

    #Boresight pointing healpix maps
    npix = hp.pixelfunc.nside2npix(nside)
    nhit_loc = np.zeros(npix)
    cos2_alpha = np.zeros(npix)
    sin2_alpha = np.zeros(npix)
    cossin_alpha = np.zeros(npix)
    weave.inline(c_code, [
        'pix_global',
        'num_pts',
        'nhit_loc',
        'parallactic_angle',
        'cos2_alpha',
        'sin2_alpha',
        'cossin_alpha'])

    return nhit_loc, cos2_alpha, sin2_alpha, cossin_alpha

def create_scan_strategy(args, experience, scan_file):
    """
    Process the args to create the scanning strategy.
    This is based originally on Neil's module.

    Parameters
    ----------
        * args: args object, contain the parameters of the scan.
        * experience: experiment object, contain the parameters
            of the experiment.
        * scan_file: InputScan object, contain the output sky
            maps to be updated.
    """
    ## Check if we have enough information to make a scan
    if not (
            (args.begin_LST and args.end_LST) or
            (args.begin_RA and args.end_RA)):
        print 'You must specify the beginning\n'
        print 'and ending of the scan in RA or in LST'
        exit(1)

    if (
            int(args.begin_LST is not None and args.end_LST is not None) + \
            int(args.begin_RA is not None and args.end_RA is not None) > 1):
        print 'You cannot specify the beginning \n'
        print 'and ending of the scan in RA and in LST'
        exit(1)

    if not (
            args.full_circle or \
            (args.az_min and args.az_max) or \
            (args.dec_min and args.dec_max)):
        print 'You must specify the range of the \n'
        print 'scan in either declination or azimuth'
        exit(1)

    if (
            int(args.full_circle) + \
            int(args.az_min is not None and args.az_max is not None) + \
            int(args.dec_min is not None and args.dec_max is not None) > 1):
        print 'You have overspecified the parameters of the scan in azimuth'
        exit(1)

    ## Figure out the elevation to run the scan
    az_mean = 0.
    az_throw = 1.
    el = args.el

    ## Define the sampling rate in Hz
    sampling_freq = args.sampling_freq

    #########################################################
    ## Define geometry of the scan
    #########################################################
    ## Figure out the azimuth bounds if provided
    if (args.az_min and args.az_max):
        az_mean = (args.az_min + args.az_max)*0.5
        az_throw = (args.az_max - args.az_min) / np.cos(el / radToDeg)

    if (args.full_circle):
        az_mean = 180.
        az_throw = 359.99

    ## If given bounds in declination, make bounds in azimuth
    ## note there is no sanity checking here!
    if(args.dec_min and args.dec_max):
        az_array = np.linspace(0., 180., endpoint=True, num=360.)
        ra_array = np.zeros(az_array.shape)
        dec_array = np.zeros(az_array.shape)
        dec_min = ephem.degrees(args.dec_min)
        dec_max = ephem.degrees(args.dec_max)
        if(args.orientation == 'west'):
            az_array += 180.

        for i in range(0, az_array.shape[0]):
            ra_array[i], dec_array[i] = \
                experience.inst.radec_of(az_array[i] / radToDeg, el / radToDeg)

        az_allowed = np.asarray(
            [az_array[i] for i in range(0, az_array.shape[0])
                if (dec_array[i] > dec_min and dec_array[i] < dec_max)])

        if (az_allowed.shape[0] < 2):
            print('Invalid combination of declination bounds and elevation.')
            exit(1)

        az_max = np.max(az_allowed)
        az_min = np.min(az_allowed)
        az_mean = (az_min + az_max) * 0.5
        az_throw = (az_max - az_min)

    #########################################################
    ## Define the timing bounds!
    #########################################################
    num_pts = 0

    if (args.begin_LST and args.end_LST):
        lst_now = float(experience.inst.sidereal_time()) / (2 * np.pi)
        begin_LST = float(ephem.hours(args.begin_LST)) / (2 * np.pi)
        end_LST = float(ephem.hours(args.end_LST)) / (2 * np.pi)
        if (begin_LST > end_LST):
            begin_LST -= 1.

        #Reset the date to correspond to the sidereal time to start
        experience.inst.date -= (
            (lst_now - begin_LST) * sidDayToSec) * ephem.second

        #Figure out how long to run the scan for
        num_pts = int((end_LST - begin_LST) * sidDayToSec * sampling_freq)

    if (args.begin_RA and args.end_RA):
        experience.inst.horizon = el * ephem.degree

        #define a fixed source at the ra, dec that we want to scan
        target_min_ra = ephem.FixedBody()
        target_max_ra = ephem.FixedBody()

        #Figure out where we are looking now
        ra_target, dec_target = experience.inst.radec_of(
            az_mean / radToDeg, el / radToDeg)

        #instantiate the targets
        target_min_ra._dec = dec_target
        target_max_ra._dec = dec_target
        target_min_ra._ra = args.begin_RA
        target_max_ra._ra = args.end_RA

        #compute initial RA
        target_min_ra.compute(experience.inst)
        target_max_ra.compute(experience.inst)
        if(args.orientation == 'east'):
            experience.inst.date = experience.inst.next_rising(target_min_ra)

            #recompute coodinates in the light of change of date
            target_min_ra.compute(experience.inst)
            target_max_ra.compute(experience.inst)

            ## Update number of time samples for the scan
            num_pts = int(
                (experience.inst.next_rising(target_max_ra) -
                    experience.inst.date) / ephem.second * sampling_freq)

        if(args.orientation == 'west'):
            experience.inst.date = experience.inst.next_setting(target_min_ra)

            #recompute coodinates in the light of change of date
            target_min_ra.compute(experience.inst)
            target_max_ra.compute(experience.inst)

            ## Update number of time samples for the scan
            num_pts = int(
                (experience.inst.next_setting(target_max_ra) -
                    experience.inst.date)/ephem.second * sampling_freq)

    #########################################################
    ## Run the scan!
    #########################################################
    pb_az_dir = 1.
    upper_az = az_mean + az_throw / 2.
    lower_az = az_mean - az_throw / 2.
    az_speed = args.sky_speed / np.cos(el / radToDeg)
    pb_az = az_mean

    ## Compute RA, Dec and parallactic angle
    pb_ra_array, pb_dec_array, parallactic_angle = compute_RA_Dec_PA(
        experience,
        num_pts,
        pb_az,
        upper_az,
        lower_az,
        pb_az_dir,
        az_speed,
        el,
        sampling_freq,
        args.HWP_frequency)

    ## Project from time to map domain
    nhit_loc, cc_loc, ss_loc, cs_loc = project_data_on_sky(
        args.nside,
        pb_ra_array,
        pb_dec_array,
        parallactic_angle)

    scan_file.nhit = scan_file.nhit + nhit_loc
    scan_file.cc = scan_file.cc + cc_loc
    scan_file.ss = scan_file.ss + ss_loc
    scan_file.cs = scan_file.cs + cs_loc

def checks_param(args):
    """
    Checks args have all the same number of entries.
    """
    for arg in args.__dict__.keys():
        if type(args.__dict__[arg]) == list:
            assert args.length_of_cycle == len(args.__dict__[arg]), \
                'You want to create %d CESes but give only %d %s' % \
                (args.length_of_cycle, len(args.__dict__[arg]), arg)
