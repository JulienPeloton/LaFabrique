#!/usr/bin/python
#####################################################################################
# Script to simulate TOD files (hdf5 format) to be processed by the map-making
# Author: Julien Peloton, j.peloton@sussex.ac.uk
# Pointing module: thanks Neil G.!
######################################################################################

import ConfigParser
import sys
import ephem
import copy

import numpy as np
import healpy as hp
import pylab as pl
import util_CMB
import inputs

from util_CMB import date_to_mjd
# import numpy as np
# import argparse
# import ephem
# import h5py
# import os
# import sys
# import copy
# from so_constants import radToDeg, sidDayToSec, dayToSec
# from so_liblst import greg_to_mjd, mjd_to_greg, date_to_greg, date_to_mjd
# from config import so_config as config
# import so_libhwmap as libhwmap
# import so_util
# import so_parse_flat_map as parse_flat_map


# sigma_to_fwhm = (8*np.log(2))**0.5
#
# r2d = 180/np.pi
# d2r = np.pi/180
# am2r = d2r/60
# r2am = 1/am2r
# as2r = am2r/60.0
# r2as = 1.0/as2r
# d2am=60.
#
# BOLOMETER_SAMPLE_RATE = 25.0e6/2**17
# ANTENNA_SAMPLE_RATE = BOLOMETER_SAMPLE_RATE / 2.0
# ANTENNA_SLOW_SAMPLE_RATE = ANTENNA_SAMPLE_RATE/100.0
# physical_constants_si = {'Tcmb': 2.725, 'c': 299792458.0, 'eps0': 8.85418782e-12, 'h': 6.626068e-34, 'k': 1.3806593e-23, 'mu0': 1.25663706e-6, 'qe': 1.60217646e-19}
# si_prefix = {'T':1.0e12,'G':1.0e9,'M':1.0e6,'k':1.0e3,'c':1.0e-2,'m':1.0e-3,'mu':1.0e-6,'n':1.0e-9,'p':1.0e-12}
#
# ## numerical constants
radToDeg = 180/np.pi
sidDayToSec = 86164.0905
# dayToSec = 86400.
# sid_hour_to_hour = sidDayToSec / dayToSec

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

    ## Load experiment technical details
    experience = instrument(scanning.name_instrument)

    ## Initialise hdf5 file
    scan_info = inputs.HealpixMapInfo(
        npix=0,
        obspix=[],
        nside=scanning.nside,
        source=scanning.name_instrument)
    scan = inputs.InputScan(scan_info)

    ## Loop over CESes and generate scans
    for n in range(scanning.number_of_days):
        ## Initialise the starting date of observation
        ## It will be updated then automatically
        position_in_the_cycle =  n % scanning.length_of_cycle
        if n == 0:
            experience.inst.date = scanning.starting_date

        ## Grab the details of this CES only
        myargs = copy.copy(scanning)
        for myarg in myargs.__dict__.keys():
            if type(myargs.__dict__[myarg]) == np.ndarray:
                myargs.__dict__[myarg] = \
                    myargs.__dict__[myarg][position_in_the_cycle]

        ## Create the scan strategy
        create_scan_strategy(myargs,experience,scan)

        ## Save the result
        # out_folder = scanning.outpath_masks
        # save_file(scan,output=out_folder)

class instrument(object):
    """ Class to handle different instruments and their specificities """
    def __init__(self,instrument='so_deep'):
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

def constant_velocity_portions(scan,threshold=0.98):
    """
    Define constant velocity portion, and build a mask to remove samples outside
    """
    mean_az = np.mean(scan['antenna0-tracker-actual-0'][:])
    max_az = np.max(scan['antenna0-tracker-actual-0'][:])

    ## Threshold in azimuth (can probably be better...)
    mask = np.abs(scan['antenna0-tracker-actual-0'][:] - mean_az)/(max_az-mean_az) > threshold

    return ~mask

def create_scan_strategy(args,experience,scan_file):
    """
    Process the args to create the scanning strategy.
    This is based on Neil's module.
    """
    ## Check if we have enough information to make a scan
    if not ((args.begin_LST and args.end_LST) or (args.begin_RA and args.end_RA)):
        print('You must specify the beginning and ending of the scan in RA or in LST')
        exit(1)

    if (int(args.begin_LST is not None and args.end_LST is not None) + \
            int(args.begin_RA is not None and args.end_RA is not None) > 1):
        print('You cannot specify the beginning and ending of the scan in RA and in LST')
        exit(1)

    if not (args.full_circle or (args.az_min and args.az_max) or (args.dec_min and args.dec_max)):
        print('You must specify the range of the scan in either declination or azimuth')
        exit(1)

    if (int(args.full_circle) + int(args.az_min is not None and args.az_max is not None) \
            + int(args.dec_min is not None and args.dec_max is not None) > 1):
        print('You have overspecified the parameters of the scan in azimuth')
        exit(1)

    ## Figure out the elevation to run the scan!
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

    ## If given bounds in declination, make bounds in azimuth - note there is no sanity checking here!
    if(args.dec_min and args.dec_max):
        az_array = np.linspace(0., 180., endpoint=True, num=360.)
        ra_array = np.zeros(az_array.shape)
        dec_array = np.zeros(az_array.shape)
        dec_min = ephem.degrees(args.dec_min)
        dec_max = ephem.degrees(args.dec_max)
        if(args.orientation == 'west'):
            az_array += 180.

        for i in range(0, az_array.shape[0]):
            ra_array[i], dec_array[i] = experience.inst.radec_of(az_array[i] / radToDeg, el / radToDeg)

        az_allowed = np.asarray([az_array[i] for i in range (0, az_array.shape[0]) \
         if (dec_array[i] > dec_min and dec_array[i] < dec_max)])

        if (az_allowed.shape[0] < 2):
            print('Invalid combination of declination bounds and elevation.')
            exit(1)

        az_max = np.max(az_allowed)
        az_min = np.min(az_allowed)
        az_mean = (az_min + az_max)*0.5
        az_throw = (az_max - az_min)

    #########################################################
    ## Define the timing bounds!
    #########################################################
    num_pts = 0

    if (args.begin_LST and args.end_LST):
        lst_now = float(experience.inst.sidereal_time()) / (2* np.pi)
        begin_LST = float(ephem.hours(args.begin_LST)) / (2 * np.pi)
        end_LST = float(ephem.hours(args.end_LST)) / (2 * np.pi)
        if (begin_LST > end_LST):
            begin_LST -= 1.

        #Reset the date to correspond to the sidereal time to start
        experience.inst.date -= ((lst_now - begin_LST)*sidDayToSec)*ephem.second

        #Figure out how long to run the scan for
        num_pts = int((end_LST - begin_LST) * sidDayToSec * sampling_freq)

    if (args.begin_RA and args.end_RA):
        experience.inst.horizon = el * ephem.degree

        #define a fixed source at the ra, dec that we want to scan
        target_min_ra = ephem.FixedBody()
        target_max_ra = ephem.FixedBody()

        #Figure out where we are looking now
        ra_target, dec_target = experience.inst.radec_of(az_mean / radToDeg, el / radToDeg)

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
            num_pts = int((experience.inst.next_rising(target_max_ra) - experience.inst.date)/ephem.second * sampling_freq)

        if(args.orientation == 'west'):
            experience.inst.date = experience.inst.next_setting(target_min_ra)

            #recompute coodinates in the light of change of date
            target_min_ra.compute(experience.inst)
            target_max_ra.compute(experience.inst)

            ## Update number of time samples for the scan
            num_pts = int((experience.inst.next_setting(target_max_ra) - experience.inst.date)/ephem.second * sampling_freq)


    ## Run the scan!
    pb_az_dir = 1.
    upper_az = az_mean + az_throw / 2.
    lower_az = az_mean - az_throw / 2.
    az_speed = args.sky_speed / np.cos(el / radToDeg)
    pb_az = az_mean

    ## Initialize arrays
    pb_ra_array = np.zeros(num_pts)
    pb_dec_array = np.zeros(num_pts)

    #Boresight pointing healpix maps
    npix = hp.pixelfunc.nside2npix(args.nside)
    # nHits_array = np.zeros(npix)
    # lambda_array = np.ones(npix)
    # cos_alpha_array = np.zeros(npix)
    # sin_alpha_array = np.zeros(npix)

    #########################################################
    ## Loop over time samples
    #########################################################
    begin_lst=str(experience.inst.sidereal_time())
    # Pad scans 10 seconds on either side
    time_padding = 10.0/86400.0
    for t in range(0, num_pts):
        ## Compute Ra, Dec and convert to degrees
        pb_ra_array[t], pb_dec_array[t] = experience.inst.radec_of(pb_az / radToDeg, el / radToDeg)

        if t == 0:
            pb_az += az_speed*pb_az_dir/sampling_freq
            experience.inst.date += ephem.second/sampling_freq
            continue

        delta_ra = pb_ra_array[t] - pb_ra_array[t-1]
        delta_dec = pb_dec_array[t] - pb_dec_array[t-1]

        #Fix coordinate singularity issues
        if(delta_ra < -np.pi):
            delta_ra += 2.*np.pi
        if(delta_ra > np.pi):
            delta_ra -= 2.*np.pi
        #Put in parallactic angle things
        pb_alpha = np.arctan((delta_dec)/(delta_ra * np.cos(pb_dec_array[t])))

        #Project to a heaplix map
        pix_global = hp.pixelfunc.ang2pix(args.nside, (np.pi/2.) - pb_dec_array[t], pb_ra_array[t])

        # cos_alpha_array[pix_global] += np.cos(2*pb_alpha)
        # sin_alpha_array[pix_global] += np.sin(2*pb_alpha)
        nHits_array[pix_global] += 1.

        ## Case to change the direction of the scan
        if(pb_az > upper_az):
            pb_az_dir = -1.
        elif(pb_az < lower_az):
            pb_az_dir = 1.

        pb_az += az_speed*pb_az_dir/sampling_freq

        ## Increment the time by one second / sampling rate
        experience.inst.date += ephem.second/sampling_freq
    end_lst=str(experience.inst.sidereal_time())

    # print num_pts, len(nHits_array[nHits_array>0])
    # # TODO: update nhits. iterate for all CES. Parallelize. Output images to see the scan!
    # hp.mollview(nHits_array)
    # pl.show()

    # #########################################################
    # ## Save in file
    # #########################################################
    # scan_file['sample_rate'] = sampling_freq
    # scan_file['antenna0-tracker-actual-0'] = pb_az_array * np.pi / 180
    # scan_file['antenna0-tracker-actual-1'] = pb_el_array * np.pi / 180
    # scan_file['antenna0-tracker-actualRaDec-0'] = pb_ra_array * np.pi / 180
    # scan_file['antenna0-tracker-actualRaDec-1'] = pb_dec_array * np.pi / 180
    # scan_file['antenna0-tracker-utc-0'] = pb_mjd_array
    #
    # scan_file['lastmjd'] = pb_mjd_array[-1] + time_padding
    #
    # mask_array = np.array(constant_velocity_portions(scan_file,threshold=0.9),dtype=int)
    # scan_file['antenna0-tracker-scan_flag-0'] = mask_array
    #
    # for M in range(experience.NUM_MOBO):
    #     scan_file['receiver-bolometers-utc-%d'%M] = pb_mjd_array
    #
    # print '+-----------------------------------+'
    # print ' CES starts at %s and finishes at %s'%(mjd_to_greg(scan_file['firstmjd']),mjd_to_greg(scan_file['lastmjd']))
    # print ' It lasts %.3f hours'%((scan_file['lastmjd']-scan_file['firstmjd'])*24)
    # print '+-----------------------------------+'

    ## Add one day before the next CES (to avoid conflict of time)
    experience.inst.date += 24 * ephem.second * 3600

    return

def save_file(scan,output='.'):
    date = scan['firstmjd']
    out_fn = os.path.join(output,mjd_to_greg(date) + '.hdf5')
    out_file = h5py.File(out_fn,'w')
    attrs = ['sample_rate','firstmjd','lastmjd']
    for k in scan.keys():
        if k in attrs:
            out_file.attrs[k] = scan[k]
        else:
            out_file.create_dataset(k, data=scan[k])
    out_file.close()

def addargs(parser):
    ''' Parse command line arguments '''

    ## ID
    parser.add_argument('--name_strategy', dest='name_strategy',
     help='Name of the strategy (used to create the output folder)', required=True)
    parser.add_argument('--name_instrument', dest='name_instrument',
     help='Name of the instrument: Polarbear, ACTpol, SO (case insensitive)', required=True)
    parser.add_argument('--nCES', dest='nCES', type=int, help='Number of CES in the strategy', required=True)
    parser.add_argument('--start_date', dest='start_date', type=str,
     help='date to start the scan. Format YYYY/MM/DD HH:MM:SS', required=True)

    ## Instrument parameters
    parser.add_argument('--sky_speed', dest='sky_speed', type=float, help='Az throw speed in d/s', required=True)
    parser.add_argument('--sampling_freq', dest='sampling_freq', type=float,
     help='Sample rate in Hz (25e6/2**17)', required=True)

    ## Scan parameters
    parser.add_argument('--el', dest='el',nargs='+', type=float, help='Elevation to run the scan', required=True)
    parser.add_argument('--az_min', dest='az_min', nargs='+', type=float,
     help='Minimum azimuth value for the scan', required=False)
    parser.add_argument('--az_max', dest='az_max', nargs='+', type=float,
     help='Maximum azimuth value for the scan', required=False)
    parser.add_argument('--dec_min', dest='dec_min', nargs='+', help='Minimum declination for the scan', required=False)
    parser.add_argument('--dec_max', dest='dec_max', nargs='+', help='Maximum declination for the scan', required=False)
    parser.add_argument('--full_circle', dest='full_circle', action='store_true',
     help='Run a full circle scan', required=False)
    parser.add_argument('--begin_LST', dest='begin_LST', nargs='+',
     help='Local sidereal time to begin the scan', required=False)
    parser.add_argument('--end_LST', dest='end_LST', nargs='+',
     help='Local sidereal time to end the scan', required=False)
    parser.add_argument('--begin_RA', dest='begin_RA', nargs='+', help='RA to begin the scan', required=False)
    parser.add_argument('--end_RA', dest='end_RA', nargs='+', help='RA to stop the scan', required=False)
    parser.add_argument('--orientation', dest='orientation', nargs='+', type=str,
     help='Scan to the east/west', required=False)

    ## HWP parameters
    parser.add_argument('--type_HWP', dest='type_HWP', type=str,
     help='<CRHWP> or <stepped> HWP', required=True)
    parser.add_argument('--angle_HWP', dest='angle_HWP', type=float,
     help='Increment angle in deg for stepped HWP (between 0 and 180 deg)', required=False)
    parser.add_argument('--frequency_HWP', dest='frequency_HWP', type=float,
     help='Frequency of the CRWHP in Hz', required=False)

    ## Output
    parser.add_argument('--output_folder', dest='output_folder', type=str, help='To store data [do not use]', default=None)
    parser.add_argument('--tag', dest='tag', type=str, help='ID for the run [use this instead]', default='no_syst')

def checks_param(args):
    """
    Checks args have all the same number of entries.
    """
    for arg in args.__dict__.keys():
        if type(args.__dict__[arg]) == list:
            assert args.length_of_cycle == len(args.__dict__[arg]), \
            'You want to create %d CESes but give only %d %s' % \
                (args.length_of_cycle,len(args.__dict__[arg]),arg)

def grabargs(args_param=None):
    ''' Parse command line arguments '''
    parser = argparse.ArgumentParser(description='Generate pointing for a particular CES')
    addargs(parser)
    args = parser.parse_args(args_param)
    return args

def main():
    print " ____   ___        _                 _       _             "
    print "/ ___| / _ \   ___(_)_ __ ___  _   _| | __ _| |_ ___  _ __ "
    print "\___ \| | | | / __| | '_ ` _ \| | | | |/ _` | __/ _ \| '__|"
    print " ___) | |_| | \__ \ | | | | | | |_| | | (_| | |_ (_) | |   "
    print "|____/ \___/  |___/_|_| |_| |_|\__,_|_|\__,_|\__\___/|_|   "
    print "                              _             "
    print " ___  ___ __ _ _ __  _ __ (_)_ __   __ _ "
    print "/ __|/ __/ _` | '_ \| '_ \| | '_ \ / _` |"
    print "\__ \ (__ (_| | | | | | | | | | | | (_| |"
    print "|___/\___\__,_|_| |_|_| |_|_|_| |_|\__, |"
    print "                                   |___/ "

    ## Grab parameters and perform various checks
    args_param = None
    args   = grabargs(args_param)
    checks_param(args)

    ## Load experiment technical details
    experience = instrument(args.name_instrument)

    print '+-------------------------------+'
    print '+ Instrument:  %s'%args.name_instrument
    print '+ Scan type:   %s'%args.name_strategy
    print '+ nCES:        %d'%args.nCES
    print '+ Sky speed:   %.2f deg/s'%args.sky_speed
    print '+ Sample rate: %.2f Hz'%args.sampling_freq
    print '+-------------------------------+'

    if args.output_folder is None:
        args.output_folder = os.path.join(args.tag+'_'+args.name_instrument+'_'+args.name_strategy,'TOD')
    else:
        args.output_folder = os.path.join(args.output_folder,'TOD')

    dic_HWP = {}
    ## Loop over CESes
    for n in range(args.nCES):
        ## Initialise the starting date of observation
        ## It will be updated then automatically
        if n == 0:
            experience.inst.date = args.start_date

        ## Grab the details of this CES only
        myargs = copy.copy(args)
        for myarg in myargs.__dict__.keys():
            if type(myargs.__dict__[myarg]) == list:
                myargs.__dict__[myarg] = myargs.__dict__[myarg][n]

        ## Define empty TOD structure a la PB
        scan = {}
        scan = create_fields_in_file(experience,scan)

        # Create the scan strategy
        s = create_scan_strategy(myargs,experience,scan)
        if s is False:
            ## that means already computed
            print '-- Skipping --'
            continue
        create_HWP(myargs,dic_HWP,scan,n)

        # Save the CES on disk
        # out_folder = os.path.join(args.output_folder,'TOD')
        out_folder = args.output_folder
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        save_file(scan,output=out_folder)
    if s is not False:
        hwp_caltable = os.path.join (config.get_cal_path(), 'hwp_caltable_%s.pkl'%args.tag)
        so_util.pickle_save(dic_HWP,hwp_caltable)

if __name__ == "__main__":
    main()
