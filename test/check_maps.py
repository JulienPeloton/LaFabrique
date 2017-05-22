import numpy as np
import healpy as hp
import subprocess
import os
import sys

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

def output(passed):
    if passed:
        print bcolors.OKGREEN + "   PASSED" + bcolors.ENDC
    else:
        print bcolors.FAIL + "   FAILED" + bcolors.ENDC
	sys.exit()


def compare_covariance_maps(freqs):
    print "Running check for covariance maps..."
    for freq in freqs:
        fn = 'IQU_nside64_test_weights_freq%s.fits' % freq
        mp1 = hp.read_map(
            os.path.join(
                'test/benchmark/masks',fn),
                field=[0,1,2],
                verbose=False)
        mp2 = hp.read_map(
            os.path.join(
                'test/test/masks',fn),
                field=[0,1,2],
                verbose=False)

        for i in [0, 1, 2]:
            mask = mp1[i] == mp1[i]
            norm = np.std(mp1[i][mask])

            diff = np.std((mp1[i][mask] - mp2[i][mask])) / norm
            if diff > 1E-6 :
                return 0
    return 1

def compare_noise_maps(freqs):
    print "Running check for noise maps..."
    for freq in freqs:
        fn = 'IQU_nside64_test_freq%s_white_noise_sim000.fits' % freq
        mp1 = hp.read_map(
            os.path.join(
                'test/benchmark/noise', fn),
                field=[0, 1, 2],
                verbose=False)
        mp2 = hp.read_map(
            os.path.join(
                'test/test/noise', fn),
                field=[0, 1, 2],
                verbose=False)

        for i in [0, 1, 2]:
            norm = np.std(mp1[i])
            diff = np.std((mp1[i] - mp2[i])) / norm

            if diff > 1E-6 :
                return 0

    return 1

def run_check():
    os.system(
        'python generate_dataset.py \
        -setup_env test/setup_env_test.ini \
        -setup_scanning test/setup_scanning_test.ini \
        -setup_instrument test/setup_instrument_test.ini')

    freqs = [
        '27_LF', '39_LF', '90_MF', '150_MF',
        '150_HF', '220_HF', '220_UHF', '270_UHF']

    output(compare_noise_maps(freqs))
    output(compare_covariance_maps(freqs))

    subprocess.call(['rm', '-r', 'test/test'])

run_check()
