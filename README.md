LaFabrique [![Build Status](https://travis-ci.org/JulienPeloton/LaFabrique.svg?branch=master)](https://travis-ci.org/JulienPeloton/LaFabrique)
==

![ScreenShot](https://github.com/JulienPeloton/LaFabrique/blob/master/additional_files/outputs.png)

## The package
This package contains scripts for generating simulated CMB datasets (current and future)
* Noise
    * Generate (inhomogeneous) noise maps
* Covariances
    * Generate covariance matrices (in the sense inverse noise weights)
* Foregrounds
    * Generate foregrounds maps (dust, synchrotron, freefree, spinningdust, ...) based on the PySM
* Frequency coverage
    * Multi-frequency
* Sky coverage
    * Scanning strategy simulator included

## Before starting
This code has the following dependencies (see the travis install section):
* numpy, pylab, scipy, etc (required)
* healpy >= 1.9.1 (required)
* h5py (required) + libhdf5 if using linux
* weave (optional if scipy <= v0.18 - required otherwise because no longer part of scipy v0.19)
* ephem (required - for scanning strategy)
* mpi4py (optional - for parallel computing)
* PySM (optional - foreground generation see https://github.com/bthorne93/PySM_public)

## Installation
We provide a setup.py for the installation. Just run:
```bash
python setup.py install
```
Make sure you have correct permissions (otherwise just add --user).
You can also directly use the code by updating manually your PYTHONPATH.
Just add in your bashrc:
```bash
LaFabriquePATH=/path/to/the/package
export PYTHONPATH=$PYTHONPATH:$LaFabriquePATH
```

## Usage
You can find ini files to have a quick usage:

```bash
python generate_dataset.py -h
python generate_dataset.py -setup_env setup_env.ini <other options>
```

The code has a parallel architecture, which can be useful for large noise
simulations at high resolution (simulation of scan strategy and
foregrounds are done serially for the moment).
For example, just use (replace mpirun by your favourite one):

```bash
mpirun -n <nproc> python generate_dataset.py
    -setup_env setup_env.ini
    -setup_scanning setup_scanning.ini
    -setup_instrument setup_instrument.ini
```

Note that at NERSC, you have to replace the python binary by python-mpi which is
optimized for parallel computing.

Here is the breakdown of the code for 100 noise simulations for 8 frequency channels on 48 processors
at a resolution of nside = 2048, and a few percent of the sky. Notice that the
Cholesky factorisation is done only once per processor for each frequency band and
for all MC simulations (corresponds to 48[proc] * 8[freq] = 384 calls below).

![ScreenShot](https://github.com/JulienPeloton/LaFabrique/blob/master/additional_files/perf_100MC_nside2048.png)

## License
GNU License (see the LICENSE file for details) covers all files
in the LaFabrique repository unless stated otherwise.
