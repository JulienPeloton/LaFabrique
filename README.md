LaFabrique
==

![ScreenShot](https://github.com/JulienPeloton/LaFabrique/blob/master/additional_files/outputs.png)

#### The package
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
    * The inputs are based on a potential future CMB ground-based experiment.
    * Include a call to ScanStrategy code (TBD)

### Before starting
This code has the following dependencies:
* numpy, pylab, etc (required)
* healpy
* h5py
* PySM (see https://github.com/bthorne93/PySM_public)

Make sure you update your PYTHONPATH to use the code.
Just add in your bashrc:
```bash
LaFabriquePATH=/path/to/the/package
export PYTHONPATH=$PYTHONPATH:$LaFabriquePATH
```

### Usage
You can find ini files to have a quick usage.
Here the breakdown of the code for 1 noise simulation for 8 frequency channels
at a resolution of nside = 2048, and few percent of the sky. Notice that the longest
step is the Cholesky factorisation which is done only once for all MC simulations then.

![ScreenShot](https://github.com/JulienPeloton/LaFabrique/blob/master/additional_files/perf_1MC_nside2048.png)

### License
GNU License (see the LICENSE file for details) covers all files
in the LaFabrique repository unless stated otherwise.
