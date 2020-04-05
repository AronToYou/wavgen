from .spcm_tools import *
## This Exception Catching is necessary so ReadTheDocs can import/build the automated documentation ##
## Otherwise, being a linux server, it lacks the .dll Spectrum Drivers, thus throwing an error ##
try:
    from .pyspcm import *
except OSError:
    print("Need to install Spectrum Card Drivers from their website!")
