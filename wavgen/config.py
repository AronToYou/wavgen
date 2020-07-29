### Parameters ###
NUMPY_MAX = int(1E5)   #: Max size of Software buffer for board transfers (in samples)
MAX_EXP = 150          #: Cap on the exposure value for ThorCam devices.
DEF_AMP = 210          #: Default maximum waveform output amplitude (milliVolts)
VERBOSE = False        #: Flag to de/activate most print messages throughout program.

DATA_MAX = int(16E4)     #: Maximum number of samples to hold in array at once
PLOT_MAX = int(1E4)      #: Maximum number of data-points to plot at once
SAMP_FREQ = int(1000E6)  #: Desired Sampling Frequency

### Constants - DO NOT CHANGE ###
SAMP_VAL_MAX = (2 ** 15 - 1)  #: Maximum digital value of sample ~~ signed 16 bits
SAMP_FREQ_MAX = int(1250E6)   #: Maximum Sampling Frequency
MHZ = SAMP_FREQ / 1E6         #: Coverts samples/seconds to MHz
# TODO: Generalize by querying hardware every time program runs.
MEM_SIZE = 4294967296  #: Size of the \*board's memory (bytes)  \*Spectrum M4i.6631-x8
