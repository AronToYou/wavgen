import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from math import pi, sin
from ctypes import c_uint16
from array import *
import random
import h5py
import easygui


### Constants ###
SAMP_VAL_MAX = (2 ** 15 - 1)  # Maximum digital value of sample ~~ signed 16 bits
SAMP_FREQ_MAX = 1250E6  # Maximum Sampling Frequency
MAX_DATA = 25E5  # Maximum number of samples to hold in array at once
N = mp.cpu_count()

### Parameter ###
SAMP_FREQ = 1000E6  # Modify if a different Sampling Frequency is required.

## Function for Parallel Processing ##
def workload(idx, sample_length, freqs, mags, phis, targs):
    load_length = int(sample_length // N + 1)
    parts = int(load_length//MAX_DATA) + 1
    portion = int(load_length//parts) + 1
    normalization = sum(mags)
    buffer = np.zeros()

    for part in range(parts):
        ## For each Wave ##
        temp_buffer = np.zeros(portion)
        for f, mag, phi, t in zip(freqs, mags, phis, targs):
            fn = f / SAMP_FREQ  # Cycles/Sample
            df = (t - f) / SAMP_FREQ if t else 0

            ## Compute the Wave ##
            for i in range(portion):
                n = i + part*portion + idx*load_length
                if n == sample_length:
                    break
                dfn = df*n / sample_length if t else 0
                temp_buffer[i] += mag*sin(2*pi*n*(fn + dfn) + phi)

        ## Normalize the Buffer ##
        for i in range(portion):
            n = i + part*portion + idx*load_length
            if n == sample_length:
                break
            self.Buffer[n] = c_uint16(SAMP_VAL_MAX * (temp_buffer[i] / normalization)).value

######### Wave Class #########
class Wave:
    """
        MEMBER VARIABLES:
            + Frequency - (Hertz)
            + Magnitude - Relative Magnitude between [0 - 1] inclusive
            + Phase ----- (Radians)
    """
    def __init__(self, freq, mag=1, phase=0):
        ## Validate ##
        assert freq > 0, ("Invalid Frequency: %d, must be positive" % freq)
        assert 0 <= mag <= 1, ("Invalid magnitude: %d, must be within interval [0,1]" % mag)
        ## Initialize ##
        self.Frequency = int(freq)
        self.Magnitude = mag
        self.Phase = phase

    def __lt__(self, other):
        return self.Frequency < other.Frequency


######### Segment Class #########
class Segment:
    """
        MEMBER VARIABLES:
            + Waves -------- List of Wave objects which compose the Segment. Sorted in Ascending Frequency.
            + Resolution --- (Hertz) The target resolution to aim for. In other words, sets the sample time (N / Fsamp)
                             and thus the 'wavelength' of the buffer (wavelength that completely fills the buffer).
                             Any multiple of the resolution will be periodic in the memory buffer.
            + SampleLength - Calculated during Buffer Setup; The length of the Segment in Samples.
            + Buffer ------- Storage location for calculated Wave.
            + Latest ------- Boolean indicating if the Buffer is the correct computation (E.g. correct Magnitude/Phase)

        USER METHODS:
            + add_wave(w) --------- Add the wave object 'w' to the segment, given it's not a duplicate frequency.
            + remove_frequency(f) - Remove the wave object with frequency 'f'.
            + plot() -------------- Plots the segment via matplotlib. Computes first if necessary.
            + randomize() --------- Randomizes the phases for each composing frequency of the Segment.
        PRIVATE METHODS:
            + _compute() - Computes the segment and stores into Buffer.
            + __str__() --- Defines behavior for --> print(*Segment Object*)
    """
    def __init__(self, freqs, resolution=1E6, sample_length=None, filename=None, targets=None):
        """
            Multiple constructors in one.
            INPUTS:
                freqs ------ A list of frequency values, from which wave objects are automatically created.
                waves ------ Alternative to above, a list of pre-constructed wave objects could be passed.
            == OPTIONAL ==
                resolution ---- Either way, this determines the...resolution...and thus the sample length.
                sample_length - Overrides the resolution parameter.
        """
        ## Validate & Sort ##
        freqs.sort()

        if sample_length is not None:
            target_sample_length = int(sample_length)
            resolution = SAMP_FREQ / target_sample_length
        else:
            assert resolution < SAMP_FREQ / 2, ("Invalid Resolution, has to be below Nyquist: %d" % (SAMP_FREQ / 2))
            target_sample_length = int(SAMP_FREQ / resolution)

        assert freqs[-1] >= resolution, ("Frequency %d is smaller than Resolution %d." % (freqs[-1], resolution))
        assert freqs[0] < SAMP_FREQ_MAX / 2, ("Frequency %d must below Nyquist: %d" % (freqs[0], SAMP_FREQ / 2))

        ## Initialize ##
        self.Waves = [Wave(f) for f in freqs]
        self.SampleLength = (target_sample_length - target_sample_length % 32)
        self.Latest       = False
        self.Buffer       = None
        self.Filename     = filename
        self.Targets      = np.zeros(len(freqs), dtype='i8') if targets is None else np.array(targets, dtype='i8')
        self.Filed        = False
        ## Report ##
        print("Sample Length: ", self.SampleLength)
        print('Target Resolution: ', resolution, 'Hz, Achieved resolution: ', SAMP_FREQ / self.SampleLength, 'Hz')

    def get_magnitudes(self):
        """ Returns an array of magnitudes,
            each associated with a particular trap.

        """
        return [w.Magnitude for w in self.Waves]

    def set_magnitude(self, idx, mag):
        """ Sets the magnitude of the indexed trap number.
            INPUTS:
                idx - Index to trap number, starting from 0
                mag - New value for relative magnitude, must be in [0, 1]
        """
        assert 0 <= mag <= 1, ("Invalid magnitude: %d, must be within interval [0,1]" % mag)
        self.Waves[idx].Magnitude = mag
        self.Latest = False

    def set_magnitudes(self, mags):
        """ Sets the magnitude of all traps.
            INPUTS:
                mags - List of new magnitudes, in order of Trap Number (Ascending Frequency).
        """
        for i, mag in enumerate(mags):
            assert 0 <= mag <= 1, ("Invalid magnitude: %d, must be within interval [0,1]" % mag)
            self.Waves[i].Magnitude = mag
        self.Latest = False

    def get_phases(self):
        return [w.Phase for w in self.Waves]

    def set_phase(self, idx, phase):
        """ Sets the magnitude of the indexed trap number.
            INPUTS:
                idx --- Index to trap number, starting from 0
                phase - New value for phase.
        """
        phase = phase % (2*pi)
        self.Waves[idx].Phase = phase
        self.Latest = False

    def set_phases(self, phases):
        """ Sets the magnitude of all traps.
            INPUTS:
                mags - List of new phases, in order of Trap Number (Ascending Frequency).
        """
        for i, phase in enumerate(phases):
            self.Waves[i].Phase = phase
        self.Latest = False

    def plot(self):
        """ Plots the Segment. Computes first if necessary.

        """
        if not self.Latest:
            self.compute()
        plt.plot(self.Buffer, '--o')
        plt.show()

    def randomize(self):
        """ Randomizes each phase.

        """
        for w in self.Waves:
            w.Phase = 2*pi*random.random()
        self.Latest = False

    def compute(self):
        """ Computes the superposition of frequencies
            and stores it in the buffer.
            We divide by the sum of relative wave magnitudes
            and scale the max value to SAMP_VAL_MAX,
            s.t. if all waves phase align, they will not exceed the max value.
        """
        F = None
        ## Checks if Redundant ##
        if self.Latest:
            return

        ## Initialize Buffer ##
        if self.SampleLength > MAX_DATA or self.Filename is not None:
            if self.Filename is None:
                msg = "You data is too large, You need to save it to file."
                if easygui.boolbox(msg, "Warning!", ['Abort', 'Save'], "images/panic.jpg"):
                    exit(-1)
                self.Filename = easygui.enterbox("Enter a filename:", "Input", "unnamed")

            while not self.Filed:
                if self.Filename is None:
                    exit(-1)
                try:
                    F = h5py.File(self.Filename, 'r')
                    if easygui.boolbox("Overwrite existing file?"):
                        F.close()
                        break
                    self.Filename = easygui.enterbox("Enter a filename or blank to abort:", "Input")
                except OSError:
                    break
            F = h5py.File(self.Filename, "w")
            F.create_dataset('frequencies', data=np.array([w.Frequency for w in self.Waves]))
            F.create_dataset('targets', data=np.array(self.Targets))
            F.create_dataset('magnitudes', data=np.array([w.Magnitude for w in self.Waves]))
            F.create_dataset('phases', data=np.array([w.Phase for w in self.Waves]))
            self.Buffer = F.create_dataset('data', shape=(self.SampleLength,), dtype='uint16')
        else:
            self.Buffer = np.zeros(self.SampleLength, dtype=np.int16)

        ## Setup Parallel Processing ##
        N = mp.cpu_count()
        samp_length = mp.Value('I', self.SampleLength)
        frequencies = mp.Array('I', [w.Frequency for w in self.Waves])
        magnitudes = mp.Array('f', [w.Magnitude for w in self.Waves])
        phases = mp.Array('f', [w.Phase for w in self.Waves])
        targets = mp.Array('I', self.Targets)
        workers = mp.Pool(N)

        ## Distribution of Work ##
        workers.starmap(workload, [(i, samp_length, frequencies, magnitudes, phases, targets) for i in range(N)])

        ## Wrapping things Up ##
        self.Latest = True  # Will be up to date after
        if F is not None:   # Also, close the file if opened
            self.Filed = True
            F.close()

    def __str__(self):
        s = "Segment with Resolution: " + str(SAMP_FREQ / self.SampleLength) + "\n"
        s += "Contains Waves: \n"
        for w in self.Waves:
            s += "---" + str(w.Frequency) + "Hz - Magnitude: " \
                + str(w.Magnitude) + " - Phase: " + str(w.Phase) + "\n"
        return s


class Waveform():
    """
        MEMBER VARIABLES:
            + Waves -------- List of Wave objects which compose the Segment. Sorted in Ascending Frequency.
            + Resolution --- (Hertz) The target resolution to aim for. In other words, sets the sample time (N / Fsamp)
                             and thus the 'wavelength' of the buffer (wavelength that completely fills the buffer).
                             Any multiple of the resolution will be periodic in the memory buffer.
            + SampleLength - Calculated during Buffer Setup; The length of the Segment in Samples.
            + Buffer ------- Storage location for calculated Wave.
            + Latest ------- Boolean indicating if the Buffer is the correct computation (E.g. correct Magnitude/Phase)

        USER METHODS:
            + add_wave(w) --------- Add the wave object 'w' to the segment, given it's not a duplicate frequency.
            + remove_frequency(f) - Remove the wave object with frequency 'f'.
            + plot() -------------- Plots the segment via matplotlib. Computes first if necessary.
            + randomize() --------- Randomizes the phases for each composing frequency of the Segment.
        PRIVATE METHODS:
            + _compute() - Computes the segment and stores into Buffer.
            + __str__() --- Defines behavior for --> print(*Segment Object*)
    """
    def __init__(self, freqs, resolution=1E6, sample_length=None, filename=None, targets=None):
        """
            Multiple constructors in one.
            INPUTS:
                freqs ------ A list of frequency values, from which wave objects are automatically created.
                waves ------ Alternative to above, a list of pre-constructed wave objects could be passed.
            == OPTIONAL ==
                resolution ---- Either way, this determines the...resolution...and thus the sample length.
                sample_length - Overrides the resolution parameter.
        """
        ## Validate & Sort ##
        freqs.sort()

        self.Filename = filename
        self.Data = []

        if sample_length is not None:
            target_sample_length = int(sample_length)
            resolution = SAMP_FREQ / target_sample_length
        else:
            assert resolution < SAMP_FREQ / 2, ("Invalid Resolution, has to be below Nyquist: %d" % (SAMP_FREQ / 2))
            target_sample_length = int(SAMP_FREQ / resolution)

        assert freqs[-1] >= resolution, ("Frequency %d is smaller than Resolution %d." % (freqs[-1], resolution))
        assert freqs[0] < SAMP_FREQ_MAX / 2, ("Frequency %d must below Nyquist: %d" % (freqs[0], SAMP_FREQ / 2))


        N = sample_length//DATA_MAX + 1
        ## Initialize ##
        self.Segments = [Segment(n) for n in range(N)]
        self.SampleLength = (target_sample_length - target_sample_length % 32)
        self.Latest       = False
        self.Buffer       = None
        self.Filename     = filename
        self.Targets      = np.zeros(len(freqs), dtype='i8') if targets is None else np.array(targets, dtype='i8')
        self.Filed        = False


######### SegmentFromFile Class #########
class SegmentFromFile(Segment):
    """ This class just provides a clean way to construct Segment objects
        from saved files.
        It shares all of the same characteristics as a Segment.
    """
    def __init__(self, filename):
        with h5py.File(filename, 'r') as f:
            freqs = f.get('frequencies')[()]
            targs = f.get('targets')[()]
            sampL = f['data'].shape[0]
            super().__init__(freqs, sample_length=sampL, filename=filename, targets=targs)
            self.Latest = True
            self.Filed = True
            self.set_phases(f['phases'])
            self.set_magnitudes(f['magnitudes'])
