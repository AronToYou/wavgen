import numpy as np
import multiprocessing as mp
from math import pi, sin
from ctypes import c_uint16
# from .card import SAMP_FREQ
import random
import h5py
import easygui


### Constants ###
SAMP_VAL_MAX = (2 ** 15 - 1)  # Maximum digital value of sample ~~ signed 16 bits
SAMP_FREQ_MAX = 1250E6  # Maximum Sampling Frequency
CPU_MAX = mp.cpu_count()

### Parameter ###
DATA_MAX = int(16E5)  # Maximum number of samples to hold in array at once
SAMP_FREQ = 1000E6

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
class Segment(mp.Process):
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
    def __init__(self, n, waves, targets, sample_length):
        """
            Multiple constructors in one.
            INPUTS:
                freqs ------ A list of frequency values, from which wave objects are automatically created.
                waves ------ Alternative to above, a list of pre-constructed wave objects could be passed.
            == OPTIONAL ==
                resolution ---- Either way, this determines the...resolution...and thus the sample length.
                sample_length - Overrides the resolution parameter.
        """
        mp.Process.__init__(self)
        self.SampleLength = sample_length
        self.Portion = n
        self.Waves = waves
        self.Targets = targets
        buf_length = min(DATA_MAX, int(sample_length - n*DATA_MAX))
        self.Buffer = np.zeros(buf_length, dtype='int16')

    def run(self):
        temp_buffer = np.zeros(len(self.Buffer), dtype=float)
        normalization = sum([w.Magnitude for w in self.Waves])

        for w, t in zip(self.Waves, self.Targets):
            f = w.Frequency
            phi = w.Phase
            mag = w.Magnitude

            fn = f / SAMP_FREQ  # Cycles/Sample
            df = (t - f) / SAMP_FREQ if t else 0

            ## Compute the Wave ##
            for i in range(len(self.Buffer)):
                n = i + DATA_MAX*self.Portion
                dfn = df*n / self.SampleLength if t else 0
                temp_buffer[i] += mag*sin(2*pi*n*(fn + dfn) + phi)

        ## Normalize the Buffer ##
        for i in range(len(self.Buffer)):
            self.Buffer[i] = c_uint16(int(SAMP_VAL_MAX * (temp_buffer[i] / normalization))).value


######### Waveform Class #########
class Waveform:
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
        self.Waves        = [Wave(f) for f in freqs]
        self.SampleLength = (target_sample_length - target_sample_length % 32)
        self.Targets      = np.zeros(len(freqs), dtype='i8') if targets is None else np.array(targets, dtype='i8')
        self.Latest       = False
        self.Filename     = filename
        self.Filed        = False

    def compute_and_save(self):
        """ Computes the superposition of frequencies
            and stores it to an .h5py file.

        """
        ## Checks if Redundant ##
        if self.Latest:
            return

        ## Check for File duplicate ##
        if self.Filename is None:
            self.Filename = easygui.enterbox("Enter a filename:", "Input", None)
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

        ## Open h5py File ##
        F = h5py.File(self.Filename, "w")
        F.create_dataset('frequencies', data=np.array([w.Frequency for w in self.Waves]))
        F.create_dataset('targets', data=np.array(self.Targets))
        F.create_dataset('magnitudes', data=np.array([w.Magnitude for w in self.Waves]))
        F.create_dataset('phases', data=np.array([w.Phase for w in self.Waves]))
        dset = F.create_dataset('data', shape=(self.SampleLength,), dtype='uint16')

        ## Setup Parallel Processing ##
        procs = []
        N = int(self.SampleLength//(DATA_MAX + 1)) + 1
        print("N: ", N)
        n = 0
        while n != N:
            for _ in range(CPU_MAX):
                print("Running N=", n)
                p = Segment(n, self.Waves, self.Targets, self.SampleLength)
                procs.append(p)
                p.start()
                n += 1
                if n == N:
                    break

            for i in range(len(procs)):
                p = procs.pop()
                p.join()
                j = (i + n)*DATA_MAX
                if n == N:
                    dset[j:] = p.Buffer
                else:
                    dset[j:j + DATA_MAX] = p.Buffer
                del p

        ## Wrapping things Up ##
        self.Latest = True  # Will be up to date after
        self.Filed = True
        F.close()

    def get_magnitudes(self):
        """ Returns an array of magnitudes,
            each associated with a particular trap.

        """
        return [w.Magnitude for w in self.Waves]

    def set_magnitudes(self, mags):
        """ Sets the magnitude of all traps.
            INPUTS:
                mags - List of new magnitudes, in order of Trap Number (Ascending Frequency).
        """
        for w, mag in zip(self.Waves, mags):
            assert 0 <= mag <= 1, ("Invalid magnitude: %d, must be within interval [0,1]" % mag)
            w.Magnitude = mag
        self.Latest = False

    def get_phases(self):
        return [w.Phase for w in self.Waves]

    def set_phases(self, phases):
        """ Sets the magnitude of all traps.
            INPUTS:
                mags - List of new phases, in order of Trap Number (Ascending Frequency).
        """
        for w, phase in zip(self.Waves, phases):
            w.Phase = phase
        self.Latest = False

    def plot(self):
        """ Plots the Segment. Computes first if necessary.

        """
        pass

    def randomize(self):
        """ Randomizes each phase.

        """
        for w in self.Waves:
            w.Phase = 2*pi*random.random()
        self.Latest = False

    def __str__(self):
        s = "Segment with Resolution: " + str(SAMP_FREQ / self.SampleLength) + "\n"
        s += "Contains Waves: \n"
        for w in self.Waves:
            s += "---" + str(w.Frequency) + "Hz - Magnitude: " \
                + str(w.Magnitude) + " - Phase: " + str(w.Phase) + "\n"
        return s


######### SegmentFromFile Class #########
class WaveformFromFile(Waveform):
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
