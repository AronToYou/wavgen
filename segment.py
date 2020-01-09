import matplotlib.pyplot as plt
import numpy as np
from math import sin, pi
import random, bisect, pickle

## Helper Class ##
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
        assert mag >= 0 and mag <= 1, ("Invalid magnitude: %d, must be within interval [0,1]" % mag)
        ## Initialize ##
        self.Frequency = freq
        self.Magnitude = mag
        self.Phase = phase

    def __lt__(self, other):
        return self.Frequency < other.Frequency


## Primary Class ##
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
            + __compute() - Computes the segment and stores into Buffer.
            + __str__() --- Defines behavior for --> print(*Segment Object*)
    """
    def __init__(self, freqs=None, waves=None, resolution=1E6, sample_length=None):
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
        if sample_length is not None:
            target_sample_length = int(sample_length)
            resolution = SAMP_FREQ / target_sample_length
        else:
            assert resolution < SAMP_FREQ / 2, ("Invalid Resolution, has to be less than Nyquist Frequency: %d" % (SAMP_FREQ / 2))
            target_sample_length = int(SAMP_FREQ / resolution)
        if freqs is None and waves is not None:
            for i in range(len(waves)):
                assert waves[i].Frequency >= resolution, ("Frequency %d was given while Resolution is limited to %d Hz." % (waves[i].Frequency, resolution))
                assert waves[i].Frequency < SAMP_FREQ / 2, ("All frequencies must below Nyquist frequency: %d" % (SAMP_FREQ / 2))
        elif freqs is not None and waves is None:
            for f in freqs:
                assert f >= resolution, ("Frequency %d was given while Resolution is limited to %d Hz." % (f, resolution))
                assert f < SAMP_FREQ_MAX / 2, ("All frequencies must below Nyquist frequency: %d" % (SAMP_FREQ / 2))
        else:
            assert False, "Must override either only 'freqs' or 'waves' input argument."
        ## Initialize ##
        if waves is None:
            self.Waves = [Wave(f) for f in freqs]
        else:
            self.Waves = waves
        self.Waves.sort(key=(lambda w: w.Frequency))
        self.SampleLength = (target_sample_length - target_sample_length % 32)
        self.Latest       = False
        self.Buffer       = None
        ## Report ##
        print("Sample Length: ", self.SampleLength)
        print('Target Resolution: ', resolution, 'Hz, Achieved resolution: ', SAMP_FREQ / self.SampleLength, 'Hz')


    def add_wave(self, w):
        for wave in self.Waves:
            if w.Frequency == wave.Frequency:
                print("Skipping duplicate: %d Hz" % w.Frequency)
                return
        resolution = SAMP_FREQ / self.SampleLength
        assert w.Frequency >= resolution, ("Resolution: %d Hz, sets the minimum allowed frequency. (it was violated)" % resolution)
        assert w.Frequency < SAMP_FREQ / 2, ("All frequencies must be below Nyquist frequency: %d" % (SAMP_FREQ / 2))
        bisect.insort(self.Waves, w)
        self.Latest = False


    def remove_frequency(self, f):
        self.Waves = [W for W in self.Waves if W.Frequency != f]
        self.Latest = False


    def set_magnitude(self, idx, mag):
        """
            Sets the magnitude of the indexed trap number.
            INPUTS:
                idx - Index to trap number, starting from 0
                mag - New value for relative magnitude, must be in [0, 1]
        """
        assert mag >= 0 and mag <= 1, ("Invalid magnitude: %d, must be within interval [0,1]" % mag)
        self.Waves[idx].Magnitude = mag
        self.Latest = False

    def set_magnitude_all(self, mags):
        """
            Sets the magnitude of all traps.
            INPUTS:
                mags - List of new magnitudes, in order of Trap Number (Ascending Frequency).
        """
        for i, mag in enumerate(mags):
            assert mag >= 0 and mag <= 1, ("Invalid magnitude: %d, must be within interval [0,1]" % mag)
            self.Waves[i].Magnitude = mag
        self.Latest = False


    def set_phase(self, idx, phase):
        """
            Sets the magnitude of the indexed trap number.
            INPUTS:
                idx --- Index to trap number, starting from 0
                phase - New value for phase.
        """
        phase = phase % (2*pi)
        self.Waves[idx].Phase = phase
        self.Latest = False


    def set_phase_all(self, phases):
        """
            Sets the magnitude of all traps.
            INPUTS:
                mags - List of new phases, in order of Trap Number (Ascending Frequency).
        """
        for i, phase in enumerate(phases):
            self.Waves[i].Phase = phase
        self.Latest = False

    def plot(self):
        """
            Plots the Segment. Computes first if necessary.
        """
        if not self.Latest:
            self.compute()
        plt.plot(self.Buffer, '--o')
        plt.show()


    def randomize(self):
        """
            Randomizes each phase.
        """
        for w in self.Waves:
            w.Phase = 2*pi*random.random()
        self.Latest = False


    def compute(self):
        """
            Computes the superposition of frequencies
            and stores it in the buffer.
            We divide by the sum of relative wave magnitudes
            and scale the max value to SAMP_VAL_MAX,
            s.t. if all waves phase align, they will not exceed the max value.
        """
        ## Checks if Redundant ##
        if self.Latest:
            return
        self.Latest = True ## Will be up to date after

        ## Initialize Buffer ##
        self.Buffer = np.zeros(self.SampleLength, dtype=np.int16)
        temp_buffer = np.zeros(self.SampleLength)

        ## Compute and Add the full wave, Each frequency at a time ##
        for w in self.Waves:
            fn = w.Frequency / SAMP_FREQ  # Cycles/Sample
            for i in range(self.SampleLength):
                temp_buffer[i] += w.Magnitude*sin(2 * pi * i * fn + w.Phase)

        ## Normalize the Buffer ##
        normalization = sum([w.Magnitude for w in self.Waves])
        for i in range(self.SampleLength):
            self.Buffer[i] = int(SAMP_VAL_MAX * (temp_buffer[i] / normalization))


    def save(self, name="unamed_segment", data_only=False):
        if data_only:
            np.savetxt(name, self.Buffer, delimiter=",")
        else:
            pickle.dump(self, open(name, "wb"))

    def __str__(self):
        s = "Segment with Resolution: " + str(SAMP_FREQ / self.SampleLength) + "\n"
        s += "Contains Waves: \n"
        for w in self.Waves:
           s += "---" + str(w.Frequency) + "Hz - Magnitude: " \
                + str(w.Magnitude) + " - Phase: " + str(w.Phase) + "\n"
        return s