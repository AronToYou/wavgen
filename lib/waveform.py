import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Slider
from multiprocessing.sharedctypes import RawArray
from math import pi, sin, cosh, tanh
from ctypes import c_int16
from time import time
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
PLOT_MAX = int(1E4)   # Maximum number of data-points to plot at once
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
    def __init__(self, n, sample_length, buffer, args):
        """
            Multiple constructors in one.
            INPUTS:
                freqs ------ A list of frequency values, from which wave objects are automatically created.
                waves ------ Alternative to above, a list of pre-constructed wave objects could be passed.
            == OPTIONAL ==
                resolution ---- Either way, this determines the...resolution...and thus the sample length.
                sample_length - Overrides the resolution parameter.
        """
        super().__init__(daemon=True)
        waves, targets = args
        self.Portion = n
        self.SampleLength = sample_length
        self.Buffer = buffer
        self.Waves = waves
        self.Targets = targets

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
            self.Buffer[i] = c_int16(int(SAMP_VAL_MAX * (temp_buffer[i] / normalization))).value


######### Waveform Class #########
class Waveform:
    """
        MEMBER VARIABLES:
            + Latest --- Boolean indicating if the Buffer is the correct computation (E.g. correct Magnitude/Phase)
            + Filename -
            + Filed ----

        USER METHODS:
            + plot() -------------- Plots the segment via matplotlib. Computes first if necessary.
            + compute_and_save
        PRIVATE METHODS:
            + __str__() --- Defines behavior for --> print(*Segment Object*)
    """
    def __init__(self, sample_length, filename=None):
        """
            Multiple constructors in one.
            INPUTS:
                freqs ------ A list of frequency values, from which wave objects are automatically created.
                waves ------ Alternative to above, a list of pre-constructed wave objects could be passed.
            == OPTIONAL ==
                resolution ---- Either way, this determines the...resolution...and thus the sample length.
                sample_length - Overrides the resolution parameter.
        """
        self.SampleLength = (sample_length - sample_length % 32)
        self.Latest       = False
        self.Filename     = filename
        self.Filed        = False

    def compute_and_save(self):
        pass

    def __compute_and_save(self, f, seg, *args):
        """ Computes the superposition of frequencies
            and stores it to an .h5py file.

        """
        ## Open h5py File ##
        dset = f.create_dataset('data', shape=(self.SampleLength,), dtype='int16')

        ## Setup Parallel Processing ##
        procs = []
        N = int(self.SampleLength//(DATA_MAX + 1)) + 1
        print("N: ", N)
        n = 0
        start_time = time()
        while n != N:
            for _ in range(CPU_MAX):
                buf_size = min(DATA_MAX, int(self.SampleLength - n * DATA_MAX))
                buffer = RawArray(c_int16, buf_size)

                p = seg(n, self.SampleLength, buffer, args)
                procs.append(p)
                p.start()
                n += 1
                if n == N:
                    break
            print("Running N = %d to %d" % (n - len(procs) + 1, n))

            for i in range(n - len(procs), n):
                p = procs.pop(0)
                p.join()

                j = i*DATA_MAX
                dset[j:j + len(p.Buffer)] = p.Buffer
                del p
            print("%d%c" % (int(n / N * 100), '%'))
        bytes_per_sec = self.SampleLength*2//(time() - start_time)
        print("Average Rate: %d bytes/second" % bytes_per_sec)

        ## Wrapping things Up ##
        self.Latest = True  # Will be up to date after
        self.Filed = True
        f.close()

    def load(self, buf, buf_start, buf_size) -> None:
        pass

    def plot(self):
        """ Plots the Segment. Computes first if necessary.

        """
        N = min(PLOT_MAX, self.SampleLength)
        xdat = np.arange(N)
        ydat = np.zeros(N, dtype='int16')
        self.load(ydat, 0, N)

        ## Figure Creation ##
        fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))

        ax1.set(facecolor='#FFFFCC')
        line1, = ax1.plot(xdat, ydat, '-')
        ax1.set_title('Use slider to scroll top plot')

        ax2.set(facecolor='#FFFFCC')
        line2, = ax2.plot(xdat, ydat, '-')
        ax2.set_title('Click & Drag on top plot to zoom in lower plot')

        ## Slider ##
        def scroll(value):
            offset = int(value)
            xscrolled = np.arange(offset, offset + N)
            self.load(ydat, offset, N)

            line1.set_data(xscrolled, ydat)
            ax1.set_xlim(xscrolled[0], xscrolled[-1])
            fig.canvas.draw()

        axspar = plt.axes([0.14, 0.94, 0.73, 0.05])
        slid = Slider(axspar, 'Scroll', valmin=0, valmax=self.SampleLength - N, valinit=0, valfmt='%d', valstep=10)
        slid.on_changed(scroll)

        ## Span Selector ##
        def onselect(xmin, xmax):
            xzoom = np.arange(int(slid.val), int(slid.val) + N)
            indmin, indmax = np.searchsorted(xzoom, (xmin, xmax))
            indmax = min(N - 1, indmax)

            thisx = xdat[indmin:indmax]
            thisy = ydat[indmin:indmax]
            line2.set_data(thisx, thisy)
            ax2.set_xlim(thisx[0], thisx[-1])
            fig.canvas.draw()

        _ = SpanSelector(ax1, onselect, 'horizontal', useblit=True, rectprops=dict(alpha=0.5, facecolor='red'))

        plt.show()

    def _get_filename(self):
        """ Checks for a filename,
            otherwise asks for one.
            Exits if necessary.

        """
        ## Checks if Redundant ##
        if self.Latest:
            return False

        ## Check for File duplicate ##
        if self.Filename is None:
            self.Filename = easygui.enterbox("Enter a filename:", "Input")
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

        return True

    def __str__(self) -> str:
        pass


class Superposition(Waveform):
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
            sample_length = int(sample_length)
            resolution = SAMP_FREQ / sample_length
        else:
            assert resolution < SAMP_FREQ / 2, ("Invalid Resolution, has to be below Nyquist: %d" % (SAMP_FREQ / 2))
            sample_length = int(SAMP_FREQ / resolution)

        assert freqs[-1] >= resolution, ("Frequency %d is smaller than Resolution %d." % (freqs[-1], resolution))
        assert freqs[0] < SAMP_FREQ_MAX / 2, ("Frequency %d must below Nyquist: %d" % (freqs[0], SAMP_FREQ / 2))

        ## Initialize ##
        self.Waves        = [Wave(f) for f in freqs]
        self.Targets      = np.zeros(len(freqs), dtype='i8') if targets is None else np.array(targets, dtype='i8')
        super().__init__(sample_length, filename=filename)

    def compute_and_save(self):
        """ Computes the superposition of frequencies
            and stores it to an .h5py file.

        """
        if not self._get_filename():
            return

        ## Open h5py File ##
        F = h5py.File(self.Filename, "w")
        F.create_dataset('frequencies', data=np.array([w.Frequency for w in self.Waves]))
        F.create_dataset('targets', data=np.array(self.Targets))
        F.create_dataset('magnitudes', data=np.array([w.Magnitude for w in self.Waves]))
        F.create_dataset('phases', data=np.array([w.Phase for w in self.Waves]))

        self.__compute_and_save(F, Segment, self.Waves, self.Targets)

    def load(self, buf, buf_start, buf_size):
        if self.Filename is not None:
            assert self.Latest and self.Filed, "Needs to be computed first!"
            with h5py.File(self.Filename, "r") as f:
                for i, dat in enumerate(f.get('data')[buf_start:buf_start+buf_size]):
                    buf[i] = dat
        else:
            temp_buffer = np.zeros(buf_size, dtype=float)
            normalization = sum([w.Magnitude for w in self.Waves])

            ## For each Wave (& potentially sweep target)
            for w, t in zip(self.Waves, self.Targets):
                f = w.Frequency
                phi = w.Phase
                mag = w.Magnitude

                fn = f / SAMP_FREQ  # Cycles/Sample
                df = (t - f) / (SAMP_FREQ*self.SampleLength) if t else 0

                ## Compute the Wave ##
                for i in range(buf_size):
                    n = buf_start + i
                    dfn = df * n
                    temp_buffer[i] += mag * sin(2*pi*n*(fn + dfn) + phi)

            ## Normalize the Buffer ##
            for i in range(buf_size):
                buf[i] = c_int16(int(SAMP_VAL_MAX * (temp_buffer[i] / normalization))).value

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


######### SuperpositionFromFile Class #########
class SuperpositionFromFile(Superposition):
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
            self.set_phases(f['phases'])
            self.set_magnitudes(f['magnitudes'])
            self.Latest = True
            self.Filed = True


######### HS1Segment Class #########
class HS1Segment(mp.Process):
    """
        MEMBER VARIABLES:
            + PulseLength -- The duration of the modulation in samples.
            + CenterFreq - Average of the lower & upper detunings.
            + SweepWidth - Frequency width of modulation.
        USER METHODS:
            + run() - Computes the waveform.
        PRIVATE METHODS:
    """
    def __init__(self, n, sample_length, buffer, args):
        """
            Multiple constructors in one.
            INPUTS:
                n ---------- Index indicating which fraction of the full wave.
                pulse_time - Time in (ms) for the pulse modulation.
        """
        super().__init__(daemon=True)
        center_freq, sweep_width = args
        self.Portion = n
        self.SampleLength = sample_length
        self.Buffer = buffer
        self.CenterFreq = center_freq
        self.SweepWidth = sweep_width

    def run(self):
        fn = self.CenterFreq / SAMP_FREQ  # Cycles/Sample

        ## Compute the Wave ##
        for i in range(len(self.Buffer)):
            n = i + DATA_MAX*self.Portion

            d = 2*n/self.SampleLength
            dfn = self.SweepWidth*tanh(d)/(2*SAMP_FREQ)

            self.Buffer[i] = c_int16(int(SAMP_VAL_MAX*sin(2*pi*n*(fn + dfn))/cosh(d))).value


class HS1(Waveform):

    def __init__(self, pulse_time, center_freq, sweep_width, filename=None):
        sample_length = int(SAMP_FREQ * pulse_time)
        super().__init__(sample_length, filename)

        self.CenterFreq = center_freq
        self.SweepWidth = sweep_width

    def compute_and_save(self):
        if not self._get_filename():
            return

        ## Open h5py File ##
        F = h5py.File(self.Filename, "w")
        F.create_dataset('parameters', data=np.array([self.CenterFreq, self.SweepWidth], dtype='int32'))

        self.__compute_and_save(F, HS1Segment, self.CenterFreq, self.SweepWidth)

    def load(self, buf, seg_start, seg_size):
        if self.Filename is not None:
            assert self.Latest and self.Filed, "Needs to be computed first!"
            with h5py.File(self.Filename, "r") as f:
                for i, dat in enumerate(f.get('data')[seg_start:seg_start + seg_size]):
                    buf[i] = dat
        else:
            fn = self.CenterFreq / SAMP_FREQ  # Cycles/Sample
            for i in range(seg_size):
                n = i + seg_start

                d = 2 * n / self.SampleLength
                dfn = self.SweepWidth * tanh(d) / (2 * SAMP_FREQ)

                buf[i] = c_int16(SAMP_VAL_MAX * sin(2 * pi * n * (fn + dfn)) / cosh(d)).value


######### HS1FromFile Class #########
class HS1FromFile(HS1):
    """ This class just provides a clean way to construct Segment objects
        from saved files.
        It shares all of the same characteristics as a Segment.
    """
    def __init__(self, filename):
        with h5py.File(filename, 'r') as f:
            params = f.get('parameters')[()]
            sampL = f['data'].shape[0]
            pulse_time = sampL / SAMP_FREQ
            self.CenterFreq = params[0]
            self.SweepWidth = params[1]

            super().__init__(pulse_time, params[0], params[1], filename=filename)
            self.Latest = True
            self.Filed = True
