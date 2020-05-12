import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Slider
from math import pi, sin, cosh, ceil, log
from time import time
from tqdm import tqdm
from sys import maxsize
import random
import h5py
import easygui
import sys
import inspect


### Parameter ###
DATA_MAX = int(16E4)  # Maximum number of samples to hold in array at once
PLOT_MAX = int(1E4)   # Maximum number of data-points to plot at once
SAMP_FREQ = 1000E6

### Constants ###
SAMP_VAL_MAX = (2 ** 15 - 1)    # Maximum digital value of sample ~~ signed 16 bits
SAMP_FREQ_MAX = 1250E6          # Maximum Sampling Frequency
CPU_MAX = mp.cpu_count()
MHZ = SAMP_FREQ / 1E6           # Coverts samples/seconds to MHz


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
    def __init__(self, sample_length, normed=True):
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
        self.PlotObjects  = []
        self.Latest       = False
        self.Filename     = None
        self.Normed       = normed

    ## PUBLIC FUNCTIONS ##

    def compute(self, p, q):
        N = min(DATA_MAX, self.SampleLength)
        wav = np.empty(N)
        q.put((p, [('waveform', wav)]))

    def config_file(self, h5py_f):
        ## Waveform Data ##
        h5py_f.create_dataset('waveform', shape=(self.SampleLength,), dtype='int16')

        ## OPTIONAL ##
        # h5py_f.create_dataset('modulation', shape=(self.SampleLength, 2), dtype='float32')
        # h5py_f['modulation'].attrs.create('legend', data=['Legendary'])
        # h5py_f['modulation'].attrs.create('title', data='Wicked Waves')
        # h5py_f['modulation'].attrs.create('y_label', data='Mega-Hurts')

        ## Meta-Data ##
        # h5py_f.attrs.create('attribute', data=self.SampleLength)

    @classmethod
    def from_file(cls, *attrs):
        ## Pre-process the args for Constructor use ##
        args = attrs
        return cls(*args)

    def compute_and_save(self, filename=None):
        """
            INPUTS:
                f ---- An h5py.File object where waveform is written
                seg -- The waveform subclass segment object
                args - Waveform specific arguments.
        """
        ## Redundancy & Input check ##
        if self.Latest:
            return
        self._check_filename(filename)

        start_time = time()  # Timer
        with h5py.File(filename, 'w') as F:
            self.config_file(F)  # Setup File Attributes

            ## Compute the Waveform ##
            if self.Normed:
                self._parallelize(F, self.compute)
            ## Unless Normalization is Required ##
            else:
                ## Then Compute the Un-normalized Waveform first ##
                with h5py.File('temp.h5', 'w') as T:
                    T.create_dataset('waveform', shape=(self.SampleLength,), dtype=float)
                    self._parallelize(T, self.compute)

                    ## Retrieve the Normalization Factor ##
                    wav = T['waveform'][()]
                    norm = max(wav.max(), abs(wav.min()))

                ## Then Normalize ##
                F['waveform'][()] = np.multiply(np.divide(wav, norm), SAMP_VAL_MAX).astype(np.int16)

        bytes_per_sec = self.SampleLength*2//(time() - start_time)
        print("Average Rate: %d bytes/second" % bytes_per_sec)

        ## Wrapping things Up ##
        self.Latest = True  # Will be up to date after
        self.Filename = filename

    def plot(self):
        """ Plots the Segment. Computes first if necessary.

        """
        assert self.Filename is not None, "Must save waveform to file first!"
        ## Don't plot if already plotted ##
        if len(self.PlotObjects):
            return

        ## Retrieve the names of each Dataset ##
        with h5py.File(self.Filename, 'r') as f:
            dsets = list(f.keys())
            if f.get('waveform') is None:
                dsets = ['data']

        ## Plot each Dataset ##
        for dset in dsets:
            self.PlotObjects.append(self._plot_span(dset))

    ## PRIVATE FUNCTIONS ##

    def _parallelize(self, f, func):
        ## Setup Parallel Processing ##
        N = ceil(self.SampleLength / DATA_MAX)  # Number of Child Processes
        print("N: ", N)
        q = mp.Queue()  # Child Process results Queue

        ## Initialize each CPU w/ a Process ##
        for p in range(min(CPU_MAX, N)):
            mp.Process(target=func, args=(p, q)).start()

        ## Collect Validation & Start Remaining Processes ##
        for p in tqdm(range(N)):
            n, data = q.get()  # Collects a Result

            i = n * DATA_MAX  # Shifts to Proper Interval

            f['waveform'][i:i + len(data)] = data  # Writes to Disk

            if p < N - CPU_MAX:  # Starts a new Process
                mp.Process(target=func, args=(p + CPU_MAX, q)).start()

    def _check_filename(self, filename):
        """ Checks for a filename,
            otherwise asks for one.
            Exits if necessary.

        """
        ## Check for File duplicate ##
        if filename is None:
            pass
        while True:
            if filename is None:
                exit(-1)
            try:
                F = h5py.File(filename, 'r')
                if easygui.boolbox("Overwrite existing file?"):
                    F.close()
                    return
                filename = easygui.enterbox("Enter a filename (blank to abort):", "Input")
            except OSError:
                return

    def _plot_span(self, dset):
        N = min(PLOT_MAX, self.SampleLength)
        with h5py.File(self.Filename, 'r') as f:
            legend = f[dset].attrs.get('legend')
            title = f[dset].attrs.get('title')
            y_label = f[dset].attrs.get('y_label')
            dtype = f[dset].dtype

        shape = N if legend is None else (N, len(legend))
        M, m = self._y_limits(dset)

        xdat = np.arange(N)
        ydat = np.zeros(shape, dtype=dtype)
        self._load(dset, ydat, 0)

        ## Figure Creation ##
        fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))

        ax1.set(facecolor='#FFFFCC')
        lines = ax1.plot(xdat, ydat, '-')
        ax1.set_ylim((m, M))
        ax1.set_title(title if title else 'Use slider to scroll top plot')

        ax2.set(facecolor='#FFFFCC')
        ax2.plot(xdat, ydat, '-')
        ax2.set_ylim((m, M))
        ax2.set_title('Click & Drag on top plot to zoom in lower plot')

        if legend is not None:
            ax1.legend(legend)
        if y_label is not None:
            ax1.set_ylabel(y_label)
            ax2.set_ylabel(y_label)

        ## Slider ##
        def scroll(value):
            offset = int(value)
            xscrolled = np.arange(offset, offset + N)
            self._load(dset, ydat, offset)

            if len(lines) > 1:
                for line, y in zip(lines, ydat.transpose()):
                    line.set_data(xscrolled, y)
            else:
                lines[0].set_data(xscrolled, ydat)

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

            ax2.clear()
            ax2.plot(thisx, thisy)
            ax2.set_xlim(thisx[0], thisx[-1])
            fig.canvas.draw()

        span = SpanSelector(ax1, onselect, 'horizontal', useblit=True, rectprops=dict(alpha=0.5, facecolor='red'))

        plt.show(block=False)

        return fig, span

    def _load(self, dset, buf, offset):
        with h5py.File(self.Filename, "r") as f:
            buf[:] = f.get(dset)[offset:offset + len(buf)]

    def _y_limits(self, dset):
        with h5py.File(self.Filename, 'r') as f:
            N = f.get(dset).shape[0]
            loops = ceil(N/DATA_MAX)

            semifinals = np.empty((loops, 2), dtype=f.get(dset).dtype)

            for i in range(loops):
                n = i*DATA_MAX

                dat = f.get(dset)[n:min(n + DATA_MAX, N)]
                semifinals[i][:] = [dat.max(), dat.min()]

        M = semifinals.transpose()[:][0].max()
        m = semifinals.transpose()[:][1].min()
        margin = max(abs(M), abs(m)) * 1E-15
        return M-margin, m+margin

    ## SPECIAL FUNCTIONS ##

    def __str__(self) -> str:
        pass


######### Subclasses ############
class Wave:
    """ Describes a Sin wave. """
    def __init__(self, freq, mag=1, phase=0):
        """
            Constructor.

            Parameters
            ----------
            freq
                Frequency of the wave.

            mag
                Magnitude within [0,1]

            phase
                Initial phase of oscillation.

        """
        ## Validate ##
        assert freq > 0, ("Invalid Frequency: %d, must be positive" % freq)
        assert 0 <= mag <= 1, ("Invalid magnitude: %d, must be within interval [0,1]" % mag)
        ## Initialize ##
        self.Frequency = int(freq)
        self.Magnitude = mag
        self.Phase = phase

    def __lt__(self, other):
        return self.Frequency < other.Frequency


######### Superposition Class #########
class Superposition(Waveform):
    """
        A static trap configuration.
    """
    def __init__(self, freqs, mags=None, phases=None, resolution=1E6, sample_length=None):
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

        if mags is None:
            mags = np.ones(len(freqs))
        if phases is None:
            phases = np.zeros(len(freqs))

        if sample_length is not None:
            sample_length = int(sample_length)
            resolution = SAMP_FREQ / sample_length
        else:
            assert resolution < SAMP_FREQ / 2, ("Invalid Resolution, has to be below Nyquist: %d" % (SAMP_FREQ / 2))
            sample_length = int(SAMP_FREQ / resolution)

        assert freqs[-1] >= resolution, ("Frequency %d is smaller than Resolution %d." % (freqs[-1], resolution))
        assert freqs[0] < SAMP_FREQ_MAX / 2, ("Frequency %d must below Nyquist: %d" % (freqs[0], SAMP_FREQ / 2))
        assert len(mags) == len(freqs) == len(phases), "Parameter size mismatch!"

        ## Initialize ##
        self.Waves = [Wave(f, m, p) for f, m, p in zip(freqs, mags, phases)]
        super().__init__(sample_length)

        self.Normed = False  # Needs to be Normalized

    def compute(self, p, q):
        N = min(DATA_MAX, self.SampleLength - p*DATA_MAX)
        waveform = np.zeros(N, dtype=float)

        ## For each Pure Tone ##
        for j, w in enumerate(self.Waves):
            f = w.Frequency
            phi = w.Phase
            mag = w.Magnitude

            fn = f / SAMP_FREQ  # Cycles/Sample
            ## Compute the Wave ##
            for i in range(N):
                n = i + p*DATA_MAX
                waveform[i] += mag * sin(2 * pi * n * fn + phi)

        ## Send the results to Parent ##
        q.put((p, waveform))

    def config_file(self, h5py_f):
        """ Computes the superposition of frequencies
            and stores it to an .h5py file.

        """
        #### Meta-Data ####
        ## Table of Contents ##
        h5py_f.attrs.create('class', data=self.__class__.__name__)
        h5py_f.attrs.create('attrs', data=['freqs', 'mags', 'phases'])

        ## Contents ##
        h5py_f.attrs.create('freqs', data=np.array([w.Frequency for w in self.Waves]))
        h5py_f.attrs.create('mags', data=np.array([w.Magnitude for w in self.Waves]))
        h5py_f.attrs.create('phases', data=np.array([w.Phase for w in self.Waves]))

        ## Waveform Data ##
        h5py_f.create_dataset('waveform', shape=(self.SampleLength,), dtype='int16')

    @classmethod
    def from_file(cls, *attrs):
        freqs, mags, phases, sample_length = attrs
        return cls(freqs, mags, phases, sample_length=sample_length)

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


class SupFromFile(Superposition):
    """ This class just provides a clean way to construct Waveform objects
        from saved files.
        It shares all of the same characteristics as a Waveform.
    """
    def __init__(self, filename):
        with h5py.File(filename, 'r') as f:
            freqs = f.attrs.get('frequencies')
            sample_length = f['waveform'].shape[0]
            super().__init__(freqs, sample_length=sample_length)

            self.set_magnitudes(f.attrs.get('magnitudes'))
            self.set_phases(f.attrs.get('phases'))

        self.Filename = filename
        self.Latest = True
        self.Filed = True


######## Sweep Class ########
class Sweep(Waveform):
    def __init__(self, config_a, config_b, sweep_time=None, sample_length=16E6):
        assert isinstance(config_a, Superposition) and isinstance(config_b, Superposition)
        assert len(config_a.Waves) == len(config_b.Waves)

        if sweep_time is not None:
            sample_length = SAMP_FREQ*sweep_time

        self.WavesA = config_a.Waves
        self.WavesB = config_b.Waves
        super().__init__(sample_length)

        self.Normed = False  # Needs to be Normalized

    def compute(self, p, q):
        N = min(DATA_MAX, self.SampleLength - p*DATA_MAX)

        ## Prepare Buffers ##
        waveform = np.empty(N, dtype=float)

        ## For each Pure Tone ##
        for j, (a, b) in enumerate(zip(self.WavesA, self.WavesB)):
            fn = a.Frequency / SAMP_FREQ  # Cycles/Sample
            dfn_inc = (b.Frequency - a.Frequency) / (SAMP_FREQ * self.SampleLength)

            phi = a.Phase
            phi_inc = (b.Phase - phi) / self.SampleLength

            mag = a.Magnitude
            mag_inc = (b.Magnitude - mag) / self.SampleLength

            ## Compute the Wave ##
            for i in range(N):
                n = i + p*DATA_MAX
                dfn = dfn_inc * n / 2  # Sweep Frequency shift
                waveform[i] += (mag + n*mag_inc) * sin(2 * pi * n * (fn + dfn) + (phi + n*phi_inc))

        ## Send the results to Parent ##
        q.put((p, waveform))

    def config_file(self, h5py_f):
        """ Computes the superposition of frequencies
            and stores it to an .h5py file.

        """
        #### Meta-Data ####
        ## Table of Contents ##
        h5py_f.attrs.create('class', data=self.__class__.__name__)
        h5py_f.attrs.create('attrs', data=['freqsA', 'magsA', 'phasesA', 'freqsB', 'magsB', 'phasesB'])

        ## Contents ##
        h5py_f.attrs.create('freqsA', data=np.array([w.Frequency for w in self.WavesA]))
        h5py_f.attrs.create('magsA', data=np.array([w.Magnitude for w in self.WavesA]))
        h5py_f.attrs.create('phasesA', data=np.array([w.Phase for w in self.WavesA]))

        h5py_f.attrs.create('freqsB', data=np.array([w.Frequency for w in self.WavesB]))
        h5py_f.attrs.create('magsB', data=np.array([w.Magnitude for w in self.WavesB]))
        h5py_f.attrs.create('phasesB', data=np.array([w.Phase for w in self.WavesB]))

        ## Waveform Data ##
        h5py_f.create_dataset('waveform', shape=(self.SampleLength,), dtype='int16')

    @classmethod
    def from_file(cls, *attrs):
        freqsA, magsA, phasesA, freqsB, magsB, phasesB, sample_length = attrs
        supA = Superposition(freqsA, magsA, phasesA, sample_length=sample_length)
        supB = Superposition(freqsB, magsB, phasesB, sample_length=sample_length)

        return cls(supA, supB, sample_length=sample_length)


class SwpFromFile(Sweep):
    """ This class just provides a clean way to construct Waveform objects
        from saved files.
        It shares all of the same characteristics as a Waveform.
    """
    def __init__(self, filename):
        with h5py.File(filename, 'r') as f:
            sample_length = f['waveform'].shape[0]

            A = Superposition(f.attrs.get('frequenciesA'))
            A.set_magnitudes(f.attrs.get('magnitudesA'))
            A.set_phases(f.attrs.get('phasesA'))

            B = Superposition(f.attrs.get('frequenciesB'))
            B.set_magnitudes(f.attrs.get('magnitudesB'))
            B.set_phases(f.attrs.get('phasesB'))

            super().__init__(A, B, sample_length=sample_length)

        self.Filename = filename
        self.Latest = True
        self.Filed = True


######### HS1 Class #########
class HS1(Waveform):
    def __init__(self, pulse_time, center_freq, sweep_width, duration=None):
        if duration:
            sample_length = int(SAMP_FREQ * duration)
        else:
            sample_length = int(SAMP_FREQ * pulse_time * 7)
        super().__init__(sample_length)

        self.Tau = pulse_time * SAMP_FREQ
        self.Center = center_freq / SAMP_FREQ
        self.BW = sweep_width / SAMP_FREQ

    def compute(self, p, q):
        N = min(DATA_MAX, self.SampleLength - p*DATA_MAX)
        waveform = np.empty(N, dtype='int16')

        ## Compute the Wave ##
        for i in range(N):
            n = i + p*DATA_MAX

            d = 2*(n - self.SampleLength/2)/self.Tau  # 2t/tau

            try:
                arg = n * self.Center + (self.Tau * self.BW / 4) * log(cosh(d))
                amp = 1 / cosh(d)
            except OverflowError:
                arg = n * self.Center + (self.Tau * self.BW / 4) * log(maxsize)
                amp = 0

            waveform[i] = int(SAMP_VAL_MAX * amp * sin(2 * pi * arg))

        ## Send results to Parent ##
        q.put((p, waveform))

    def config_file(self, h5py_f):
        #### Meta-Data ####
        ## Table of Contents
        h5py_f.attrs.create('class', data=self.__class__.__name__)
        h5py_f.attrs.create('attrs', data=['pulse_time', 'center_freq', 'sweep_width'])

        ## Contents ##
        h5py_f.attrs.create('pulse_time', data=self.Tau / SAMP_FREQ)
        h5py_f.attrs.create('center_freq', data=self.Center * SAMP_FREQ)
        h5py_f.attrs.create('sweep_width', data=self.BW * SAMP_FREQ)

        ## Waveform Data ##
        h5py_f.create_dataset('waveform', shape=(self.SampleLength,), dtype='int16')

    @classmethod
    def from_file(cls, *attrs):
        pulse_time, center_freq, sweep_width, sample_length = attrs

        return cls(pulse_time, center_freq, sweep_width, sample_length/SAMP_FREQ)


######### FromFile Class #########
def from_file(filename, path=None):
    """ This class just provides a clean way to construct Waveform objects
        from saved files.
        It shares all of the same characteristics as a Waveform.
    """
    classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)

    class_name = None
    attrs = []
    sample_length = 0
    with h5py.File(filename, 'r') as f:
        ## Maneuver to relevant Data location ##
        dat = f.get(path) if path else f

        assert dat is not None, "Invalid path"

        ## Waveform's Python Class name ##
        class_name = dat.attrs.get('class')
        sample_length = dat['waveform'].shape[0]

        ## Attributes ##
        for attr in dat.attrs.get('attrs'):
            attrs.append(dat.attrs.get(attr))
        attrs.append(sample_length)  # Including the sample_length

    obj = None
    ## Find the proper Class & Construct it ##
    for name, cls in classes:
        if class_name == name:
            obj = cls.from_file(*attrs)
            break
    if obj is None:
        obj = Waveform()

    ## Indicates already saved to File ##
    obj.Latest = True
    obj.Filed = True
    obj.Filename = filename

    return obj
