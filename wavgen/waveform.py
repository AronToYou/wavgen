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
    def __init__(self, sample_length):
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

        with h5py.File(filename, 'w') as F:
            self.config_file(F)
            F.attrs.create('class', data=self.__class__, dtype='int32')

            ## Setup Parallel Processing ##
            N = ceil(self.SampleLength / DATA_MAX)    # Number of Child Processes
            print("N: ", N)
            q = mp.Queue()                            # Child Process results Queue
            start_time = time()                       # Timer

            ## Initialize each CPU w/ a Process ##
            for p in range(min(CPU_MAX, N)):
                mp.Process(target=self.compute, args=(p, q)).start()

            ## Collect Validation & Start Remaining Processes ##
            for p in tqdm(range(N)):
                n, rslts = q.get()                  # Collects a Result

                i = n * DATA_MAX                    # Writes it to Disk
                for dset, data in rslts:
                    F[dset][i:i + len(data)] = data

                if p < N - CPU_MAX:                 # Starts a new Process
                    mp.Process(target=self.compute, args=(p + CPU_MAX, q)).start()

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

    def _check_filename(self, filename):
        """ Checks for a filename,
            otherwise asks for one.
            Exits if necessary.

        """
        ## Check for File duplicate ##
        if filename is None:
            if self.Filename is None:
                filename = easygui.enterbox("Enter a filename (blank to abort):", "Input")
            else:
                return
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
        print(M, m)

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
                print(semifinals.shape)

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
    def __init__(self, freqs, resolution=1E6, sample_length=None, targets=None):
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
        super().__init__(sample_length)

    def compute(self, p, q):
        normalization = sum([w.Magnitude for w in self.Waves])
        N = min(DATA_MAX, self.SampleLength - p*DATA_MAX)

        ## Prepare Buffers ##
        temp_buffer = np.zeros(N, dtype=float)
        waveform = np.empty(N, dtype='int16')
        fs = np.empty((N, len(self.Waves)), dtype='float64')

        ## For each Pure Tone ##
        for j, (w, t) in enumerate(zip(self.Waves, self.Targets)):
            f = w.Frequency
            phi = w.Phase
            mag = w.Magnitude

            fn = f / SAMP_FREQ  # Cycles/Sample
            dfn_inc = (t - f) / (SAMP_FREQ * self.SampleLength) if t else 0

            ## Compute the Wave ##
            for i in range(N):
                n = i + p*DATA_MAX
                dfn = dfn_inc * n / 2  # Sweep Frequency shift
                temp_buffer[i] += mag * sin(2 * pi * n * (fn + dfn) + phi)
                fs[i][j] = (fn + dfn) * SAMP_FREQ / 1E6

        ## Normalize the Buffer ##
        for i in range(N):
            waveform[i] = int(SAMP_VAL_MAX * (temp_buffer[i] / normalization))

        ## Send the results to Parent ##
        dat = [('waveform', waveform), ('fs', fs)]
        q.put((p, dat))

    def config_file(self, h5py_f):
        """ Computes the superposition of frequencies
            and stores it to an .h5py file.

        """
        ## Meta-Data ##
        h5py_f.attrs.create('frequencies', data=np.array([w.Frequency for w in self.Waves]))
        h5py_f.attrs.create('targets', data=np.array(self.Targets))
        h5py_f.attrs.create('magnitudes', data=np.array([w.Magnitude for w in self.Waves]))
        h5py_f.attrs.create('phases', data=np.array([w.Phase for w in self.Waves]))

        ## Waveform Data ##
        h5py_f.create_dataset('waveform', shape=(self.SampleLength,), dtype='int16')
        h5py_f.create_dataset('fs', shape=(self.SampleLength, len(self.Waves)), dtype='float64')
        h5py_f['fs'].attrs.create('legend', data=["%.2fMHz" % (w.Frequency / 1E6) for w in reversed(self.Waves)])

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
            if f.get('targets') is None:
                targs = f.attrs.get('targets')
                freqs = f.attrs.get('frequencies')
                sampL = f['waveform'].shape[0]
            else:
                targs = f.get('targets')[()]
                freqs = f.get('frequencies')[()]
                sampL = f['data'].shape[0]

            super().__init__(freqs, sample_length=sampL, targets=targs)
            # self.set_phases(f['phases'])
            # self.set_magnitudes(f['magnitudes'])
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
        modulation = np.empty((N, 2), dtype=float)

        last_arg = 0

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

            freq_n = (arg - last_arg)*MHZ if last_arg else (self.Center - self.BW/2)*MHZ
            normed_amp = (amp*self.BW + self.Center - self.BW/2)*MHZ
            modulation[i] = [freq_n, normed_amp]
            last_arg = arg

        ## Send results to Parent ##
        dat = [('waveform', waveform), ('modulation', modulation)]
        q.put((p, dat))

    def config_file(self, h5py_f):
        ## Meta-Data ##
        h5py_f.attrs.create('Tau', data=self.Tau)
        h5py_f.attrs.create('Center', data=self.Center)
        h5py_f.attrs.create('BW', data=self.BW)

        ## Waveform Data ##
        h5py_f.create_dataset('waveform', shape=(self.SampleLength,), dtype='int16')
        h5py_f.create_dataset('modulation', shape=(self.SampleLength, 2), dtype='float32')
        h5py_f['modulation'].attrs.create('legend', data=['Instantaneous Frequency', 'Amplitude (normalized)'])
        h5py_f['modulation'].attrs.create('title', data='HS1 Pulse Parameters')
        h5py_f['modulation'].attrs.create('y_label', data='MHz')


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
            self.Center = params[0]
            self.BW = params[1]

            super().__init__(pulse_time, params[0], params[1])
            self.Latest = True
            self.Filed = True
