import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Slider
from math import pi, sin, cosh, ceil, log
from .utilities import Wave
from sys import maxsize
from time import time
from tqdm import tqdm
from .config import *
import warnings
import inspect
import easygui
import random
import h5py
import sys
import os


## Suppresses matplotlib.widgets.Slider warning ##
warnings.filterwarnings("ignore", category=UserWarning)


######### Waveform Class #########
class Waveform:
    """
    Basic Waveform object.

    Note
    ----
    All other defined waveform objects (below) extend *this* class.

    Attributes
    ----------
    cls.OpenTemps : list of int, **Class object**
        Tracks the number of Waveforms not explicitly saved to file. (temporarily saved)
        Necessary because, even if not explicitly asked to save to file, the system
        employs temporary files which make handling any sized waveform simple.
    SampleLength : int
        How long the waveform is in 16-bit samples.
    Amplitude : float
        Fraction of the maximum card output voltage,
        to which the waveform is normalized to.
        (AKA relative amplitude between other `Waveform` objects)
    PlotObjects : list
        List of matplotlib objects, so that they aren't garbage collected.
    Latest : bool
        Indicates if the data reflects the most recent waveform definition.
    Filename : str
        The name of the file where the waveform is saved.
    Path : str
        The HDF5 pathway to where **this** waveform's root exists.
        Used in the case where a single HDF5 file contains a database
        of several waveforms (Very efficient space wise).
    """
    OpenTemps = 0

    def __init__(self, sample_length, amp=1.0):
        """
        Parameters
        ----------
        sample_length : int
            Sets the `SampleLength`.
        amp : float, optional
            Amplitude of waveform relative to maximum output voltage.
        """
        self.SampleLength = (sample_length - sample_length % 32)
        self.Amplitude    = amp
        self.PlotObjects  = []
        self.Latest       = False
        self.Filename     = None
        self.Path         = ''

    def __del__(self):
        """Deletes the temporary file used by unsaved (temporary) Waveforms."""
        if self.Filename == 'temporary.h5':
            Waveform.OpenTemps -= 1
            if Waveform.OpenTemps == 0:
                os.remove('temporary.h5')
                os.remove('temp.h5')

    def compute(self, p, q):
        """ Calculates the *p*\ th portion of the entire waveform.

        Note
        ----
        This is the function dispatched to :ref:`parallel processes <parallel>`.
        The *p* argument indicates the interval of the whole waveform to calculate.

        Parameters
        ----------
        p : int
            Index, **starting from 0**, indicating which interval of the whole waveform
            should be calculated. Intervals are size :const:`DATA_MAX` in samples.
        q : :obj:`multiprocessing.Queue`
            A *Queue* object shared by multiple processes.
            Each process places there results here once done,
            to be collected by the parent process.
        """
        N = min(DATA_MAX, self.SampleLength - p*DATA_MAX)
        wav = np.empty(N)
        q.put((p, wav))

    def config_file(self, h5py_f):
        ## Necessary to determine subclass when loading from file ##
        h5py_f.attrs.create('class', data=self.__class__.__name__)

        ## Waveform Data ##
        return h5py_f.create_dataset('waveform', shape=(self.SampleLength,), dtype='int16')

    @classmethod
    def from_file(cls, **kwargs):
        return cls(**kwargs)

    def compute_waveform(self, filename=False, group='', cpus=None):
        """
        Computes the waveform to disk.
        If no filename is given, then waveform data will be destroyed upon object cleanup.

        Parameters
        ----------
        filename : str, optional
            Searches for an HDF5 database file with the given name. If none exists, then one is created.
            Within the database, waveform will be saved to the dataset located at ``self.Path``.
        group : str, optional
            Describes a path in the HDF5 database to save the particular waveform dataset.
        cpus : int, optional
            Sets the desired number of CPUs to utilized for the calculation. Will round down if too
            large a number given.
        Note
        ----
        The `filename` parameter does not need to include a file-extension; only a name.
        """
        ## Redundancy & Input check ##
        write_mode = self._check_filename(filename, group)
        if write_mode is None:
            return

        ## Open HDF5 files for Writing ##
        with h5py.File(self.Filename, write_mode) as F:
            if self.Path != '':
                F = F.create_group(self.Path) if F.get(self.Path) is None else F.get(self.Path)
            wav = self.config_file(F)  # Setup File Attributes
            # TODO: Validate that this nesting of HDF5 files is safe.
            with h5py.File('temp.h5', 'w') as T:
                temp = T.create_dataset('waveform', shape=(self.SampleLength,), dtype=float)
                self._compute_waveform(wav, temp, cpus)

        ## Wrapping things Up ##
        self.Latest = True  # Will be up to date after

    def load(self, buffer, offset, size):
        """
        Loads a portion of the waveform.

        Parameters
        ----------
        buffer : numpy or h5py array
            Location to load data into.
        offset : int
            Offset from the waveforms beginning in samples.
        size : int
            How much waveform to load in samples.
        """
        if not self.Latest:
            self.compute_waveform()
        with h5py.File(self.Filename, 'r') as f:
            try:
                buffer[()] = f.get(self.Path + '/waveform')[offset:offset + size]
            except TypeError:
                dat = f.get(self.Path + '/waveform')[offset:offset + size]
                for i in range(size):
                    buffer[i] = dat[i]

    def plot(self):
        """
        Plots the Segment. Computes first if necessary.
        """
        if len(self.PlotObjects):  # Don't plot if already plotted
            return
        if not self.Latest:        # Compute before Plotting
            self.compute_waveform()

        ## Retrieve the names of each Dataset ##
        with h5py.File(self.Filename, 'r') as f:
            if self.Path != '':
                f = f.get(self.Path)
            dsets = list(f.keys())

        ## Plot each Dataset ##
        for dset in dsets:
            self.PlotObjects.append(self._plot_span(dset))

    def rms2(self):
        """ Calculates the Mean Squared value of the Waveform.

            Returns
            -------
            float
                Mean Squared sample value, normalized to be within [0, 1].
        """
        buf = np.empty(DATA_MAX, dtype=np.int64)
        rms2, so_far = 0, 0
        for i in range(self.SampleLength // DATA_MAX):
            self.load(buf, so_far, DATA_MAX)

            rms2 += buf.dot(buf) / self.SampleLength
            so_far += DATA_MAX

        remain = self.SampleLength % DATA_MAX
        buf = np.empty(remain, dtype=np.int64)
        self.load(buf, so_far, remain)

        return (rms2 + buf.dot(buf) / self.SampleLength) / (self.Amplitude * SAMP_VAL_MAX)**2

    ## PRIVATE FUNCTIONS ##

    def _compute_waveform(self, wav, temp, cpus):
        start_time = time()  # Timer

        ## Compute the Waveform ##
        self._parallelize(temp, self.compute, cpus)

        ## Determine the Normalization Factor ##
        norm = (SAMP_VAL_MAX * self.Amplitude) / max(temp[()].max(), abs(temp[()].min()))

        ## Then Normalize ##
        wav[()] = np.multiply(temp[()], norm).astype(np.int16)

        ## Wrapping things Up ##
        bytes_per_sec = self.SampleLength * 2 // (time() - start_time)
        print("Average Rate: %d bytes/second" % bytes_per_sec)

    def _parallelize(self, buffer, func, cpus):
        ## Number of Parallel Processes ##
        cpus_max = mp.cpu_count()
        cpus = min(cpus_max, cpus) if cpus else int(0.75*cpus_max)

        ## Total Processes to-do ##
        N = ceil(self.SampleLength / DATA_MAX)  # Number of Child Processes
        print("N: ", N)
        q = mp.Queue()  # Child Process results Queue

        ## Initialize each CPU w/ a Process ##
        for p in range(min(cpus, N)):
            mp.Process(target=func, args=(p, q)).start()

        ## Collect Validation & Start Remaining Processes ##
        for p in tqdm(range(N)):
            n, data = q.get()  # Collects a Result

            i = n * DATA_MAX  # Shifts to Proper Interval

            buffer[i:i + len(data)] = data  # Writes to Disk

            if p < N - cpus:  # Starts a new Process
                mp.Process(target=func, args=(p + cpus, q)).start()

    def _check_filename(self, filename, group):
        """ Checks filename for duplicate.
            In the case of no filename, configures the temporary file.

            Parameters
            ----------
            filename : str
                Potential name for HDF5 file to write to.
            group : str
                Potential path in HDF5 file hierarchy to write to.

            Returns
            -------
            char : {'a', 'w'}
                Returns a character corresponding to append or truncate
                mode on the file to write to.
        """
        if self.Latest and not (filename or group):
            return None
        while filename:
            filename, _ = os.path.splitext(filename)
            filename = filename + '.h5'
            try:
                F = h5py.File(filename, 'r')
                if self.Filename == filename or \
                        easygui.boolbox("Overwrite existing file?"):
                    break
                filename = easygui.enterbox("Enter a different filename (blank to abort):", "Input")
            except OSError:
                break

        if filename is None:
            exit(-1)
        elif not (filename or self.Filename):
            self.Filename = 'temporary.h5'
            self.Path = str(id(self))
            Waveform.OpenTemps += 1
        else:
            if filename:
                self.Filename = filename
            if group:
                self.Path = group

        return 'w' if self.Path == '' else 'a'

    def _plot_span(self, dset):
        N = min(PLOT_MAX, self.SampleLength)
        name = self.Path + '/' + dset
        with h5py.File(self.Filename, 'r') as f:
            legend = f[name].attrs.get('legend')
            title = f[name].attrs.get('title')
            y_label = f[name].attrs.get('y_label')
            dtype = f[name].dtype

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
        with h5py.File(self.Filename, 'r') as f:
            buf[:] = f.get(self.Path + '/' + dset)[offset:offset + len(buf)]

    def _y_limits(self, dset):
        name = self.Path + '/' + dset
        with h5py.File(self.Filename, 'r') as f:
            N = f.get(name).shape[0]
            loops = ceil(N/DATA_MAX)

            semifinals = np.empty((loops, 2), dtype=f.get(name).dtype)

            for i in range(loops):
                n = i*DATA_MAX

                dat = f.get(name)[n:min(n + DATA_MAX, N)]
                semifinals[i][:] = [dat.max(), dat.min()]

        M = semifinals.transpose()[:][0].max()
        m = semifinals.transpose()[:][1].min()
        margin = max(abs(M), abs(m)) * 1E-15
        return M-margin, m+margin


######### Subclasses ############
class Superposition(Waveform):
    """
        A static trap configuration.
    """
    def __init__(self, freqs, mags=None, phases=None, sample_length=int(16E3),
                 resolution=None, milliseconds=None, amp=1.0):
        """
            Parameters
            ----------
            freqs : list of int
                A list of frequency values, from which wave objects are automatically created.
            mags : list of float, optional
                Vector representing relative magnitude of each trap, within [0,1] (in order of increasing frequency).
            phases : list of float, optional
                Vector representing initial phases of each trap tone, within [0, 2*pi]
                (in order of increasing frequency).
            sample_length : int, optional
                Length of waveform in samples.
            resolution : int, optional
                **Overrides sample_length.** Sets the resolution of the waveform in Hertz.
            milliseconds : float, optional
                **Overrides sample_length.** Length of waveform in milliseconds.
            amp : float, optional
                Amplitude of waveform relative to maximum output voltage.
        """
        ## Validate & Sort ##
        freqs.sort()

        if mags is None:
            mags = np.ones(len(freqs))
        if phases is None:
            phases = np.zeros(len(freqs))

        if milliseconds:
            sample_length = int(SAMP_FREQ*milliseconds/1000)
        elif resolution:
            assert resolution < SAMP_FREQ / 2, ("Invalid Resolution, has to be below Nyquist: %d" % (SAMP_FREQ / 2))
            sample_length = int(SAMP_FREQ / resolution)

        assert freqs[-1]*sample_length >= SAMP_FREQ, "Frequency is below resolution. Increase sample length."
        assert freqs[0] < SAMP_FREQ / 2, ("Frequency %d must below Nyquist: %d" % (freqs[0], SAMP_FREQ / 2))
        assert len(mags) == len(freqs) == len(phases), "Parameter size mismatch!"

        ## Initialize ##
        self.Waves = [Wave(f, m, p) for f, m, p in zip(freqs, mags, phases)]
        super().__init__(sample_length, amp)

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
        ## Contents ##
        h5py_f.attrs.create('freqs', data=np.array([w.Frequency for w in self.Waves]))
        h5py_f.attrs.create('mags', data=np.array([w.Magnitude for w in self.Waves]))
        h5py_f.attrs.create('phases', data=np.array([w.Phase for w in self.Waves]))
        h5py_f.attrs.create('sample_length', data=self.SampleLength)

        ## Table of Contents ##
        h5py_f.attrs.create('keys', data=['freqs', 'mags', 'phases', 'sample_length'])

        return super().config_file(h5py_f)

    def get_magnitudes(self):
        """ Returns an array of magnitudes,
            each associated with a particular trap.
        """
        return [w.Magnitude for w in self.Waves]

    def set_magnitudes(self, mags):
        """ Sets the magnitude of each trap.

        Parameters
        ----------
        mags : list of float
            New magnitudes (**[0, 1]**) in order of Trap Number (Ascending Frequency).
        """
        for w, mag in zip(self.Waves, mags):
            assert 0 <= mag <= 1, ("Invalid magnitude: %d, must be within interval [0,1]" % mag)
            w.Magnitude = mag
        self.Latest = False

    def get_phases(self):
        return [w.Phase for w in self.Waves]

    def set_phases(self, phases):
        """ Sets the relative phase of each traps.

        Parameters
        ----------
        phases : list of float
            New phases (**radians**) in order of Trap Number (Ascending Frequency).

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


def even_spacing(ntraps, center, spacing, mags=None, phases=None, periods=1, amp=1.0):
    """ Wrapper function which makes defining equally spaced traps simple.

        Parameters
        ----------
        ntraps : int
            Number of optical traps.
        center : int
            Mean or center frequency of the traps.
        spacing : int
            Frequency spacing between traps.
        mags : list of float, optional
            Vector representing relative magnitude of each trap, within [0,1]
            (in order of increasing frequency).
        phases : list of float, optional
            Vector representing initial phases of each trap tone, within [0, 2*pi]
            (in order of increasing frequency).
        periods : int, optional
            Number of full periods of the entire waveform to calculate.
        amp : float, optional
            Amplitude of waveform relative to maximum output voltage.

        Returns
        -------
        :obj:`Superposition`
            Packages the input parameters into a :obj:`Superposition` object.

    """
    freqs = [int(center + spacing*(i - (ntraps-1)/2)) for i in range(ntraps)]
    N = int(SAMP_FREQ * (2 - ntraps % 2) // spacing) * periods

    return Superposition(freqs, mags=mags, phases=phases, sample_length=N, amp=amp)


######## Sweep Class ########
class Sweep(Waveform):
    def __init__(self, config_a, config_b, sweep_time=None, sample_length=int(16E6)):
        assert isinstance(config_a, Superposition) and isinstance(config_b, Superposition)
        assert len(config_a.Waves) == len(config_b.Waves)

        if sweep_time is not None:
            sample_length = int(SAMP_FREQ*sweep_time)

        self.WavesA = config_a.Waves  #: list of :obj"`Wave` : Initial trap configuration
        self.WavesB = config_b.Waves  #: list of :obj"`Wave` : Final trap configuration
        self.Damp = (config_b.Amplitude / config_a.Amplitude - 1) / sample_length  #: float : Change in amplitude

        super().__init__(sample_length, max(config_a.Amplitude, config_b.Amplitude))

    def compute(self, p, q):
        N = min(DATA_MAX, self.SampleLength - p*DATA_MAX)
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
                waveform[i] += (1 + n*self.Damp) * (mag + n*mag_inc) * sin(2 * pi * n * (fn + dfn) + (phi + n*phi_inc))

        ## Send the results to Parent ##
        q.put((p, waveform))

    def config_file(self, h5py_f):
        ## Contents ##
        h5py_f.attrs.create('freqsA', data=np.array([w.Frequency for w in self.WavesA]))
        h5py_f.attrs.create('magsA', data=np.array([w.Magnitude for w in self.WavesA]))
        h5py_f.attrs.create('phasesA', data=np.array([w.Phase for w in self.WavesA]))

        h5py_f.attrs.create('freqsB', data=np.array([w.Frequency for w in self.WavesB]))
        h5py_f.attrs.create('magsB', data=np.array([w.Magnitude for w in self.WavesB]))
        h5py_f.attrs.create('phasesB', data=np.array([w.Phase for w in self.WavesB]))

        h5py_f.attrs.create('sample_length', data=self.SampleLength)

        ## Table of Contents ##
        h5py_f.attrs.create('keys', data=['freqsA', 'magsA', 'phasesA', 'freqsB', 'magsB', 'phasesB', 'sample_length'])

        return super().config_file(h5py_f)

    @classmethod
    def from_file(cls, **kwargs):
        freqsA, magsA, phasesA, freqsB, magsB, phasesB, sample_length = kwargs.values()
        supA = Superposition(freqsA, magsA, phasesA)
        supB = Superposition(freqsB, magsB, phasesB)

        return cls(supA, supB, sample_length=sample_length)


######### HS1 Class #########
class HS1(Waveform):
    def __init__(self, pulse_time, center_freq, sweep_width, duration=None, amp=1.0):
        """
        Parameters
        ----------
        pulse_time : float
            TODO
        center_freq : int
            TODO
        sweep_width : int
            TODO
        duration : float, optional
            TODO
        amp : float, optional
            Amplitude of waveform relative to maximum output voltage.
        """
        if duration:
            sample_length = int(SAMP_FREQ * duration)
        else:
            sample_length = int(SAMP_FREQ * pulse_time * 7)
        super().__init__(sample_length, amp)

        self.Tau = pulse_time * SAMP_FREQ
        self.Center = center_freq / SAMP_FREQ
        self.BW = sweep_width / SAMP_FREQ

    def compute(self, p, q):
        N = min(DATA_MAX, self.SampleLength - p*DATA_MAX)
        waveform = np.empty(N, dtype=float)

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

            waveform[i] = amp * sin(2 * pi * arg)

        ## Send results to Parent ##
        q.put((p, waveform))

    def config_file(self, h5py_f):
        ## Contents ##
        h5py_f.attrs.create('pulse_time', data=self.Tau / SAMP_FREQ)
        h5py_f.attrs.create('center_freq', data=self.Center * SAMP_FREQ)
        h5py_f.attrs.create('sweep_width', data=self.BW * SAMP_FREQ)
        h5py_f.attrs.create('duration', data=self.SampleLength / SAMP_FREQ)

        ## Table of Contents ##
        h5py_f.attrs.create('keys', data=['pulse_time', 'center_freq', 'sweep_width', 'duration'])

        return super().config_file(h5py_f)


######### FromFile Class #########
def from_file(filename, path=None):
    """ This class just provides a clean way to construct Waveform objects
        from saved files.
        It shares all of the same characteristics as a Waveform.

        Parameters
        ----------
        filename : str
            TODO
        path : str, optional
            TODO
    """
    classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)

    kwargs = {}
    with h5py.File(filename, 'r') as f:
        ## Maneuver to relevant Data location ##
        dat = f.get(path) if path else f
        assert dat is not None, "Invalid path"

        ## Waveform's Python Class name ##
        class_name = dat.attrs.get('class')

        ## Extract the Arguments ##
        for key in dat.attrs.get('keys'):
            kwargs[key] = dat.attrs.get(key)

    obj = None
    ## Find the proper Class & Construct it ##
    for name, cls in classes:
        if class_name == name:
            obj = cls.from_file(**kwargs)
            break
    assert obj, "The retrieved 'class' attribute matches no module class"

    ## Configure Status ##
    obj.Latest = True
    obj.Filed = True
    obj.Filename = filename
    obj.Path = path if path else ''

    return obj
