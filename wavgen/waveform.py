"""Waveform Module

Contained here is the :class:`Waveform` base class.
All user waveforms are defined here as extensions of this base.
The base allows operations s.a. :meth:`~Waveform.compute`,
:meth:`~Waveform.load`, & :meth:`~Waveform.plot` to be generalized across any & all defined waveforms.

See the :ref:`How-To <define>` guide for complete instructions on defining new user waveforms.
"""

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Slider
from math import pi, sin, cosh, ceil, log
from .utilities import Wave
from sys import maxsize
from time import time
from tqdm import tqdm
import warnings
import inspect
import easygui
import random
import h5py
import sys
import os


## Suppresses matplotlib.widgets.Slider warning ##
warnings.filterwarnings("ignore", category=UserWarning)

### Parameter - can be changed. ###
DATA_MAX = int(16E4)     #: Maximum number of samples to hold in array at once
PLOT_MAX = int(1E4)      #: Maximum number of data-points to plot at once
SAMP_FREQ = int(1000E6)  #: Desired Sampling Frequency

### Constants - Should NOT be changed. ###
SAMP_VAL_MAX = (2 ** 15 - 1)  #: Maximum digital value of sample ~~ signed 16 bits
SAMP_FREQ_MAX = int(1250E6)   #: Maximum Sampling Frequency
CPU_MAX = mp.cpu_count()      #: Number of physical cores for multi-threading
MHZ = SAMP_FREQ / 1E6         #: Coverts samples/seconds to MHz


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
    Filed : bool
        Indicates if the waveform has been saved to a file.
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
            Sets the `Amplitude`. Defaults to 1.
        """
        self.SampleLength = (sample_length - sample_length % 32)
        self.Amplitude    = amp
        self.PlotObjects  = []
        self.Latest       = False
        self.Filed        = False
        self.Filename     = 'temporary.h5'
        self.Path         = ''

    def __del__(self):
        """Deletes the temporary file used by unsaved (temporary) Waveforms."""
        if self.Filename == 'temporary.h5':
            Waveform.OpenTemps -= 1
            if Waveform.OpenTemps == 0:
                os.remove('temporary.h5')
                os.remove('temp.h5')

    def compute(self, p, q):
        """
        Calculates the `p`th portion of the entire waveform.

        Note
        ----
        This is the function dispatched to processes when employing :ref:`paralellism <parallel>`.
        The `p` argument indicates the interval of the whole waveform to calculate.

        Parameters
        ----------
        p : int
            Index, **starting from 0**, indicating which interval of the whole waveform
            should be calculated. Intervals are size :const:`DATA_MAX` in samples.
        q : :obj:`multiprocessing.Queue`
            A `Queue` object shared by multiple processes.
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
    def from_file(cls, *attrs):
        ## Pre-process the args for Constructor use ##
        args = attrs
        return cls(*args)

    def compute_waveform(self, filename=None):
        """
        Computes the entire waveform.

        Parameters
        ----------
        filename : str, optional
            If provided, will save the waveform to file on disk named as such.
        """
        ## Redundancy & Input check ##
        if self.Latest:
            return
        write_mode = self._check_filename(filename)

        with h5py.File(self.Filename, write_mode) as F:
            if self.Path != '':
                if F.get(self.Path) is None:
                    F.create_group(self.Path)
                F = F.get(self.Path)
            wav = self.config_file(F)  # Setup File Attributes
            with h5py.File('temp.h5', 'w') as T:
                temp = T.create_dataset('waveform', shape=(self.SampleLength,), dtype=float)
                self._compute_waveform(wav, temp)

        ## Wrapping things Up ##
        self.Latest = True  # Will be up to date after
        self.Filed = True  # Is on file

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
            self.compute_waveform(self.Filename)
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

    def _compute_waveform(self, wav, temp):
        start_time = time()  # Timer

        ## Compute the Waveform ##
        self._parallelize(temp, self.compute)

        ## Determine the Normalization Factor ##
        norm = (SAMP_VAL_MAX * self.Amplitude) / max(temp[()].max(), abs(temp[()].min()))

        ## Then Normalize ##
        wav[()] = np.multiply(temp[()], norm).astype(np.int16)

        ## Wrapping things Up ##
        bytes_per_sec = self.SampleLength * 2 // (time() - start_time)
        print("Average Rate: %d bytes/second" % bytes_per_sec)

    def _parallelize(self, buffer, func, cpus=CPU_MAX):
        ## Setup Parallel Processing ##
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

    def _check_filename(self, filename):
        """ Checks for a filename,
            otherwise asks for one.
            Exits if necessary.

            Parameters
            ----------
            filename : str
                Potential name for file to write to.

            Returns
            -------
            char : {'a', 'w'}
                Returns a character corresponding to append or truncate
                mode on the file to write to.
        """
        ## Check for File duplicate ##
        if filename is None:
            self.Path = str(id(self))
            Waveform.OpenTemps += 1
            return 'a'

        while True:
            if filename is None:
                exit(-1)
            try:
                F = h5py.File(filename, 'r')
                if (self.Filed and self.Filename == filename) or \
                        self.Filed != '' or \
                        easygui.boolbox("Overwrite existing file?"):
                    F.close()
                    break
                filename = easygui.enterbox("Enter a filename (blank to abort):", "Input")
            except OSError:
                break

        self.Filename = filename
        if self.Path == '':
            return 'w'
        return 'a'

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
######### Superposition Class #########
class Superposition(Waveform):
    """
        A static trap configuration.
    """
    def __init__(self, freqs, mags=None, phases=None, resolution=1E6, sample_length=None, amp=1):
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
        """ Computes the superposition of frequencies
            and stores it to an .h5py file.

        """
        ## Table of Contents ##
        h5py_f.attrs.create('attrs', data=['freqs', 'mags', 'phases'])

        ## Contents ##
        h5py_f.attrs.create('freqs', data=np.array([w.Frequency for w in self.Waves]))
        h5py_f.attrs.create('mags', data=np.array([w.Magnitude for w in self.Waves]))
        h5py_f.attrs.create('phases', data=np.array([w.Phase for w in self.Waves]))

        ## Waveform Data ##
        return super().config_file(h5py_f)

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


def even_spacing(ntraps, center, spacing, mags=None, phases=None, periods=1, amp=1):
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

        Returns
        -------
        :obj:`Superposition`
            Packages the input parameters into a :obj:`Superposition` object.

    """
    freqs = [center + spacing*(i - (ntraps-1)/2) for i in range(ntraps)]
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
        """ Computes the superposition of frequencies
            and stores it to an .h5py file.

        """
        ## Table of Contents ##
        h5py_f.attrs.create('attrs', data=['freqsA', 'magsA', 'phasesA', 'freqsB', 'magsB', 'phasesB'])

        ## Contents ##
        h5py_f.attrs.create('freqsA', data=np.array([w.Frequency for w in self.WavesA]))
        h5py_f.attrs.create('magsA', data=np.array([w.Magnitude for w in self.WavesA]))
        h5py_f.attrs.create('phasesA', data=np.array([w.Phase for w in self.WavesA]))

        h5py_f.attrs.create('freqsB', data=np.array([w.Frequency for w in self.WavesB]))
        h5py_f.attrs.create('magsB', data=np.array([w.Magnitude for w in self.WavesB]))
        h5py_f.attrs.create('phasesB', data=np.array([w.Phase for w in self.WavesB]))

        ## Waveform Data ##
        return super().config_file(h5py_f)

    @classmethod
    def from_file(cls, *attrs):
        freqsA, magsA, phasesA, freqsB, magsB, phasesB, sample_length = attrs
        supA = Superposition(freqsA, magsA, phasesA, sample_length=sample_length)
        supB = Superposition(freqsB, magsB, phasesB, sample_length=sample_length)

        return cls(supA, supB, sample_length=sample_length)


######### HS1 Class #########
class HS1(Waveform):
    def __init__(self, pulse_time, center_freq, sweep_width, duration=None, amp=1):
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
        ## Table of Contents
        h5py_f.attrs.create('attrs', data=['pulse_time', 'center_freq', 'sweep_width'])

        ## Contents ##
        h5py_f.attrs.create('pulse_time', data=self.Tau / SAMP_FREQ)
        h5py_f.attrs.create('center_freq', data=self.Center * SAMP_FREQ)
        h5py_f.attrs.create('sweep_width', data=self.BW * SAMP_FREQ)

        ## Waveform Data ##
        return super().config_file(h5py_f)

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
    obj.Path = path if path else ''

    return obj
