import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import SpanSelector, Slider
from easygui import buttonbox, multenterbox
from .utilities import verboseprint
from math import ceil
from time import time
from tqdm import tqdm
from .config import *
import h5py
import os


######### Waveform Class #########
class Waveform:
    """ Basic Waveform object.

    Attention
    ---------
    All other defined waveform objects (below) extend *this* class;
    therefore, they all share these attributes, at the least.

    Attributes
    ----------
    cls.OpenTemps : int, **Class Object**
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
    FilePath : str
        The name of the file where the waveform is saved.
    GroupPath : str
        The HDF5 pathway to where **this** waveform's root exists.
        Used in the case where a single HDF5 file contains a database
        of several waveforms (Very efficient space wise).
    """
    OpenTemps = 0
    PlottedTemps = 0

    def __init__(self, sample_length, amp=1.0):
        """
        Parameters
        ----------
        sample_length : int
            Sets the `SampleLength`.
        amp : float, optional
            Amplitude of waveform relative to maximum output voltage.
        """
        if sample_length % 32:
            verboseprint("Sample length is being truncated to align with 32 samples.")
        self.SampleLength = int(sample_length - sample_length % 32)
        self.Amplitude    = amp
        self.PlotObjects  = []
        self.Latest       = False
        self.FilePath     = None
        self.GroupPath    = ''

    def __del__(self):
        """Deletes the temporary file used by unsaved (temporary) Waveforms."""
        if self.FilePath == 'temporary.h5' and Waveform.OpenTemps > 0:
            Waveform.OpenTemps -= 1
            if Waveform.OpenTemps == 0:
                os.remove('temporary.h5')

    def __eq__(self, other):
        self, other = self.__dict__, other.__dict__  # For cleanliness

        def comp_attr(A, B):
            """Compares a single Attribute between 2 objects"""
            if isinstance(A, list):
                return np.array([comp_attr(a, b) for a, b in zip(A, B)]).all()
            return A == B

        keys = list(set(self.keys()) - set(Waveform(0).__dict__.keys()))  # We remove Parent Attributes
        return np.array([comp_attr(self.get(key), other.get(key)) for key in keys]).all()

    def compute(self, p, q):
        """ Calculates the *p*\ th portion of the entire waveform.

        Note
        ----
        This is the function dispatched to :doc:`parallel processes <../info/parallel>`.
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
        q.put((p, wav, max(wav.max(), abs(wav.min()))))

    def config_file(self, h5py_f):
        ## Necessary to determine subclass when loading from file ##
        h5py_f.attrs.create('class', data=self.__class__.__name__)

        ## Determine a title attribute for Plot titles ##
        if self.FilePath == 'temporary.h5':
            name = 'temporary waveform %d' % Waveform.PlottedTemps
            Waveform.PlottedTemps += 1
        else:
            if self.GroupPath != '':
                name = os.path.basename(self.GroupPath)
            else:
                name = os.path.splitext(os.path.basename(self.FilePath))[0]

        ## Waveform Data Buffer ##
        dataset = h5py_f.create_dataset('waveform', shape=(self.SampleLength,), dtype='int16')
        dataset.attrs.create('title', name)

        return dataset  # Return the dataset for waveform data-points

    @classmethod
    def from_file(cls, **kwargs):
        return cls(**kwargs)

    def compute_waveform(self, filepath=False, grouppath=False, cpus=None):
        """
        Computes the waveform to disk.
        If no filepath is given, then waveform data will be destroyed upon object cleanup.

        Parameters
        ----------
        filepath : str, optional
            Searches for an HDF5 database file with the given name. If none exists, then one is created.
            If not provided, saves the waveform to a temporary file.
        grouppath : str, optional
            Describes a path to a group in the HDF5 database for saving this particular waveform dataset.
        cpus : int, optional
            Sets the desired number of CPUs to utilized for the calculation. Will round down if too
            large a number given.

        Note
        ----
        The `filepath` parameter does not need to include a file-extension; only a name.
        """
        ## Redundancy & Input check ##
        f, g = self._check_savepath(filepath, grouppath)
        if f is None:
            return
        self.FilePath, self.GroupPath = f, g

        ## Open HDF5 files for Writing ##
        mode = 'a' if self.GroupPath != '' else 'w'
        with h5py.File(self.FilePath, mode) as F:
            if self.GroupPath != '':
                F = F.require_group(self.GroupPath)
            wav = self.config_file(F)  # Setup File Attributes
            # TODO: Validate that this nesting of HDF5 files is safe.
            with h5py.File('temp.h5', 'w') as T:
                temp = T.create_dataset('waveform', shape=(self.SampleLength,), dtype=float)
                self._compute_waveform(wav, temp, cpus)
            F.file.flush()    # Flush all calculations to disk
        os.remove('temp.h5')  # Remove the temporary

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
        with h5py.File(self.FilePath, 'r') as f:
            try:
                buffer[()] = f.get(self.GroupPath + '/waveform')[offset:offset + size]
            except TypeError:
                dat = f.get(self.GroupPath + '/waveform')[offset:offset + size]
                for i in range(size):
                    buffer[i] = dat[i]

    def plot(self, ends=False):
        """ Plots the Segment. Computes first if necessary.
        """
        if len(self.PlotObjects):  # Don't plot if already plotted
            return
        if not self.Latest:        # Compute before Plotting
            self.compute_waveform()

        ## Retrieve the names of each Dataset ##
        with h5py.File(self.FilePath, 'r') as f:
            if self.GroupPath != '':
                f = f.get(self.GroupPath)
            dsets = list(f.keys())

        ## Plot each Dataset ##
        for dset in dsets:
            if ends:
                self.PlotObjects.append(self._plot_ends(dset))
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
        max_val = self._parallelize(temp, self.compute, cpus)

        ## Calculate Normalization Factor ##
        norm = (SAMP_VAL_MAX * self.Amplitude) / max_val

        ## Then Normalize ##
        try:
            wav[()] = np.multiply(temp[()], norm).astype(np.int16)
        except Exception as err:
            print(err)
            print('NumPy not big enough?')
            for n in range(0, self.SampleLength, DATA_MAX):
                N = min(DATA_MAX, self.SampleLength - n)
                wav[n:n+N] = np.multiply(temp[n:n+N], norm).astype(np.int16)

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
            mp.Process(target=func, args=(p, q), daemon=True).start()

        ## Collect Validation & Start Remaining Processes ##
        max_val = 0
        for p in tqdm(range(N)):
            n, data, cur_max_val = q.get()  # Collects a Result

            max_val = max(cur_max_val, max_val)  # Tracks the result's greatest absolute value

            i = n * DATA_MAX  # Shifts to Proper Interval

            buffer[i:i + len(data)] = data  # Writes to Disk

            if p < N - cpus:  # Starts a new Process
                mp.Process(target=func, args=(p + cpus, q), daemon=True).start()

        return max_val

    def _check_savepath(self, filepath, grouppath):
        """ Checks specified location for pre-existing waveform.

            If the desired data-path is discovered to be already occupied,
            the options are offered to Overwrite, Abort, or try a new location.

            Parameters
            ----------
            filepath : str
                Potential name for HDF5 file to write to.
            grouppath : str
                Potential path in HDF5 file hierarchy to write to.

            Returns
            -------
            (str, str)
                Returns a tuple of strings indicating the path-to-file & path-to-hdf5 group.
        """
        ## Format the file extension for uniformity ##
        filepath = os.path.splitext(filepath)[0] + '.h5' if filepath else filepath

        ## Check if: the waveform has already been saved; it needs an updated calculation ##
        if not (filepath or grouppath) or (self.FilePath == filepath and self.GroupPath == grouppath):
            if self.Latest:
                return None, None
            elif not (self.FilePath or self.GroupPath):
                Waveform.OpenTemps += 1
                return 'temporary.h5', str(id(self))
            return filepath, grouppath

        ## Check the specified Location for Vacancy ##
        while True:
            # Determine a FilePath
            if not os.access(filepath, os.F_OK):  # FilePath available!
                break
            elif grouppath:  # Check group
                with h5py.File(filepath, 'r') as F:
                    if F.get(grouppath) is None:
                        break
                # TRY NEW GROUP OR FILE
            choices = ['Overwrite', 'NewLocation', 'Abort']
            msg = (('Group: \'%s\' in\n' % grouppath) if grouppath else '') + \
                  'File: %s\nis already occupied!\nHow would you like to proceed?' % filepath
            choice = buttonbox(msg, choices=choices)
            if choice == 'Overwrite':
                break
            elif choice == 'NewLocation':
                filepath, grouppath = multenterbox('Enter a new Location: ', '', ['FilePath', 'GroupPath'])
                filepath = os.path.splitext(filepath)[0] + '.h5' if filepath else exit(-1)
            else:
                exit(-1)

        return filepath, grouppath

    def _plot_span(self, dset):
        N = min(PLOT_MAX, self.SampleLength)
        name = self.GroupPath + '/' + dset
        with h5py.File(self.FilePath, 'r') as f:
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

            ax1.set_xlim(xscrolled[0] - 100, xscrolled[-1] + 100)
            fig.canvas.draw()

        slid = None
        if N != self.SampleLength:  # Only include a scroller if Waveform is large enough
            axspar = plt.axes([0.14, 0.94, 0.73, 0.05])
            slid = Slider(axspar, 'Scroll', valmin=0, valmax=self.SampleLength - N, valinit=0, valfmt='%d', valstep=10)
            slid.on_changed(scroll)

        ## Span Selector ##
        def onselect(xmin, xmax):
            if xmin == xmax:
                return
            pos = int(slid.val) if slid else 0
            xzoom = np.arange(pos, pos + N)
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

    def _plot_ends(self, dset):
        N = PLOT_MAX // 32
        N += 1 if N%2 else 0

        name = self.GroupPath + '/' + dset
        with h5py.File(self.FilePath, 'r') as f:
            legend = f[name].attrs.get('legend')
            title = f[name].attrs.get('title')
            y_label = f[name].attrs.get('y_label')
            dtype = f[name].dtype

        shape = N if legend is None else (N, len(legend))
        M, m = self._y_limits(dset)

        xdat = np.arange(N)
        end_dat, begin_dat = np.zeros(shape, dtype=dtype), np.zeros(shape, dtype=dtype)
        self._load(dset, begin_dat, 0)
        self._load(dset, end_dat, self.SampleLength - N)

        ## Figure Creation ##
        fig = plt.figure(figsize=(10, 4), constrained_layout=True)
        fig.suptitle(title if title else "Waveform", fontsize=14)

        gs = GridSpec(2, 4, figure=fig)
        end_ax = fig.add_subplot(gs[0, :2])
        begin_ax = fig.add_subplot(gs[0, 2:])
        cat_ax = fig.add_subplot(gs[1, :])

        ## Plotting the Waveform End ##
        begin_ax.set(facecolor='#FFFFCC')
        begin_ax.plot(xdat, begin_dat, '-')
        begin_ax.set_ylim((m, M))
        begin_ax.yaxis.tick_right()
        begin_ax.set_title("First %d samples" % N)

        ## Plotting the Waveform End ##
        end_ax.set(facecolor='#FFFFCC')
        end_ax.plot(xdat, end_dat, '-')
        end_ax.set_ylim((m, M))
        end_ax.set_title("Last %d samples" % N)

        cat_ax.set(facecolor='#FFFFCC')
        cat_ax.plot(np.arange(2*N), np.concatenate((end_dat, begin_dat)), '-')
        cat_ax.set_ylim((m, M))
        cat_ax.set_title("Above examples concatenated")

        if legend is not None:
            end_ax.legend(legend)
        if y_label is not None:
            end_ax.set_ylabel(y_label)
            cat_ax.set_ylabel(y_label)

        plt.show(block=False)

        return fig

    def _load(self, dset, buf, offset):
        with h5py.File(self.FilePath, 'r') as f:
            buf[:] = f.get(self.GroupPath + '/' + dset)[offset:offset + len(buf)]

    def _y_limits(self, dset):
        name = self.GroupPath + '/' + dset
        with h5py.File(self.FilePath, 'r') as f:
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
