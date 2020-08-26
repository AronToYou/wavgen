import numpy as np
import multiprocessing as mp
from easygui import buttonbox, multenterbox
from .utilities import verboseprint, plot_waveform
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
    DataPath : str
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
        self.DataPath    = ''

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

    def config_dset(self, dset):
        ## Contents ##
        dset.attrs.create('sample_length', data=self.SampleLength)

        ## Table of Contents ##
        dset.attrs.create('keys', data=['sample_length'])

        return dset

    @classmethod
    def from_file(cls, **kwargs):
        return cls(**kwargs)

    def compute_waveform(self, filepath=False, datapath=False, cpus=None):
        """
        Computes the waveform to disk.
        If no filepath is given, then waveform data will be destroyed upon object cleanup.

        Parameters
        ----------
        filepath : str, optional
            Searches for an HDF5 database file with the given name. If none exists, then one is created.
            If not provided, saves the waveform to a temporary file.
        datapath : str, optional
            Describes a path to a group in the HDF5 database for saving this particular waveform dataset.
        cpus : int, optional
            Sets the desired number of CPUs to utilized for the calculation. Will round down if too
            large a number given.

        Note
        ----
        The `filepath` parameter does not need to include a file-extension; only a name.
        """
        ## Redundancy & Input check ##
        if not self._valid_savepath(filepath, datapath):
            return

        ## Open HDF5 files for Writing ##
        with h5py.File(self.FilePath, 'a', libver='latest') as F:
            dset = F.require_dataset(self.DataPath, shape=(self.SampleLength,), dtype='int16')
            dset.attrs.create('class', data=self.__class__.__name__)  # For determining the Constructor
            wav = self.config_dset(dset)  # Setup Dataset Attributes

            with h5py.File('temp.h5', 'w', libver='latest') as T:
                temp = T.create_dataset('waveform', shape=(self.SampleLength,), dtype=float)
                self._compute_waveform(wav, temp, cpus)

            F.flush()    # Flush all calculations to disk
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
        with h5py.File(self.FilePath, 'r', libver='latest') as f:
            try:
                buffer[()] = f.get(self.DataPath)[offset:offset + size]
            except TypeError:
                dat = f.get(self.DataPath)[offset:offset + size]
                for i in range(size):
                    buffer[i] = dat[i]

        # buf[:] = f.get(self.DataPath)[offset:offset + len(buf)]

    def plot(self):
        """ Convenient alias for :func:`~wavgen.utilities.plot_waveform` """
        plot_waveform(self)

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
        except Exception as err:  #TODO: Speed this up!
            print(err)
            print('NumPy not big enough?')
            for n in range(0, self.SampleLength, DATA_MAX):
                N = min(DATA_MAX, self.SampleLength - n)
                wav[n:n+N] = np.multiply(temp[n:n+N], norm).astype(np.int16)

        ## Wrapping things Up ##
        bytes_per_sec = self.SampleLength * 2 // (time() - start_time)
        print("\tAverage Rate: %d bytes/second" % bytes_per_sec)

    def _parallelize(self, buffer, func, cpus):
        ## Number of Parallel Processes ##
        cpus_max = mp.cpu_count()
        cpus = min(cpus_max, cpus) if cpus else int(0.75*cpus_max)

        ## Total Processes to-do ##
        N = ceil(self.SampleLength / DATA_MAX)  # Number of Child Processes
        print("\tNumber of child processes: ", N)
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

    def _valid_savepath(self, filepath, datapath):
        """ Checks specified location for pre-existing waveform.

            If the desired data-path is discovered to be already occupied,
            the options are offered to Overwrite, Abort, or try a new location.

            Parameters
            ----------
            filepath : str
                Potential name for HDF5 file to write to.
            datapath : str
                Potential path in HDF5 file hierarchy to write to.

            Returns
            -------
            (str, str)
                Returns a tuple of strings indicating the path-to-file & path-to-hdf5 group.
        """
        assert (filepath and datapath) or not(filepath or datapath), 'Both a File & Dataset need to be specified!'

        ## Format the file extension for uniformity ##
        filepath = os.path.splitext(filepath)[0] + '.h5' if filepath else filepath

        ## Checks: -if the waveform has already been saved; -if it needs an updated calculation ##
        if not filepath or (self.FilePath == filepath and self.DataPath == datapath):
            if not self.FilePath:
                Waveform.OpenTemps += 1
                Waveform.PlottedTemps += 1
                self.FilePath, self.DataPath = 'temporary.h5', 'Temporary Waveform %d' % Waveform.PlottedTemps
            return not self.Latest

        def occupied():
            """Returns whether the specified Location is occupied"""
            try:
                f = h5py.File(filepath, 'r', libver='latest')
                f.get(datapath).name
            except (OSError, AttributeError):
                return False
            return True

        choices = ['Overwrite', 'NewLocation', 'Abort']
        msg = (('Group: \'%s\' in\n' % datapath) if datapath else '') + \
            'File: %s\nis already occupied!\nHow would you like to proceed?' % filepath

        while occupied():
            choice = buttonbox(msg, choices=choices)
            if choice == 'Overwrite':
                with h5py.File(filepath, 'a', libver='latest') as f:
                    del f[datapath]
            elif choice == 'NewLocation':
                filepath, datapath = multenterbox('Enter a new Location: ', '', ['FilePath', 'DataPath'])
                filepath = os.path.splitext(filepath)[0] + '.h5' if filepath else exit(-1)
            else:
                exit(-1)

        self.FilePath, self.DataPath = filepath, datapath
        return True
