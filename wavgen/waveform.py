from math import pi, sin, cosh, log
from .waveform_base import Waveform
from .utilities import Wave
from easygui import msgbox
from sys import maxsize
from .config import *
from math import inf
import numpy as np
import random


class Superposition(Waveform):
    """ A static trap configuration.

    Attributes
    ----------
    Waves : list of :class:`~wavgen.utilities.Wave`
        The list of composing pure tones. Each object holds a frequency,
        magnitude, and relative phase.

    Hint
    ----
    There are now 3 relevant **amplitudes** here:
    :meth:`Output Voltage Limit <wavgen.card.Card.setup_channels>`,
    :attr:`Waveform Amplitude <wavgen.waveform.Waveform.Amplitude>`, & the
    :attr:`amplitudes of each pure tone <wavgen.utilities.Wave.Magnitude>`
    composing a superposition. Each one is expressed as a fraction of the previous.

    See Also
    --------
    :func:`even_spacing`
        An alternative constructor for making :class:`Superposition` objects.
    """
    def __init__(self, freqs, mags=None, phases=None, sample_length=None, amp=1.0):
        """ Provides several options for defining waveform duration.

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
        amp : float, optional
            Amplitude of waveform relative to maximum output voltage.
        """
        freqs.sort()

        ## Find the LeastCommonMultiple ##
        if sample_length is None:
            lcm = inf
            for f in freqs:
                digits = 0
                while f%10 == 0:
                    f = f // 10
                    digits += 1
                lcm = min(digits, lcm)
            sample_length = (SAMP_FREQ / 10**lcm) * 32 * REPEAT
            msg = "Waveform will not be an integer # of periods.\nYou may want to calculate a sample length manually"
        if sample_length % 1:
            msgbox(msg, "Warning")
        else:
            sample_length = int(sample_length)

        ## Applies passed Magnitudes or Phases ##
        if mags is None:
            mags = np.ones(len(freqs))
        if phases is None:
            phases = np.zeros(len(freqs))

        assert freqs[-1] >= SAMP_FREQ/sample_length, "Frequency is below resolution. Increase sample length."
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
        q.put((p, waveform, max(waveform.max(), abs(waveform.min()))))

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
        """
        Returns
        -------
        list of float
            Value of :attr:`~wavgen.utilities.Wave.Magnitude` for each pure tone,
            in order of increasing frequency.
        """
        return [w.Magnitude for w in self.Waves]

    def set_magnitudes(self, mags):
        """ Sets the :attr:`~wavgen.utilities.Wave.Magnitude` of each pure tone.

        Parameters
        ----------
        mags : list of float
            Each new magnitude, limited to (**[0, 1]**), ordered by ascending frequency).
        """
        for w, mag in zip(self.Waves, mags):
            assert 0 <= mag <= 1, ("Invalid magnitude: %d, must be within interval [0,1]" % mag)
            w.Magnitude = mag
        self.Latest = False

    def get_phases(self):
        return [w.Phase for w in self.Waves]

    def set_phases(self, phases):
        """ Sets the relative phase of each pure tone.

        Parameters
        ----------
        phases : list of float
            New phases, expressed as (**radians**), ordered by ascending frequency.

        """
        for w, phase in zip(self.Waves, phases):
            w.Phase = phase
        self.Latest = False

    def randomize(self):
        """ Randomizes each pure tone's phase.
        """
        for w in self.Waves:
            w.Phase = 2*pi*random.random()
        self.Latest = False


def even_spacing(ntraps, center, spacing, mags=None, phases=None, sample_length=None, amp=1.0):
    """ Wrapper function which simplifies defining :class:`~wavgen.waveform.Superposition` objects
     to describe equally spaced traps.

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
    sample_length : int, optional
            Length of waveform in samples.
    amp : float, optional
        Amplitude of waveform relative to maximum output voltage.

    Returns
    -------
    :class:`~wavgen.waveform.Superposition`

    """
    freqs = [int(center + spacing*(i - (ntraps-1)/2)) for i in range(ntraps)]
    N = sample_length if sample_length else int(SAMP_FREQ * (2 - ntraps % 2) // spacing) * 32 * REPEAT

    return Superposition(freqs, mags=mags, phases=phases, sample_length=N, amp=amp)


class Sweep(Waveform):
    """ Describes a waveform which smoothly modulates from one :class:`~wavgen.waveform.Superposition`
    to another.

    Attributes
    ----------
    WavesA, WavesB : list of :class:`~wavgen.utilities.Wave`
        Basically full descriptions of 2 :class:`~wavgen.waveform.Superposition` objects;
        i.e. 2 lists of pure tones, including each frequency, magnitude, & phase.
    Damp : float
        Expresses the *change in* :attr:`~wavgen.waveform.Waveform.Amplitude`
        as the waveform modulates from initial to final configuration.

    Warning
    -------
    Sometimes, roughly 1 out of 8 times, the calculation of a ``Sweep`` object will silently fail;
    resulting in correct number of data points, except all 0-valued. To avoid the uncertainty of re-calculation,
    be sure to save ``Sweep`` objects to named files. Also, check calculations with ``Sweep.plot()``.
    """
    def __init__(self, config_a, config_b, sweep_time=None, sample_length=int(16E6)):
        """ Allows for defining the duration in terms of milliseconds or samples.

        Parameters
        ----------
        config_a, config_b : :class:`~wavgen.waveform.Superposition`
            These play the initial & final configurations of the Sweep form,
            going from **A** to **B** respectively.
        sweep_time : float, optional
            The time, in milliseconds, that the waveform will spend to complete
            the entire modulation. **Overrides sample_length**
        sample_length : int, optional
            Otherwise, one can simply fix the length to an integer number of samples.
        """
        assert isinstance(config_a, Superposition) and isinstance(config_b, Superposition)
        assert len(config_a.Waves) == len(config_b.Waves)

        if sweep_time:
            sample_length = int(SAMP_FREQ*sweep_time/1000)

        self.WavesA = config_a.Waves
        self.WavesB = config_b.Waves
        self.Damp = (config_b.Amplitude / config_a.Amplitude - 1) / sample_length

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
        q.put((p, waveform, max(waveform.max(), abs(waveform.min()))))

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
    """ Embodies a Hyperbolic-Secant Pulse.

    Attributes
    ----------
    Tau : float
        Characteristic length of pulse; expressed in samples.
    Center : float
        The frequency at which the sweep is centered about; expressed as oscillations per sample.
    BW : float
        Bandwith or width of the range the frequency is swooped across; expressed as oscillations per sample.

    See Also
    --------
    `B Peaudecerf et al 2019 New J. Phys. 21 013020 (Section 3.1) <file:../_static/pap1.pdf>`_
        Relevant context. Used to verify functional form.

    `M. Khudaverdyan et al 2005 Phys. Rev. A 71, 031404(R) <file:../_static/pap2.pdf>`_
        Slightly more relevant...yet less useful.
    """
    def __init__(self, pulse_time, center_freq, sweep_width, duration=None, amp=1.0):
        """
        Parameters
        ----------
        pulse_time : float
            Sets the characteristic time.
        center_freq : int
            The frequency sweep is centered about this value.
        sweep_width : int
            How wide, in frequency, the sweep swoops.
        duration : float, optional
            Used to fix the waveform duration, while the pulse width itself is unaffected.
            Otherwise, we follow a recommendation from the first reference above.
        amp : float, optional
            Amplitude of waveform relative to maximum output voltage.
        """
        if duration:
            sample_length = int(SAMP_FREQ * duration)
        else:
            sample_length = int(SAMP_FREQ * pulse_time * 5)
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

            d = 2*(n - self.SampleLength/2)/self.Tau  # 2(t - T/2)/tau

            try:
                arg = n * self.Center + (self.Tau * self.BW / 4) * log(cosh(d))
                amp = 1 / cosh(d)
            except OverflowError:
                arg = n * self.Center + (self.Tau * self.BW / 4) * log(maxsize)
                amp = 0

            waveform[i] = amp * sin(2 * pi * arg)

        ## Send results to Parent ##
        q.put((p, waveform, max(waveform.max(), abs(waveform.min()))))

    def config_file(self, h5py_f):
        ## Contents ##
        h5py_f.attrs.create('pulse_time', data=self.Tau / SAMP_FREQ)
        h5py_f.attrs.create('center_freq', data=self.Center * SAMP_FREQ)
        h5py_f.attrs.create('sweep_width', data=self.BW * SAMP_FREQ)
        h5py_f.attrs.create('duration', data=self.SampleLength / SAMP_FREQ)

        ## Table of Contents ##
        h5py_f.attrs.create('keys', data=['pulse_time', 'center_freq', 'sweep_width', 'duration'])

        return super().config_file(h5py_f)
