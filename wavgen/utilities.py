import sys
import h5py
import time
import inspect
import numpy as np
import matplotlib.pyplot as plt
from instrumental import u
from .config import MAX_EXP
from scipy.optimize import curve_fit
from .spectrum import SPCSEQ_END, SPCSEQ_ENDLOOPALWAYS, SPCSEQ_ENDLOOPONTRIG


class Wave:
    """ Describes a Sin wave.

    Attributes
    ----------
    Frequency : int
        The frequency of the tone in Hertz.
    Magnitude : float
        A fraction, in [0,1], indicating the tone's amplitude as a fraction of the comprising parent Waveform's.
    Phase : float
        The initial phase, in [0, 2*pi], that the Wave begins with at the comprising parent Waveform's start.
    """
    def __init__(self, freq, mag=1.0, phase=0.0):
        ## Validate ##
        assert freq > 0, ("Invalid Frequency: %d, must be positive" % freq)
        assert 0 <= mag <= 1, ("Invalid magnitude: %d, must be within interval [0,1]" % mag)
        ## Initialize ##
        self.Frequency = freq
        self.Magnitude = mag
        self.Phase = phase

    def __lt__(self, other):
        return self.Frequency < other.Frequency


class Step:
    """ Describes 1 step in a control sequence.

    Attributes
    ----------
    CurrentStep : int
        The Sequence index for this step, or step number.
    SegmentIndex : int
        The index into the segmented board memory, to house the associated Waveform.
    Loops : int
        Number of times the Waveform is looped before checking
         the :attr:`~wavgen.utilities.Step.Transition`.
    NextStep : int
        The Sequence index for the next step, or which step this one will transition *to*.
    Transition : {None, 'trigger', 'end'}
        Accepts a keyword which sets the Transition Behavior, options:

        ``None``
            Transitions to :attr:`~wavgen.utilities.Step.NextStep` after looping
            the Waveform :attr:`~wavgen.utilities.Step.Loops` times.

        ``'trigger'``
            Will Transition after looping set number of times,
            but only if a :doc:`trigger <../how-to/trigger>` event occurs.

        ``'end'``
            Terminates the sequence after the set number of loops have occurred. Stops the card output.

    Hint
    ----
    All above indices begin at 0.
    """
    Trans = {  # Dictionary of Condition keywords to Register Value Constants
        None      : SPCSEQ_ENDLOOPALWAYS,
        'trigger' : SPCSEQ_ENDLOOPONTRIG,
        'end'     : SPCSEQ_END
    }

    def __init__(self, cur, seg, loops, nxt, tran=None):
        self.CurrentStep = cur
        self.SegmentIndex = seg
        self.Loops = loops
        self.NextStep = nxt
        self.Transition = self.Trans.get(tran)

        assert self.Transition is not None, "Invalid keyword for Condition."


## FUNCTIONS ##
def from_file(filename, path=None):
    """ Extracts parameters from a HDF5 dataset and constructs the corresponding Waveform object.

        Parameters
        ----------
        filename : str
            The name of the :doc:`HDF5 <../info/hdf5>` file.
        path : str, optional
            Path to a specific dataset in the HDF5 database.

        Returns
        -------
        :class:`~wavgen.waveform.Waveform`
            Marshals extracted parameters into correct Waveform subclass constructor, returning the resulting object.
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


def gaussian1d(x, x0, w, amp, offset):
    """ Parameterized 1-Dimensional Gaussian.

    Parameters
    ----------
    x : ndarray or scalar
        Input value to evaluate the Gaussian at.
    x0 : scalar
        Mean value or x-offset
    w : scalar
        Standard-Deviation or width.
    amp : scalar
        Amplitude or height.
    offset : scalar
        Vertical offset of Gaussian.

    Returns
    -------
    same as :obj:`x`
        The value of the Gaussian at :obj:`x`.
    """
    if w == 0:
        return 0
    return amp * np.exp(-2 * (x - x0) ** 2 / (w ** 2)) + offset


def gaussianarray1d(x, x0_vec, w_vec, amp_vec, offset, ntraps):
    """ Superposition of parameterized 1-Dimensional Gaussians.

    Parameters
    ----------
    x : ndarray or scalar
        Domain or input values to evaluate the Gaussians across.
    x0_vec : sequence of scalar
        Mean value or x-offset
    w_vec : sequence of scalar
        Standard-Deviation or width.
    amp_vec : sequence of scalar
        Amplitude or height.
    offset : sequence of scalar
        Vertical offset of Gaussian.
    ntraps : int
        Number of Gaussians or length of above parameter arrays.

    Returns
    -------
    same as :obj:`x`
        The value of the Gaussian superposition for each value in :obj:`x`.
    """
    array = np.zeros(np.shape(x))
    for k in range(ntraps):
        array = array + gaussian1d(x, x0_vec[k], w_vec[k], amp_vec[k], 0)
    return array + offset


def wrapper_fit_func(x, ntraps, *args):
    """ Wraps :func:`gaussianarray1d` for :func:`scipy.optimize.curve_fit` fitting.
    """
    a, b, c = list(args[0][:ntraps]), list(args[0][ntraps:2 * ntraps]), list(args[0][2 * ntraps:3 * ntraps])
    offset = args[0][-1]
    return gaussianarray1d(x, a, b, c, offset, ntraps)


def extract_peaks(which_cam, image, ntraps):
    """ Finds the value & location of each Gaussian Peak.

    Given a matrix of pixel values,
    locates each trap peak and records its position along x-axis and pixel value.

    Parameters
    ----------
    which_cam : bool
        *True* or *False* selects Pre- or Post- chamber cameras respectively.
    image : 2d ndarray
        Pixel matrix obtained from camera driver.
    ntraps : int
        Number of peaks (traps) to search for.

    Returns
    -------
    ndarray
        Pixel value at each peak.
    list
        Compiled parameter list for passing with :func:`wrapper_fit_func`.

    """
    threshes = [0.5, 0.6]
    margin = 10
    threshold = np.max(image) * threshes[which_cam]
    im = image.transpose()

    x_len = len(im)
    peak_locs = np.zeros(x_len)
    peak_vals = np.zeros(x_len)

    ## Trap Peak Detection ##
    for i in range(x_len):
        if i < margin or x_len - i < margin:
            peak_locs[i] = 0
            peak_vals[i] = 0
        else:
            peak_locs[i] = np.argmax(im[i])
            peak_vals[i] = max(im[i])

    ## Trap Range Detection ##
    first = True
    pos_first, pos_last = 0, 0
    left_pos = 0
    for i, p in enumerate(peak_vals):
        if p > threshold:
            left_pos = i
        elif left_pos != 0:
            if first:
                pos_first = (left_pos + i) // 2
                first = False
            pos_last = (left_pos + i) // 2
            left_pos = 0

    ## Separation Value ##
    separation = (pos_last - pos_first) / ntraps  # In Pixels

    ## Initial Guesses ##
    means0 = np.linspace(pos_first, pos_last, ntraps).tolist()
    waists0 = (separation * np.ones(ntraps) / 2).tolist()
    ampls0 = (max(peak_vals) * 0.7 * np.ones(ntraps)).tolist()
    _params0 = [means0, waists0, ampls0, [0.06]]
    params0 = [item for sublist in _params0 for item in sublist]

    return peak_vals, params0


def plot_image(which_cam, image, ntraps, step_num=0, fit=None, guess=False):
    """ Scans image for peaks, then plots the 1-dimensional gaussian profiles.

    Parameters
    ----------
    which_cam : bool
        `True` or `False` selects Pre- or Post- chamber cameras respectively.
        Returns None is passed.
    image : 2d ndarray
        Pixel matrix obtained from camera driver.
    ntraps : int
        Number of peaks (traps) to search for.
    step_num : int, optional
        Indicates current iteration of :meth:`~wavgen.card.Card.stabilize_intensity`
    fit : list of scalar, optional
        Parameters found as result of fitting. Plots if given.
    guess : bool, optional
        Whether to plot initial fitting guess or not.
    """
    if which_cam is None:
        return
    peak_vals, params0 = extract_peaks(which_cam, image, ntraps)

    pos_first, pos_last = params0[0], params0[ntraps-1]
    xdata = np.arange(image.shape[1])
    margin = 10

    plt.figure()
    plt.suptitle("Trap Intensities")

    plt.plot(xdata, peak_vals)
    if guess:
        plt.plot(xdata, wrapper_fit_func(xdata, ntraps, params0), '--r')  # Initial Guess
    if fit is not None:
        plt.plot(xdata, wrapper_fit_func(xdata, ntraps, fit))  # Fit

    plt.xlim((pos_first - margin, pos_last + margin))
    plt.legend(["Data", "Guess", "Fit"])
    if step_num:
        plt.title("Iteration: %d" % step_num)
    plt.ylabel("Pixel Value (0-255)")
    plt.xlabel("Pixel X-Position")
    plt.show(block=False)
    print("Fig_Newton")


# noinspection PyUnboundLocalVariable
def analyze_image(which_cam, cam, ntraps, step_num=0, iterations=20):
    """ Fits 1d Gaussians across image x-axis & returns the peak values.

    Parameters
    ----------
    which_cam : bool
        `True` or `False` selects Pre- or Post- chamber cameras respectively.
    cam : :obj:`instrumental.drivers.cameras.uc480`
        The camera object opened by :obj:`instrumental` module.
    ntraps : int
        Number of peaks (traps) to search for.
    step_num : int, optional
        Indicates current iteration of :meth:`~wavgen.card.Card.stabilize_intensity`
    iterations : int, optional
        How many times the peak values should be averaged.

    Returns
    -------
    ndarray
        Peak values of each Gaussian in image.
    """
    trap_powers = np.zeros(ntraps)
    for _ in range(iterations):
        image = cam.latest_frame()
        peak_vals, params0 = extract_peaks(which_cam, image, ntraps)

        ## Fitting ##
        xdata = np.arange(image.shape[1])
        try:
            popt, pcov = curve_fit(lambda x, *params_0: wrapper_fit_func(x, ntraps, params_0),
                                   xdata, peak_vals, p0=params0)
            trap_powers = np.add(trap_powers, np.frombuffer(popt[2 * ntraps:3 * ntraps]))
        except RuntimeError:
            plot_image(which_cam, image, ntraps, step_num, guess=True)
            return np.ones(ntraps)

    plot_image(which_cam, image, ntraps, step_num, popt)
    return np.multiply(trap_powers, 1/iterations)


# noinspection PyProtectedMember
def fix_exposure(cam, slider, verbose=False):
    """ Automatically adjusts camera exposure.

    Parameters
    ----------
    cam : :obj:`instrumental.drivers.cameras.uc480`
        The camera object opened by :obj:`instrumental` module.
    slider : :obj:`matplotlib.widgets.Slider`
        Slider which sets camera exposure.
    verbose : bool, optional
        Verbosity!
    """
    margin = 10
    exp_t = MAX_EXP / 2
    cam._set_exposure(exp_t * u.milliseconds)
    time.sleep(0.5)
    print("Fetching Frame")
    im = cam.latest_frame()
    x_len = len(im)
    print("Fetching Exposure")
    exp_t = cam._get_exposure()

    right, left = MAX_EXP, 0
    inc = right / 10
    for _ in range(10):
        ## Determine if Clipping or Low-Exposure ##
        gap = 255
        for i in range(x_len):
            if i < margin or x_len - i < margin:
                continue
            else:
                gap = min(255 - max(im[i]), gap)

        ## Make Appropriate Adjustment ##
        if gap == 0:
            if verbose:
                print("Clipping at: ", exp_t)
            right = exp_t
        elif gap > 50:
            if verbose:
                print("Closing gap: ", gap, " w/ exposure: ", exp_t)
            left = exp_t
        else:
            if verbose:
                print("Final Exposure: ", exp_t)
            return

        if inc < 0.01:
            exp_t -= inc if gap == 0 else -inc
        else:
            exp_t = (right + left) / 2
            inc = (right - left) / 10

        slider.set_val(exp_t)
        time.sleep(0.5)
        im = cam.latest_frame()
