import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from instrumental import u
from spectrum import SPCSEQ_END, SPCSEQ_ENDLOOPALWAYS, SPCSEQ_ENDLOOPONTRIG
from .config import MAX_EXP


class Wave:
    """ Describes a Sin wave. """
    def __init__(self, freq, mag=1, phase=0):
        """
            Constructor.

            Parameters
            ----------
            freq : int
                Frequency of the wave.

            mag : float
                Magnitude within [0,1]

            phase : float
                Initial phase of oscillation.

        """
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
    """
    NOTE: Indexes start at 0!!

    Attributes
    ----------
    CurrentStep : int
        The Sequence index for this step.
    SegmentIndex : int
        The index into the Segment array for the associated Wave.
    Loops : int
        Number of times the Wave is looped before checking continue Condition.
    NextStep : int
        The Sequence index for the next step.
    Condition : {None, 'trigger', 'end'}, optional
        A keyword to indicate: if a trigger is necessary for the step
        to continue to the next, or if it should be the last step.
        ['trigger', 'end'] respectively.
        Defaults to None, meaning the step continues after looping 'Loops' times.
    """
    Conds = {  # Dictionary of Condition keywords to Register Value Constants
        None      : SPCSEQ_ENDLOOPALWAYS,
        'trigger' : SPCSEQ_ENDLOOPONTRIG,
        'end'     : SPCSEQ_END
    }

    def __init__(self, cur, seg, loops, nxt, cond=None):
        self.CurrentStep = cur
        self.SegmentIndex = seg
        self.Loops = loops
        self.NextStep = nxt
        self.Condition = self.Conds.get(cond)

        assert self.Condition is not None, "Invalid keyword for Condition."


## FUNCTIONS ##
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
    """ Wraps :obj:`gaussianarray1d` for :obj:`scipy.optimize.curve_fit` fitting.
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
        `True` or `False` selects Pre- or Post- chamber cameras respectively.
    image : 2d ndarray
        Pixel matrix obtained from camera driver.
    ntraps : int
        Number of peaks (traps) to search for.

    Returns
    -------
    ndarray
        Pixel value at each peak.
    list
        Compiled parameter list for passing with :obj:`wrapper_fit_func`.

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
        Indicates current iteration of :mod:`~wavgen.card.Card.stabilize_intensity`
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
        Indicates current iteration of :mod:`~wavgen.card.Card.stabilize_intensity`
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
