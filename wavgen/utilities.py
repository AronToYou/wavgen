import os
import sys
import h5py
import time
import inspect
import numpy as np
import matplotlib.pyplot as plt
from .config import *
from math import ceil
from instrumental import u
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import SpanSelector, Slider
from .spectrum import SPCSEQ_END, SPCSEQ_ENDLOOPALWAYS, SPCSEQ_ENDLOOPONTRIG


rp = [1.934454997984215 , 2.8421067958595616 , 2.677047569335915 , 1.1721824508892977 , 6.158065366794917 ,
    3.2691669970332335 , 2.636275384021578 , 1.6254638780707589 , 4.919003540925028 , 1.6084971058993613 ,
    5.2499387038268575 , 2.3688357219496265 , 4.713893357925578 , 5.223088585470364 , 0.3257672775855246 ,
    2.9571038289407126 , 2.4258010454280505 , 4.084691833872798 , 6.1867748426923335 , 5.200604534623386 ,
    3.3056812953203925 , 4.189137888598024 , 1.7650458297661427 , 4.080234513102615 , 0.6054340441874929 ,
    1.6794564559420377 , 2.385531129338364 , 5.400612735688388 , 4.978163766484847 , 5.335873096123345 ,
    0.9273414057111622 , 2.4193737371833834 , 2.8777346889035185 , 6.214778264445415 , 3.758998982400149 ,
    3.7838618270241438 , 0.60809445869596 , 0.1507635470741596 , 4.371624180280478 , 4.539661740808455 ,
    0.3847626491973457 , 6.145153550108536 , 1.008385520345513 , 5.852133555294753 , 0.016620198470431467 ,
    2.0158660597106937 , 1.7261705033296812 , 5.223710321703292 , 2.2220833343473436 , 2.9180968688523863 ,
    2.122206092376529 , 5.402785161537129 , 5.478771156577643 , 2.291512850266888 , 1.5715835663916051 ,
    2.255249593007268 , 1.571931477334538 , 1.3993650740616836 , 0.6011622182733365 , 3.1927489491586014 ,
    4.381746015200942 , 1.974081456041723 , 1.393542167751563 , 5.521906837731298 , 5.612290110455913 ,
    2.31118503089683 , 4.829965025115874 , 0.3421538142269762 , 4.555158230853398 , 1.6134448025783288 ,
    6.157248240200644 , 5.027656526405459 , 0.295901526406544 , 5.502983369799478 , 4.472320872860696 ,
    1.7618458333352276 , 4.41379605495804 , 4.6652622669145725 , 3.379174996566024 , 2.9970834472120313 ,
    4.886226685869682 , 4.340847582571988 , 0.3684494418446467 , 3.3447731714626525 , 0.3569784383241427 ,
    0.2362652137260263 , 4.420022732699935 , 6.263528358483921 , 6.2277672316776505 , 6.0305138883226554 ,
    2.5228306972997183 , 0.29710864827838496 , 0.5164352609138518 , 3.079335706611155 , 0.7796787693888715 ,
    2.9068441712875255 , 3.3802818513629718 , 0.16738916961106443 , 1.7466706296839072 , 0.7532941316251239]
""" 100 Pre-calculated Random Phases in range [0, 2pi] """


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
        self.Frequency = int(freq)
        self.Magnitude = mag
        self.Phase = phase

    def __lt__(self, other):
        return self.Frequency < other.Frequency

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return "Frequency: %d; Magnitude: %f; Phase: %f" % (self.Frequency, self.Magnitude, self.Phase)


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
def from_file(filepath, datapath):
    """ Extracts parameters from a HDF5 dataset and constructs the corresponding Waveform object.

        Parameters
        ----------
        filepath : str
            The name of the :doc:`HDF5 <../info/hdf5>` file.
        datapath : str
            Path to a specific dataset in the HDF5 database.

        Returns
        -------
        :class:`~wavgen.waveform.Waveform`
            Marshals extracted parameters into correct Waveform subclass constructor, returning the resulting object.
    """
    classes = inspect.getmembers(sys.modules['wavgen.waveform'], inspect.isclass)

    filepath = os.path.splitext(filepath)[0] + '.h5'

    kwargs = {}
    with h5py.File(filepath, 'r') as f:
        ## Maneuver to relevant Data location ##
        dat = f.get(datapath)
        assert dat is not None, "Invalid datapath"

        ## Waveform's Python Class name ##
        class_name = dat.attrs.get('class')

        ## Extract the Arguments ##
        try:
            for key in dat.attrs.get('keys'):
                kwargs[key] = dat.attrs.get(key)
        except TypeError:
            pass

    obj = None
    ## Find the proper Class & Construct it ##
    for name, cls in classes:
        if class_name == name:
            obj = cls.from_file(**kwargs)
            break
    assert obj, "The retrieved 'class' attribute matches no module class"

    ## Configure Status ##
    obj.Latest = True
    obj.FilePath = filepath
    obj.DataPath = datapath

    return obj


def y_limits(wav):
    with h5py.File(wav.FilePath, 'r') as f:
        data = f.get(wav.DataPath)
        N = data.shape[0]
        loops = ceil(N/DATA_MAX)

        semifinals = np.empty((loops, 2), dtype=data.dtype)

        for i in range(loops):
            n = i*DATA_MAX

            dat = data[n:min(n + DATA_MAX, N)]
            semifinals[i][:] = [dat.max(), dat.min()]

    M = semifinals.transpose()[:][0].max().astype(np.int32)
    m = semifinals.transpose()[:][1].min().astype(np.int32)
    margin = (M - m) * 5E-2

    return M+margin, m-margin


def plot_waveform(wav):
    original = wav  # Dirty Fix: Waveform that doesn't disappear on function return

    ## Determine the Concatenated Waveform ##
    if isinstance(wav, list):
        original = wav[0]  # Part 2 of the Dirty Fix

        sample_length = sum([w.SampleLength for w in wav])
        layout = h5py.VirtualLayout(shape=(sample_length,), dtype='int16')
        so_far = 0
        for w in wav:
            with h5py.File(w.FilePath, 'r') as f:
                dset = f.get(w.DataPath)
                vdset = h5py.VirtualSource(dset)
            layout[so_far:so_far + w.SampleLength] = vdset[()]
            so_far += w.SampleLength

        plot_waveform.cats += 1
        name = 'Concatenated Waveform %d' % plot_waveform.cats
        with h5py.File('temporary.h5', 'a', libver='latest') as f:
            ## Mimics a stored Waveform object ##
            dset = f.create_virtual_dataset(name, layout)
            dset.attrs.create('class', data='Waveform')
            dset.attrs.create('sample_length', data=sample_length)
            dset.attrs.create('keys', data=['sample_length'])

        wav = from_file('temporary.h5', name)  # Then creates the Waveform object from the mimic

    ## Plot Dataa & Parameters ##
    N = min(wav.SampleLength, PLOT_MAX)
    xdat = np.arange(N)
    ydat = np.zeros(N, dtype='int16')
    wav.load(ydat, 0, N)
    M, m = y_limits(wav)

    ## Figure Creation ##
    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))

    ax1.set(facecolor='#FFFFCC')
    lines = ax1.plot(xdat, ydat, '-')
    ax1.set_ylim((m, M))
    ax1.set_title(os.path.basename(wav.DataPath))
    ax1.set_ylabel('Sample Value')

    ax2.set(facecolor='#FFFFCC')
    ax2.plot(xdat, ydat, '-')
    ax2.set_ylim((m, M))
    ax2.set_title('Click & Drag on top plot to zoom in lower plot')
    ax2.set_ylabel('Sample Value')

    ## Slider ##
    def scroll(value):
        offset = int(value)
        xscrolled = np.arange(offset, offset + N)
        wav.load(ydat, offset, N)

        if len(lines) > 1:
            for line, y in zip(lines, ydat.transpose()):
                line.set_data(xscrolled, y)
        else:
            lines[0].set_data(xscrolled, ydat)

        ax1.set_xlim(xscrolled[0] - 100, xscrolled[-1] + 100)
        fig.canvas.draw()

    slid = None
    if N != wav.SampleLength:  # Only include a scroller if Waveform is large enough
        axspar = plt.axes([0.14, 0.94, 0.73, 0.05])
        slid = Slider(axspar, 'Scroll', valmin=0, valmax=wav.SampleLength - N, valinit=0, valfmt='%d', valstep=10)
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
    original.PlotObjects.append(span)

    plt.show(block=False)


plot_waveform.cats = 0


def plot_ends(wav):
    N = PLOT_MAX // 32
    N += 1 if N % 2 else 0

    M, m = y_limits(wav)

    xdat = np.arange(N)
    end_dat, begin_dat = np.zeros(N, dtype='int16'), np.zeros(N, dtype='int16')
    wav.load(begin_dat, 0, N)
    wav.load(end_dat, wav.SampleLength - N, N)

    ## Figure Creation ##
    fig = plt.figure(figsize=(10, 4), constrained_layout=True)
    fig.suptitle(os.path.basename(wav.DataPath), fontsize=14)

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
    end_ax.set_ylabel('Sample Value')

    cat_ax.set(facecolor='#FFFFCC')
    cat_ax.plot(np.arange(2 * N), np.concatenate((end_dat, begin_dat)), '-')
    cat_ax.set_ylim((m, M))
    cat_ax.set_title("Above examples concatenated")
    cat_ax.set_ylabel('Sample Value')

    plt.show(block=False)


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
    verboseprint("Fig_Newton")


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
    verboseprint("Fetching Frame")
    im = cam.latest_frame()
    x_len = len(im)
    verboseprint("Fetching Exposure")
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
                verboseprint("Clipping at: ", exp_t)
            right = exp_t
        elif gap > 50:
            if verbose:
                verboseprint("Closing gap: ", gap, " w/ exposure: ", exp_t)
            left = exp_t
        else:
            if verbose:
                verboseprint("Final Exposure: ", exp_t)
            return

        if inc < 0.01:
            exp_t -= inc if gap == 0 else -inc
        else:
            exp_t = (right + left) / 2
            inc = (right - left) / 10

        slider.set_val(exp_t)
        time.sleep(0.5)
        im = cam.latest_frame()

verboseprint = print if VERBOSE else lambda *a, **k: None
""" Print function which works only when global VERBOSE parameter is set. """

debugprint = print if DEBUG else lambda *a, **k: None
""" Print function which works only when global DEBUG parameter is set. """
