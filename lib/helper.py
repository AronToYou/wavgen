import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from instrumental import u

MAX_EXP = 150    # Maximum value for Thorcam exposure

# noinspection PyPep8Naming
def gaussian1d(x, x0, w0, A, offset):
    """ Returns intensity profile of 1d gaussian beam
        x0:  x-offset
        w0:  waist of Gaussian beam
        A:   Amplitude
        offset: Global offset
    """
    if w0 == 0:
        return 0
    return A * np.exp(-2 * (x - x0) ** 2 / (w0 ** 2)) + offset


# noinspection PyPep8Naming
def gaussianarray1d(x, x0_vec, wx_vec, A_vec, offset, ntraps):
    """ Returns intensity profile of trap array
        x0_vec: 1-by-ntraps array of x-offsets of traps
        wx_vec: 1-by-ntraps array of waists of traps
        A_vec:  1-by-ntraps array of amplitudes of traps
        offset: global offset
        ntraps: Number of traps

    """
    array = np.zeros(np.shape(x))
    for k in range(ntraps):
        array = array + gaussian1d(x, x0_vec[k], wx_vec[k], A_vec[k], 0)
    return array + offset


def wrapper_fit_func(x, ntraps, *args):
    """ Juggles parameters in order to be able to fit a list of parameters

    """
    a, b, c = list(args[0][:ntraps]), list(args[0][ntraps:2 * ntraps]), list(args[0][2 * ntraps:3 * ntraps])
    offset = args[0][-1]
    return gaussianarray1d(x, a, b, c, offset, ntraps)


def guess_image(which_cam, image, ntraps):
    """ Scans the given image for the 'ntraps' number of trap intensity peaks.
        Then extracts the 1-dimensional gaussian profiles across the traps and
        returns a list of the amplitudes.

    """
    threshes = [0.5, 0.6]
    ## Image Conditioning ##
    margin = 10
    threshold = np.max(image)*threshes[which_cam]
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
        elif p < threshold and left_pos != 0:
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

    xdata = np.arange(x_len)
    plt.figure()
    plt.plot(xdata, peak_vals)
    plt.plot(xdata, wrapper_fit_func(xdata, ntraps, params0), '--r')  # Initial Guess
    plt.xlim((pos_first - margin, pos_last + margin))
    plt.legend(["Data", "Guess", "Fit"])
    plt.title("Trap Intensities")
    plt.ylabel("Pixel Value (0-255)")
    plt.xlabel("Pixel X-Position")
    plt.show(block=False)


def analyze_image(which_cam, image, ntraps, iteration=0, verbose=False):
    """ Scans the given image for the 'ntraps' number of trap intensity peaks.
        Then extracts the 1-dimensional gaussian profiles across the traps and
        returns a list of the amplitudes.

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

    ## Fitting ##
    if verbose:
        print("Fitting...")
    xdata = np.arange(x_len)
    success = True
    try:
        popt, pcov = curve_fit(lambda x, *params_0: wrapper_fit_func(x, ntraps, params_0),
                           xdata, peak_vals, p0=params0)
    except RuntimeError:
        success = False


    if verbose:
        print("Fit!")
        plt.figure()
        plt.plot(xdata, peak_vals)                                            # Data
        if iteration:
            plt.plot(xdata, wrapper_fit_func(xdata, ntraps, params0), '--r')  # Initial Guess
            if success:
                plt.plot(xdata, wrapper_fit_func(xdata, ntraps, popt))            # Fit
            plt.title("Iteration: %d" % iteration)
        else:
            plt.title("Final Product")

        plt.xlim((pos_first - margin, pos_last + margin))
        plt.legend(["Data", "Guess", "Fit"])
        plt.show(block=False)
        print("Fig_Newton")
    if success:
        trap_powers = np.frombuffer(popt[2 * ntraps:3 * ntraps])
        return trap_powers
    else:
        return np.ones(ntraps)


# noinspection PyProtectedMember
def fix_exposure(cam, slider, verbose=False):
    """ Given the opened camera object and the Slider
        object connected to the camera's exposure,
        adjusts the exposure to just below clipping.
        *Binary Search*
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
