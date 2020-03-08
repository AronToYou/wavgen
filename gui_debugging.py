## For Cam Control ##
from instrumental import instrument, u
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
from scipy.optimize import curve_fit
## Other ##
import time
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

MAX_EXP = 150



def stabilize_intensity(which_cam, cam, verbose=False):
    """ Given a UC480 camera object (instrumental module) and
        a number indicating the number of trap objects,
        applies an iterative image analysis to individual trap adjustment
        in order to achieve a nearly homogeneous intensity profile across traps.

    """
    L = 0.5  # Correction Rate
    mags = np.ones(12)            ### !
    ntraps = len(mags)
    iteration = 0
    while iteration < 5:
        iteration += 1
        print("Iteration ", iteration)

        im = cam.latest_frame()
        try:
            trap_powers = analyze_image(which_cam, im, ntraps, iteration, verbose)
        except (AttributeError, ValueError) as e:
            print("No Bueno, error occurred during image analysis:\n", e)
            break

        mean_power = trap_powers.mean()
        rel_dif = 100 * trap_powers.std() / mean_power
        print(f'Relative Power Difference: {rel_dif:.2f} %')
        if rel_dif < 0.8:
            print("WOW")
            break

        deltaP = [mean_power - P for P in trap_powers]
        dmags = [(dP / abs(dP)) * sqrt(abs(dP)) * L for dP in deltaP]
        mags = np.add(mags, dmags)
        print("Magnitudes: ", mags)
        break
        # self._update_magnitudes(mags)
    _ = analyze_image(im, ntraps, verbose=verbose)


def _run_cam(which_cam, verbose=False):

    names = ['ThorCam', 'ChamberCam']  # False, True
    ## https://instrumental-lib.readthedocs.io/en/stable/uc480-cameras.html ##
    cam = instrument(names[which_cam])

    ## Cam Live Stream ##
    cam.start_live_video(framerate=10 * u.hertz)
    exp_t = cam._get_exposure()

    ## Create Figure ##
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ## Animation Frame ##
    def animate(i):
        if cam.wait_for_frame():
            im = cam.latest_frame()
            ax1.clear()
            ax1.imshow(im)

    ## Button: Automatic Exposure Adjustment ##
    def find_exposure(event):
        fix_exposure(cam, set_exposure, verbose)

    ## Button: Intensity Feedback ##
    def stabilize(event):  # Wrapper for Intensity Feedback function.
        im = cam.latest_frame()
        print(analyze_image(which_cam, im, 12, 1, True))
        # stabilize_intensity(which_cam, cam, verbose)

    def snapshot(event):
        im = cam.latest_frame()
        guess_image(which_cam, im, 12)

    def switch_cam(event):
        nonlocal cam, which_cam
        cam.close()

        which_cam = not which_cam
        
        cam = instrument(names[which_cam])
        cam.start_live_video(framerate=10 * u.hertz)


    # ## Button: Pause ##
    # def playback(event):
    #     if playback.running:
    #         spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
    #         playback.running = 0
    #     else:
    #         spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER)
    #         playback.running = 1

    # playback.running = 1

    ## Slider: Exposure ##
    def adjust_exposure(exp_t):
        cam._set_exposure(exp_t * u.milliseconds)

    ## Button Construction ##
    axspos = plt.axes([0.56, 0.0, 0.13, 0.05])
    axstab = plt.axes([0.7, 0.0, 0.1, 0.05])
    # axstop = plt.axes([0.81, 0.0, 0.12, 0.05])
    axplot = plt.axes([0.81, 0.0, 0.09, 0.05])   ### !
    axswch = plt.axes([0.91, 0.0, 0.09, 0.05])
    axspar = plt.axes([0.14, 0.9, 0.73, 0.05])

    correct_exposure = Button(axspos, 'AutoExpose')
    stabilize_button = Button(axstab, 'Stabilize')
    # pause_play = Button(axstop, 'Pause/Play')
    plot_snapshot = Button(axplot, 'Plot')
    switch_cameras = Button(axswch, 'Switch')
    set_exposure = Slider(axspar, 'Exposure', valmin=0.1, valmax=MAX_EXP, valinit=exp_t.magnitude)

    correct_exposure.on_clicked(find_exposure)
    stabilize_button.on_clicked(stabilize)
    # pause_play.on_clicked(playback)
    plot_snapshot.on_clicked(snapshot)
    switch_cameras.on_clicked(switch_cam)
    set_exposure.on_changed(adjust_exposure)

    ## Begin Animation ##
    _ = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()
    plt.close(fig)


########## Helper Functions ###############
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
    threshes = [0.5, 0.65]
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
    popt, pcov = curve_fit(lambda x, *params_0: wrapper_fit_func(x, ntraps, params_0),
                           xdata, peak_vals, p0=params0)
    if verbose:
        print("Fit!")
        plt.figure()
        plt.plot(xdata, peak_vals)                                            # Data
        if iteration:
            plt.plot(xdata, wrapper_fit_func(xdata, ntraps, params0), '--r')  # Initial Guess
            plt.plot(xdata, wrapper_fit_func(xdata, ntraps, popt))            # Fit
            plt.title("Iteration: %d" % iteration)
        else:
            plt.title("Final Product")

        plt.xlim((pos_first - margin, pos_last + margin))
        plt.legend(["Data", "Guess", "Fit"])
        plt.show(block=False)
        print("Fig_Newton")
    trap_powers = np.frombuffer(popt[2 * ntraps:3 * ntraps])
    return trap_powers


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
        time.sleep(1)
        im = cam.latest_frame()



_run_cam(True, True)