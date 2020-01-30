## For Card Control ##
from lib.pyspcm import *
from lib.spcm_tools import *
## For Cam Control ##
from instrumental import instrument, u
import matplotlib.animation as animation
from matplotlib.widgets import Button
from scipy.optimize import curve_fit
## Other ##
import sys
import time
import easygui
import matplotlib.pyplot as plt
import numpy as np
from math import sin, pi, sqrt
import random
import bisect
import pickle
import warnings


## Warning Suppression ##
warnings.filterwarnings("ignore", category=FutureWarning, module="instrumental")

### Constants ###
SAMP_VAL_MAX = (2 ** 15 - 1)  # Maximum digital value of sample ~~ signed 16 bits
SAMP_FREQ_MAX = 1250E6  # Maximum Sampling Frequency

### Parameter ###
SAMP_FREQ = 1000E6  # Modify if a different Sampling Frequency is required.


# noinspection PyTypeChecker,PyUnusedLocal
class OpenCard:
    """ Class designed for Opening, Configuring, running the Spectrum AWG card.

        CLASS VARIABLES:
            + hCard ---- The handle to the open card. For use with Spectrum API functions.
            + ModeBook - Dictionary for retrieving board register constants from key phrases.
        MEMBER VARIABLES:
            + ModeReady - Indicator of setup_mode()
            + ChanReady - Indicator of setup_channels()
            + BufReady  - Indicator of setup_buffer()
            + Mode      - Most recent mode card was configured to
            + Segments  - List of Segment objects

        USER METHODS:
            + set_mode(mode) ------------------------------ Set the card operation mode, e.g. multiple, continuous.
            + setup_channels(amplitude, ch0, ch1, filter) - Activates chosen channels and Configures Triggers.
            + setup_buffer() ------------------------------ Transfers the waveform to Board Memory
            + load_segments(segs) ------------------------- Appends a set of segments to the current set.
            + clear_segments() ---------------------------- Clears out current set of Segments.
            + reset_card() -------------------------------- Resets all of the cards configuration. Doesn't close card.
        PRIVATE METHODS:
            + _error_check() ------------------------------- Reads the card's error register.
            + _compute_and_load(seg, Ptr, buf, fsamp) ------ Computes a Segment and Transfers to Card.
    """
    ## Handle on card ##
    # We make this a class variable because there is only 1 card in the lab.
    # This simplifies enforcing 1 instance.
    hCard = None
    ModeBook = {  # Dictionary of Mode Names to Register Value Constants
        'continuous': SPC_REP_STD_CONTINUOUS,
        'multi': SPC_REP_STD_MULTI
        # 'sequence'  : SPC_REP_STD_SEQUENCE, --> Card doesn't possess feature :'(
    }

    def __init__(self, mode='continuous'):
        """ Just Opens the card in the given mode.
            INPUTS:
                mode  - Name for card output mode. limited support :)
            'single' or 'multiple' mode only (not yet supported)
                loops - Number of times the buffer is looped, 0 = infinity
        """
        assert self.hCard is None, "Card opened twice!"
        self.hCard = spcm_hOpen(create_string_buffer(b'/dev/spcm0'))  # Opens Card
        self._error_check()
        self.ModeReady = True
        self.ChanReady = False
        self.BufReady = False
        self.Mode = mode
        self.Segments = None
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_RESET)
        ## Setup Mode ##
        mode = self.ModeBook.get(mode)  # ModeBook is class object, look above
        if mode is None:
            print('Invalid mode phrase, possible phrases are: ')
            print(list(self.ModeBook.keys()))
            exit(1)
        spcm_dwSetParam_i32(self.hCard, SPC_CARDMODE, mode)
        if mode is 'continuous':
            loops = 0
            spcm_dwSetParam_i64(self.hCard, SPC_LOOPS, int64(loops))
        # elif mode is 'single':
        # elif mode is 'multiple':
        self._error_check()
        self.ModeReady = True

    def __exit__(self, exception_type, exception_value, traceback):
        print("in __exit__")
        spcm_vClose(self.hCard)

    ################# PUBLIC FUNCTIONS #################

    #### Segment Object Handling ####
    def clear_segments(self):
        self.Segments = None

    def load_segments(self, segs):
        if self.Segments is None:
            self.Segments = segs
        else:
            self.Segments.extend(segs)

    #### Basic Card Configuration Functions ####
    def set_mode(self, mode):
        if self.Mode != mode:
            self.BufReady = False
        spcm_dwSetParam_i32(self.hCard, SPC_CARDMODE, self.ModeBook.get(mode))
        self.Mode = mode
        self.ModeReady = True

    def setup_channels(self, amplitude=200, ch0=False, ch1=True, use_filter=False):
        """ Performs a Standard Initialization for designated Channels & Trigger
            INPUTS:
                amplitude -- Sets the Output Amplitude ~~ RANGE: [80 - 2000](mV) inclusive
                ch0 -------- Bool to Activate Channel0
                ch1 -------- Bool to Activate Channel1
                use_filter - Bool to Activate Output Filter
        """
        ## Input Validation ##
        if ch0 and ch1:
            print('Multi-Channel Support Not Yet Supported!')
            print('Defaulting to Ch1 only.')
            ch0 = False
        assert 80 <= amplitude <= 200, "Amplitude must within interval: [80 - 2000]"
        if amplitude != int(amplitude):
            amplitude = int(amplitude)
            print("Rounding amplitude to required integer value: ", amplitude)

        ## Channel Activation ##
        CHAN = 0x00000000
        amp = int32(amplitude)
        if ch0:
            spcm_dwSetParam_i32(self.hCard, SPC_ENABLEOUT0, 1)
            CHAN = CHAN ^ CHANNEL0
            spcm_dwSetParam_i32(self.hCard, SPC_AMP0, amp)
            spcm_dwSetParam_i64(self.hCard, SPC_FILTER0, int64(use_filter))
        if ch1:
            spcm_dwSetParam_i32(self.hCard, SPC_ENABLEOUT1, 1)
            CHAN = CHAN ^ CHANNEL1
            spcm_dwSetParam_i32(self.hCard, SPC_AMP1, amp)
            spcm_dwSetParam_i64(self.hCard, SPC_FILTER1, int64(use_filter))
        spcm_dwSetParam_i32(self.hCard, SPC_CHENABLE, CHAN)

        ## Trigger Config ##
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_ORMASK, SPC_TMASK_SOFTWARE)
        ## Necessary? Doesn't Hurt ##
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_ANDMASK,   0)
        spcm_dwSetParam_i64(self.hCard, SPC_TRIG_DELAY,     int64(0))
        spcm_dwSetParam_i32(self.hCard, SPC_TRIGGEROUT,     0)
        ############ ???? ###########
        self._error_check()
        self.ChanReady = True

    def setup_buffer(self):
        """ Calculates waves contained in Segments,
            configures the board memory buffer,
            then transfers to the board.
        """
        ## Validate ##
        assert self.ChanReady and self.ModeReady, "The Mode & Channels must be configured before Buffer!"
        assert len(self.Segments) > 0, "No Segments defined! Nothing to put in Buffer."

        ## Gather Information from Board ##
        num_chan = int32(0)  # Number of Open Channels
        mem_size = int64(0)  # Total Memory ~ 4.3 GB
        mode = int32(0)  # Operation Mode
        spcm_dwGetParam_i32(self.hCard, SPC_CHCOUNT, byref(num_chan))
        spcm_dwGetParam_i64(self.hCard, SPC_PCIMEMSIZE, byref(mem_size))
        spcm_dwGetParam_i32(self.hCard, SPC_CHCOUNT, byref(mode))

        seg = self.Segments[0]  # Only supports single segments currently

        ## Sets up a local Software Buffer for Transfer to Board ##
        buf_size = uint64(seg.SampleLength * 2 * num_chan.value)  # Calculates Buffer Size in Bytes
        pv_buf = pvAllocMemPageAligned(buf_size.value)  # Allocates space on PC
        pn_buf = cast(pv_buf, ptr16)  # Casts pointer into something usable

        ## Configures and Loads the Buffer ##
        spcm_dwSetParam_i64(self.hCard, SPC_MEMSIZE, int64(seg.SampleLength))
        self._compute_and_load(seg, pn_buf, pv_buf, buf_size)

        ## Clock ##
        spcm_dwSetParam_i32(self.hCard, SPC_CLOCKMODE, SPC_CM_INTPLL)  # Sets out internal Quarts Clock For Sampling
        spcm_dwSetParam_i64(self.hCard, SPC_SAMPLERATE, int64(int(SAMP_FREQ)))  # Sets Sampling Rate
        spcm_dwSetParam_i32(self.hCard, SPC_CLOCKOUT, 0)  # Disables Clock Output
        check_clock = int64(0)
        spcm_dwGetParam_i64(self.hCard, SPC_SAMPLERATE, byref(check_clock))  # Checks Sampling Rate
        print("Achieved Sampling Rate: ", check_clock.value)

        self._error_check()
        self.BufReady = True

    def wiggle_output(self, timeout=0, cam=True, verbose=False):
        """ Performs a Standard Output for configured settings.
            INPUTS:
                -- OPTIONAL --
                timeout - How long the output streams in Milliseconds.
                cam ----- Indicates whether to use Camera GUI.
            OUTPUTS:
                WAVES! (This function itself actually returns void)
        """
        if self.ChanReady and self.ModeReady and not self.BufReady:
            print("Psst..you need to reconfigure the buffer after switching modes.")
        assert self.BufReady and self.ChanReady and self.ModeReady, "Card not fully configured"
        if self.Mode == 'continuous':
            print("Looping Signal for ", timeout / 1000 if timeout else "infinity", " seconds...")
        if timeout != 0:
            WAIT = M2CMD_CARD_WAITREADY
        else:
            WAIT = 0
        spcm_dwSetParam_i32(self.hCard, SPC_TIMEOUT, timeout)
        dwError = spcm_dwSetParam_i32(self.hCard, SPC_M2CMD,
                                      M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | WAIT)
        count = 0
        while dwError == ERR_CLOCKNOTLOCKED:
            count += 1
            time.sleep(0.1)
            self._error_check(halt=False)
            dwError = spcm_dwSetParam_i32(self.hCard, SPC_M2CMD,
                                          M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | WAIT)
            if count == 10:
                break

        if dwError == ERR_TIMEOUT:
            print("timeout!")
            spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
        elif cam:
            self._run_cam(verbose)
        elif timeout == 0:
            easygui.msgbox('Stop Card?', 'Infinite Looping!')
            spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
        self._error_check()

    def stabilize_intensity(self, cam, verbose=False):
        """ Given a UC480 camera object (instrumental module) and
            a number indicating the number of trap objects,
            applies an iterative image analysis to individual trap adjustment
            in order to achieve a nearly homogeneous intensity profile across traps.

        """
        prev_magnitudes = self.Segments[0].get_magnitudes()
        ntraps = len(prev_magnitudes)
        iteration = 0
        while True:
            iteration += 1
            print("Iteration ", iteration)
            while not cam.wait_for_frame():
                pass
            print("Auto: ", cam.auto_exposure)
            im = cam.latest_frame()
            try:
                new_magnitudes = analyze_image(im, ntraps, verbose)
            except (AttributeError, ValueError) as e:
                print("No Bueno, error occurred: ", e)
                break
            print("New: ", new_magnitudes)
            print("Prev: ", prev_magnitudes)
            normalization = sqrt(np.linalg.norm(new_magnitudes, 1) * np.linalg.norm(prev_magnitudes, 1))
            similarity = new_magnitudes.dot(prev_magnitudes) / normalization
            if similarity > 0.95:
                print("WOW")
            print("similarity: ", similarity)
            # self._update_magnitudes(new_magnitudes)
            # prev_magnitudes = new_magnitudes
            break


    def reset_card(self):
        """ Wipes Card Configuration clean

        """
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_RESET)
        self.ModeReady = False
        self.ChanReady = False
        self.BufReady = False

    ################# PRIVATE FUNCTIONS #################

    def _error_check(self, halt=True):
        """ Checks the Error Register. If Occupied:
                -Prints Error
                -Closes the Card and exits program
        """
        ErrBuf = create_string_buffer(ERRORTEXTLEN)  # Buffer for returned Error messages
        if spcm_dwGetErrorInfo_i32(self.hCard, None, None, ErrBuf) != ERR_OK and halt:
            sys.stdout.write("{0}\n".format(ErrBuf.value))
            spcm_vClose(self.hCard)
            exit(1)

    def _compute_and_load(self, seg, ptr, buf, buf_size):
        """
            Computes the superposition of frequencies
            and stores it in the buffer.
            We divide by the number of waves and scale to the SAMP_VAL_MAX
            s.t. if all waves phase align, they will not exceed the max value.
            INPUTS:
                seg   - Segment which describes what frequencies & how many samples to compute.
                ptr   - Pointer to buffer to access the data.
                buf   - The buffer passed to the Spectrum Transfer Function.
                fsamp - The Sampling Frequency
        """
        ## Clear out Previous Values ##
        for i in range(seg.SampleLength):
            ptr[i] = 0

        ## Computes if Necessary, then copies Segment to software Buffer ##
        if not seg.Latest:
            seg.compute()
        for i in range(seg.SampleLength):
            ptr[i] = seg.Buffer[i]

        ## Do a Transfer ##
        spcm_dwDefTransfer_i64(self.hCard, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, int32(0), buf, uint64(0), buf_size)
        print("Doing a transfer...")
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
        print("Done")

    def _update_magnitudes(self, new_magnitudes):
        """ Subroutine used by stabilize_intensity()
            Turns off card, modifies each tone's magnitude, then lights it back up.

        """
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
        self.Segments[0].set_magnitude_all(new_magnitudes)
        self.setup_buffer()
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER)

    def _run_cam(self, verbose=False):
        """ Fires up the camera stream (ThorLabs UC480),
            then plots frames at a modifiable framerate in a Figure.
            Additionally, sets up special button functionality on the Figure.

        """
        ## https://instrumental-lib.readthedocs.io/en/stable/uc480-cameras.html ##
        ## ^^LOOK HERE^^ for driver documentation ##

        ## If you have problems here ##
        ## then see above doc &      ##
        ## Y:\E6\Software\Python\Instrument Control\ThorLabs UC480\cam_control.py ##
        cam = instrument('ThorCam')

        ## Cam Live Stream ##
        cam.start_live_video(framerate=10 * u.hertz, exposure_time=3*u.milliseconds)

        ## Fix Exposure ##
        fix_exposure(cam, verbose)

        ## Create Figure ##
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        ## Animation Frame ##
        def animate(i):
            if cam.wait_for_frame():
                im = cam.latest_frame()
                ax1.clear()
                ax1.imshow(im)

        ## Button: Intensity Feedback ##
        def stabilize(event):  # Wrapper for Intensity Feedback function.
            self.stabilize_intensity(cam, verbose)

        ## Button: Pause ##
        def playback(event):
            if playback.running:
                spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
                playback.running = 0
            else:
                spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER)
                playback.running = 1
        playback.running = 1

        ## Button Construction ##
        axstab = plt.axes([0.7, 0.0, 0.1, 0.05])
        axstop = plt.axes([0.81, 0.0, 0.15, 0.05])
        stabilize_button = Button(axstab, 'Stabilize')
        pause_play = Button(axstop, 'Pause/Play')
        stabilize_button.on_clicked(stabilize)
        pause_play.on_clicked(playback)

        ## Begin Animation ##
        _ = animation.FuncAnimation(fig, animate, interval=100)
        plt.show()
        plt.close(fig)
        self._error_check()
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)


######### Wave Class #########
class Wave:
    """
        MEMBER VARIABLES:
            + Frequency - (Hertz)
            + Magnitude - Relative Magnitude between [0 - 1] inclusive
            + Phase ----- (Radians)
    """
    def __init__(self, freq, mag=1, phase=0):
        ## Validate ##
        assert freq > 0, ("Invalid Frequency: %d, must be positive" % freq)
        assert 0 <= mag <= 1, ("Invalid magnitude: %d, must be within interval [0,1]" % mag)
        ## Initialize ##
        self.Frequency = freq
        self.Magnitude = mag
        self.Phase = phase

    def __lt__(self, other):
        return self.Frequency < other.Frequency


######### Segment Class #########
class Segment:
    """
        MEMBER VARIABLES:
            + Waves -------- List of Wave objects which compose the Segment. Sorted in Ascending Frequency.
            + Resolution --- (Hertz) The target resolution to aim for. In other words, sets the sample time (N / Fsamp)
                             and thus the 'wavelength' of the buffer (wavelength that completely fills the buffer).
                             Any multiple of the resolution will be periodic in the memory buffer.
            + SampleLength - Calculated during Buffer Setup; The length of the Segment in Samples.
            + Buffer ------- Storage location for calculated Wave.
            + Latest ------- Boolean indicating if the Buffer is the correct computation (E.g. correct Magnitude/Phase)

        USER METHODS:
            + add_wave(w) --------- Add the wave object 'w' to the segment, given it's not a duplicate frequency.
            + remove_frequency(f) - Remove the wave object with frequency 'f'.
            + plot() -------------- Plots the segment via matplotlib. Computes first if necessary.
            + randomize() --------- Randomizes the phases for each composing frequency of the Segment.
        PRIVATE METHODS:
            + _compute() - Computes the segment and stores into Buffer.
            + __str__() --- Defines behavior for --> print(*Segment Object*)
    """
    def __init__(self, freqs=None, waves=None, resolution=1E6, sample_length=None):
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
        if sample_length is not None:
            target_sample_length = int(sample_length)
            resolution = SAMP_FREQ / target_sample_length
        else:
            assert resolution < SAMP_FREQ / 2, ("Invalid Resolution, has to be below Nyquist: %d" % (SAMP_FREQ / 2))
            target_sample_length = int(SAMP_FREQ / resolution)
        if freqs is None and waves is not None:
            for i in range(len(waves)):
                assert waves[i].Frequency >= resolution, "Frequency must be greater than Resolution."
                assert waves[i].Frequency < SAMP_FREQ / 2, ("Frequencies must below Nyquist: %d" % (SAMP_FREQ / 2))
        elif freqs is not None and waves is None:
            for f in freqs:
                assert f >= resolution, ("Frequency %d is smaller than Resolution %d." % (f, resolution))
                assert f < SAMP_FREQ_MAX / 2, ("Frequencies must below Nyquist: %d" % (SAMP_FREQ / 2))
        else:
            assert False, "Must override either only 'freqs' or 'waves' input argument."
        ## Initialize ##
        if waves is None:
            self.Waves = [Wave(f) for f in freqs]
        else:
            self.Waves = waves
        self.Waves.sort(key=(lambda w: w.Frequency))
        self.SampleLength = (target_sample_length - target_sample_length % 32)
        self.Latest       = False
        self.Buffer       = None
        ## Report ##
        print("Sample Length: ", self.SampleLength)
        print('Target Resolution: ', resolution, 'Hz, Achieved resolution: ', SAMP_FREQ / self.SampleLength, 'Hz')


    def add_wave(self, w):
        """ Given a Wave object,
            adds to current Segment as long as it's not a duplicate.

        """
        for wave in self.Waves:
            if w.Frequency == wave.Frequency:
                print("Skipping duplicate: %d Hz" % w.Frequency)
                return
        resolution = SAMP_FREQ / self.SampleLength
        assert w.Frequency >= resolution, ("Resolution: %d Hz, sets the minimum frequency." % resolution)
        assert w.Frequency < SAMP_FREQ / 2, ("Frequencies must be below Nyquist: %d" % (SAMP_FREQ / 2))
        bisect.insort(self.Waves, w)
        self.Latest = False


    def remove_frequency(self, f):
        """ Given an input frequency,
            searches current Segment and removes a matching frequency if found.

        """
        self.Waves = [W for W in self.Waves if W.Frequency != f]
        self.Latest = False


    def get_magnitudes(self):
        """ Returns an array of magnitudes,
            each associated with a particular trap.

        """
        return [w.Magnitude for w in self.Waves]

    def set_magnitude(self, idx, mag):
        """ Sets the magnitude of the indexed trap number.
            INPUTS:
                idx - Index to trap number, starting from 0
                mag - New value for relative magnitude, must be in [0, 1]
        """
        assert 0 <= mag <= 1, ("Invalid magnitude: %d, must be within interval [0,1]" % mag)
        self.Waves[idx].Magnitude = mag
        self.Latest = False

    def set_magnitudes(self, mags):
        """ Sets the magnitude of all traps.
            INPUTS:
                mags - List of new magnitudes, in order of Trap Number (Ascending Frequency).
        """
        for i, mag in enumerate(mags):
            assert 0 <= mag <= 1, ("Invalid magnitude: %d, must be within interval [0,1]" % mag)
            self.Waves[i].Magnitude = mag
        self.Latest = False


    def set_phase(self, idx, phase):
        """ Sets the magnitude of the indexed trap number.
            INPUTS:
                idx --- Index to trap number, starting from 0
                phase - New value for phase.
        """
        phase = phase % (2*pi)
        self.Waves[idx].Phase = phase
        self.Latest = False


    def set_phase_all(self, phases):
        """ Sets the magnitude of all traps.
            INPUTS:
                mags - List of new phases, in order of Trap Number (Ascending Frequency).
        """
        for i, phase in enumerate(phases):
            self.Waves[i].Phase = phase
        self.Latest = False

    def plot(self):
        """ Plots the Segment. Computes first if necessary.

        """
        if not self.Latest:
            self.compute()
        plt.plot(self.Buffer, '--o')
        plt.show()


    def randomize(self):
        """ Randomizes each phase.

        """
        for w in self.Waves:
            w.Phase = 2*pi*random.random()
        self.Latest = False


    def compute(self):
        """ Computes the superposition of frequencies
            and stores it in the buffer.
            We divide by the sum of relative wave magnitudes
            and scale the max value to SAMP_VAL_MAX,
            s.t. if all waves phase align, they will not exceed the max value.
        """
        ## Checks if Redundant ##
        if self.Latest:
            return
        self.Latest = True  # Will be up to date after

        ## Initialize Buffer ##
        self.Buffer = np.zeros(self.SampleLength, dtype=np.int16)
        temp_buffer = np.zeros(self.SampleLength)

        ## Compute and Add the full wave, Each frequency at a time ##
        for w in self.Waves:
            fn = w.Frequency / SAMP_FREQ  # Cycles/Sample
            for i in range(self.SampleLength):
                temp_buffer[i] += w.Magnitude*sin(2 * pi * i * fn + w.Phase)

        ## Normalize the Buffer ##
        normalization = sum([w.Magnitude for w in self.Waves])
        for i in range(self.SampleLength):
            self.Buffer[i] = int(SAMP_VAL_MAX * (temp_buffer[i] / normalization))


    def save(self, name="unnamed_segment", data_only=False):
        if data_only:
            np.savetxt(name, self.Buffer, delimiter=",")
        else:
            pickle.dump(self, open(name, "wb"))

    def __str__(self):
        s = "Segment with Resolution: " + str(SAMP_FREQ / self.SampleLength) + "\n"
        s += "Contains Waves: \n"
        for w in self.Waves:
           s += "---" + str(w.Frequency) + "Hz - Magnitude: " \
                + str(w.Magnitude) + " - Phase: " + str(w.Phase) + "\n"
        return s


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


def is_clipping(image):
    """ Given an image (2d numpy array),
        returns a boolean indicating if the imaged
        traps saturating the camera pixels.

    """
    margin = 10
    im = image.transpose()
    x_len = len(im)

    for i in range(x_len):
        if i < margin or x_len - i < margin:
            continue
        else:
            if max(im[i]) >= 255:
                return True
    return False


def analyze_image(image, ntraps, verbose=False):
    ## Image Conditioning ##
    margin = 10
    threshold = np.max(image)*0.7
    im = image.transpose()

    ## Plot Image Quadrants ##
    if verbose:
        plt.figure()
        plt.subplot(221)
        plt.imshow(im[:len(im) // 2, :])
        plt.subplot(222)
        plt.imshow(im[len(im) // 2:, :])
        plt.subplot(223)
        plt.imshow(im[:, :len(im[0]) // 2])
        plt.subplot(224)
        plt.imshow(im[:, len(im[0]) // 2:])
        plt.show(block=False)
        print("After fig")

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
    for i, p in enumerate(peak_vals):
        if p > threshold:
            if first:
                pos_first = i
                first = False
            pos_last = i
    ## Separation Value ##
    separation = (pos_last - pos_first) / ntraps  # In Pixels

    ## Initial Guesses ##
    means0 = np.linspace(pos_first, pos_last, ntraps).tolist()
    waists0 = (separation * np.ones(ntraps) / 2).tolist()
    ampls0 = (max(peak_vals) * 0.7 * np.ones(ntraps)).tolist()
    _params0 = [means0, waists0, ampls0, [0.06]]
    params0 = [item for sublist in _params0 for item in sublist]

    ## Fitting ##
    if verbose: print("Fitting...")
    xdata = np.arange(x_len)
    popt, pcov = curve_fit(lambda x, *params_0: wrapper_fit_func(x, ntraps, params_0),
                           xdata, peak_vals, p0=params0)
    if verbose:
        print("Fit!")
        plt.figure()
        plt.plot(xdata, peak_vals)                                        # Data
        plt.plot(xdata, wrapper_fit_func(xdata, ntraps, params0), '--r')  # Initial Guess
        plt.plot(xdata, wrapper_fit_func(xdata, ntraps, popt))            # Fit

        plt.xlim((pos_first - margin, pos_last + margin))
        plt.legend(["Data", "Guess", "Fit"])
        plt.show(block=False)
        print("Fig_NEwton")
    ampls = list(popt[2 * ntraps:3 * ntraps])
    if verbose: print("Amps: ", ampls)
    mags = [min(ampls) / A for A in ampls]
    return np.array(mags)


# noinspection PyProtectedMember
def fix_exposure(cam, verbose=False):
    """ Given an opened camera object,
        adjusts the exposure until no clipping is present.

    """
    exp_t = cam._get_exposure()
    while is_clipping(cam.latest_frame()):
        if verbose: print("Clipping at: ", exp_t)
        exp_t *= 0.95
        cam._set_exposure(exp_t)
    if verbose: print("Final Exposure: ", exp_t.magnitude)