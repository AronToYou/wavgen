## For Card Control ##
from lib.pyspcm import *
from lib.spcm_tools import *
## For Cam Control ##
from instrumental import instrument, u
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
## Submodules ##
from .helper import fix_exposure, analyze_image
## Other ##
import sys
import time
import easygui
import matplotlib.pyplot as plt
import numpy as np
import warnings


## Warning Suppression ##
warnings.filterwarnings("ignore", category=FutureWarning, module="instrumental")

### Parameter ###
SAMP_FREQ = 1000E6  # Modify if a different Sampling Frequency is required.


# noinspection PyTypeChecker,PyUnusedLocal
class Card:
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
        'multi'     : SPC_REP_STD_MULTI,
        'sequential': SPC_REP_STD_SEQUENCE
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
        self.ProgrammedSequence = False if mode == 'sequential' else True
        self.Mode = mode
        self.Waveforms = None

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
        self._error_check()
        self.ModeReady = True


    def __exit__(self, exception_type, exception_value, traceback):
        print("in __exit__")
        spcm_vClose(self.hCard)

    ################# PUBLIC FUNCTIONS #################

    #### Segment Object Handling ####
    def load_waveforms(self, wavs):
        self.Waveforms = wavs

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

        assert 80 <= amplitude <= 240, "Amplitude must within interval: [80 - 2000]"
        if amplitude != int(amplitude):
            amplitude = int(amplitude)
            print("Rounding amplitude to required integer value: ", amplitude)

        ## Channel Activation ##
        CHAN = 0x00000000
        amp = int32(amplitude)
        if ch0:
            spcm_dwSetParam_i32(self.hCard, SPC_ENABLEOUT0, 1)
            CHAN = CHAN ^ CHANNEL0
            spcm_dwSetParam_i32(self.hCard, SPC_AMP0,       amp)
            spcm_dwSetParam_i64(self.hCard, SPC_FILTER0,    int64(use_filter))
        if ch1:
            spcm_dwSetParam_i32(self.hCard, SPC_ENABLEOUT1, 1)
            CHAN = CHAN ^ CHANNEL1
            spcm_dwSetParam_i32(self.hCard, SPC_AMP1,       amp)
            spcm_dwSetParam_i64(self.hCard, SPC_FILTER1,    int64(use_filter))
        spcm_dwSetParam_i32(self.hCard, SPC_CHENABLE,       CHAN)

        ## Trigger Config ##
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_ORMASK,    SPC_TMASK_SOFTWARE)
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
        assert len(self.Waveforms) > 0, "No Segments defined! Nothing to put in Buffer."

        ## Gather Information from Board ##
        num_chan = int32(0)  # Number of Open Channels
        mem_size = uint64(0)  # Total Memory ~ 4.3 GB
        spcm_dwGetParam_i32(self.hCard, SPC_CHCOUNT,    byref(num_chan))
        spcm_dwGetParam_i64(self.hCard, SPC_PCIMEMSIZE, byref(mem_size))
#######################################################################################################################
        ## Configures Memory Size & Divisions ##
        num_segs = 0
        for wav in self.Waveforms:
            num_segs += wav.NumSegments
        print("Num Segments: ", num_segs)

        if self.Mode == 'sequential':
            board_segs = int32(2)
            buf_size = max([wav.SampleLength for wav in self.Waveforms])*2*num_chan.value
            while board_segs.value < num_segs:
                board_segs.value <<= 1
            assert buf_size <= mem_size.value / board_segs.value, "One of the segments is too large!"

            spcm_dwSetParam_i32(self.hCard, SPC_SEQMODE_MAXSEGMENTS,    board_segs)
            spcm_dwSetParam_i32(self.hCard, SPC_SEQMODE_STARTSTEP,      0)
        else:
            buf_size = self.Waveforms[0].SampleLength*2*num_chan.value
            spcm_dwSetParam_i64(self.hCard, SPC_MEMSIZE,                int64(self.Waveforms[0].SampleLength))

        ## Sets up a local Software Buffer for Transfer to Board ##
        pv_buf = pvAllocMemPageAligned(buf_size)  # Allocates space on PC
        pn_buf = cast(pv_buf, ptr16)  # Casts pointer into something usable

        ## Loads each necessary Segment ##
        if self.Mode == 'sequential':
            for i, wav in enumerate(self.Waveforms):
                spcm_dwSetParam_i32(self.hCard, SPC_SEQMODE_WRITESEGMENT, i)
                spcm_dwSetParam_i32(self.hCard, SPC_SEQMODE_SEGMENTSIZE, seg.SampleLength) ###### Right Here! ###
                self._error_check()

                buf_size = seg.SampleLength * 2 * num_chan.value  # Calculates Segment Size in Bytes
                self._compute_and_load(seg, pn_buf, pv_buf, uint64(buf_size))
        else:
            buf_size = self.Segments[0].SampleLength * 2 * num_chan.value
            self._compute_and_load(self.Segments[0], pn_buf, pv_buf, uint64(buf_size))
#######################################################################################################################
        ## Clock ##
        spcm_dwSetParam_i32(self.hCard, SPC_CLOCKMODE,  SPC_CM_INTPLL)  # Sets out internal Quarts Clock For Sampling
        spcm_dwSetParam_i64(self.hCard, SPC_SAMPLERATE, int64(int(SAMP_FREQ)))  # Sets Sampling Rate
        spcm_dwSetParam_i32(self.hCard, SPC_CLOCKOUT,   0)  # Disables Clock Output
        check_clock = int64(0)
        spcm_dwGetParam_i64(self.hCard, SPC_SAMPLERATE, byref(check_clock))  # Checks Sampling Rate
        print("Achieved Sampling Rate: ", check_clock.value)

        self._error_check()
        self.BufReady = True

    def wiggle_output(self, timeout=0, cam=True, verbose=False):
        """ Performs a Standard Output for configured settings.
            INPUTS:
                -- OPTIONAL --
                + timeout - How long the output streams in Milliseconds.
                + cam ----- Indicates whether to use Camera GUI.
            OUTPUTS:
                WAVES! (This function itself actually returns void)
        """
        if self.ChanReady and self.ModeReady and not self.BufReady:
            print("Psst..you need to reconfigure the buffer after switching modes.")
        assert self.BufReady and self.ChanReady and self.ModeReady, "Card not fully configured"
        assert self.ProgrammedSequence, "If your using 'sequential' mode, you must us 'load_sequence()'."

        WAIT = 0
        if self.Mode == 'continuous':
            print("Looping Signal for ", timeout / 1000 if timeout else "infinity", " seconds...")
            if timeout != 0:
                WAIT = M2CMD_CARD_WAITREADY
            spcm_dwSetParam_i32(self.hCard, SPC_TIMEOUT, timeout)

        dwError = spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | WAIT)
        count = 0
        while dwError == ERR_CLOCKNOTLOCKED:
            count += 1
            time.sleep(0.1)
            self._error_check(halt=False)
            dwError = spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | WAIT)
            if count == 10:
                break

        if dwError == ERR_TIMEOUT:
            print("timeout!")
        elif cam:
            self._run_cam(verbose)
        elif self.Mode == 'sequential':
            while True:
                if easygui.boolbox('Send Trigger?', 'Running Sequence', ['exit', 'trigger']):
                    break
                spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_FORCETRIGGER)
        elif timeout == 0:
            easygui.msgbox('Stop Card?', 'Infinite Looping!')

        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
        print("End?")
        self._error_check()


    def load_sequence(self, steps, dump=True):
        """ Given a list of steps

        """
        assert self.Mode == 'sequential', "Cannot load sequence unless in Sequential mode."
        for step in steps:
            cur = step.CurrentStep
            seg = step.SegmentIndex
            loop = step.Loops
            nxt = step.NextStep
            cond = step.Condition
            reg_upper = int32(cond | loop)
            reg_lower = int32(nxt << 16 | seg)
            print("Step %.2d: 0x%08x_%08x\n" % (cur, reg_upper.value, reg_lower.value))
            spcm_dwSetParam_i64m(self.hCard, SPC_SEQMODE_STEPMEM0 + cur, reg_upper, reg_lower)
        self.ProgrammedSequence = True

        if dump:
            print("\nDump!:\n")
            for i in range(len(steps)):
                temp = uint64(0)
                spcm_dwGetParam_i64(self.hCard, SPC_SEQMODE_STEPMEM0 + i, byref(temp))
                print("Step %.2d: 0x%08x_%08x\n" % (i, int32(temp.value >> 32).value, int32(temp.value).value))


    def stabilize_intensity(self, cam, verbose=False):
        """ Given a UC480 camera object (instrumental module) and
            a number indicating the number of trap objects,
            applies an iterative image analysis to individual trap adjustment
            in order to achieve a nearly homogeneous intensity profile across traps.

        """
        L = 0.5  # Correction Rate
        mags = self.Segments[0].get_magnitudes()
        ntraps = len(mags)
        iteration = 0
        while True:
            iteration += 1
            print("Iteration ", iteration)

            im = cam.latest_frame()
            try:
                ampls = analyze_image(im, ntraps, iteration, verbose)
            except (AttributeError, ValueError) as e:
                print("No Bueno, error occurred during image analysis:\n", e)
                break

            rel_dif = 100 * np.std(np.array(ampls)) / np.mean(np.array(ampls))
            print(f'Relative Difference: {rel_dif:.2f} %')
            if rel_dif < 0.8:
                print("WOW")
                break

            ampls = [min(ampls)*L / A - L + 1 for A in ampls]
            mags = np.multiply(mags, ampls)
            mags = [mag + 1 - max(mags) for mag in mags]  # Shift s.t. ALL <= 1
            print("Magnitudes: ", mags)
            self._update_magnitudes(mags)
        _ = analyze_image(im, ntraps, verbose=verbose)



    def reset_card(self):
        """ Wipes Card Configuration clean

        """
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_RESET)
        self.ModeReady = False
        self.ChanReady = False
        self.BufReady = False

    ################# PRIVATE FUNCTIONS #################

    def _error_check(self, halt=True):
        """ Checks the Error Register.
        If Occupied:
                -Prints Error
                -Optionally closes the Card and exits program
                -Or returns False
        Else:   -Returns True
        """
        ErrBuf = create_string_buffer(ERRORTEXTLEN)  # Buffer for returned Error messages
        if spcm_dwGetErrorInfo_i32(self.hCard, None, None, ErrBuf) != ERR_OK:
            sys.stdout.write("Warning: {0}".format(ErrBuf.value))
            if halt:
                spcm_vClose(self.hCard)
                exit(1)
            return False
        return True

    def load(self, seg, ptr, buf, buf_size):
        """
            Computes the superposition of frequencies
            and stores it in the buffer.
            We divide by the number of waves and scale to the SAMP_VAL_MAX
            s.t. if all waves phase align, they will not exceed the max value.
            INPUTS:
                seg   ---- Segment which describes what frequencies & how many samples to compute.
                ptr   ---- Pointer to buffer to access the data.
                buf   ---- The buffer passed to the Spectrum Transfer Function.
                buf_size - Size of the buffer in bytes.
        """
        ## Computes if Necessary, then copies Segment to software Buffer ##
        if not seg.Latest:
            seg.compute()
        for i in range(seg.SampleLength):
            ptr[i] = seg.Buffer[i]

        ## Do a Transfer ##
        spcm_dwDefTransfer_i64(self.hCard, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, int32(0), buf, uint64(0), buf_size)
        print("Doing a transfer...%d bytes" % buf_size.value)
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
        print("Done")

    def _update_magnitudes(self, new_magnitudes):
        """ Subroutine used by stabilize_intensity()
            Turns off card, modifies each tone's magnitude, then lights it back up.

        """
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
        self.Segments[0].set_magnitudes(new_magnitudes)
        self.setup_buffer()
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER)
        time.sleep(1)

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

        ## Button: Exposure Adjustment ##
        def find_exposure(event):
            fix_exposure(cam, set_exposure, verbose)

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

        ## Slider: Exposure ##
        def adjust_exposure(exp_t):
            cam._set_exposure(exp_t * u.milliseconds)

        ## Button Construction ##
        axspos = plt.axes([0.56, 0.0, 0.13, 0.05])
        axstab = plt.axes([0.7,  0.0, 0.1,  0.05])
        axstop = plt.axes([0.81, 0.0, 0.12, 0.05])
        axspar = plt.axes([0.14, 0.9, 0.73, 0.05])
        correct_exposure = Button(axspos, 'AutoExpose')
        stabilize_button = Button(axstab, 'Stabilize')
        pause_play       = Button(axstop, 'Pause/Play')
        set_exposure     = Slider(axspar, 'Exposure', valmin=0.1, valmax=30, valinit=exp_t.magnitude)
        correct_exposure.on_clicked(find_exposure)
        stabilize_button.on_clicked(stabilize)
        pause_play.on_clicked(playback)
        set_exposure.on_changed(adjust_exposure)

        ## Begin Animation ##
        _ = animation.FuncAnimation(fig, animate, interval=100)
        plt.show()
        plt.close(fig)
        self._error_check()