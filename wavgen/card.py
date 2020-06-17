## For Card Control ##
from spectrum import *
## For Cam Control ##
from instrumental import instrument, u
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
## Submodules ##
from .utilities import fix_exposure, analyze_image, plot_image, Step
from .waveform import Superposition
## Other ##
from math import log2, ceil, sqrt
import sys
from time import time, sleep
import easygui
import matplotlib.pyplot as plt
import numpy as np
import warnings


## Suppresses a Deprecation warning from instrumental ##
warnings.filterwarnings("ignore", category=FutureWarning, module="instrumental")

### Parameter ###
MEM_SIZE = 4294967296  #: Size of the *board's memory (bytes)  *Spectrum M4i.6631-x8
SAMP_FREQ = 1000E6     #: Modify if a different Sampling Frequency is required.
NUMPY_MAX = int(1E5)   #: Max size of Software buffer for board transfers (in samples)
MAX_EXP = 150          #: Cap on the exposure value for ThorCam devices.
DEF_AMP = 210          #: Default maximum waveform output amplitude (milliVolts)
VERBOSE = False        #: Flag to de/activate most print messages throughout program.


# noinspection PyTypeChecker,PyUnusedLocal,PyProtectedMember
class Card:
    """ Class designed for Opening, Configuring, running the Spectrum AWG card.

    ATTRIBUTES
    ----------
    hCard
        The handle to the open card. For use with Spectrum API functions.
    ChanReady, BufReady, Sequence : Bool
        Indicators of card configuration completion.
    Wave : :obj:`Superposition`
        Object containing a trap configuration's waveform parameters.
        Used when optimizing the parameters for homogeneous trap intensity.
    """
    hCard = None

    def __init__(self):
        """ Establishes connection to Spectrum card.
            The returned object is a handle to that connection.
        """
        assert self.hCard is None, "Card opened twice!"

        self.hCard = spcm_hOpen(create_string_buffer(b'/dev/spcm0'))
        """Handle to card device. *Class object* See `spectrum.pyspcm.py`"""
        self._error_check()

        self.ChanReady = False  #: bool : Indicates channels are setup.
        self.BufReady = False   #: bool : Indicates the card buffer is configured & loaded with waveform data.
        self.Sequence = None    #: bool : T/F indicates if sequence steps have been loaded. None implies straight mode.
        self.Wave = None        #: :obj:`Superposition` : Waveform to optimize.

        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_RESET)  # Clears the card's configuration

        self._error_check()

    def __del__(self):
        print("in __exit__")
        spcm_vClose(self.hCard)

    ################# PUBLIC FUNCTIONS #################

    def setup_channels(self, amplitude=DEF_AMP, ch0=False, ch1=True, use_filter=False):
        """ Performs a Standard Initialization for designated Channels & Trigger.

        INPUTS:
        -------
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

        assert 80 <= amplitude <= 250, "Amplitude must within interval: [80 - 2000]"
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

        # TODO: Trigger Config
        # spcm_dwSetParam_i32(self.hCard, SPC_TRIG_ORMASK,    SPC_TMASK_SOFTWARE)
        # ## Necessary? Doesn't Hurt ##
        # spcm_dwSetParam_i32(self.hCard, SPC_TRIG_ANDMASK,   0)
        # spcm_dwSetParam_i64(self.hCard, SPC_TRIG_DELAY,     int64(0))
        # spcm_dwSetParam_i32(self.hCard, SPC_TRIGGEROUT,     0)
        ############ Only relevant for 'continuous' mode? ###########

        self._error_check()
        self.ChanReady = True

    def load_waveforms(self, wavs, offset=0):
        """ Writes a set of waveforms as a single block to card.

            Note
            ----
            This operation will wipe any waveforms that were previously on the card.

            Parameters
            ----------
            wavs : :obj:`Waveform`, list of :obj:`Waveform`
                The given waves will be transferred to board memory in order.
            offset : int, optional
                If data already exists on the board, you can partially overwrite it
                by indicating where to begin writing (in bytes from the mem start).
                **You cannot exceed the set size of the pre-existing data**
        """
        ## Sets channels to default mode if no user setting ##
        if not self.ChanReady:
            self.setup_channels()

        ## Saves single waveforms for optimization functions ##
        if not isinstance(wavs, list) or len(wavs) == 1:
            self.Wave = wavs
            wavs = wavs if isinstance(wavs, list) else [wavs]  # wavs needs to be a list

        spcm_dwSetParam_i32(self.hCard, SPC_CARDMODE, SPC_REP_STD_CONTINUOUS)  # Sets the mode

        ## Define the Size of required Board Memory ##
        sample_length = sum([wav.SampleLength for wav in wavs])
        assert sample_length * 2 <= MEM_SIZE, "Waves exceed board capacity by %d bytes" % (sample_length * 2 - MEM_SIZE)
        if offset:
            cur_mem_size = int64(0)
            spcm_dwGetParam_i64(self.hCard, SPC_MEMSIZE, byref(cur_mem_size))
            assert offset + sample_length <= cur_mem_size.value, "Your waveform exceeds the previously set mem capacity"
        else:
            spcm_dwSetParam_i64(self.hCard, SPC_MEMSIZE, int64(sample_length))

        ## Sets up a local Software Buffer then Transfers to Board ##
        pv_buf = pvAllocMemPageAligned(NUMPY_MAX * 2)  # Allocates space on PC
        pn_buf = cast(pv_buf, ptr16)  # Casts pointer into something usable

        self._write_segment(wavs, pv_buf, pn_buf, offset)

        ## Setup the Clock & Wrap Up ##
        self._setup_clock()
        self._error_check()
        self.BufReady = True
        self.Sequence = None

    def load_sequence(self, segments=None, steps=None):
        """
        Given a list of waveform/index tuples
        or
        list of :class:`Step`
        will add new or overwrite existing data on board memory
        with the new data.

        Parameters
        ----------
        segments : list of :class:`Waveform`, list of (int, :class:`Waveform`)
            Waveform objects to each be written to a board segment.
            If paired with `int`s indicating segment index,
            then the board memory will not be wiped/re-divided,
            but instead indexed (allowing partial overwrites).
        steps : list of :class:`Step`
            Each step is written to associated segment on board memory
            (defining that segments transition rule).

        Examples
        --------
        ::
            hCard  # Opened & configured Board handle
            myWaves = [(0, wav0), (2, wav2), (3, wav3)]
            mySteps = [step1, step3]
            hCard.load_sequence(myWaves, mySteps)
        """
        assert steps is not None or segments is not None, "No data given to load!"
        if not self.ChanReady:  # Sets channels to default mode if no user setting
            self.setup_channels()

        indices = None
        if isinstance(segments[0], tuple):
            indices = [i for i, _ in segments]
            segments = [wav for _, wav in segments]

        spcm_dwSetParam_i32(self.hCard, SPC_CARDMODE, SPC_REP_STD_SEQUENCE)
        self._transfer_sequence(segments, indices)  # per index
        self._transfer_steps(steps)  # Load the Sequence Steps to the Board

        ## Wrap Up ##
        self._setup_clock()
        self._error_check()
        self.BufReady = True
        self.Sequence = False
        self._transfer_steps(steps)  # Loads the sequence steps to card

    def wiggle_output(self, duration=None, cam=None, block=True):
        """ Performs a Standard Output for configured settings.

        Parameters
        ----------
        duration : int, float, **straight mode only**
            Pass an integer to loop waveforms said number of times.
            Pass a float to loop waveforms for said number of milliseconds.
            Defaults to looping an infinite number of times.
        cam : bool, optional
            Indicates whether to use Camera GUI.
            `True` or `False` selects Pre- or Post- chamber cameras respectively.
        block : bool, optional
            Stops the card on function exit?

        Returns
        -------
        None
            WAVES! (This function itself actually returns void)
        """
        assert self.BufReady, "Card not fully configured"
        assert duration is None or self.Sequence is None, "Duration cannot be set for sequences."

        ## Timed or Looped mode determination ##
        if isinstance(duration, float):  # Timed Mode
            if VERBOSE:
                print("Looping Signal for ", duration / 1000 if duration else "infinity", " seconds...")
            spcm_dwSetParam_i32(self.hCard, SPC_TIMEOUT, duration)
        elif isinstance(duration, int):  # Looped Mode
            spcm_dwSetParam_i64(self.hCard, SPC_LOOPS, int64(duration))
        elif duration is not None:       # Invalid Option
            spcm_vClose(self.hCard)
            assert True, "Invalid input for steps"

        ## Sets blocking command appropriately ##
        WAIT = M2CMD_CARD_WAITREADY if block and cam is None else 0

        ## Start card, try again if clock-not-locked ##
        dwError = spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | WAIT)
        count = 0
        while dwError == ERR_CLOCKNOTLOCKED:
            count += 1
            sleep(0.1)
            self._error_check(halt=False, print_err=False)
            dwError = spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | WAIT)
            if count == 10:
                break

        ## Special Cases GUI/blocking ##
        if cam is not None:       # GUI Mode
            self._run_cam(cam)
            spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
        elif block and not self.Sequence and duration is None:  # Infinite Looping until stopped
            easygui.msgbox('Stop Card?', 'Infinite Looping!')
            spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)

        self._error_check()

    def stabilize_intensity(self, wav, cam=None, which_cam=None):
        """ Balances power across traps.

        Applies an iterative update to the magnitude vector
        (corresponding to the trap array) upon image analysis.
        Converges to homogeneous intensity profile across traps.

        Example
        -------
        You could access this algorithm
        by passing a waveform object::
            myWave = Superposition([79, 80, 81])  # Define a waveform
            hCard.stabilize_intensity(myWave)  # Pass it into the optimizer
        or through the gui_, offering camera view during the process::
            # Define a wave & load the board memory.
            hCard.wiggle_output(self, cam=True)
            # Use button on GUI

        Parameters
        ----------
        wav : :obj:`Superposition`
            The waveform object whose magnitudes will be optimized.
        which_cam : bool, optional
            `True` or `False` selects Pre- or Post- chamber cameras respectively.
        cam : :obj:`instrumental.drivers.cameras.uc480`
            The camera object opened by :obj:`instrumental` module.
        """
        assert isinstance(wav, Superposition), "Only Superpositions of pure tones can be optimized!"

        if cam is None:  # If we're not in GUI mode
            cam = self._run_cam()  # Retrieves a cam
            which_cam = 0
            self.load_waveforms(wav)
            self.wiggle_output(block=False)  # Outputs the given wave

        L = 0.2  # Correction Rate
        mags = wav.get_magnitudes()
        ntraps = len(mags)
        step_num, rel_dif = 0, 1
        while step_num < 5:
            step_num += 1
            print("Iteration ", step_num)

            trap_powers = analyze_image(which_cam, cam, ntraps, step_num)

            mean_power = trap_powers.mean()
            rel_dif = 100 * trap_powers.std() / mean_power
            print(f'Relative Power Difference: {rel_dif:.2f} %')

            ## Chain of performance thresholds ##
            if rel_dif < 0.1:
                print("WOW")
                break
            elif rel_dif < 0.36:
                L = 0.001
            elif rel_dif < 0.5:
                L = 0.01
            elif rel_dif < 2:
                L = 0.05
            elif rel_dif < 5:
                L = 0.1

            deltaM = [(mean_power - P)/P for P in trap_powers]
            dmags = [L * dM / sqrt(abs(dM)) for dM in deltaM]
            wav.set_magnitudes(np.add(mags, dmags))
            self._update_magnitudes(wav)

        for i in range(5):
            if rel_dif > 0.5:
                break
            sleep(2)

            # im = np.zeros(cam.latest_frame().shape)
            # for _ in range(10):
            #     imm = cam.latest_frame()
            #     for _ in range(9):
            #         imm = np.add(imm, cam.latest_frame())
            #     imm = np.multiply(imm, 0.1)
            #
            #     im = np.add(im, imm)
            # im = np.multiply(im, 0.1)

            trap_powers = analyze_image(which_cam, cam, ntraps)
            dif = 100 * trap_powers.std() / trap_powers.mean()
            print(f'Relative Power Difference: {dif:.2f} %')

        plot_image(which_cam, cam.latest_frame(), ntraps)
        cam.close()

    def reset_card(self):
        """ Wipes Card Configuration clean.
        """
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_RESET)
        self.ChanReady = False
        self.BufReady = False

    ################# PRIVATE FUNCTIONS #################

    def _error_check(self, halt=True, print_err=True):
        """
        Checks the Error Register.

        Parameters
        ----------
        halt : bool, optional
            Will halt program on discovery of error code. *Default:*True
        print_err=True : bool, optional
            Will print the error code.
        """
        ErrBuf = create_string_buffer(ERRORTEXTLEN)  # Buffer for returned Error messages
        if spcm_dwGetErrorInfo_i32(self.hCard, None, None, ErrBuf) != ERR_OK:
            if print_err:
                sys.stdout.write("Warning: {0}".format(ErrBuf.value))
            if halt:
                spcm_vClose(self.hCard)
                exit(1)
            return False
        return True

    def _transfer_sequence(self, wavs, indices):
        """
        Tries to write each waveform, from a set, to an indicated board memory segment.

        Parameters
        ----------
        wavs : list of (int, :obj:`Waveform`) tuples
            Each tuple contains an index & waveform.
            The waveform is written to the board memory segment with the given index.
        """
        if indices is None:  # Divides board memory
            segs = 1     # Indicates an undivided board memory as single segment
            while segs < len(wavs):
                segs *= 2  # Halves all segments, doubling available segments

            ## Splits the Board Memory ##
            spcm_dwSetParam_i32(self.hCard, SPC_SEQMODE_MAXSEGMENTS, segs)
            spcm_dwSetParam_i32(self.hCard, SPC_SEQMODE_STARTSTEP, 0)
        else:
            ## Splits the Board Memory ##
            segs = int32(0)
            spcm_dwGetParam_i32(self.hCard, SPC_SEQMODE_MAXSEGMENTS, byref(segs))
            assert max(indices) < segs.value, "Index out of range!"
            segs = segs.value

        ## Checks that each wave can fit in the allowed segments ##
        limit = MEM_SIZE / (segs * 2)  # Segment capacity in samples
        for wav in wavs:
            assert wav.Sample_Length <= limit, "%i waves limits each segment to %i samples." % (len(wavs), limit)

        ## Sets up a local Software Buffer for Transfer to Board ##
        pv_buf = pvAllocMemPageAligned(NUMPY_MAX*2)  # Allocates space on PC
        pn_buf = cast(pv_buf, ptr16)              # Casts pointer into something usable

        # Writes each waveform from the sequence to a corresponding segment on Board Memory ##
        for itr, (idx, wav) in enumerate(zip(indices, wavs)):
            transfer_size = wav.SampleLength*2
            print("Transferring Seg %d of size %d bytes..." % (itr, transfer_size))
            start = time()
            spcm_dwSetParam_i32(self.hCard, SPC_SEQMODE_WRITESEGMENT, idx)
            spcm_dwSetParam_i32(self.hCard, SPC_SEQMODE_SEGMENTSIZE,  int32(transfer_size))
            self._error_check()

            self._write_segment([wav], pv_buf, pn_buf)

            print("%d%c" % (int(100*(itr+1)/segs), '%'))
            print("Average Transfer rate: %d bytes/second" % (transfer_size // (time() - start)))

    def _write_segment(self, wavs, pv_buf, pn_buf, offset=0):
        """
        Writes set of waveforms consecutively into a single segment
        of board memory.
        Breaks down the transfer into manageable chunks.

        Parameters
        ----------
        wavs : list of :obj:`Waveform`
            Waveforms to be written to the current segment.
        pv_buf : :obj:`ctypes.Array`
            Local contiguous PC buffer for transferring to Board.
        pn_buf : :obj:`ctypes.Pointer(int16)`
            Usable pointer to buffer, cast as correct data type.
        offset : int, optional
            Passed from :func:`load_waveforms`, see description there.
        """
        total_so_far = offset
        for wav in wavs:
            size = wav.SampleLength * 2
            so_far = 0
            for n in range(ceil(size / NUMPY_MAX)):
                ## Decides chunk size & loads wave data to PC buffer ##
                seg_size_part = min(NUMPY_MAX, size - n * NUMPY_MAX)
                wav.load(pn_buf, so_far, seg_size_part)  # Fills the Buffer

                ## Do a Transfer to Board ##
                spcm_dwDefTransfer_i64(self.hCard, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, 0, pv_buf,
                                       uint64(total_so_far), uint64(seg_size_part))
                spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)

                ## Track transfer progress ##
                so_far += seg_size_part
                total_so_far += seg_size_part

    def _transfer_steps(self, steps):
        """ Writes all sequence steps passed,
        whether or not they overwrite previously written steps.

            Parameters
            ----------
            steps : list of :obj:`Step`
                Sequence steps to write.
        """
        self.Sequence = True

        for step in steps:
            cur = step.CurrentStep
            seg = step.SegmentIndex
            loop = step.Loops
            nxt = step.NextStep
            cond = step.Condition
            reg_upper = int32(cond | loop)
            reg_lower = int32(nxt << 16 | seg)
            if VERBOSE:
                print("Step %.2d: 0x%08x_%08x\n" % (cur, reg_upper.value, reg_lower.value))
            spcm_dwSetParam_i64m(self.hCard, SPC_SEQMODE_STEPMEM0 + cur, reg_upper, reg_lower)

        if VERBOSE:
            print("\nDump!:\n")
            for i in range(len(steps)):
                temp = uint64(0)
                spcm_dwGetParam_i64(self.hCard, SPC_SEQMODE_STEPMEM0 + i, byref(temp))
                print("Step %.2d: 0x%08x_%08x\n" % (i, int32(temp.value >> 32).value, int32(temp.value).value))
                print("Also: %16x\n" % temp.value)

    def _setup_clock(self):
        """
        Attempts to achieve the desired sampling frequency (defined by `SAMP_FREQ` global parameter).
        """
        spcm_dwSetParam_i32(self.hCard, SPC_CLOCKMODE, SPC_CM_INTPLL)  # Sets out internal Quarts Clock For Sampling
        spcm_dwSetParam_i64(self.hCard, SPC_SAMPLERATE, int64(int(SAMP_FREQ)))  # Sets Sampling Rate
        spcm_dwSetParam_i32(self.hCard, SPC_CLOCKOUT, 0)  # Disables Clock Output
        check_clock = int64(0)
        spcm_dwGetParam_i64(self.hCard, SPC_SAMPLERATE, byref(check_clock))  # Checks Sampling Rate
        if VERBOSE:
            print("Achieved Sampling Rate: ", check_clock.value)

    def _update_magnitudes(self, wav):
        """
        Turns off card, modifies each tone's magnitude, then lights it back up.

        Parameters
        ----------
        wav : :obj:`Superposition`
            Waveform with updated magnitudes.
        """
        # FIXME: borken
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
        self.load_waveforms(wav)
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER)
        sleep(1)

    def _run_cam(self, which_cam=None):
        """
        Fires up the camera stream (ThorLabs UC480)

        Parameters
        ----------
        which_cam : bool
            Chooses between displaying the Pre- or Post- chamber ThorCams.
            Passing nothing returns a camera object for silent use (not displaying).

        Returns
        -------
        :obj:`instrumental.drivers.cameras.uc480`, None
            Only returns if no selection for `which_cam` is made.
        """
        ## https://instrumental-lib.readthedocs.io/en/stable/uc480-cameras.html ##
        ## ^^LOOK HERE^^ for driver documentation ##

        ## If you have problems here ##
        ## then see above doc &      ##
        ## Y:\E6\Software\Python\Instrument Control\ThorLabs UC480\cam_control.py ##
        # TODO: Integrate button for saving optimized waveforms

        names = ['ThorCam', 'ChamberCam']  # First one is default for stabilize_intensity(wav)
        cam = instrument(names[which_cam])

        ## Cam Live Stream ##
        cam.start_live_video(framerate=10 * u.hertz)

        ## No-Display mode ##
        if which_cam is None:
            return cam

        ## Create Figure ##
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        ## Animation Frame ##
        def animate(i):
            if cam.wait_for_frame():
                im = cam.latest_frame()
                ax1.clear()
                if which_cam:
                    im = im[300:501, 300:501]
                ax1.imshow(im)

        ## Button: Automatic Exposure Adjustment ##
        def find_exposure(event):
            fix_exposure(cam, set_exposure)

        ## Button: Intensity Feedback ##
        def stabilize(event):  # Wrapper for Intensity Feedback function.
            self.stabilize_intensity(self.Wave, which_cam, cam)

        def snapshot(event):
            im = cam.latest_frame()
            plot_image(which_cam, im, 12, guess=True)

        def switch_cam(event):
            nonlocal cam, which_cam
            cam.close()

            which_cam = not which_cam

            cam = instrument(names[which_cam])
            cam.start_live_video(framerate=10 * u.hertz)

        ## Slider: Exposure ##
        def adjust_exposure(exp_t):
            cam._set_exposure(exp_t * u.milliseconds)

        ## Button Construction ##
        correct_exposure = Button(plt.axes([0.56, 0.0, 0.13, 0.05]), 'AutoExpose')
        stabilize_button = Button(plt.axes([0.7, 0.0, 0.1, 0.05]), 'Stabilize')
        plot_snapshot = Button(plt.axes([0.81, 0.0, 0.09, 0.05]), 'Plot')
        switch_cameras = Button(plt.axes([0.91, 0.0, 0.09, 0.05]), 'Switch')
        set_exposure = Slider(plt.axes([0.14, 0.9, 0.73, 0.05]), 'Exposure', 0.1, MAX_EXP, 20)

        correct_exposure.on_clicked(find_exposure)
        stabilize_button.on_clicked(stabilize)
        plot_snapshot.on_clicked(snapshot)
        switch_cameras.on_clicked(switch_cam)
        set_exposure.on_changed(adjust_exposure)

        ## Begin Animation ##
        _ = animation.FuncAnimation(fig, animate, interval=100)
        plt.show()
        cam.close()
        plt.close(fig)
        self._error_check()
