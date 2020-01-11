from lib.pyspcm import *
from lib.spcm_tools import *
import sys
import time, easygui
import matplotlib.pyplot as plt
import numpy as np
from math import sin, pi
import random, bisect, pickle

### Constants ###
SAMP_VAL_MAX = (2 ** 15 - 1)  ## Maximum digital value of sample ~~ signed 16 bits

SAMP_FREQ_MAX = 1250E6  ## Maximum Sampling Frequency
### Paremeter ###
SAMP_FREQ = 1000E6  ## Modify if a different Sampling Frequency is required.


## Otherwise, why would one not use the max?

class OpenCard:
    """
        Class designed for Opening, Configuring, running the Spectrum AWG card.

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
            + __error_check() ------------------------------- Reads the card's error register.
            + __compute_and_load(seg, Ptr, buf, fsamp) ------ Computes a Segment and Transfers to Card.
    """
    ## Handle on card ##
    # We make this a class variable because there is only 1 card in the lab.
    # This simplifies enforcing 1 instance.
    hCard = None
    ModeBook = {  ## Dictionary of Mode Names to Register Value Constants
        'continuous': SPC_REP_STD_CONTINUOUS,
        'multi': SPC_REP_STD_MULTI
        # 'sequence'  : SPC_REP_STD_SEQUENCE, --> Card doesn't possess feature :'(
    }

    def __init__(self, mode='continuous', loops=0):
        """
            Just Opens the card in the given mode.
            INPUTS:
                mode  - Name for card output mode. limited support :)
            'single' or 'multiple' mode only (not yet supported)
                loops - Number of times the buffer is looped, 0 = infinity
        """
        assert self.hCard is None, "Card opened twice!"
        self.hCard = spcm_hOpen(create_string_buffer(b'/dev/spcm0'))  # Opens Card
        self.__error_check()
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
        self.__error_check()
        self.ModeReady = True

    def __exit__(self, exception_type, exception_value, traceback):
        print("in __exit__")
        spcm_vClose(self.hCard)

    ## Segment Handling ##
    def clear_segments(self):
        self.Segments = None

    def load_segments(self, segs):
        if self.Segments is None:
            self.Segments = segs
        else:
            self.Segments.extend(segs)

    ################# Basic Card Configuration Functions #################
    def set_mode(self, mode):
        if self.Mode != mode:
            self.BufReady = False
        spcm_dwSetParam_i32(self.hCard, SPC_CARDMODE, self.ModeBook.get(mode))
        self.Mode = mode
        self.ModeReady = True

    def setup_channels(self, amplitude=200, ch0=False, ch1=True, use_filter=False):
        """
            Performs a Standard Initialization for designated Channels & Trigger
            INPUTS:
                amplitude -- Sets the Output Amplitude ~~ RANGE: [80 - 2000](mV) inclusive
                ch0 -------- Bool to Activate Channel0
                ch1 -------- Bool to Activate Channel1
                use_filter - Bool to Activate Output Filter
        """
        ### Input Validation ###
        if ch0 and ch1:
            print('Multi-Channel Support Not Yet Supported!')
            print('Defaulting to Ch1 only.')
            ch0 = False
        assert 80 <= amplitude <= 2000, "Amplitude must within interval: [80 - 2000]"
        if amplitude != int(amplitude):
            amplitude = int(amplitude)
            print("Rounding amplitude to required integer value: ", amplitude)

        ######### Channel Activation ##########
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

        ######### Trigger Config ###########
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_ORMASK, SPC_TMASK_SOFTWARE)
        ########## Necessary? Doesn't Hurt ##################
        spcm_dwSetParam_i32(self.hCard, SPC_TRIG_ANDMASK,   0)
        spcm_dwSetParam_i64(self.hCard, SPC_TRIG_DELAY,     int64(0))
        spcm_dwSetParam_i32(self.hCard, SPC_TRIGGEROUT,     0)
        ############ ???? ####################################
        self.__error_check()
        self.ChanReady = True

    def setup_buffer(self):
        """
            Calculates waves contained in Segments,
            configures the board memory buffer,
            then transfers to the board.
        """
        ## Validate ##
        assert self.ChanReady and self.ModeReady, "The Mode & Channels must be configured before Buffer!"
        assert len(self.Segments) > 0, "No Segments defined! Nothing to put in Buffer."

        #### Gather Information from Board ####
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
        self.__compute_and_load(seg, pn_buf, pv_buf, buf_size)

        ########## Clock ############
        spcm_dwSetParam_i32(self.hCard, SPC_CLOCKMODE, SPC_CM_INTPLL)  # Sets out internal Quarts Clock For Sampling
        spcm_dwSetParam_i64(self.hCard, SPC_SAMPLERATE, int64(int(SAMP_FREQ)))  # Sets Sampling Rate
        spcm_dwSetParam_i32(self.hCard, SPC_CLOCKOUT, 0)  # Disables Clock Output
        check_clock = int64(0)
        spcm_dwGetParam_i64(self.hCard, SPC_SAMPLERATE, byref(check_clock))  # Checks Sampling Rate
        print("Achieved Sampling Rate: ", check_clock.value)

        self.__error_check()
        self.BufReady = True

    def wiggle_output(self, timeout=0):
        """
            Performs a Standard Output for configured settings.
            INPUTS:
                -- OPTIONAL --
                timeout  - How long the output streams in Milliseconds
            OUTPUTS:
                WAVES!
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
            self.__error_check(halt=False)
            dwError = spcm_dwSetParam_i32(self.hCard, SPC_M2CMD,
                                          M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | WAIT)
            if count == 10:
                break
        if timeout == 0:
            easygui.msgbox('Stop Card?', 'Infinite Looping!')
            spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
        elif dwError == ERR_TIMEOUT:
            print("timeout!")
            spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_STOP)
        self.__error_check()

    ################# Miscellaneous #################

    def reset_card(self):
        """
            Wipes Card Configuration clean
        """
        spcm_dwSetParam_i32(self.hCard, SPC_M2CMD, M2CMD_CARD_RESET)
        self.ModeReady = False
        self.ChanReady = False
        self.BufReady = False

    def __error_check(self, halt=True):
        """
            Checks the Error Register. If Occupied:
                -Prints Error
                -Closes the Card and exits program
        """
        ErrBuf = create_string_buffer(ERRORTEXTLEN)  # Buffer for returned Error messages
        if spcm_dwGetErrorInfo_i32(self.hCard, None, None, ErrBuf) != ERR_OK and halt:
            sys.stdout.write("{0}\n".format(ErrBuf.value))
            spcm_vClose(self.hCard)
            exit(1)

    def __compute_and_load(self, seg, ptr, buf, buf_size):
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

####################### Class Implementations ########################


## Helper Class ##
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
        assert mag >= 0 and mag <= 1, ("Invalid magnitude: %d, must be within interval [0,1]" % mag)
        ## Initialize ##
        self.Frequency = freq
        self.Magnitude = mag
        self.Phase = phase

    def __lt__(self, other):
        return self.Frequency < other.Frequency


## Primary Class ##
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
            + __compute() - Computes the segment and stores into Buffer.
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
            assert resolution < SAMP_FREQ / 2, ("Invalid Resolution, has to be less than Nyquist Frequency: %d" % (SAMP_FREQ / 2))
            target_sample_length = int(SAMP_FREQ / resolution)
        if freqs is None and waves is not None:
            for i in range(len(waves)):
                assert waves[i].Frequency >= resolution, ("Frequency %d was given while Resolution is limited to %d Hz." % (waves[i].Frequency, resolution))
                assert waves[i].Frequency < SAMP_FREQ / 2, ("All frequencies must below Nyquist frequency: %d" % (SAMP_FREQ / 2))
        elif freqs is not None and waves is None:
            for f in freqs:
                assert f >= resolution, ("Frequency %d was given while Resolution is limited to %d Hz." % (f, resolution))
                assert f < SAMP_FREQ_MAX / 2, ("All frequencies must below Nyquist frequency: %d" % (SAMP_FREQ / 2))
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
        for wave in self.Waves:
            if w.Frequency == wave.Frequency:
                print("Skipping duplicate: %d Hz" % w.Frequency)
                return
        resolution = SAMP_FREQ / self.SampleLength
        assert w.Frequency >= resolution, ("Resolution: %d Hz, sets the minimum allowed frequency. (it was violated)" % resolution)
        assert w.Frequency < SAMP_FREQ / 2, ("All frequencies must be below Nyquist frequency: %d" % (SAMP_FREQ / 2))
        bisect.insort(self.Waves, w)
        self.Latest = False


    def remove_frequency(self, f):
        self.Waves = [W for W in self.Waves if W.Frequency != f]
        self.Latest = False


    def set_magnitude(self, idx, mag):
        """
            Sets the magnitude of the indexed trap number.
            INPUTS:
                idx - Index to trap number, starting from 0
                mag - New value for relative magnitude, must be in [0, 1]
        """
        assert mag >= 0 and mag <= 1, ("Invalid magnitude: %d, must be within interval [0,1]" % mag)
        self.Waves[idx].Magnitude = mag
        self.Latest = False

    def set_magnitude_all(self, mags):
        """
            Sets the magnitude of all traps.
            INPUTS:
                mags - List of new magnitudes, in order of Trap Number (Ascending Frequency).
        """
        for i, mag in enumerate(mags):
            assert mag >= 0 and mag <= 1, ("Invalid magnitude: %d, must be within interval [0,1]" % mag)
            self.Waves[i].Magnitude = mag
        self.Latest = False


    def set_phase(self, idx, phase):
        """
            Sets the magnitude of the indexed trap number.
            INPUTS:
                idx --- Index to trap number, starting from 0
                phase - New value for phase.
        """
        phase = phase % (2*pi)
        self.Waves[idx].Phase = phase
        self.Latest = False


    def set_phase_all(self, phases):
        """
            Sets the magnitude of all traps.
            INPUTS:
                mags - List of new phases, in order of Trap Number (Ascending Frequency).
        """
        for i, phase in enumerate(phases):
            self.Waves[i].Phase = phase
        self.Latest = False

    def plot(self):
        """
            Plots the Segment. Computes first if necessary.
        """
        if not self.Latest:
            self.compute()
        plt.plot(self.Buffer, '--o')
        plt.show()


    def randomize(self):
        """
            Randomizes each phase.
        """
        for w in self.Waves:
            w.Phase = 2*pi*random.random()
        self.Latest = False


    def compute(self):
        """
            Computes the superposition of frequencies
            and stores it in the buffer.
            We divide by the sum of relative wave magnitudes
            and scale the max value to SAMP_VAL_MAX,
            s.t. if all waves phase align, they will not exceed the max value.
        """
        ## Checks if Redundant ##
        if self.Latest:
            return
        self.Latest = True ## Will be up to date after

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


    def save(self, name="unamed_segment", data_only=False):
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