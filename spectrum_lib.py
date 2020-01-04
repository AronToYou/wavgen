from pyspcm import *
from spcm_tools import *
from math import sin, pi
import sys
import matplotlib.pyplot as plt
import numpy as np

### Constants ###
SAMP_VAL_MAX  = (2 ** 16 - 1) ## Maximum digital value of sample ~~ 16 bits
SAMP_FREQ_MAX = 1250E6        ## Maximum Sampling Frequency

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
            + set_mode(mode) ------------------------------ Set the card operation mode, e.g. single, multiple, continuous, sequential.
            + setup_channels(amplitude, ch0, ch1, filter) - Activates chosen channels and Configures Triggers. (Only uses default trigger setting)
            + setup_buffer(sampling_frequency) ------------ Divides the Card Memory, processes each Segment, & transfers to Card Memory.
            + load_segments(segs) ------------------------- Appends a set of segments to the current set.
            + clear_segments() ---------------------------- Clears out current set of Segments.
            + reset_card() -------------------------------- Resets all of the cards configuration. Doesn't close card.
        PRIVATE METHODS:
            + __error_check() ------------------------------- Reads the card's error register. Prints error & closes card/program when necessary.
            + __compute_and_load(seg, Ptr, buf, fsamp) ------ Computes a Segment and Transfers to Card.
    """
    ## Handle on card ##
    # We make this a class variable because there is only 1 card in the lab, thus only
    #   every 1 instance of the 'card' class. This makes enforcing 1 instance simple.
    hCard = None
    ModeBook = {  ## Dictionary of Mode Names to Register Value Constants
        'continuous': SPC_REP_STD_CONTINUOUS,
        'multi'     : SPC_REP_STD_MULTI
        #'sequence'  : SPC_REP_STD_SEQUENCE, --> Card doesn't posess feature :'(
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
        self.hCard = spcm_hOpen (create_string_buffer(b'/dev/spcm0'))  # Opens Card
        self.__error_check()
        self.ModeReady = True
        self.ChanReady = False
        self.BufReady = False
        self.Mode = mode
        self.Segments = None
        spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_CARD_RESET)
        ## Setup Mode ##
        mode = self.ModeBook.get(mode)  ## ModeBook is class object, look above
        if mode is None:
            print('Invalid mode phrase, possible phrases are: ')
            print(list(self.ModeBook.keys()))
            exit(1)
        spcm_dwSetParam_i32(self.hCard, SPC_CARDMODE, mode)
        if mode is 'continuous':
            loops = 0
            spcm_dwSetParam_i64(self.hCard, SPC_LOOPS, int64(loops))
        ##elif mode is 'single':
        ##elif mode is 'multiple':
        self.__error_check()
        self.ModeReady = True

    def __exit__(self, exception_type, exception_value, traceback):
        print("in __exit__")
        spcm_vClose(self.hCard)

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


    def setup_channels(self, amplitude=2000, ch0=False, ch1=True, filter=False):
        """
            Performs a Standard Initialization for designated Channels & Trigger
            INPUTS:
                amplitude - Sets the Output Amplitude ~~ RANGE: [80 - 2500](mV) inclusive
                ch0 ------- Bool to Activate Channel0
                ch1 ------- Bool to Activate Channel1
                filter ---- Bool to Activate Output Filter
        """
        ### Input Validation ###
        if ch0 and ch1:
            print('Multi-Channel Support Not Yet Supported!')
            print('Defaulting to Ch1 only.')
            ch0 = False
        assert amplitude >= 80 and amplitude <= 2000, "Amplitude must within interval: [80 - 2500]"
        if amplitude != int(amplitude):
            amplitude = int(amplitude)
            print("Rounding amplitude to required integer value: ", amplitude)

        ######### Channel Activation ##########
        CHAN = 0x00000000
        amp = int32(amplitude)
        if ch0:
            spcm_dwSetParam_i32 (self.hCard, SPC_ENABLEOUT0, 1)
            CHAN = CHAN ^ CHANNEL0
            spcm_dwSetParam_i32 (self.hCard, SPC_AMP0,       amp)
            spcm_dwSetParam_i64 (self.hCard, SPC_FILTER0,    int64(filter))
        if ch1:
            spcm_dwSetParam_i32 (self.hCard, SPC_ENABLEOUT1, 1)
            CHAN = CHAN ^ CHANNEL1
            spcm_dwSetParam_i32 (self.hCard, SPC_AMP1,       amp)
            spcm_dwSetParam_i64 (self.hCard, SPC_FILTER1,    int64(filter))
        spcm_dwSetParam_i32 (self.hCard,     SPC_CHENABLE,   CHAN)

        ######### Trigger Config ###########
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_ORMASK,      SPC_TMASK_SOFTWARE)
        ########## Necessary? Doesn't Hurt ##################
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_ANDMASK,     0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_CH_ORMASK0,  0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_CH_ORMASK1,  0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_CH_ANDMASK0, 0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIG_CH_ANDMASK1, 0)
        spcm_dwSetParam_i32 (self.hCard, SPC_TRIGGEROUT,       0)
        ############ ???? ####################################
        self.__error_check()
        self.ChanReady = True


    def setup_buffer(self, sampling_frequency=SAMP_FREQ_MAX):
        """
            Calculates waves contained in Segments,
            configures the board memory buffer,
            then transfers to the board.
            INPUTS:
                sampling_frequency - For overriding the board output sampling frequency from the max. (Hertz)
        """
        assert self.ChanReady and self.ModeReady, "The Mode & Channels must be configured before Buffer!"
        assert len(self.Segments) > 0, "No Segments defined! Nothing to put in Buffer."

        #### Gather Information from Board ####
        num_chan = int32(0)  # Number of Open Channels
        mem_size = int64(0)  # Total Memory ~ 4.3 GB
        mode     = int32(0)  # Operation Mode
        spcm_dwGetParam_i32 (self.hCard, SPC_CHCOUNT,    byref(num_chan)) ## Should always be 1 -- i.e. multi is not supported yet!
        spcm_dwGetParam_i64 (self.hCard, SPC_PCIMEMSIZE, byref(mem_size))                                    ## (But it could be)
        spcm_dwGetParam_i32 (self.hCard, SPC_CHCOUNT,    byref(mode))

        #### Determines how many Sectors to divide the Board Memory into ####
        num_segs = 1
        while (len(self.Segments) < num_segs): num_segs *= 2  # Memory can only be divided into Powers of 2

        #### Calculates Sample-Length for each Segment ####
        MaxSampLen = 0
        for i, seg in enumerate(self.Segments):
            SampLen = int(sampling_frequency / seg.Resolution)  # Sets Sample Length s.t. the target resolution is roughly true
            SampLen = SampLen - (SampLen % 32) + 32      # Constrains the memory to be 64 byte aligned
            print('Segment ', i, ' - Sampling Length: ', SampLen)
            print('Target Resolution: ', seg.Resolution, 'Hz, Achieved resolution: ', sampling_frequency / SampLen, 'Hz')
            if SampLen > MaxSampLen:
                MaxSampLen = SampLen
            seg.SampleLength = SampLen
            if mode != self.ModeBook.get('sequential'): break # We only use 1 Segment for other modes

        #### Sets up a local Software Buffer for Transfer to Board ####
        buf_size = uint64(MaxSampLen * 2 * num_chan.value)  # Calculates Buffer Size in Bytes
        pv_buf = pvAllocMemPageAligned(buf_size.value)  # Allocates space on PC
        pn_buf = cast(pv_buf, ptr16)  # Casts pointer into something usable

        #### Configures and Loads the Buffer ####
        #### Mode specific setup
        if mode == self.ModeBook.get('continuous'):
            seg = self.Segments[0]
            if num_segs > 1:
                print("Continuous mode is set. Only using 1st Segment.")
            spcm_dwSetParam_i64(self.hCard, SPC_MEMSIZE, int64(seg.SampLength))
            self.__compute_and_load(seg, pn_buf, pv_buf, sampling_frequency) ## Wave calculation
        elif mode == self.ModeBook.get('sequence'):
            spcm_dwSetParam_i32(self.hCard,  SPC_SEQMODE_MAXSEGMENTS, int32(num_segs))
            for i, seg in enumerate(self.Segements):
                spcm_dwSetParam_i32 (self.hCard,   SPC_SEQMODE_WRITESEGMENT, i)
                spcm_dwSetParam_i32 (self.hCard,   SPC_SEQMODE_SEGMENTSIZE,  seg.SampleLength)
                self.__compute_and_load(seg, pn_buf, pv_buf, sampling_frequency) ## Wave calculation

        ########## Clock ############
        spcm_dwSetParam_i32(self.hCard, SPC_CLOCKMODE, SPC_CM_INTPLL)  # Sets out internal Quarts Clock For Sampling
        print("fsamp: ", sampling_frequency)
        spcm_dwSetParam_i64(self.hCard, SPC_SAMPLERATE, int64(int(sampling_frequency)))  # Sets Sampling Rate
        spcm_dwSetParam_i32(self.hCard, SPC_CLOCKOUT, 0)  # Disables Clock Output

        self.__error_check()
        self.BufReady = True


    ################## Outputs the Wave #######################

    def wiggle_output(self, timeout=0):
        """
            Performs a Standard Initialization for designated Channels & Trigger
            INPUTS:
                hCard - The handle to the opened hardware card
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
        spcm_dwSetParam_i32(self.hCard, SPC_TIMEOUT, timeout)
        dwError = spcm_dwSetParam_i32(self.hCard, SPC_M2CMD,
                                      M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_CARD_WAITREADY)
        if dwError == ERR_TIMEOUT:
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
        self.BufReady  = False


    def __error_check(self):
        """
            Checks the Error Register. If Occupied:
                -Prints Error
                -Closes the Card and exits program
        """
        ErrBuf = create_string_buffer(ERRORTEXTLEN)  # Buffer for returned Error messages
        if spcm_dwGetErrorInfo_i32(self.hCard, None, None, ErrBuf) != ERR_OK:
            sys.stdout.write("{0}\n".format(ErrBuf.value))
            spcm_vClose(self.hCard)
            exit(1)


    def __compute_and_load(cls, seg, ptr, buf, fsamp):
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
        ## Compute and Add the full wave, Each frequency at a time
        for f in seg.Frequencies:
            fn = f / fsamp  # Cycles/Sample
            for i in range(seg.SampleLength):
                ptr[i] += sin(2 * pi * i * fn) / len(seg.Frequencies) ## Divide by number of waves (To Avoid Clipping)
        ## Scale and round each value
        for i in range(seg.SampleLength):
            ptr[i] = int(SAMP_VAL_MAX*ptr[i])
        ## Do a Transfer
        bytes = uint64(seg.SampleLength*2*1) # The 1 is a reminder to support Multiple Channels
        spcm_dwDefTransfer_i64 (self.hCard, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, int32(0), buf, uint64(0), bytes)
        print("Doing a transfer...")
        spcm_dwSetParam_i32 (self.hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
        print("Done")


####################### Class Implementations ########################
class Segment:
    """
        Member Variables:
            + Frequencies -- (Hertz) List of frequencies which compose the Segment
            + Resolution --- (Hertz) The target resolution to aim for. In other words, sets the sample time (N / Fsamp)
                             and thus the 'wavelength' of the buffer (wavelength that completely fills the buffer).
                             Any multiple of the resolution will be periodic in the memory buffer.
            + SampleLength - Calculated during Buffer Setup; The length of the Segment in Samples.
    """
    def __init__(self, frequencies=[], resolution=1E6):
        assert resolution < SAMP_FREQ_MAX / 2, ("Invalid Resolution, has to be less than Nyquist Frequency: %d" % (SAMP_FREQ_MAX / 2))
        for i in range(len(frequencies)):
            assert frequencies[i] >= resolution, ("Frequency %d was given while Resolution is limited to %d Hz." % (frequencies[i], resolution))
            assert frequencies[i] < SAMP_FREQ_MAX / 2, ("All frequencies must below Nyquist frequency: %d" % (SAMP_FREQ_MAX / 2))
        ### Initialize
        self.Frequencies = frequencies
        self.Resolution  = resolution
        self.SampleLength = None


    def add_frequency(self, f):
        for freq in self.frequencies:
            if f == freq:
                print("Skipping duplicate: %d Hz" % f)
                return
        assert f >= self.Resolution, ("Resolution: %d Hz, sets the minimum allowed frequency. (it was violated)" % self.Resolution)
        assert f < SAMP_FREQ_MAX / 2, ("All frequencies must be below Nyquist frequency: %d" % (SAMP_FREQ_MAX / 2))
        self.Frequencies.append(f)


    def remove_frequency(self, f):
        self.Frequencies = [F for F in self.Frequencies if F != f]


    def __str__(self):
        s = "Segment with Resolution: " + str(self.Resolution) + "\n"
        s += "Contains Frequencies: \n"
        for f in self.Frequencies:
           s += "---" + str(f) + "Hz\n"
        return s