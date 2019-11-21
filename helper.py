from pyspcm import *
from spcm_tools import *
from math import sin, pi
import sys
import matplotlib.pyplot as plt
import numpy as np

### Constants ###
SAMP_VAL_MAX  = (2 ** 16 - 1) ## Maximum digital value of sample ~~ 16 bits
SAMP_FREQ_MAX = 1.25E9        ## Maximum Sampling Frequency

class OpenCard:
    """
        Class designed for Opening & Configuring the Spectrum AWG card.

        Class Variable:
            + hCard ---- The handle to the open card. For use with Spectrum API functions.
            + ModeBook - Dictionary for retrieving board register constants from key phrases.
        List of Member Variables:
            + ModeReady
            + ChanReady
            + BufReady
            + Segments - List of Segment objects

        List of Methods (see implementations for details regarding arguments):
            + setup_mode:       Set the card operation mode, e.g. single, multiple, continuous, loop#
            + setup_channel:    Activates chosen channels and Configures Triggers. (Only uses default trigger setting)
            +
            + error_check:      Reads the card's error register. Prints error & closes card/program when necessary.
            + reset_card:       Resets all of the cards configuration. Doesn't close card.
    """
    ## Handle on card ##
    # We make this a class variable because there is only 1 card in the lab, thus only
    #   every 1 instance of the 'card' class. This makes enforcing 1 instance simple.
    hCard = None
    ModeBook = {  ## Dictionary of Mode Names to Register Value Constants
        'continuous': SPC_REP_STD_CONTINUOUS,
        'sequence'  : SPC_REP_STD_SEQUENCE,
    }

    def __init__(self, Segs=[]):
        """
            Just Opens the card and resets the configuration
            INPUTS:
                Segs - For passing defined Segments on creation
        """
        if hCard is not None:
            print('Card already open!')
        hCard = spcm_hOpen (create_string_buffer(b'/dev/spcm0'))  # Opens Card
        self.error_check()
        self.ModeReady = False
        self.ChanReady = False
        self.BufReady = False
        self.Segments = Segs
        spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_RESET)

    ################# Basic Card Configuration Functions #################

    def setup_mode(self, Mode='continuous', Loops=0):
        """
            Sets the Card mode
            Call without arguments for Continuous looping of Buffer
            INPUTS:
                Mode  - Name for card output mode. limited support :)
                Loops - Number of times the buffer is looped, 0 = infinity
        """
        if Loops != int(Loops):
            Loops = int(Loops)
            print('Rounding loops to required integer value: ', Loops)
        Mode = ModeBook.get(Mode)  ## ModeBook is class object, look above
        if Mode is None:
            print('Invalid mode phrase, possible phrases are: ')
            print(list(ModeBook.keys()))
            return
        spcm_dwSetParam_i32 (hCard, SPC_CARDMODE, Mode)
        spcm_dwSetParam_i64 (hCard, SPC_LOOPS,    int64(Loops))
        self.error_check()
        self.ModeReady = True

    def setup_channels(self, Amplitude=2500, Ch0=False, Ch1=True, Filter=False):
        """
            Performs a Standard Initialization for designated Channels & Trigger
            INPUTS:
                Amplitude - Sets the Output Amplitude ~~ RANGE: [80 - 2500](mV) inclusive
                Ch0 ------- Bool to Activate Channel0
                Ch1 ------- Bool to Activate Channel1
                Filter ---- Bool to Activate Output Filter
        """
        ### Input Validation ###
        if Ch0 and Ch1:
            print('Multi-Channel Support Not Yet Supported!')
            print('Defaulting to Ch1 only.')
            Ch0 = False
        if Amplitude < 80 or Amplitude > 2500:
            print("Amplitude must within interval: [80 - 2500]")
        if not (Amplitude != int(Amplitude)):
            Amplitude = int(Amplitude)
            print("Rounding amplitude to required integer value: ", Amplitude)

        ######### Channel Activation ##########
        CHAN = 0x00000000
        Amp = int32(Amplitude)
        if Ch0:
            spcm_dwSetParam_i32 (hCard, SPC_ENABLEOUT0, 1)
            CHAN = CHAN ^ CHANNEL0
            spcm_dwSetParam_i32 (hCard, SPC_AMP0,       Amp)
            spcm_dwSetParam_i64 (hCard, SPC_FILTER1,    Filter)
        if Ch1:
            spcm_dwSetParam_i32 (hCard, SPC_ENABLEOUT1, 1)
            CHAN = CHAN ^ CHANNEL1
            spcm_dwSetParam_i32 (hCard, SPC_AMP1,       Amp)
            spcm_dwSetParam_i64 (hCard, SPC_FILTER2,    Filter)
        spcm_dwSetParam_i32 (hCard,     SPC_CHENABLE,   CHAN)


        ######### Trigger Config ###########
        spcm_dwSetParam_i32 (hCard, SPC_TRIG_ORMASK,      SPC_TMASK_SOFTWARE)
        ########## Necessary? Doesn't Hurt ##################
        spcm_dwSetParam_i32 (hCard, SPC_TRIG_ANDMASK,     0)
        spcm_dwSetParam_i32 (hCard, SPC_TRIG_CH_ORMASK0,  0)
        spcm_dwSetParam_i32 (hCard, SPC_TRIG_CH_ORMASK1,  0)
        spcm_dwSetParam_i32 (hCard, SPC_TRIG_CH_ANDMASK0, 0)
        spcm_dwSetParam_i32 (hCard, SPC_TRIG_CH_ANDMASK1, 0)
        spcm_dwSetParam_i32 (hCard, SPC_TRIGGEROUT,       0)
        ############ ???? ####################################
        self.error_check()
        self.ChanReady = True

    def setupBuffer(self, SamplingFrequency=SAMP_FREQ_MAX):
        """
            Configures
            INPUTS:
                SamplingFrequency - For overriding the board output sampling frequency from the max. (Hertz)
        """
        if not self.ChanReady or not self.ModeReady:
            print('The Mode & Channels must be configured before Buffer!')
            return
        if len(self.Segments) == 0:
            print('No Segments defined! Nothing to put in Buffer.')
            return

        NumChan = int32(0)  # Number of Open Channels
        MemSize = int64(0)  # Total Memory ~ 4.3 GB
        Mode    = int32(0)  # Operation Mode

        #### Gather Information from Board ####
        spcm_dwGetParam_i32 (hCard, SPC_CHCOUNT,    byref(NumChan))  # Number of Open Channels
        spcm_dwGetParam_i64 (hCard, SPC_PCIMEMSIZE, byref(MemSize))  # Physical Memory Size in Samples
        spcm_dwGetParam_i32 (hCard, SPC_CHCOUNT,    byref(Mode))     # Number of Open Channels

        #### Determines the Number Sectors to divide the Board Memory ####
        NumSegs = 1
        while (len(self.Segments) < NumSegs): NumSegs *= 2  # Memory can only be divided into Powers of 2

        #### Calculates Max and Sample-Length for each Segment ####
        MaxSampLen = 0
        for i, Seg in enumerate(self.Segments):
            if i == NumSegs:
                break  # Skips other Segments if in continuous mode
            SampLen = int(SamplingFrequency / Seg.Resolution)  # Sets Sample Length s.t. the target resolution is roughly true
            SampLen = SampLen - (SampLen % 32) + 32      # Constrains the memory to be 64 byte aligned
            print('Segment ', i, ' - Sampling Length: ', SampLen)
            print('Target Resolution: ', Seg.Resolution, 'Hz, Achieved resolution: ', SamplingFrequency / SampLen, 'Hz')
            if SampLen > MaxSampLen:
                MaxSampLen = SampLen
            Seg.SampleLength = SampLen

        #### Sets up a User Buffer for Transfer to Board ####
        BufSize = uint64(MaxSampLen * 2 * numChan.value)  # Calculates Buffer Size in Bytes
        pvBuffer = pvAllocMemPageAligned(BufSize.value)  # Allocates space on PC
        pnBuffer = cast(pvBuffer, ptr16)  # Casts pointer into something usable

        #### Configures and Loads the Buffer ####
        if   Mode == ModeBook.get('continuous'):
            if NumSegs > 1:
                print("Continuous mode is set. Only using 1st Segment.")
            spcm_dwSetParam_i64 (hCard, SPC_MEMSIZE, int64(SampLength))
            self.compute_and_load(self.Segments[0], pnBuffer, pvBuffer, SamplingFrequency)
        elif Mode == ModeBook.get('sequence')
            spcm_dwSetParam_i32(hCard,  SPC_SEQMODE_MAXSEGMENTS, int32(NumSegs))
            for i, Seg in enumerate(self.Segements):
                spcm_dwSetParam_i32 (hCard,   SPC_SEQMODE_WRITESEGEMENT, i)
                spcm_dwSetParam_i32 (hCard,   SPC_SEQMODE_SEGEMENTSIZE,  Seg.SampleLength)
                self.compute_and_load(Seg, pnBuffer, pvBuffer, SamplingFrequency)


        ########## Clock ############
        spcm_dwSetParam_i32(hCard, SPC_CLOCKMODE, SPC_CM_INTPLL)  # Sets out internal Quarts Clock For Sampling
        spcm_dwSetParam_i64(hCard, SPC_SAMPLERATE, int64(SamplingFrequency))  # Sets Sampling Rate
        spcm_dwSetParam_i32(hCard, SPC_CLOCKOUT, 0)  # Disables Clock Output


    ################# Miscellaneous #################

    def error_check(self):
        """
            Checks the Error Register. If Occupied:
                -Prints Error
                -Closes the Card and exits program
        """
        ErrBuf = create_string_buffer(ERRORTEXTLEN)  # Buffer for returned Error messages
        if spcm_dwGetErrorInfo_i32(hCard, None, None, ErrBuf) != ERR_OK:
            sys.stdout.write("{0}\n".format(ErrBuf.value))
            spcm_vClose(hCard)
            self.ModeReady = False
            self.ChanReady = False
            self.BufReady  = False
            exit()

    def reset_card(self):
        """
            Wipes Card Configuration clean
        """
        spcm_dwSetParam_i32(hCard, SPC_M2CMD, M2CMD_CARD_RESET)
        self.ModeReady = False
        self.ChanReady = False
        self.BufReady  = False

    @classmethod
    def compute_and_load(cls, Seg, Ptr, Buf, fSamp):
        """
            Computes the superposition of frequencies
            and stores it in the buffer.
            We divide by the number of waves and scale to the SAMP_VAL_MAX
            s.t. if all waves phase align, they will just reach the max value.
            INPUTS:
                seg   - Segment which describes what frequencies & how many samples to compute.
                ptr   - Pointer to buffer to access the data.
                buf   - The buffer passed to the Spectrum Transfer Function.
                fSamp - The Sampling Frequency
        """
        ## Clear out Previous Values ##
        for i in range(Seg.SampleLength):
            Ptr[i] = 0
        ## Compute and Add the full wave, Each frequency at a time
        for f in Seg.Frequencies:
            F = f / fSamp
            for i in range(Seg.SampleLength):
                Ptr[i] += sin(2 * pi * i * F) / Seg.SampleLength ## Divide by number of waves (To Avoid Clipping)
        ## Scale and round each value
        for i in range(Seg.SampleLength):
            Ptr[i] = int(SAMP_VAL_MAX*Ptr[i])

        spcm_dwDefTransfer_i64 (hCard, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, int32(0), Buf, uint64(0), uint64(Seg.SampleLength*2*1)) ## The 1 is a reminder to support Multiple Channels
        spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)


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
    def __init__(self, Frequencies=[], Resolution=1E6):
        for i in range(len(Frequencies)):
            if Frequencies[i] < Resolution:
                print("Resolution: ", Resolution, "Hz, sets the minimum allowed frequency. (it was violated)")
                print("Decreasing Resolution to: ", Frequencies[i], "Hz, in order to satisfy requirement")
                Resolution = Frequencies[i]
            if Frequencies[i] > SAMP_FREQ_MAX / 2:
                print("All frequencies must below Nyquist frequency: ", SAMP_FREQ_MAX / 2)
                print("Removing offending frequency, ", Frequencies[i], "Hz, from Segment")
                Frequencies[i] = 0
        ### Initialize
        self.Frequencies = Frequencies
        self.Resolution  = Resolution
        self.SampleLength = None

    def add_frequency(self, f):
        for freq in self.frequencies:
            if f == freq:
                print("Already in List!")
                return
        if f < self.Resolution:
            print("Resolution: ", Resolution, "Hz, sets the minimum allowed frequency. (it was violated)")
        elif f > SAMP_FREQ_MAX / 2:
            print("All frequencies must below Nyquist frequency: ", SAMP_FREQ_MAX / 2)
        else:
            self.Frequencies.append(f)

    def remove_frequency(self, f):
        self.Frequencies = [F for F in self.Frequencies if F != f]

    def __str__(self):
        s = "Segment with Resolution: " + str(self.Resolution) + "\n"
        s += "Contains Frequencies: \n"
        for f in Frequencies:
           s += "---" + str(f) + "Hz\n"
        return s

def wiggleOutput(hCard, time = 10000):
    """
        Performs a Standard Initialization for designated Channels & Trigger
        INPUTS:
            hCard - The handle to the opened hardware card
            time  - How long the output streams in Milliseconds
        OUTPUTS:
            NULL
    """
    print("Looping Signal for ", time/1000 if time else "infinity", " seconds...")
    spcm_dwSetParam_i32(hCard, SPC_TIMEOUT, time)  # Runs for 10 seconds
    dwError = spcm_dwSetParam_i32(hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_CARD_WAITREADY)
    if dwError == ERR_TIMEOUT:
        spcm_dwSetParam_i32(hCard, SPC_M2CMD, M2CMD_CARD_STOP)
    error_check()
