from pyspcm import *
from spcm_tools import *
from math import sin, pi
import sys
import matplotlib.pyplot as plt
import numpy as np

### Constants ###
MAX = (2 ** 16 - 1) ## Maximum value of digital output ~~ 16 bits

### Global Variables ###


class card:
    #### Class Variables ####
    # Configuration Indicators
    modeReady = False
    chanReady = False
    bufReady = False
    # Handle on card
    hCard = None

    def __init__(self):
        """
            Just Opens the card and resets the configuration
        """
        if hCard is not None:
            print('There can only be 1 open card --> 1 card instance!')
        hCard = spcm_hOpen(create_string_buffer(b'/dev/spcm0'))  # Opens Card
        errorCheck()
        spcm_dwSetParam_i32(hCard, SPC_M2CMD, M2CMD_CARD_RESET)

################# Basic Card Configuration Functions #################
### There is no 'card' class since there is only 1 card!
def openCard():



def resetCard():
    """
        Wipes clean Card Configuration
    """
    if not isOpen:
        print('No card open to reset! Noting done.')
        return
    spcm_dwSetParam_i32(hCard, SPC_M2CMD, M2CMD_CARD_RESET)
    global modeReady = False
    global chanReady = False
    global bufReady  = False


def setupMode(mode='continuous', loops=0):
    """
        Sets the Card mode
        Call without arguments for Continuous looping of Buffer
        INPUTS:
            mode  - Card output mode, limited support :)
            loops - Number of times the buffer is looped, 0 = infinity
    """
    if not isOpen:
        print('Card not open! Nothing done.')
        return
    if loops != int(loops):
        loops = int(loops)
        print('Rounding loops to required integer value: ', loops)
    switch = {
        'continuous' : SPC_REP_STD_CONTINUOUS,

    }
    mode = switch.get(mode)
    spcm_dwSetParam_i32(hCard, SPC_CARDMODE,    mode)  ## SPC_REP_STD_CONTINUOUS) # Sets Continuous Mode (Loops Memory)
    spcm_dwSetParam_i64(hCard, SPC_LOOPS,       int64(loops))  # Sets Number of Loops to 0; meaning infinite
    errorCheck()
    global modeReady = True

def setupChannel(ch0 = False, ch1 = True):
    """
        Performs a Standard Initialization for designated Channels & Trigger
        INPUTS:
            ch0   - Set to True to Activate Channel0
            ch1   - Set to True to Activate Channel1
    """
    if not isOpen:
        print('Card not open to setup! Nothing done.')
        return
    ######### Channels ##########
    CHAN = 0x00000000
    if (ch0):
        spcm_dwSetParam_i32(hCard, SPC_ENABLEOUT0, 1)
        CHAN = CHAN ^ CHANNEL0
    if (ch1):
        spcm_dwSetParam_i32(hCard, SPC_ENABLEOUT1, 1)
        CHAN = CHAN ^ CHANNEL1
    spcm_dwSetParam_i32(hCard, SPC_CHENABLE, CHAN)
    # spcm_dwSetParam_i64 (hCard, SPC_FILTER1,    1)   # Enables A Filter on Channel 1

    ######### Trigger ###########
    spcm_dwSetParam_i32(hCard, SPC_TRIG_ORMASK, SPC_TMASK_SOFTWARE)
    ########## Necessary? Reset already right?! ##################
    spcm_dwSetParam_i32(hCard, SPC_TRIG_ANDMASK, 0)
    spcm_dwSetParam_i32(hCard, SPC_TRIG_CH_ORMASK0, 0)
    spcm_dwSetParam_i32(hCard, SPC_TRIG_CH_ORMASK1, 0)
    spcm_dwSetParam_i32(hCard, SPC_TRIG_CH_ANDMASK0, 0)
    spcm_dwSetParam_i32(hCard, SPC_TRIG_CH_ANDMASK1, 0)
    spcm_dwSetParam_i32(hCard, SPC_TRIGGEROUT, 0)
    ############ ???? ####################################
    errorCheck()
    global chanReady = True

####################### Class Implementations ########################
class wave:
    def __init__(self, freq, amp):
        ### Input Validation
        if amp <= 80 or amp > 2500:
            print("Amplitude must within interval: [80 - 2500]")
            spcm_vClose(hCard)
            exit()
        if not (amp != int(amp)):
            amp = int(amp)
            print("Rounding amplitude to required integer value: ", amp)
        if freq <= 0 or freq > sampMax.value / 2:
            print("Frequency must be positive & below Nyquist frequency: ", sampMax.value / 2)
            spcm_vClose(hCard)
            exit()
        ### Initialize
        self.frequency = freq  # (Hertz)
        self.amplitude = amp   # (milliVolts)

    @property
    def period(self):  # Period of Oscillation (in samples) (assuming max sampling frequency)
        return 1.25E9 / self.frequency


class buffer:
    def __init__(self, waves):
        self.waves = []
        self.addWaves(waves)

    def addWaves(self,waves):
        if type(waves) is list:
            for w in waves:
                if not (type(w) is wave):
                    print("Something here isn't a wave...")
            self.waves.extend(waves)
        elif type(waves) is wave:
            self.waves.append(waves)
        else:
            print ("That's not  wave...")

    def calculateBuffer(self, plot=False):


    @classmethod
    def setupBuffer(res = 1E6):
        """
            Configures the Buffer while aiming for
            target frequency resolution set by argument 'res'
        """
        numChan = int32(0)  # Number of Open Channels
        sampMax = int64(0)  # Maximum Sampling Rate = 1.25 GHz
        memSize = int64(0)  # Total Memory ~ 4.3 GB

        #### Gather Information ####
        spcm_dwGetParam_i32(hCard, SPC_CHCOUNT, byref(numChan))  # Number of Open Channels
        spcm_dwGetParam_i64(hCard, SPC_PCISAMPLERATE, byref(sampMax))  # Maximum Sampling Rate
        spcm_dwGetParam_i64(hCard, SPC_PCIMEMSIZE, byref(memSize))  # Physical Memory Size in Samples
        print("Open Channels: ", numChan.value)
        numSamples = int(sampMax.value / resolution)  # Sets Sample Length s.t. the target resolution is roughly true
        numSamples = numSamples - (numSamples % 32) + 32  # Constrains the memory to be 64 byte aligned
        print('Achieved resolution: ', sampMax.value / numSamples)

        ########## Clock ############
        spcm_dwSetParam_i32(hCard, SPC_CLOCKMODE, SPC_CM_INTPLL)  # Sets out internal Quarts Clock For Sampling
        spcm_dwSetParam_i64(hCard, SPC_SAMPLERATE, sampMax)  # Sets Sampling Rate
        spcm_dwSetParam_i32(hCard, SPC_CLOCKOUT, 0)  # Disables Clock Output

        #### Set Amplifier Gains ####
        enb0 = int32(0)
        enb1 = int32(0)
        spcm_dwGetParam_i32(hCard, SPC_ENABLEOUT0,          byref(enb0))  # Checks if Channel 0 is Enabled
        spcm_dwGetParam_i32(hCard, SPC_ENABLEOUT1,          byref(enb1))  # Checks if Channel 1 is Enabled
        if enb0.value: spcm_dwSetParam_i32(hCard, SPC_AMP0, int32(2500))  # Sets Channel 0 Amplifier Gain
        if enb0.value: spcm_dwSetParam_i32(hCard, SPC_AMP0, int32(2500))  # Sets Channel 0 Amplifier Gain

        #### Configure Buffer ####
        spcm_dwSetParam_i64(hCard, SPC_MEMSIZE, int64(numSamples))  # Fixes the On-Board Memory Size
        bufSize = uint64(numSamples * 2 * numChan.value)  # Calculates Buffer Size in Bytes
        pvBuffer = pvAllocMemPageAligned(bufSize.value)  # Allocates space on PC
        pnBuffer = cast(pvBuffer, ptr16)  # Casts pointer into something usable



'''
    Generates a wave of desired frequency and places it in a contiguous buffer
    INPUTS: 
        hCard - The handle to the opened hardware card
        freq - Your desired frequency in Hertz ~~~~~~~~ RANGE: [0  ~ 100E6) soft upper limit
        amp  - Amplitude of your wave in Millivolts ~~~ RANGE: [80 - 2500] inclusive
    OUTPUTS:
        pvBuffer - A pointer to the Allocated Buffer Memory (To be given to DMA transfer function)
        bufSize  - The size of the buffer in bytes, (Also for DMA transfer function)
'''
def wave(hCard, freq, amp, plot=False):


    #### Calculate the Data ####
    bufDat = []
    relDat = [MAX * sin(2 * pi * (i / period / 100)) for i in range(numSamples*100)]
    zeros   = [0 for _ in range(99)]
    for i in range(0, numSamples):        # Calculates the Buffer Values
        pnBuffer[i] = int(MAX * sin(2 * pi * (i / period)))
        bufDat.append(pnBuffer[i])
        bufDat = bufDat + zeros
    # MatPlotLib stuff
    x = np.linspace(0, len(bufDat), len(bufDat))
    if plot:
        plt.plot(x, bufDat, x, relDat)
        plt.scatter(x, bufDat)
        plt.xlim((0, 10000))
        plt.show()

    return pvBuffer, bufSize


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
    errorCheck()


def writeToBoard(hCard, pvBuffer, qwBufferSize):
    sys.stdout.write("Starting the DMA transfer and waiting until data is in board memory\n")
    spcm_dwDefTransfer_i64(hCard, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, int32(0), pvBuffer, uint64(0), qwBufferSize)
    spcm_dwSetParam_i32(hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
    sys.stdout.write("... data has been transferred to board memory\n")

def errorCheck():
    ErrBuf = create_string_buffer(ERRORTEXTLEN)  # Buffer for returned Error messages
    if spcm_dwGetErrorInfo_i32(hCard, None, None, ErrBuf) != ERR_OK:
        sys.stdout.write("{0}\n".format(ErrBuf.value))
        spcm_vClose(hCard)
        exit()