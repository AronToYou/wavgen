from pyspcm import *
from spcm_tools import *
from math import sin, pi
import sys
import matplotlib.pyplot as plt
import numpy as np

MAX = 2 ** 15
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
    numChan = int32(0) # Number of Open Channels
    sampMax = int64(0) # Maximum Sampling Rate = 1.25 GHz
    memSize = int64(0) # Total Memory ~ 4.3 GB
    numSamples = KILO_B(64) # Number of Samples per Oscillation
    period  = numSamples
    numCycles = 1                  # Number of Oscillations to be put in Buffer

    #### Gather Information ####
    spcm_dwGetParam_i32(hCard, SPC_CHCOUNT,         byref(numChan)) # Number of Open Channels
    spcm_dwGetParam_i64(hCard, SPC_PCISAMPLERATE,   byref(sampMax)) # Maximum Sampling Rate
    spcm_dwGetParam_i64(hCard, SPC_PCIMEMSIZE,      byref(memSize)) # Physical Memory Size in Samples
    print("Open Channels: ", numChan.value)

    ### Input Validation
    if amp <= 80 or amp > 2500:
        print("Amplitude must within interval: [80 - 2500]")
        spcm_vClose(hCard)
        exit()
    if freq <= 0 or freq > sampMax.value/2:
        print("Frequency must be positive & below Nyquist frequency: ", sampMax.value/2)
        spcm_vClose(hCard)
        exit()

    ### Memory Adjustment
    fsamp = int64(numSamples * freq)
    while fsamp.value > sampMax.value:
        numSamples  -= 32
        fsamp.value -= 32*freq
    if numSamples < 32:
        fsamp = int64(sampMax.value)
        numCycles = find_lcm(fsamp.value, freq) / fsamp.value
        period  = fsamp.value / freq
        numSamples = int(find_lcm(period*numCycles, 32))
    # Check if our scheme overflows the memory
    print("Period After: ", period)
    print("numCycle: ", numCycles)
    print("Adjusted numCycle: ", max(1, numSamples/period))
    print("NumSamples: ", numSamples)
    if numSamples > memSize.value:
        print("Dude, we need a better solution...like FIFO")
        spcm_vClose(hCard)
        exit()


    ########## Clock ############
    spcm_dwSetParam_i32 (hCard, SPC_CLOCKMODE,       SPC_CM_INTPLL)  # Sets out internal Quarts Clock For Sampling
    spcm_dwSetParam_i64 (hCard, SPC_SAMPLERATE,      fsamp)  # Sets Sampling Rate
    spcm_dwSetParam_i32 (hCard, SPC_CLOCKOUT,        0)  # Disables Clock Output
    spcm_dwGetParam_i64 (hCard, SPC_SAMPLERATE,     byref(fsamp))
    ####### Sanity Check ########
    print("Frequency you want: ", freq)
    print("Frequency you get: ", fsamp.value / period)

    #### Set Amplifier Gains ####
    enb0 = int32(0)
    enb1 = int32(0)
    spcm_dwGetParam_i32(hCard, SPC_ENABLEOUT0, byref(enb0))  # Checks if Channel 0 is Enabled
    spcm_dwGetParam_i32(hCard, SPC_ENABLEOUT1, byref(enb1))  # Checks if Channel 1 is Enabled
    spcm_dwSetParam_i32(hCard, SPC_AMP0, int32(80 + (amp - 80)*enb0.value))  # Sets Channel 0 Amplifier Gain
    spcm_dwSetParam_i32(hCard, SPC_AMP1, int32(80 + (amp - 80)*enb1.value))  # Sets Channel 1 Amplifier Gain

    #### Configure Buffer ####
    spcm_dwSetParam_i64(hCard, SPC_MEMSIZE, int64(numSamples)) # Fixes the On-Board Memory Size
    bufSize = uint64(numSamples * 2 * numChan.value) # Calculates Buffer Size in Bytes
    pvBuffer = pvAllocMemPageAligned(bufSize.value)                    # Allocates space on PC
    pnBuffer = cast(pvBuffer, ptr16)                                   # Casts pointer into something usable

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

'''
    Performs a Standard Initialization for designated Channels & Trigger
    INPUTS: 
        hCard - The handle to the opened hardware card
        time  - How long the output streams in Milliseconds
    OUTPUTS:
        NULL
'''
def wiggleOutput(hCard, time = 10000):
    print("Looping Signal for ", time/1000 if time else "infinity", " seconds...")
    spcm_dwSetParam_i32(hCard, SPC_TIMEOUT, time)  # Runs for 10 seconds
    dwError = spcm_dwSetParam_i32(hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_CARD_WAITREADY)
    if dwError == ERR_TIMEOUT:
        spcm_dwSetParam_i32(hCard, SPC_M2CMD, M2CMD_CARD_STOP)
    errorCheck(hCard)

'''
    Performs a Standard Initialization for designated Channels & Trigger
    INPUTS: 
        hCard - The handle to the opened hardware card
        ch0   - Set to True to Activate Channel0
        ch1   - Set to True to Activate Channel1
    OUTPUTS:
        NULL
'''
def setupChannel(hCard, ch0 = False, ch1 = True):
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

def writeToBoard(hCard, pvBuffer, qwBufferSize):
    sys.stdout.write("Starting the DMA transfer and waiting until data is in board memory\n")
    spcm_dwDefTransfer_i64(hCard, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, int32(0), pvBuffer, uint64(0), qwBufferSize)
    spcm_dwSetParam_i32(hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
    sys.stdout.write("... data has been transferred to board memory\n")

def errorCheck(hCard):
    ErrBuf = create_string_buffer(ERRORTEXTLEN)  # Buffer for returned Error messages
    if spcm_dwGetErrorInfo_i32(hCard, None, None, ErrBuf) != ERR_OK:
        sys.stdout.write("{0}\n".format(ErrBuf.value))
        spcm_vClose(hCard)
        exit()

'''
    Returns the Greatest Common Denominator between 2 numbers x & y
'''
def find_gcd(x, y):
    while (y):
        x, y = y, x%y
    return x
'''
    Returns the Least Common Multiple between x a& y
'''
def find_lcm(x, y):
    G = find_gcd(x, y)
    return x*y/G