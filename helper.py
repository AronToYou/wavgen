from pyspcm import *
from spcm_tools import *
from math import sin, pi
import sys

MAX = 2 ** 16
MEM_FRACTION = 0.01
'''
    Generates a wave of desired frequency and places it in a contiguous buffer
    INPUTS: 
        hCard - The handle to the opened hardware card
        freq - Your desired frequency in Hertz
        amp  - Amplitude of your wave in Millivolts
    OUTPUTS:
        pvBuffer - A pointer to the Allocated Buffer Memory (To be given to DMA transfer function)
        bufSize  - The size of the buffer in bytes, (Also for DMA transfer function)
'''
def wave(hCard, freq, amp):
    numChan = int32(0)
    sampMax = int64(0)
    memSize = int64(0)
    numSamples = int64(KILO_B(64))
    #### Gather Information ####
    spcm_dwGetParam_i32(hCard, SPC_CHCOUNT,         byref(numChan)) # Number of Open Channels
    spcm_dwGetParam_i64(hCard, SPC_PCISAMPLERATE,   byref(sampMax)) # Physical Memory Size in Samples
    spcm_dwGetParam_i64(hCard, SPC_PCIMEMSIZE,      byref(memSize)) # Maximum Sampling Rate
    ### Input Validation
    if freq <= 0 or freq > sampMax.value/2:
        print("Frequency must be positive & below: ", sampMax.value/2)
        spcm_vClose(hCard)
        exit()
    ### Memory Adjustment
    fsamp = int64(numSamples.value*freq)
    while (fsamp.value > sampMax.value):
        numSamples.value -= KILO_B(1)
        fsamp.value      -= KILO_B(1)*freq

    ########## Clock ############
    spcm_dwSetParam_i32(hCard, SPC_CLOCKMODE, SPC_CM_INTPLL)  # Sets out internal Quarts Clock For Sampling
    spcm_dwSetParam_i64(hCard, SPC_SAMPLERATE, fsamp)  # Sets Sampling Rate to 50MHz
    spcm_dwSetParam_i32(hCard, SPC_CLOCKOUT, 0)  # Disables Clock Output

    spcm_dwGetParam_i64 (hCard, SPC_SAMPLERATE,     byref(fsamp))

    print("Frequency you want: ", freq)
    print("Frequency you get: ", fsamp.value / numSamples.value)

    #### Set Parameters ####
    spcm_dwSetParam_i32(hCard, SPC_AMP1, int32(amp))  # Enables Amplifier 1 for Output
    ###
    spcm_dwSetParam_i64(hCard, SPC_MEMSIZE, numSamples) # Fixes the On-Board Memory Size
    ###
    bufSize = uint64(numSamples.value * 2 * numChan.value)             # Allocates the Buffer
    pvBuffer = pvAllocMemPageAligned(bufSize.value)
    pnBuffer = cast(pvBuffer, ptr16)


    for i in range(0, numSamples.value, numChan.value):        # Calculates the Buffer Values
        for j in range(numChan.value):
            pnBuffer[i+j] = int(MAX * sin(2 * pi * (i / numSamples.value)))

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
    print("Looping Signal for ", time/1000, " seconds...")
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