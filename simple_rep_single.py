


                     # EXAMPLE PROGRAM FROM SPECTRUM #




# **************************************************************************
#
# simple_rep_single.py                           (c) Spectrum GmbH , 11/2009
#
# **************************************************************************
#
# Example for all SpcMDrv based analog replay cards. 
# Shows a simple standard mode example using only the few necessary commands
#
# Information about the different products and their drivers can be found
# online in the Knowledge Base:
# https://www.spectrum-instrumentation.com/en/platform-driver-and-series-differences
#
# Feel free to use this source for own projects and modify it in any kind
#
# Documentation for the API as well as a detailed description of the hardware
# can be found in the manual for each device which can be found on our website:
# https://www.spectrum-instrumentation.com/en/downloads
#
# Further information can be found online in the Knowledge Base:
# https://www.spectrum-instrumentation.com/en/knowledge-base-overview
#
# **************************************************************************
#


from pyspcm import *
from spcm_tools import *
import sys
from math import sin, pi

#
# **************************************************************************
# main 
# **************************************************************************
#

# open card
# uncomment the second line and replace the IP address to use remote
# cards like in a generatorNETBOX
hCard = spcm_hOpen (create_string_buffer (b'/dev/spcm0'))
#hCard = spcm_hOpen (create_string_buffer (b'TCPIP::192.168.1.10::inst0::INSTR'))
if hCard == None:
    sys.stdout.write("no card found...\n")
    exit ()


# read type, function and sn and check for D/A card
lCardType = int32 (0)
spcm_dwGetParam_i32 (hCard, SPC_PCITYP, byref (lCardType))
lSerialNumber = int32 (0)
spcm_dwGetParam_i32 (hCard, SPC_PCISERIALNO, byref (lSerialNumber))
lFncType = int32 (0)
spcm_dwGetParam_i32 (hCard, SPC_FNCTYPE, byref (lFncType))

sCardName = szTypeToName (lCardType.value)
if lFncType.value == SPCM_TYPE_AO:
    sys.stdout.write("Found: {0} sn {1:05d}\n".format(sCardName,lSerialNumber.value))
else:
    sys.stdout.write("This is an example for D/A cards.\nCard: {0} sn {1:05d} not supported by example\n".format(sCardName,lSerialNumber.value))
    exit ()


# set samplerate to 1 MHz (M2i) or 50 MHz, no clock output
if ((lCardType.value & TYP_SERIESMASK) == TYP_M4IEXPSERIES) or ((lCardType.value & TYP_SERIESMASK) == TYP_M4XEXPSERIES):
    spcm_dwSetParam_i64 (hCard, SPC_SAMPLERATE, int64(MEGA(50)))
else:
    spcm_dwSetParam_i64 (hCard, SPC_SAMPLERATE, MEGA(1))
spcm_dwSetParam_i32 (hCard, SPC_CLOCKOUT,   0)

# setup the mode
qwChEnable = uint32 (2)
llMemSamples = int64 (KILO_B(64))
llLoops = int64 (0) # loop continuously
spcm_dwSetParam_i32 (hCard, SPC_CARDMODE,    SPC_REP_STD_CONTINUOUS)
spcm_dwSetParam_i32 (hCard, SPC_CHENABLE,    qwChEnable)
spcm_dwSetParam_i64 (hCard, SPC_MEMSIZE,     llMemSamples)
spcm_dwSetParam_i64 (hCard, SPC_LOOPS,       llLoops)
spcm_dwSetParam_i64 (hCard, SPC_ENABLEOUT1,  int64(1))

lSetChannels = int32 (0)
spcm_dwGetParam_i32 (hCard, SPC_CHCOUNT,     byref (lSetChannels))
lBytesPerSample = int32 (0)
spcm_dwGetParam_i32 (hCard, SPC_MIINST_BYTESPERSAMPLE,  byref (lBytesPerSample))

# setup the trigger mode
# (SW trigger, no output)
spcm_dwSetParam_i32 (hCard, SPC_TRIG_ORMASK,      SPC_TMASK_SOFTWARE)
spcm_dwSetParam_i32 (hCard, SPC_TRIG_ANDMASK,     0)
spcm_dwSetParam_i32 (hCard, SPC_TRIG_CH_ORMASK0,  0)
spcm_dwSetParam_i32 (hCard, SPC_TRIG_CH_ORMASK1,  0)
spcm_dwSetParam_i32 (hCard, SPC_TRIG_CH_ANDMASK0, 0)
spcm_dwSetParam_i32 (hCard, SPC_TRIG_CH_ANDMASK1, 0)
spcm_dwSetParam_i32 (hCard, SPC_TRIGGEROUT,       0)

lChannel = int32 (1)
spcm_dwSetParam_i32 (hCard, SPC_AMP0 + lChannel.value * (SPC_AMP1 - SPC_AMP0), int32 (1000))

# setup software buffer
qwBufferSize = uint64 (llMemSamples.value * lBytesPerSample.value * lSetChannels.value)
pvBuffer = pvAllocMemPageAligned (qwBufferSize.value)

# calculate the data
pnBuffer = cast  (pvBuffer, ptr16)
for i in range (0, llMemSamples.value, 1):
   pnBuffer[i] = int(16384*sin(2*pi*(i / llMemSamples.value)))


# we define the buffer for transfer and start the DMA transfer
sys.stdout.write("Starting the DMA transfer and waiting until data is in board memory\n")
spcm_dwDefTransfer_i64 (hCard, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, int32 (0), pvBuffer, uint64 (0), qwBufferSize)
spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
sys.stdout.write("... data has been transferred to board memory\n")

# We'll start and wait until the card has finished or until a timeout occurs
spcm_dwSetParam_i32 (hCard, SPC_TIMEOUT, 10000)
sys.stdout.write("\nStarting the card and waiting for ready interrupt\n(continuous and single restart will have timeout)\n")
dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_CARD_WAITREADY)
if dwError == ERR_TIMEOUT:
    spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_STOP)

spcm_vClose (hCard);

