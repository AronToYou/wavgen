######### Board API #########
from pyspcm import *
from spcm_tools import *
from helper import *

########## Initialization  ##########
hCard = spcm_hOpen(create_string_buffer (b'/dev/spcm0')) # Opens Card
errorCheck(hCard)
spcm_dwSetParam_i32(hCard, SPC_M2CMD, M2CMD_CARD_RESET)

########## Configuration ############
###### Card Mode #
spcm_dwSetParam_i32(hCard, SPC_CARDMODE,   SPC_REP_STD_CONTINUOUS) ## SPC_REP_STD_CONTINUOUS) # Sets Continuous Mode (Loops Memory)
spcm_dwSetParam_i64(hCard, SPC_LOOPS,      int64(0))               # Sets Number of Loops to 0; meaning infinite

###### Channel #
setupChannel(hCard)

###### Memory/Wave #
pvBuffer, qwBufferSize = wave(hCard, 10000, 1000)

###### Write Data to Board #
writeToBoard(hCard, pvBuffer, qwBufferSize)

########## Arm & Ignite the Fireworks ################
wiggleOutput(hCard)

################################################################################################
spcm_vClose(hCard)

print("Success! -- Done")