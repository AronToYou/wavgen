######### Board API #########
from pyspcm import *
from spcm_tools import *
from helper import *

######## Card Initialization ########
openCard()  ## Opens the Card!


########## Configuration ############
###### Card Mode #


###### Channel #
setupChannel(hCard)

###### Memory/Wave #
pvBuffer, qwBufferSize = wave(hCard, freq=int(100E6), amp=2000, plot=False)

###### Write Data to Board #
writeToBoard(hCard, pvBuffer, qwBufferSize)

########## Arm & Ignite the Fireworks ################
wiggleOutput(hCard, time = 1)

################################################################################################
spcm_vClose(hCard)

print("Success! -- Done")