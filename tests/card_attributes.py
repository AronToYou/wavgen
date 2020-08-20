from wavgen import *
from wavgen.spectrum import *

dwCard = Card()
hCard = dwCard.hCard

max_loops = int32(0)

spcm_dwGetParam_i32(hCard, SPC_SEQMODE_AVAILMAXLOOP, byref(max_loops))

print(max_loops.value)