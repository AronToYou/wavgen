from spectrum_lib import *

card = OpenCard()
card.setup_channels()

### Continuous Mode Test
frequencies = [78E6, 79E6, 80E6, 81E6, 82E6]
segmentA = Segment(frequencies)

card.load_segments([segmentA])
card.setup_buffer()
card.wiggle_output(timeout = 3000)

features = int32(0)
spcm_dwGetParam_i32(OpenCard.hCard, SPC_PCIFEATURES, byref(features))
features = features.value & SPCM_FEAT_SEQUENCE
print("Features?: ", features)

###  Sequential Mode Test
freq = [500E3]
segmentB = Segment(freq, resolution=500E3)

card.set_mode('sequential')
card.load_segments([segmentB])
card.setup_buffer()

#### Program for Sequential Mode ####
    #### Step 1 ####
step = int64(0)
step_seg = int64(0)
loop = int64(1000)
next_step = int64(1)
condition = SPCSEQ_ENDLOOPALWAYS

value = (condition << 32) | (loop << 32) | (next_step << 16) | (step_seg)
spcm_dwSetParam_i64(OpenCard.hCard, SPC_SEQMODE_STEPMEM0 + step, value)

    #### Step 2 ####
step = int64(1)
step_seg = int64(1)
loop = int64(1000)
next_step = int64(0)
condition = SPCSEQ_ENDLOOPALWAYS

value = (condition << 32) | (loop << 32) | (next_step << 16) | (step_seg)
spcm_dwSetParam_i64(OpenCard.hCard, SPC_SEQMODE_STEPMEM0 + step, value)

card.wiggle_output(timeout=10000)

print("Success! -- Done")