from spectrum_lib import *

## Open Card/Configure
card = OpenCard(mode='continuous', loops=0)
card.setup_channels(amplitude=200, ch0=False, ch1=True)

### Choose Superpositions
freq = [78E6, 79E6, 80E6, 81E6, 82E6]
segmentA = Segment(freqs = freq)

## Load Card
card.load_segments([segmentA])
card.setup_buffer()

## Run the Card
card.wiggle_output(timeout = 10000)

print("Done -- Success!")


