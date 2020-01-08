from spectrum_lib import *


## Open Card/Configure
card = OpenCard(mode='continuous', loops=0)
card.setup_channels(amplitude=200, ch0=True, ch1=False, use_filter=False)
#max value of amplitude: 2000mV

### Choose Superpositions
freq = [85E6 + i*1E6 for i in range(17)] #[91E6 + i*0.2E6 for i in range(21)]
segmentA = Segment(freqs=freq, waves=None, sample_length=16E3)
segmentA.randomize()


## Load Card
card.load_segments([segmentA])
card.setup_buffer()
segmentA.plot()

## Run the Card
card.wiggle_output(timeout=0)

"""
## Randomize the Phases and try again ##
segmentA.randomize()
segmentA.plot()
card.setup_buffer()

card.wiggle_output(timeout = 5000)


## Set all but one component's Magnitude to 0 ##
new_mags = np.zeros(len(segmentA.Waves), dtype=int)
new_mags[0] = 1
segmentA.set_magnitude_all(new_mags)
segmentA.plot()
card.setup_buffer()

card.wiggle_output(timeout = 10000)
"""
## Done! ##
print("Done -- Success!")


