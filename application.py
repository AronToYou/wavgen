from spectrum_lib import *


## Open Card/Configure
card = OpenCard(mode='continuous', loops=0)
# max value of amplitude: 2000mV. Do not exceed
for i in range(4):
    card.setup_channels(amplitude=(100 + i*100), ch0=True, ch1=False, use_filter=False)
    freq = [100E6]
    segmentA = Segment(freqs=freq, waves=None, sample_length=16E3)
    card.load_segments([segmentA])
    segmentA.randomize()
    card.setup_buffer()
    card.wiggle_output(timeout=8000)
    time.sleep(1)
"""
## Choose Superpositions
freq = [82E6 + i*3E6 for i in range(15)]
segmentA = Segment(freqs=freq, waves=None, sample_length=16E3)
segmentA.randomize()


## Load Card
card.load_segments([segmentA])
card.setup_buffer()
segmentA.plot()

## Run the Card
card.wiggle_output(timeout=0)

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


