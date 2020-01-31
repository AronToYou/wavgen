from lib.spectrum_lib import *

         #### IMPORTANT NOTES ####
# max value of amplitude: 2000mV. Do not exceed.
# For a given maximum amplitude of the card output, Vout,
# the amplitude of each pure tone will be (Vout / N) if
# N traps are output simultaneously.

# Define Waveform #
freq = [90E6 + j*1.0E6 for j in range(1)]
segmentA = Segment(freqs=freq, waves=None, sample_length=16E3)
segmentA.randomize()

# Open Card/Configure #
card = OpenCard()
card.setup_channels()
card.load_segments([segmentA])
card.setup_buffer()
# Let it Rip #
card.wiggle_output(timeout=0, cam=False, verbose=True)
"""
## Set all but one component's Magnitude to 0 ##
new_mags = np.zeros(len(segmentA.Waves), dtype=int)
new_mags[0] = 1
segmentA.set_magnitudes(new_mags)
segmentA.plot()
card.setup_buffer()

card.wiggle_output(timeout = 10000)
"""
## Done! ##
print("Done -- Success!")


