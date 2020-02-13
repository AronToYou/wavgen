from lib.spectrum_lib import *

         #### IMPORTANT NOTES ####
# max value of amplitude: 2000mV. Do not exceed.
# For a given maximum amplitude of the card output, Vout,
# the amplitude of each pure tone will be (Vout / N) if
# N traps are output simultaneously.

## 20 Random Phases within [0, 2pi] ##
r = [2.094510589860613, 5.172224588379723, 2.713365750754814, 2.7268654021553975, 1.   /
     9455621726067513, 2.132845902763719, 5.775685169342227, 4.178303582622483, 1.971  /
     4912917733933, 1.218844007759545, 4.207174369712666, 2.6609861484752124, 3.41140  /
     54221128125, 1.0904071328591276, 1.0874359520279866, 1.538248528697041, 0.501676  /
     9726252504, 2.058427862897829, 6.234202186024447, 5.665480185178818]

# Define Waveform #
freq = [90E6 + j*0.5E6 for j in range(1)]
segmentA = Segment(freqs=freq, waves=None, sample_length=16E3)
segmentA.set_phase_all(r[:len(freq)])

# Open Card/Configure #
card = OpenCard()
card.setup_channels(amplitude=350)
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


