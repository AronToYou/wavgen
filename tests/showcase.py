from wavgen.waveform import *
from wavgen.card import *

## Define some Parameters ##
freqs = [78E6 + i*1E6 for i in range(5)]
duration = 2E-6  # (seconds)
center = 80E6  # (Hertz)
bandwidth = 10E6  # (Hertz)

## Create & Compute some Waveforms ##
wave = Superposition(freqs, sample_length=int(16E3))
wave.compute_waveform()

another_wave = even_spacing(5, int(80E6), int(2E6))

pulse = HS1(duration, center, bandwidth)
pulse.compute_waveform()

wildcard = Sweep(wave, another_wave, sample_length=16E3)

## Open the Spectrum card ##
dwCard = Card(mode='sequential')
dwCard.setup_channels(amplitude=300, ch0=False, ch1=True, filter=True)
dwCard.load_waveforms([pulse, wave, wildcard])
# dwCard.load_sequence()
# dwCard.setup_buffer() DECPRECATED
dwCard.wiggle_output()
