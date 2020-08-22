from wavgen import *

if __name__ == '__main__':
    ## Define some Parameters ##
    freqs = [78E6 + i * 1E6 for i in range(5)]
    pulse_time = 3E-6
    bandwidth = 19E6
    center = 10E6

    ## Create & Compute some Waveforms ##
    wave = Superposition(freqs)
    another_wave = even_spacing(5, int(80E6), int(2E6))
    # pulse = HS1(pulse_time, center, bandwidth)
    # wildcard = Sweep(wave, another_wave, sample_length=16E4)

    # This computes the waveforms to a temporary file, deleted upon exit
    for wav in [wave, another_wave]:  # , pulse, wildcard]:
        wav.compute_waveform()

    wave.plot(ends=True)
    another_wave.plot(ends=True)
    # pulse.plot()
    # wildcard.plot()

    import matplotlib.pyplot as plt
    plt.show()
