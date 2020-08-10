from wavgen import *

if __name__ == '__main__':
    ## Define some Parameters ##
    freqs = [78E6 + i * 1E6 for i in range(5)]
    pulse_time = 3E-6
    bandwidth = 19E6
    center = 10E6

    ## Create & Compute some Waveforms ##
    wave = Superposition(freqs, sample_length=int(16E3))
    another_wave = even_spacing(5, int(80E6), int(2E6), periods=1)
    pulse = HS1(pulse_time, center, bandwidth)
    wildcard = Sweep(wave, another_wave, sample_length=16E4)

    filename = 'single_output_test'
    wave.compute_waveform(filename, 'wave')
    another_wave.compute_waveform(filename, 'another_wave')
    pulse.compute_waveform(filename, 'pulse')
    wildcard.compute_waveform(filename, 'wildcard')

    wave = from_file(filename, 'wave')
    another_wave = from_file(filename, 'another_wave')
    pulse = from_file(filename, 'pulse')
    wildcard = from_file(filename, 'wildcard')

    wave.plot()
    another_wave.plot()
    pulse.plot()
    wildcard.plot()

    import matplotlib.pyplot as plt
    plt.show()

