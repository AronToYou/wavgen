from wavgen import *
import os

if __name__ == '__main__':

    filename = 'waveform_saving'  # Location for our HDF5 file

    # If we have already computed the Waveforms...
    if os.access(filename + '.h5', os.F_OK):  # ...retrieve the Waveforms from file.
        wave = from_file(filename, 'wave')
        another_wave = from_file(filename, 'another_wave')
        pulse = from_file(filename, 'pulse')
        wildcard = from_file(filename, 'wildcard')
    else:  # Otherwise we need to compute them now.
        ## Define some Parameters ##
        freqs = [78E6 + i * 1E6 for i in range(5)]
        pulse_time = 3E-6
        bandwidth = 19E6
        center = 10E6

        ## Define a few various Waveforms ##
        wave = Superposition(freqs)
        another_wave = even_spacing(5, int(80E6), int(2E6))
        pulse = HS1(pulse_time, center, bandwidth)
        wildcard = Sweep(wave, another_wave, sample_length=16E2)

        ## Compute them to File ##
        wave.compute_waveform(filename, 'wave')
        another_wave.compute_waveform(filename, 'another_wave')
        pulse.compute_waveform(filename, 'pulse')
        wildcard.compute_waveform(filename, 'wildcard')

    wave.plot()
    another_wave.plot()
    pulse.plot()
    wildcard.plot()

    import matplotlib.pyplot as plt
    plt.show()
