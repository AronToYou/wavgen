from wavgen.waveform import *
import matplotlib.pyplot as plt

MEM_SIZE = 4_294_967_296  # Board Memory
PLOT_MAX = int(1E4)

## 20 Random Phases within [0, 2pi] ##
r = [2.094510589860613, 5.172224588379723, 2.713365750754814, 2.7268654021553975, 1.   /
     9455621726067513, 2.132845902763719, 5.775685169342227, 4.178303582622483, 1.971  /
     4912917733933, 1.218844007759545, 4.207174369712666, 2.6609861484752124, 3.41140  /
     54221128125, 1.0904071328591276, 1.0874359520279866, 1.538248528697041, 0.501676  /
     9726252504, 2.058427862897829, 6.234202186024447, 5.665480185178818]

###### This script is used to define, calculate, a save specific waveforms to files ######

## Calculating a few different HS1 Pulses ##

## A set of Waveforms which describe a smooth oscillation between two trap arrangements ##
## Frequency Sets to Switch Between ##

if __name__ == '__main__':

    freq_A = [90E6 + j*1E6 for j in range(10)]
    freq_B = [90E6 + j*2E6 for j in range(10)]
    phases = r[:len(freq_A)]
    sweep_size = MEM_SIZE // 8
    assert (sweep_size % 32) == 0, "Not 32 bit aligned."

    # ## Stationary Waveforms ##
    # A = Superposition(freq_A, sample_length=16E4)
    # A.set_phases(phases)
    # A.compute_and_save('A.h5')
    #
    # B = Superposition(freq_B, sample_length=16E4)
    # B.set_phases(phases)
    # B.compute_and_save('B.h5')

    # # Sweeping Waveforms ##
    # AB = Sweep(A, B, sample_length=sweep_size)
    # AB.compute_and_save('AB.h5')
    #
    # BA = Sweep(B, A, sample_length=sweep_size)
    # BA.compute_and_save('BA.h5')

    ## HS1 Pulse ##
    center_freq = 90E6
    sweep_width = 20E6
    pulse_time = 4E-3

    hs1 = HS1(pulse_time, center_freq, sweep_width)
    hs1.compute_and_save("HS1.h5")
    del hs1

    hs1 = from_file('HS1.h5')
    hs1.plot()

    plt.show()
