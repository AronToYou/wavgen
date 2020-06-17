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
    sweep_size = MEM_SIZE // 8
    assert (sweep_size % 32) == 0, "Not 32 bit aligned."

    ## Stationary Waveforms ##
    # A = from_file('A.h5')
    A = Superposition(freq_A, sample_length=int(16E4), amp=0.25)
    A.set_phases(r[:len(freq_A)])
    A.compute_waveform()
    A.plot()

    # B = from_file('B.h5')
    B = Superposition(freq_B, sample_length=int(16E4), amp=0.5)
    B.set_phases(r[len(freq_A):len(freq_B)])
    B.compute_waveform()
    B.plot()

    ## Sweeping Waveforms ##
    # AB = from_file('AB.h5')
    AB = Sweep(A, B, sample_length=int(16E5))
    AB.compute_waveform()
    AB.plot()

    # BA = from_file('BA.h5')
    BA = Sweep(B, A, sample_length=sweep_size)
    BA.compute_waveform('BA.h5')
    BA.plot()

    C = even_spacing(5, int(90E6), int(1E6), phases=r[:5], periods=2)
    C.plot()

    ## HS1 Pulse ##
    center_freq = 90E6
    sweep_width = 20E6
    pulse_time = 4E-3

    # hs1 = from_file('HS1.h5')
    hs1 = HS1(pulse_time, center_freq, sweep_width)
    hs1.compute_waveform("HS1.h5")
    hs1.plot()

    plt.show()
