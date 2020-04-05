from lib import *

MEM_SIZE = 4_294_967_296  # Board Memory
PLOT_MAX = int(1E4)
###### This script is used to define, calculate, a save specific waveforms to files ######

## Calculating a few different HS1 Pulses ##

## A set of Waveforms which describe a smooth oscillation between two trap arrangements ##
## Frequency Sets to Switch Between ##

if __name__ == '__main__':
    center_freq = 90E6
    sweep_width = 20E6
    pulse_time = 0.4

    hs1 = HS1(pulse_time, center_freq, sweep_width)
    hs1.compute_and_save("./waveforms/hs1_test.h5py")
    hs1.plot()

#     freq_A = [90E6 + j*1E6 for j in range(10)]
#     freq_B = [90E6 + j*2E6 for j in range(10)]
#     sweep_size = MEM_SIZE // 8
#     assert (sweep_size % 32) == 0, "Not 32 bit aligned."
#
#     ## Stationary Waveforms ##
#     A = Superposition(freq_A, sample_length=16E4)
#     A.set_phases(phases)
#     A.compute_and_save('A.h5')
#
#     B = Superposition(freq_B, sample_length=16E4)
#     B.set_phases(phases)
#     B.compute_and_save('B.h5')
#
#     ## Sweeping Waveforms ##
#     AB = Superposition(freq_A, sample_length=16E4, target=freq_B)
#     AB.set_phases(phases)
#     AB.compute_and_save('AB.h5')
#
#     BA = Superposition(freq_B, sample_length=16E4, target=freq_A)
#     BA.set_phases(phases)
#     sweep_BA.compute_and_save('BA.h5')
