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

    # hs1 = HS1(pulse_time, center_freq, sweep_width, filename="hs1_test.h5py")
    # print("Computing HS1 pulse...")
    # hs1.compute_and_save()
    # print("Done!")
    hs1 = HS1(pulse_time, center_freq, sweep_width, "./waveforms/hs1_test.h5py")
    hs1.compute_and_save()
    hs1.plot()

#     freq_A = [90E6 + j*1E6 for j in range(10)]
#     freq_B = [90E6 + j*2E6 for j in range(10)]
#     sweep_size = MEM_SIZE // 8
#     assert (sweep_size % 32) == 0, "Not 32 bit aligned."
#
#     ## Stationary Waveforms ##
#     print("Preparing A")
#     stable_A = Superposition(freq_A, sample_length=16E4, filename='./waveforms/stable_A.h5py')
#     print("Computing A...")
#     stable_A.compute_and_save()
#     print("Done with A!")
#     phases = stable_A.get_phases()
#
#     print("Preparing B...")
#     stable_B = Superposition(freqs=freq_B, sample_length=16E4, filename='./waveforms/stable_B.h5py')
#     stable_B.set_phases(phases)
#     print("Computing B...")
#     stable_B.compute_and_save()
#     print("Done with B!")
#
#
#     ## Sweeping Waveforms ##
#     print("Preparing AB")
#     sweep_AB = Superposition(freqs=freq_A, sample_length=sweep_size, targets=freq_B, filename='./waveforms/stable_AB.h5py')
#     sweep_AB.set_phases(phases)
#     print("Computing AB...")
#     sweep_AB.compute_and_save()
#     print("Done with AB!")
#
#     print("Preparing BA")
#     sweep_BA = Superposition(freqs=freq_B, sample_length=sweep_size, targets=freq_A, filename='./waveforms/stable_BA.h5py')
#     sweep_BA.set_phases(phases)
#     print("Computing BA...")
#     sweep_BA.compute_and_save()
#     print("Done with BA!")
