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
#     A = Superposition(freq_A, sample_length=16E4, filename='A.h5')
#     A.set_phases(phases)
#     print("Computing A...")
#     A.compute_and_save()
#     print("Done with A!")
#
#     print("Preparing B")
#     B = Superposition(freq_B, sample_length=16E4, filename='B.h5')
#     B.set_phases(phases)
#     print("Computing B...")
#     B.compute_and_save()
#     print("Done with B!")#
#
#     ## Sweeping Waveforms ##
#     print("Preparing AB")
#     AB = Superposition(freq_A, sample_length=16E4, filename='AB.h5', target=freq_B)
#     AB.set_phases(phases)
#     print("Computing AB...")
#     AB.compute_and_save()
#     print("Done with AB!")
#
#     print("Preparing BA")
#     BA = Superposition(freq_B, sample_length=16E4, filename='BA.h5', target=freq_A)
#     BA.set_phases(phases)
#     print("Computing BA...")
#     sweep_BA.compute_and_save()
#     print("Done with BA!")
