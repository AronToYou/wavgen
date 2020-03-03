from lib import *

###### This script is used to define, calculate, a save specific waveforms to files ######

## Calculating a few different HS1 Pulses ##




## A set of Waveforms which describe a smooth oscillation between two trap arrangements ##
# Frequency Sets to Switch Between ##
# freq_A = [90E6 + j*0.5E6 for j in range(10)]  # Note: diffraction efficiency roughly maximized at 90MHz. Use this as center
# freq_B = [90E6 + j*1E6 for j in range(10)]

# ## Stationary Waveforms ##
# print("Preparing A")
# stable_A = WaveformFromFile('./waveforms/stable_A.h5py')  # Waveform
# phases = stable_A.get_phases()
# print("Computing A...")
# stable_A.compute_and_save()
# print("Done with A!")
#
# print("Preparing B...")
# stable_B = Waveform(freqs=freq_B, sample_length=16E4, filename='./waveforms/stable_B.h5py')
# stable_B.set_phases(phases)
# print("Computing B...")
# stable_B.compute_and_save()
# print("Done with B!")
#
#
# ## Sweeping Waveforms ##
# print("Preparing AB")
# sweep_AB = Waveform(freqs=freq_A, sample_length=8E8, targets=freq_B, filename='./waveforms/stable_AB.h5py')
# sweep_AB.set_phases(phases)
# print("Computing AB...")
# sweep_AB.compute_and_save()
# print("Done with AB!")
#
# print("Preparing BA")
# sweep_BA = Waveform(freqs=freq_B, sample_length=8E8, targets=freq_A, filename='./waveforms/stable_BA.h5py')
# sweep_BA.set_phases(phases)
# print("Computing BA...")
# sweep_BA.compute_and_save()
# print("Done with BA!")
