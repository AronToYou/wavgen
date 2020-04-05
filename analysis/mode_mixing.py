import matplotlib.pyplot as plt
import numpy as np
from math import pi, sin

## Parameters and Data Structures ##
T = 10          # Number of Traps
N = int(16E4)   # (samples) WindowLength
SF = 1250E6     # (Hz) Sampling Frequency
center = 100E6  # (Hz) Center Frequency of traps
spacing = 1E6   # (Hz) Spacing between trap frequencies

# Derived #
w_0 = [(2*pi)*(center + (i - T//2)*spacing) for i in range(T)]  # (radians/sample) frequency of each trap
phi_0 = np.zeros(T)                                             # Phase of each trap
A_0 = np.ones(T)                                                # Amplitude of each trap

t = np.arange(0, N / SF, 1 / SF)  # radial samples
assert len(t) == N


## Helper Functions ##
# noinspection PyPep8Naming
def superimpose(w, phi, amp=None):
    """
        Calculates the combined waveform.
    """
    waveform = np.zeros(N)
    A = (np.ones(T) if amp is None else amp)
    for i in range(T):
        theta = np.add(np.multiply(t, w[i]), phi[i])
        waveform = np.add(waveform, np.multiply(A[i], np.exp(1j * theta)))
    return waveform


def mix_signals(attr_0):
    """
        Assumes the input array is sorted.
        Returns the 1st & 2nd order mixed values.
    """
    # 1st-Order Signal Mixing #
    attr_1 = np.array([attr_0[i] - attr_0[j] for i in range(1, T) for j in range(i)])

    # 2nd-Order #
    attr_2 = []

    for i in range(T):
        for j in range(len(attr_1)):
            attr_2.extend([attr_0[i] + attr_1[j], attr_0[i] - attr_1[j]])

    return attr_1, np.array(attr_2)


def loop_phase_configurations(amps, rates, w):
    _, w_mix = mix_signals(w)

    for i, r in enumerate(rates):
        for j, a in enumerate(amps):
            # print("Iteration: ", 1 + j + i * 5)

            phase = np.array([a * 0.5 * (1 + sin(r * t / T)) for t in range(T)])
            _, phase_mix = mix_signals(phase)

            wave_mix = superimpose(w_mix, phase_mix)
            wave_mix_ft = np.fft.fft(wave_mix)

            w_all = np.concatenate((w, w_mix))
            phase_all = np.concatenate((phase, phase_mix))

            wave_all = superimpose(w_all, phase_all)
            wave_all_ft = np.fft.fft(wave_all)

            # idx = 1 + (j + i * 5) % 6

            # tit = "r=%.2f a=%.2f" % (r, a)
            # plt.title(tit)
    print("Done!")


## Analysis & Plotting ##
# 0th-Order
waveform_0 = superimpose(w_0, phi_0, A_0)
ft_0 = np.fft.fft(waveform_0)

# Exploration of Initial Phases
# amps = [pi/4, pi/2, pi, 2*pi, 4*pi]
# rates = [pi/4, pi/2, pi, 2*pi, 4*pi]
# loop_phase_configurations(amps, rates, w_2)

# High-Order Mixing
w_1, w_2 = mix_signals(w_0)  # Obtain 1st & 2nd Order mixing frequencies

amp = [pi, 2*pi]
rate = np.linspace(0, 8*pi, 300)
itr = 0
for a in amp:
    itr += 1
    print("Iteration: ", itr)
    rms = []
    rnd = 0
    for r in rate:
        rnd += 1
        print("Round: ", rnd)
        phi_0 = np.array([a * 0.5*(1 + sin(r * t / T)) for t in range(T)])
        phi_1, phi_2 = mix_signals(phi_0)

        mixture = superimpose(w_1, phi_1)
        rms.append(np.sqrt(np.mean(mixture.real**2 + mixture.imag**2)))
    plt.figure()
    plt.plot(rate, rms)
plt.show()
