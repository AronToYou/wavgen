import matplotlib.pyplot as plt
import numpy as np
from math import exp, pi, cos, sin
from random import random

## Parameters and Data Structures ##
T = 10          # Number of Traps
N = int(16E4)   # (samples) WindowLength
SF = 1250E6     # (Hz) Sampling Frequency
center = 100E6  # (Hz) Center Frequency of traps
spacing = 1E6 # (Hz) Spacing between trap frequencies

# Derived #
w = [(2*pi)*(center + (i - T//2)*spacing) for i in range(T)]  # (radians/sample) frequency of each trap
phi = np.array([2*pi*abs(sin(pi*i/T)) for i in range(T)])     # Phase of each trap
A = np.ones(T)                                                # Amplitude of each trap
waveform = np.zeros(N)                                        # The sum of all waveforms



## Waveform Function ##
t = np.arange(0, N / SF, 1 / SF)  # radial samples
assert len(t) == N


def wave(w, phi):
    theta = np.add(np.multiply(t, w), phi)
    return np.exp(1j * theta)



## Waveform Calculation ##
for i in range(T):
    waveform = np.add(waveform, wave(w[i], phi[i]))

## Non-Linearities ##
# 1st-Order #
E1 = np.zeros(N)

w_1 = np.array([w[j] - w[i] for i in range(T) for j in range(i + 1, T)])
phi_1 = np.array([phi[j] - phi[i] for i in range(T) for j in range(i + 1, T)])

for w_mix, phi_mix in zip(w_1, phi_1):
    E1 = np.add(E1, wave(w_mix, phi_mix))

# 2nd-Order #
E2 = np.zeros(N)

w_2 = []
phi_2 = []

for w_t, phi_t in zip(w, phi):
    for w_mix, phi_mix in zip(w_1, phi_1):
        w_2.extend([w_t + w_mix, w_t - w_mix])
        phi_2.extend([phi_t + phi_mix, phi_t - phi_mix])

for w_mix, phi_mix in zip(w_2, phi_2):
    E2 = np.add(E2, wave(w_mix, phi_mix)) 