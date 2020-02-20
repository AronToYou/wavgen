from lib.spectrum_lib import *
from random import random
import matplotlib.pyplot as plt
import pyvisa as vs

## Open Card/Configure ##
card = OpenCard(mode='continuous')
card.setup_channels(amplitude=200)


## For communicating with Instruments ##
rm = vs.ResourceManager()

scope = rm.open_resource(rm.list_resources()[0])    # Oscilloscope
print(scope.query("MEASUrement:MEAS1:VALue?"))

rf_meter = rm.open_resource("ASRL9::INSTR")         # RF Power Meter
print(rf_meter.query("pwr?"))

## Parameter ranges the table supports ##
ntraps = np.array([5])  # np.array([i for i in range(20)])
centers = np.array([90E6])  # np.array([85E6 + 1E6*i for i in range(30)])
spacings = np.array([1E6])  # np.array([0.1E6*(1 + i) for i in range(20)])

## The Look-Up Table ##
table = np.zeros((len(ntraps), len(centers), len(spacings)))


## Sub-Routines ##
def optimize_phases(segment, N):

    peaks = []
    rmses = []
    # Run & Take Measurements #
    for _ in range(20):
        phases = np.array([2 * pi * random() for _ in range(N)])
        segment.set_phases(phases)

        card.clear_segments()
        card.load_segments([segment])
        card.setup_buffer()

        card.wiggle_output(timeout=0, cam=False, verbose=False, stop=False)
        peak = scope.query("MEASUrement:MEAS1:VALue?")
        rms = rf_meter.query("*TRG")
        card.stop_output()

        peaks.append(peak)
        rmses.append(rms)

    plt.scatter(peaks, rmses)
    plt.show()


## Main ##
for N in ntraps:
    phases = np.zeros(N)

    for fc in centers:
        for sep in spacings:
            freq = [(fc - sep*(N-1)/2) + i*sep for i in range(N)]
            segment = Segment(freqs=freq, sample_length=16E4)
            optimize_phases(segment, N)

## Done! ##
print("Done -- Success!")
