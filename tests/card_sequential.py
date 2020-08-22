from wavgen import *
from time import sleep
from math import ceil
import os

if __name__ == '__main__':

    filename = 'card_sequential'  # Location for our HDF5 file
    step_time = 100.0  # Milliseconds

    # If we have already computed the Waveforms...
    if os.access(filename + '.h5', os.F_OK):  # ...retrieve the Waveforms from file.
        A = from_file(filename, 'A')
        AB = from_file(filename, 'AB')
        B = from_file(filename, 'B')
        BA = from_file(filename, 'BA')
    else:
        freq_A = [90E6 + j * 1E6 for j in range(10)]
        phases = rp[:len(freq_A)]
        samples = 32*SAMP_FREQ // int(1E6)

        print("Samples: ", samples)
        samples += 32 - samples%32
        print("Samples: ", samples)

        ## Define 2 Superposition objects ##
        A = Superposition(freq_A, phases=phases, sample_length=samples)  # One via the default constructor...
        B = even_spacing(10, int(94.5E6), int(2E6), phases=phases, sample_length=samples)  # ...the other with a useful constructor wrapper helper
        assert A.SampleLength == B.SampleLength, "Sanity Check"

        ## 2 Sweep objects. One in each direction between the 2 previously defined waves ##
        AB = Sweep(A, B, sweep_time=step_time)
        BA = Sweep(B, A, sweep_time=step_time)

        BA.compute_waveform(filename, 'BA')
        B.compute_waveform(filename, 'B')
        A.compute_waveform(filename, 'A')
        AB.compute_waveform(filename, 'AB')
        A.plot()
        B.plot()
        AB.plot()
        BA.plot()

    segments = [A, AB, B, BA]
    segs = len(segments)
    loops = int(step_time*SAMP_FREQ/samples)
    steps = [Step(i, i, (i+1)%2*(loops - 1) + 1, (i+1)%segs) for i in range(segs)]

    dwCard = Card()
    dwCard.load_sequence(segments, steps)
    dwCard.wiggle_output()
    sleep(100)
    dwCard.stop_card()
