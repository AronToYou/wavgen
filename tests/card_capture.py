from wavgen import *
import instrumental
import os

if __name__ == '__main__':

    filename = 'card_capture'  # Location for our HDF5 file

    # If we have already computed the Waveforms...
    if os.access(filename + '.h5', os.F_OK):  # ...retrieve the Waveforms from file.
        A = from_file(filename, 'A')
        B = from_file(filename, 'B')
        AB = from_file(filename, 'AB')
    else:
        ## Define Waveform parameters ##
        ntraps = 15
        freq_A = [90E6 + j * 1E6 for j in range(ntraps)]
        phases = rp[:ntraps]

        ## Construct 2 Superposition objects & a Sweep ##
        A = Superposition(freq_A, phases=phases[:len(freq_A)])  # One via the default constructor...
        B = even_spacing(ntraps, int(102E6), int(2E6), phases=phases)  # ...the other with a useful constructor wrapper helper
        AB = Sweep(A, B, sweep_time=10)

        ## Compute all the Sample Points ##
        A.compute_waveform(filename, 'A')
        B.compute_waveform(filename, 'B')
        AB.compute_waveform(filename, 'AB')

    ## Set up the Card ##
    dwCard = Card()
    dwCard.setup_channels(300)

    ## Consecutively play each Waveform ##
    dwCard.load_waveforms(A)
    dwCard.wiggle_output()

    dwCard.load_waveforms(AB)
    dwCard.wiggle_output()

    dwCard.load_waveforms(B)
    dwCard.wiggle_output()

    print("Done!")
