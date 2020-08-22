from wavgen import *

if __name__ == '__main__':

    ntraps = 10
    freq_A = [90E6 + j*1E6 for j in range(2)]
    phases = rp[:ntraps]

    ## Define 2 Superposition objects ##
    B = Superposition(freq_A, phases=phases[:len(freq_A)], resolution=int(1E3))  # One via the default constructor...
    A = even_spacing(ntraps, int(94.5E6), int(2E6), phases=phases, periods=1000)  # ...the other with a useful constructor wrapper helper

    A.compute_waveform()
    A.plot()
    # B.compute_waveform()

    ## Setting up the Card ##
    dwCard = Card()

    dwCard.load_waveforms(A)
    dwCard.wiggle_output()

    # dwCard.load_waveforms(B)
    # dwCard.wiggle_output()  # duration=50000000)

    print("Done!")
