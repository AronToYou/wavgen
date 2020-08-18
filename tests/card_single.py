from wavgen import *

if __name__ == '__main__':

    freq_A = [90E6 + j*1E6 for j in range(10)]
    phases = rp[:len(freq_A)]

    ## Define 2 Superposition objects ##
    A = Superposition(freq_A, phases=phases, resolution=int(1E6))  # One via the default constructor...
    B = even_spacing(10, int(94.5E6), int(2E6), phases=phases)  # ...the other with a useful constructor wrapper helper

    B.compute_waveform()
    A.compute_waveform()

    print("A: ", A.SampleLength)
    print("B: ", B.SampleLength)

    dwCard = Card()

    dwCard.load_waveforms(A)
    dwCard.wiggle_output(duration=5000.0)

    dwCard.load_waveforms(B)
    dwCard.wiggle_output(duration=5000)

    print("Done!")
