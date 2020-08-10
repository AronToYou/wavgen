from wavgen import *

if __name__ == '__main__':

    freq_A = [90E6 + j*1E6 for j in range(10)]
    phases = rp[:len(freq_A)]

    sweep_size = MEM_SIZE // 8

    ## Define 2 Superposition objects ##
    A = Superposition(freq_A, phases=phases, resolution=int(1E6))  # One via the default constructor...
    B = even_spacing(10, int(94.5E6), int(2E6), phases=phases)  # ...the other with a useful constructor wrapper helper

    ## 2 Sweep objects. One in each direction between the 2 previously defined waves ##
    AB = Sweep(A, B, sweep_time=10.0)
    BA = Sweep(B, A, sweep_time=10.0)

    BA.compute_waveform()
    B.compute_waveform()
    A.compute_waveform()
    AB.compute_waveform()

    dwCard = Card()
    dwCard.load_sequence()