from wavgen import *
from time import sleep
import easygui
import os

if __name__ == '__main__':

    step_time = 100.0  # Milliseconds

    ntraps= 7
    freq_A = [97E6 + j * 1E6 for j in range(ntraps)]
    phases = rp[:ntraps]

    ## Define 2 Superposition objects ##
    A = Superposition(freq_A, phases=phases)  # One via the default constructor...
    B = even_spacing(ntraps, int(100E6), int(2E6), phases=phases)  # ...the other with a useful constructor wrapper helper

    A.compute_waveform()
    B.compute_waveform()

    segments = [A, B]
    loops = int(MEM_SIZE//8 / A.SampleLength)
    steps = [Step(0, 0, loops, 1), Step(1, 1, loops, 0)]

    dwCard = Card()
    dwCard.load_sequence(segments, steps)
    dwCard.wiggle_output()
    easygui.msgbox('Looping')
    dwCard.stop_card()
