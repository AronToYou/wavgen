import wavgen as wv
import easygui

         #### IMPORTANT NOTES ####
# max value of amplitude: 2000mV. Do not exceed.
# For a given maximum amplitude of the card output, Vout,
# the amplitude of each pure tone will be (Vout / N) if
# N traps are output simultaneously.

## 20 Random Phases within [0, 2pi] ##
r = [2.094510589860613, 5.172224588379723, 2.713365750754814, 2.7268654021553975, 1.   /
     9455621726067513, 2.132845902763719, 5.775685169342227, 4.178303582622483, 1.971  /
     4912917733933, 1.218844007759545, 4.207174369712666, 2.6609861484752124, 3.41140  /
     54221128125, 1.0904071328591276, 1.0874359520279866, 1.538248528697041, 0.501676  /
     9726252504, 2.058427862897829, 6.234202186024447, 5.665480185178818]

if __name__ == '__main__':
    # Define Waveform #
    freq = [80E6 + j*1E6 for j in range(10)]
    A = wv.waveform.Superposition(freq, sample_length=int(16E3))
    A.set_phases(r[:len(freq)])
    A.compute_waveform('./gey.h5')
    #
    # A = wv.utilities.from_file('./waveforms/A.h5')
    # AB = wv.utilities.from_file('./waveforms/AB.h5')
    # B = wv.utilities.from_file('./waveforms/B.h5')
    # BA = wv.utilities.from_file('./waveforms/BA.h5')
    #
    # segments = [A, AB, B, BA]
    #
    # a = wv.utilities.Step(0, 0, 10000, 1)
    # ab = wv.utilities.Step(1, 1, 1, 2)
    # b = wv.utilities.Step(2, 2, 10000, 3)
    # ba = wv.utilities.Step(3, 3, 1, 0)
    #
    # steps = [a, ab, b, ba]

    # Open Card/Configure #
    print('begin')
    card = wv.Card()
    print('opened')
    card.setup_channels(amplitude=240)
    print('channel')
    card.load_waveforms(A)
    # card.load_sequence(segments, steps)

    # Let it Rip #
    card.wiggle_output(block=False)
    easygui.msgbox("Done?")

    ## Done! ##
    print("Done -- Success!")
