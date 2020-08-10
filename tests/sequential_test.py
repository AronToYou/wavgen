from wavgen.waveform import *

MEM_SIZE = 4_294_967_296  # Board Memory

r = [2.094510589860613, 5.172224588379723, 2.713365750754814, 2.7268654021553975, 1.   /
     9455621726067513, 2.132845902763719, 5.775685169342227, 4.178303582622483, 1.971  /
     4912917733933, 1.218844007759545, 4.207174369712666, 2.6609861484752124, 3.41140  /
     54221128125, 1.0904071328591276, 1.0874359520279866, 1.538248528697041, 0.501676  /
     9726252504, 2.058427862897829, 6.234202186024447, 5.665480185178818]

if __name__ == '__main__':

    freq_A = [90E6 + j*1E6 for j in range(10)]
    phases = r[:len(freq_A)]

    sweep_size = MEM_SIZE // 8
    assert (sweep_size % 32) == 0, "Not 32 bit aligned."

    ## 2 Configurations of pure tones ##
    A = Superposition(freq_A, phases=phases, resolution=int(1E6))
    B = even_spacing(10, int(94.5E6), int(2E6), phases=phases)

    ## Sweeps, in both directions, between the two Configurations ##
    AB = Sweep(A, B, sweep_time=100.0)
    BA = Sweep(B, A, sweep_time=100.0)


    times = [time()]
    A.compute_waveform()
    times.append(time())

    times.append(time())
    B.compute_waveform()
    times.append(time())

    times.append(time())
    AB.compute_waveform()
    times.append(time())

    times.append(time())
    BA.compute_waveform()
    times.append(time())

    ## Performance Metrics ##
    print("DATA_MAX: ", DATA_MAX)
    print(A.SampleLength*2 / (times[1] - times[0]), " bytes/second")
    print((times[2] - times[1])*1000, " ms")
    print(B.SampleLength*2 / (times[3] - times[2]), " bytes/second")
    print((times[4] - times[3])*1000, " ms")
    print(AB.SampleLength*2 / (times[5] - times[4]), " bytes/second")
    print((times[6] - times[5]) * 1000, " ms")
    print(BA.SampleLength*2 / (times[7] - times[6]), " bytes/second")

    print("Total time: ", times[7] - times[0], " seconds")
