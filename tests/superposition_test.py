from lib import Superposition, SuperpositionFromFile
from time import time
from lib.waveform import DATA_MAX

MEM_SIZE = 4_294_967_296  # Board Memory

r = [2.094510589860613, 5.172224588379723, 2.713365750754814, 2.7268654021553975, 1.   /
     9455621726067513, 2.132845902763719, 5.775685169342227, 4.178303582622483, 1.971  /
     4912917733933, 1.218844007759545, 4.207174369712666, 2.6609861484752124, 3.41140  /
     54221128125, 1.0904071328591276, 1.0874359520279866, 1.538248528697041, 0.501676  /
     9726252504, 2.058427862897829, 6.234202186024447, 5.665480185178818]

if __name__ == '__main__':
    freq_A = [90E6 + j*1E6 for j in range(10)]
    freq_B = [90E6 + j*2E6 for j in range(10)]
    phases = r[:len(freq_A)]

    sweep_size = MEM_SIZE // 8
    assert (sweep_size % 32) == 0, "Not 32 bit aligned."

    # A = SuperpositionFromFile('A.h5')
    A = Superposition(freq_A, sample_length=16E4)
    A.set_phases(phases)

    times = [time()]
    A.compute_and_save('../scratch/A.h5')
    times.append(time())

    # B = SuperpositionFromFile('B.h5')
    B = Superposition(freq_B, sample_length=16E5)
    B.set_phases(phases)

    times.append(time())
    B.compute_and_save('../scratch/B.h5')
    times.append(time())

    # AB = SuperpositionFromFile('AB.h5')
    AB = Superposition(freq_A, sample_length=16E6, targets=freq_B)
    AB.set_phases(phases)

    times.append(time())
    AB.compute_and_save('../scratch/AB.h5')
    times.append(time())

    print("DATA_MAX: ", DATA_MAX)
    print(32E4 / (times[1] - times[0]), " bytes/second")
    print((times[2] - times[1])*1000, " ms")
    print(32E5 / (times[3] - times[2]), " bytes/second")
    print((times[4] - times[3])*1000, " ms")
    print(32E6 / (times[5] - times[4]), " bytes/second")
    print("Total time: ", times[-1] - times[0], " seconds")

    A.plot()
    B.plot()
    AB.plot()
    import matplotlib.pyplot as plt
    plt.show()
