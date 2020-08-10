from wavgen import *
from time import time

if __name__ == '__main__':

    freq_A = [90E6 + j*1E6 for j in range(10)]
    phases = rp[:len(freq_A)]

    sweep_size = MEM_SIZE // 8

    ## First Superposition defined with a list of frequencies ##
    A = Superposition(freq_A, phases=phases, resolution=int(1E6))

    ## Another Superposition made with Wrapper Function ##
    B = even_spacing(10, int(94.5E6), int(2E6), phases=phases)

    ## A Sweep between the 2 previously defined stationary waves ##
    AB = Sweep(A, B, sweep_time=10.0)

    times = [time()]
    B.compute_waveform()
    times.append(time())

    times.append(time())
    A.compute_waveform()
    times.append(time())

    times.append(time())
    AB.compute_waveform()
    times.append(time())

    ## Performance Metrics ##
    print("DATA_MAX: ", DATA_MAX)
    print(32E4 / (times[1] - times[0]), " bytes/second")
    print((times[2] - times[1])*1000, " ms")
    print(32E5 / (times[3] - times[2]), " bytes/second")
    print((times[4] - times[3])*1000, " ms")
    print(32E6 / (times[5] - times[4]), " bytes/second")
    print("Total time: ", times[-1] - times[0], " seconds")

    ## Plotting of our Waveforms for Validation ##
    AB.plot()
    A.plot()
    B.plot()
    import matplotlib.pyplot as plt
    plt.show()
