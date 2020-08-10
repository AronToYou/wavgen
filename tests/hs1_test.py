from wavgen.waveform import *
from time import time


MEM_SIZE = 4_294_967_296  # Board Memory

if __name__ == '__main__':

    ## Define the Pulse ##
    pulse_time = 3E-6
    BW = 19E6
    center = 10E6

    ## Create it ##
    pulse = HS1(pulse_time, center, BW)

    ## Compute it ##
    start = time()
    pulse.compute_waveform()
    print("Total time: %dms" % ((time() - start)*1000))

    ## Plot! ##
    pulse.plot()
    import matplotlib.pyplot as plt
    plt.show()
