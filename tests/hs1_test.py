from lib import HS1, HS1FromFile
from time import time

MEM_SIZE = 4_294_967_296  # Board Memory


if __name__ == '__main__':

    pulse_time = 5E-6
    BW = 40E6
    center = 100E6

    A = HS1(pulse_time, center, BW)

    start = time()
    A.compute_and_save('../scratch/hs1.h5')
    print("Total time: %dms" % ((time() - start)*1000))

    A.plot()
    import matplotlib.pyplot as plt
    plt.show()
