import h5py
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
from math import sin, pi
from ctypes import c_int16

MAX = (2 ** 15 - 1)

# def targ(array):
#     N = len(array)
#     for j in range(N):
#         array[j] = c_int16(int(MAX * sin(2 * pi * j / N)))
#     print(array)


class Seg(mp.Process):

    def __init__(self, array):
        super().__init__()
        self.Buf = array

    def run(self):
        N = len(self.Buf)
        for j in range(N):
            self.Buf[j] = c_int16(int(MAX * sin(2 * pi * j / N)))
        print(self.Buf)


if __name__ == '__main__':
    f = h5py.File('test.h5py', 'w')
    dset = f.create_dataset('data', shape=(20,), dtype='int16')

    procs = []
    data = []
    for i in range(mp.cpu_count()):
        dat = RawArray(c_int16, 10)
        data.append(dat)

        # p = mp.Process(target=targ, args=(dat,))
        p = Seg(dat)
        procs.append(p)
        p.start()

    for p in procs:
        p.join()
    dset[0:10] = procs.pop().Buf
    print(dset[()])
    plt.plot(dset[()])
    plt.show()
    f.close()

    with h5py.File('test.h5py', 'r') as f:
        print(f['data'][()])
        plt.plot(f['data'][()])
        plt.show()