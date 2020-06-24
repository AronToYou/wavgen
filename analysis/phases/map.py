import os
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from tqdm import tqdm


def p_plot(pmap):
    npi = np.linspace(-2, 2, len(pmap))

    plt.figure(figsize=(8, 5))
    plt.pcolor(npi, npi, pmap, cmap='Blues')
    # plt.title(r"$(A_{rms}^2 / A_{max})$", fontsize=24)
    # plt.xlabel("$\phi_1 / \pi$", fontsize=25)
    # plt.ylabel("$\phi_2 / \pi$", fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplots_adjust(top=1.1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.show()


def p_save(pmap, r, m):
    npi = np.linspace(-2, 2, len(pmap))

    fig = plt.figure(figsize=(8, 5))
    plt.pcolor(npi, npi, pmap, cmap='Blues')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplots_adjust(top=1.1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)

    name = "" % ()
    plt.savefig("%04d_%06d" % (r, m), bbox_inches='tight',
                pad_inches=0)

    fig.clf()
    plt.close()
    del npi

############################################################

## Only 1 random set is sampled ##
phi = np.random.default_rng().uniform(high=2 * pi, size=30)


def power_iter(T, sep, c, p1, p2, res=500, M=1000):
    """ For a given parameter configuration,
        takes a random sample of phases,
        then graphs the score landscape for varying 2 relative phases.

        ==INPUTS==
            T ---- List of TrapNumbers
            sep -- List of Trap Seperations (Mhz).
            c ---- List of Center Frequencies (MHz).
            p1/2 - Indexes for which traps should have their phases mapped.
        ==OPTIONAL==
            res -- How many times we increment a phase / pixels on axis.
            M ---- Sampling Frequency, in units (1/us) (per microsecond)
    """
    f = lambda x: x - int(x) if x - int(x) else 1
    N = int((2 - T % 2) / (f(sep) * f(c)))

    samples = np.linspace(0, N * 2 * pi, N * M)
    f = np.array([c + (n - (T - 1) / 2) * sep for n in range(T)])

    ## Constant ##
    freqs = np.concatenate((f[:p1], f[p1 + 1:p2], f[p2 + 1:]))
    waves = np.sin(np.add(np.outer(samples, freqs), phi[:len(freqs)])).sum(axis=1)

    ## Mapped Over ##
    w1 = np.multiply(f[p1], samples)
    w2 = np.multiply(f[p2], samples)

    phase_map = np.linspace(-pi, pi, res)

    scores = np.empty((res, res))

    for i, dp1 in tqdm(enumerate(phase_map)):
        for j, dp2 in enumerate(phase_map):
            DoF = np.add(
                np.sin(np.add(w1, dp1)),
                np.sin(np.add(w2, dp2)))

            form = np.add(waves, DoF)
            form = np.divide(form, form.max())

            scores[i, j] = form.dot(form) / len(form)

    return scores

##########################################################

## New Sample on each call ##
gtr = np.random.default_rng()


def pmap_3(T, sep, c, res=1000, M=1000):
    """ ==INPUTS==
        T, sep, c - Define the trap configuration.
        res --- Samples per phase. [0, 2pi]
        M ----- Samples per '1Hz' period.
    """
    phi = gtr.uniform(high=2 * pi, size=T)
    f = lambda x: x - int(x) if x - int(x) else 1
    N = int((2 - T % 2) / (f(sep) * f(c)))

    samples = np.linspace(0, N * 2 * pi, N * M)
    freqs = np.array([c + (n - (T - 1) / 2) * sep for n in range(T)])

    rock = np.outer(freqs, samples)  # The Phases
    rock = np.add(phi, rock.transpose()).transpose()
    block = np.sin(rock)  # The Waves

    score = np.empty((res,) * 2, dtype=float)  # Output
    inc = 2 * pi / res

    for i in tqdm(range(res)):
        for j in range(res):
            form = block.sum(axis=0)
            norm = np.divide(form, form.max())

            score[j, i] = norm.dot(norm) / N

            rock[0] = np.add(rock[0], inc)
            block[0] = np.sin(rock[0])
        rock[1] = np.add(rock[1], inc)
        block[1] = np.sin(rock[1])

    return score

#################################################################


def iter_params(T, sep, c, res, M):
    os.chdir('C:/Users/aronw/Desktop/wavgen/Docs/source/_static/maps/')
    for t in T:
        for s in sep:
            for c_i in c:
                name = '%02d_%1d_%02d_%02d' % (t, s, int(100 * (s - int(s))), c_i)
                try:
                    os.mkdir(name)
                except FileExistsError:
                    pass
                os.chdir('./' + name)

                i = 1
                fold = 'Samp_%02d' % i
                while os.access('./' + 'Samp_%02d' % i, os.F_OK):
                    i += 1
                    fold = 'Samp_%02d' % i
                os.mkdir(fold)
                os.chdir('./' + fold)

                for r in res:
                    for m in M:
                        dat = pmap_3(t, s, c_i, r, m)
                        p_save(dat, r, m)
                        del dat
                os.chdir('../..')

##########################################################################


def dig(f):
    i = 0
    while f % 1:
        f = f * 10
        i += 1
    N = (f % 10) / 10 ** i
    return N if N else 1

##########################################################################


if __name__ == '__main__':
    T = [3, 4, 5]
    sep = [0.5, 1, 1.5, 2]
    c = [10, 50, 80, 90]

    res = [50, 100, 500]
    M = [10, 100, 1000, 10000]

    iter_params(T, sep, c, res, M)
