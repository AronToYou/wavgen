import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import PCG64, Generator
from psutil import virtual_memory
from time import time
from math import ceil, sqrt, pi
from multiprocessing import Process, Queue, cpu_count

cpus = cpu_count()
max_rolls = 1000

waves = None
wave = None

## Helper ##
def dig(f):
    i = 0
    while f % 1:
        f = f * 10
        i += 1
    N = (f % 10) / 10 ** i
    return N if N else 1


def power_func(T, sep, c):
    """ Returns the P(phi) function,
        given a waveform parameter configuration.
        
        ==Given==
        T --- number of traps
        sep - spacing between traps (MHz)
        c --- center of traps (MHz)

        ==Returns==
        P(phi) -- Power as a function of trap relative phases
    """
    N = int(1000 * (2 - T % 2) // sep)
    samples = np.linspace(0, 2*pi, N)  # Enough for 1 period (assuming integer c).
    freqs = np.array([c + (n - (T-1)/2)*sep for n in range(T)])

    waves = np.outer(samples, freqs[1:T])
    wave = np.sin(np.multiply(freqs[0], samples))  # The first wave has no relative phase.

    return waves, wave


def power(waves, wave, queue, rolls, gtr):
    assert rolls <= max_rolls, "Something fishy..."
    phase_sets = gtr.uniform(high=2*pi, size=(rolls, 1, waves.shape[1]))
    dat = np.array([waves, ]*rolls)
    # print(dat.nbytes//1E6, " MB")
    forms = np.add(wave, np.sin(np.add(dat, phase_sets)).sum(axis=2))   # Un-Normalized sum of waves

    peaks = np.expand_dims(forms.max(axis=1), axis=1)
    forms = np.divide(forms, peaks)                                     # Normalized for Max Peak

    scores = []
    for form in forms:
        scores.append(form.dot(form) / len(form))  # Proportional to Power (Vrms^2)

    if queue is None:
        return phase_sets[np.argmax(scores), 0, :], np.array(scores)
    queue.put((phase_sets[np.argmax(scores), 0, :], np.array(scores)))


def power_iter(queue, T, sep, c, rolls):
    """ Returns the P(phi) function,
        given a waveform parameter configuration.

        ==Given==
        queue - for inter-process communication
        T ----- number of traps
        sep --- spacing between traps (MHz)
        c ----- center of traps (MHz)
        rolls - number of random phase sets sampled

        ==Returns==
        null
    """
    name = '%d/%.1f/%.1f' % (T, sep, c)
    N = int((2 - T % 2) / min(dig(sep), dig(c)))  # waveform is periodic to N*1MHz or N*1000samples
    assert N < 1000

    samples = np.linspace(0, 2 * pi * N, 1000 * N)  # Enough for 1 period (assuming integer c).

    freqs = np.array([c + (n - (T - 1) / 2) * sep for n in range(T)])

    waves = np.outer(samples, freqs[1:T])
    wave = np.sin(np.multiply(freqs[0], samples))  # The first wave has no relative phase.

    gtr = np.random.default_rng()
    scores = []
    top = 0
    ideal = None
    inc, val = [], []
    for i in range(rolls):
        phases = gtr.uniform(high=2 * pi, size=T - 1)

        form = np.add(wave, np.sin(np.add(waves, phases)).sum(axis=1))  # Un-Normalized sum of waves
        form = np.divide(form, form.max())                              # Normalized for Max Peak

        scores.append(form.dot(form) / len(form))  # Proportional to Power (Vrms^2)
        if scores[-1] > top:
            ideal = phases
            top = scores[-1]
            inc.append(i)
            val.append(top)

    plt.plot(inc, val)
    plt.title('%d traps, spaced %.1f MHz, centered at %.1f MHz' % (T, sep, c))
    plt.xlabel('# of Rolls')
    plt.ylabel('Score (RMS/Peak)')
    plt.show(block=False)

    queue.put((name, ideal, np.array(scores)))


#### Optimizers ####
def find_optimal_single(T, sep, c, rolls):
    """ Given a waveform parameter set,
        rolls many dice in parallel, up to a accumulative limit; 
        sampling random phase sets on each roll.
        
        Returns the top score & the top scoring set.
    """
    start = time()
    waves, wave = power_func(T, sep, c)

    
    #### Parallel ####
    if rolls > max_rolls*cpus:
        num_procs = rolls // max_rolls
        part = max_rolls
        rem = rolls % max_rolls
    else:
        num_procs = cpus
        part = rolls // cpus
        rem = rolls % cpus

    scores = Queue()
    gtr = np.random.default_rng()
    rand_state = gtr.bit_generator.jumped()

    args = (waves, wave, scores, part + rem, gtr)  # Handles the remainder on first run

    ## Startup the initial batch of Processes ##
    for _ in range(cpus):
        Process(target=power, args=args).start()
        args = (waves, wave, scores, part, Generator(rand_state))
        rand_state = rand_state.jumped()

    top = 0
    ideal = None
    score_data = []
    for p in range(num_procs, 0, -1):  # For each finished Process
        phase_set, score_set = scores.get()  # Collect the result
        set_top = score_set.max()

        if set_top > top:  # Track the overall most ideal Phases
            top = set_top
            ideal = phase_set

        score_data.append(score_set)  # Accumulate the scores

        if p > cpus:  # Begin a new Process
            Process(target=power, args=args).start()
            args = (waves, wave, scores, part, Generator(rand_state))
            rand_state = rand_state.jumped()

    # print('Param Set %.2fms per roll for %d traps.' % (1000*(time() - start)/rolls, T))

    return ideal, np.concatenate(score_data)
    

def find_optimal_rolls(ntraps, separations, centers, rolls):
    """ Given lists of potential parameters,
        for each possible combination of parameters,
        rolls the dice 'rolls' number of times & samples random phase sets.
        
        Returns the top score & the top scoring set of phases.

        NOTE: Parallelizes on rolls.
    """
    num_configs = len(separations)*len(centers)
    entries = sum(ntraps)*num_configs

    last = time()
    times = [last]
    prog_count = 0
    with h5py.File("rolls_op_phases.hdf5", 'w') as o, h5py.File("rolls_scores.hdf5", 'w') as s:
        for T in ntraps:
            for sep in separations:
                for c in centers:
                    ideal, scores = find_optimal_single(T, sep, c, rolls)

                    name = '%d/%.1f/%.1f' % (T, sep, c)
                    o.create_dataset(name, data=ideal)
                    s.create_dataset(name, data=scores)

                    prog_count += T
                    if time() - last > 60:
                        print('Running...  {:.1%}'.format(prog_count/entries))
                        last = time()

            times.append(time())

    elapsed = times[-1] - times[0]
    print('Total time: %d minutes %d seconds.' % (elapsed // 60, elapsed % 60))
    rates = [1000*(times[i+1] - times[i]) / (num_configs*rolls) for i in range(len(ntraps))]
    plt.plot(ntraps, rates)
    plt.xlabel('Number of Traps')
    plt.ylabel('Average time per Roll (ms)')
    plt.show(block=False)


def find_optimal_params(ntraps, separations, centers, rolls):
    """ Given a waveform parameter set,
        rolls the dice a number of times & samples random phase sets.

        Returns the top score & the top scoring set.
    """
    num_configs = len(separations) * len(centers)
    entries = sum(ntraps)*num_configs

    prog_count = 0
    results = Queue()
    procs = []
    for T in ntraps:
        for sep in separations:
            for c in centers:
                procs.append(Process(target=power_iter, args=(results, T, sep, c, rolls)))

    with h5py.File("op_phases.hdf5", 'w') as o, h5py.File("scores.hdf5", 'w') as s:
        last = time()
        times = [last]

        print('procs: ', len(procs))
        print('cpus: ', cpus)
        for _ in range(cpus):
            procs.pop(0).start()

        for T in ntraps:
            for _ in range(num_configs):
                name, ideal, scores = results.get()

                o.create_dataset(name, data=ideal)
                s.create_dataset(name, data=scores)

                prog_count += T
                if time() - last > 60:
                    print('Running...  {:.1%}'.format(prog_count / entries))
                    last = time()

                if len(procs) > 0:
                    procs.pop(0).start()

            times.append(time())

    elapsed = times[-1] - times[0]
    print('Total time: %d minutes %d seconds.' % (elapsed // 60, elapsed % 60))
    rates = [1000 * (times[i + 1] - times[i]) / (num_configs*rolls) for i in range(len(ntraps))]
    plt.plot(ntraps, rates)
    plt.xlabel('Number of Traps')
    plt.ylabel('Average time per Roll (ms)')
    plt.show(block=False)


if __name__ == '__main__':
    ntraps = np.arange(3, 20)
    separations = [1, 0.5, 0.1]  # MHz
    centers = [80, 85, 90, 95, 100]  # Mhz
    rolls = 120000

    # find_optimal_single(5, 1, 90, rolls)

    # find_optimal_rolls(ntraps, separations, centers, rolls)

    find_optimal_params(ntraps, separations, centers, rolls)  # 4.37

    print('Done!')
