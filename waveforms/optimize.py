import matplotlib.pyplot as plt
import numpy as np
from numpy.random import PCG64, Generator
from time import time
from math import ceil, sqrt, pi
from multiprocessing import Process, Queue, cpu_count

cpus = cpu_count()
max_rolls = 4000

## Helper ##
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
    N = 1000 // sep
    samples = np.linspace(0, 2*pi, N)  # Enough for 1 period (assuming integer c).
    freqs = np.array([c + (n - (T-1)/2)*sep for n in range(T)])

    waves = np.outer(samples, freqs[1:T])
    wave = np.sin(np.multiply(freqs[0], samples))  # The first wave has no relative phase.

    def power(queue, rolls, gtr):
        phase_sets = gtr.uniform(high=2*pi, size=(rolls, 1, T-1))
        dat = np.array([waves, ]*rolls)
        forms = np.add(wave, np.sin(np.add(dat, phase_sets)).sum(axis=2))   # Un-Normalized sum of waves

        peaks = np.expand_dims(forms.max(axis=1), axis=1)
        forms = np.divide(forms, peaks)                                           # Normalized for Max Peak

        scores = []
        for form in forms:
            scores.append(form.dot(form) / len(form))  # Proportional to Power (Vrms^2)

        queue.put((phase_sets[np.argmax(scores), 0, :], np.array(scores)))

    return power


#### Optimizers ####
def find_optimal_fast(params):
    """ Given a waveform parameter set,
        rolls many dice in parallel, up to a accumulative limit; 
        sampling random phase sets on each roll.
        
        Returns the top score & the top scoring set.
    """
    T, sep, c, rolls = params
    power = power_func(T, sep, c)

    
    #### Parallel ####
    if rolls > max_rolls*cpus:
        num_procs = rolls // max_rolls
        part = max_rolls
        rem = rolls % max_rolls
    else:
        num_procs = cpus
        part = rolls // cpus
        rem = rolls % cpus

    scores = Queue(cpus)
    gtr = np.random.default_rng()
    rand_state = gtr.bit_generator.jumped()

    args = (scores, part + rem, gtr)  # Handles the remainder on first run

    ## Startup the initial batch of Processes ##
    for i in range(cpus):
        Process(target=power, args=args).start()
        args = (scores, part, Generator(rand_state))
        rand_state = rand_state.jumped()

    top = 0
    ideal = None
    score_data = []
    for p in range(num_procs, 0, -1):
        phase_set, score_set = scores.get()
        set_top = score_set.max()

        if set_top > top:
            top = set_top
            ideal = phase_set

        score_data.append(score_set)

        if p > cpus:
            Process(target=power, args=args).start()
            args = (scores, part, Generator(rand_state))
            rand_state = rand_state.jumped()

    return ideal, np.concatenate(score_data)
    

def find_optimal(params):
    """ Given a waveform parameter set,
        rolls the dice a number of times & samples random phase sets.
        
        Returns the top score & the top scoring set.
    """
    N, sep, c, rolls = params
    power = power_func(N, sep, c)

    scores = []
    top = 0
    ideal = np.ones(N-1)
    
    #### Iterative ####
    for _ in range(rolls):
        phase_set = np.random(N-1)
        scores.append(power(phase_set))
        
        if scores[-1] > top:
            top = scores[-1]
            ideal = phases

    return top, ideal

if __name__ == '__main__':
    T = 5
    sep = 1
    c = 90
    rolls = 40000
    params = (T, sep, c, rolls)
    start = time()
    ideal_phi, data = find_optimal_fast(params)
    print("Average rate of %f.2ms per roll" % (1E3*(time() - start)/rolls))

    plt.hist(data, bins=np.linspace(data.min(), data.max(), ceil(sqrt(rolls))))
    plt.title("Total")
    plt.show(block=False)