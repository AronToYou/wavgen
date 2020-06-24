from waveform import *


def optimize_phases(freqs, rolls):
    """ Returns the P(phi) function,
        given a waveform parameter configuration.

        Parameters
        ----------
        freqs : list of int
            List of pure tones composing the Superposition.
        rolls : int
            Number of random phase sets to sample.

        Returns
        -------
        list of float, float
            First is a list of optimal relative phases.
            Second is the score, or mean square value, the optimal set received.
    """
    N = 10000
    frac = N / SAMP_FREQ
    samples = np.linspace(0, 2 * pi * frac, N)  # Enough for 1 period (assuming integer c).

    waves = np.outer(samples, freqs[1:])
    wave = np.sin(np.multiply(freqs[0], samples))  # The first wave has no relative phase.

    gtr = np.random.default_rng()
    top, score = 0, 0
    ideal = None
    for _ in tqdm(range(rolls)):
        phases = gtr.uniform(high=2 * pi, size=len(freqs) - 1)

        form = np.add(wave, np.sin(np.add(waves, phases)).sum(axis=1))  # Un-Normalized sum of waves
        form = np.divide(form, max(form.max(), abs(form.min())))        # Normalized for Max Peak

        score = form.dot(form) / len(form)  # Proportional to Power (Vrms^2)
        if score > top:
            ideal = phases
            top = score

    ideal = ideal.tolist()
    ideal.insert(0, 0)
    return ideal, score


if __name__ == '__main__':
    ntraps = 5
    f_min = 75E6
    f_max = 105E6

    runs = 300
    data = np.empty((5, runs))

    for i in range(runs):
        print('Run: ', i+1)
        freqs_A = [f_min + (f_max - f_min) * np.random.random() for i in range(ntraps)]
        freqs_B = [f_min + (f_max - f_min) * np.random.random() for i in range(ntraps)]

        phi_A, score_A = optimize_phases(freqs_A, 1000)
        phi_B, score_B = optimize_phases(freqs_B, 1000)

        A = Superposition(freqs_A, phases=phi_A)
        B = Superposition(freqs_B, phases=phi_B)

        AB = Sweep(A, B, sample_length=int(16E5))

        data[0][i] = A.rms2()
        data[1][i] = score_A
        data[2][i] = B.rms2()
        data[3][i] = score_B
        data[4][i] = AB.rms2()

    difa = np.multiply(np.divide(np.subtract(data[0][:], data[1][:]), np.add(data[0][:], data[1][:])), 2)
    difb = np.multiply(np.divide(np.subtract(data[2][:], data[3][:]), np.add(data[2][:], data[3][:])), 2)

    ave = np.multiply(np.add(data[0][:], data[2][:]), 0.5)
    difc = np.multiply(np.divide(np.subtract(data[4][:], ave), np.add(data[4][:], ave)), 2)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,15))

    ax1.hist(np.concatenate([difa, difb]))
    ax1.set_title('Scores Errors')

    ax2.hist(data[:][:4].flatten())
    ax2.set_title('Stable Scores')

    ax3.hist(data[:][4])
    ax3.set_title('Sweep Scores')

    ax4.hist(difc)
    ax4.set_title('Sweep Reduction')
    plt.show()

    with h5py.File('sweep_scores', 'a') as f:
        i = 0
        while f.get('g' + str(i)):
            i += 1
        f = f.create_group('g' + str(i))
        dset = f.create_dataset('data', shape=(5, runs), dtype=float, data=data)
        dset.dims[0].label = 'rms2_a'
        dset.dims[1].label = 'score_a'
        dset.dims[2].label = 'rms2_b'
        dset.dims[3].label = 'score_b'
        dset.dims[4].label = 'rms2_ab'
