import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
rc('text', usetex=False)

means = []
sep_set = [0.1, 0.5, 1.0]
seps = [[] for _ in range(3)]

ntraps = np.arange(3, 20)
separations = [1, 0.5, 0.1]  # MHz
centers = [80, 85, 90, 95, 100]  # Mhz
rolls = 120000

with h5py.File('scores.hdf5', 'r') as f:
    T_bins = [[] for _ in range(len(ntraps))]

    for i, T in enumerate(ntraps):
        for j, sep in enumerate(separations):
            for k, c in enumerate(centers):
                name = '%d/%.1f/%.1f' % (T, sep, c)
                dat = f[name][()]

                T_bins[i].append(dat)

    for T in f.keys():
        seps = f[T]
        for i, sep in enumerate(seps.keys()):
            cs = seps[sep]
            for c in cs.keys():
                dat = cs[c]

                means.append(dat[()])

total = np.concatenate(means)

# plt.hist(np.concatenate(means), bins=500)
# plt.show()

# for i, sep in enumerate(seps):
    # plt.hist(np.concatenate(sep), bins=500)
    # plt.title('%.1f' % sep_set[i])
    # plt.show()

# for k, c in enumerate(c_bins):
#     plt.hist(np.concatenate(c), bins=500)
#     plt.title('%d' % centers[k])
#     plt.show()

high, low = total.max(), total.min()
hists = np.empty((len(T_bins), 500))
bins = None

top_bin = 0
for i in range(len(T_bins)):
    T_bins[i] = np.concatenate(T_bins[i])
    hists[i], bins = np.histogram(T_bins[i], range=(low, high), bins=500, density=True)
    top = hists[i].max()
    if top > top_bin:
        top_bin = top

# plt.hist(T_bins, range=(low, high), bins=500, density=True, histtype='stepfilled')
# plt.title('Probability Density of $V_{RMS}^2 / V_{peak}$')
# plt.show()

fig = plt.figure()
fig.suptitle('Probabiliy Density of $V_{RMS}^2 / V_{peak}$')
ax1 = fig.add_subplot(1, 1, 1)

## Animation Frame ##
def animate(i):
    ax1.clear()
    ax1.plot(bins[1:], hists[i])
    plt.ylim((0, top_bin))
    # ax1.hist(T_bins[i], range=(low, high), bins=500, density=True)
    plt.title('%d traps' % ntraps[i])

# writ = animation.ImageMagickFileWriter(2)
# writ.setup(fig, 'hist_animation.gif')
# for i in range(len(T_bins)):
#     animate(i)
#     writ.grab_frame()
# writ.finish()

ani = animation.FuncAnimation(fig, animate, frames=np.arange(len(T_bins)), interval=500)
plt.show()
plt.close(fig)
