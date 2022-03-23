import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from bcm import BCMSynapse
from itertools import cycle, islice


def periodic(freq, n_seconds, dt, stim_strength):
    size = int(n_seconds / dt)  # number of timesteps in seconds
    a = np.zeros((size, 1))
    idxs = list(range(0, size, round((1.0 / freq) / dt)))
    a[idxs, :] = stim_strength
    # print(a.shape)
    # print(a.flatten().tolist())
    return a


if __name__ == "__main__":
    # Testing initial implementation
    n_presynaptic = 1

    n_seconds = 10  # length of simulation in seconds
    dt = 0.01  # duration of each timestep in seconds
    stim_strength = 1.0  # strength of stimulus

    freqs_to_try = [
        0.001,
        0.01,
        0.1,
        0.2,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        5.0,
        10.0,
        15.0,
        20.0,
        30.0,
    ]
    weight_change_pct = []
    weight_change_pct2 = []

    for freq in freqs_to_try:

        input = periodic(freq, n_seconds, dt, stim_strength)
        syn = BCMSynapse(epsilon=0.02, y_0=0.5, window_len=100)
        w, y, theta = syn.forward(input, n_seconds)
        output_weights = w.squeeze()
        weight_change_pct.append(output_weights[-1] / output_weights[0])

        # input = periodic(freq, n_seconds, dt, stim_strength * 2)
        # syn = BCMSynapse(epsilon=0.02, y_0=0.5, window_len=100)
        # w, y, theta = syn.forward(input, n_seconds)
        # output_weights = w.squeeze()
        # weight_change_pct2.append(output_weights[-1] / output_weights[0])

    print(weight_change_pct, weight_change_pct2)

    plt.clf()
    ax = plt.gca()
    # time = np.cumsum(np.full_like(input.flatten(), dt))
    # plot = sns.lineplot(x=time, y=w.squeeze(), ax=ax)

    ax.set(xscale="log")

    plot = sns.regplot(x=freqs_to_try, y=weight_change_pct, ax=ax, order=3)
    # plot = sns.regplot(x=freqs_to_try, y=weight_change_pct2, ax=ax, order=3)
    # plot.set_xlabel("time (s)")
    # plot.set_ylabel("w (a.u)")
    plot.set_xlabel("stimulation frequency")
    plot.set_ylabel("weight change factor")

    plot.set_xlim([freqs_to_try[0], freqs_to_try[-1]])
    plt.savefig("images/single_synapse_freq_vs_weightchange", dpi=300)
