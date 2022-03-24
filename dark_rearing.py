import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from bcm import BCMSynapse
from itertools import cycle, islice
import pandas as pd


def periodic(freq, n_seconds, dt, stim_strength):
    size = int(n_seconds / dt)  # number of timesteps in seconds
    a = np.zeros((size, 1))
    idxs = list(range(0, size, round((1.0 / freq) / dt)))
    a[idxs, :] = stim_strength
    return a


def exp1():
    # Testing initial implementation
    n_presynaptic = 1

    n_seconds = 20  # length of simulation in seconds
    dt = 0.01  # duration of each timestep in seconds
    stim_strength = 1.0  # strength of stimulus

    freqs_to_try = [
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
        40.0,
    ]

    data = []
    data2 = []

    for _ in range(1):

        for freq in freqs_to_try:

            light_history = np.ones((round(n_seconds / dt) * 24, 1))
            input = periodic(freq, n_seconds, dt, stim_strength=2.0)

            input = np.concatenate((light_history, input))
            print(input.shape)

            syn = BCMSynapse(epsilon=0.02, y_0=1.0, window_len=5000)
            w, y, theta = syn.forward(input, n_seconds)
            output_weights = w.squeeze()

            data.append(
                {
                    "stim_freq": freq,
                    "weight_change_pct": output_weights[-1] / output_weights[0],
                    "label": "light-reared",
                }
            )

            data2.append(
                pd.DataFrame(
                    {
                        "y": y.squeeze(),
                        "w": w.squeeze(),
                        "theta": theta.squeeze(),
                        "t": np.cumsum(np.full_like(input, dt)),
                        "label": "light-reared",
                        "freq": freq,
                    }
                )
            )

            dark_history = np.zeros((round(n_seconds / dt) * 24, 1))
            input = periodic(freq, n_seconds, dt, stim_strength=2.0)

            input = np.concatenate((dark_history, input))
            print(input.shape)

            syn = BCMSynapse(epsilon=0.02, y_0=1.0, window_len=5000)
            w, y, theta = syn.forward(input, n_seconds)
            output_weights = w.squeeze()

            data.append(
                {
                    "stim_freq": freq,
                    "weight_change_pct": output_weights[-1] / output_weights[0],
                    "label": "dark-reared",
                }
            )

            data2.append(
                pd.DataFrame(
                    {
                        "y": y.squeeze(),
                        "w": w.squeeze(),
                        "theta": theta.squeeze(),
                        "t": np.cumsum(np.full_like(input, dt)),
                        "label": "dark-reared",
                        "freq": freq,
                    }
                )
            )

    data = pd.DataFrame(data)
    data.dropna(inplace=True)
    print(data)
    print(data["weight_change_pct"].describe())

    data2 = pd.concat(data2)

    plt.clf()
    plot = sns.lineplot(data=data, x="stim_freq", y="weight_change_pct", hue="label",)
    plot.set_xlabel("stimulation frequency")
    plot.set_ylabel("weight change factor")
    plot.set_xlim([freqs_to_try[0], freqs_to_try[-1]])
    plot.set_xscale("log")
    plt.savefig("images/single_synapse_freq_vs_weightchange_light_v_dark", dpi=150)

    plt.clf()
    plot = sns.lineplot(data=data2, x="t", y="y", hue="label", style="freq")
    plot.set_xlabel("time (s)")
    plot.set_ylabel("output")
    plot.set_ylim(0, 5)
    plt.savefig(f"images/output", dpi=150)


if __name__ == "__main__":
    exp1()
