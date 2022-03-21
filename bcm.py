import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class BCMSynapse(object):
    def __init__(
        self, 
        epsilon,
        y_0, 
        window_len, 
    ):    
        """Constructor

        Args:
            epsilon (float): Weight decay term
            y_baseline (float): Baseline postsynaptic activity 
            window_len (int): Length (in timesteps) of temporal window over which 
                average postsynaptic activity is calculated
        """

        self._epsilon = epsilon
        self._y_0 = y_0
        self._window_len = window_len

    def _step(self, input, w, y, theta):
        """Single step of weight updates and output calculation/

        Args:
            input (np.array): Vector of presynaptic neur. activities
            w (np.array): Weight vector on previous timestep
            y (np.array): History of output
            theta (float): Current threshold 
        Returns:
            new_w (np.array): Updated weight vector
            new_y (float): Postsynaptic neuron output
            new_theta (float): Updated threshold
        """
        new_y = np.dot(w, input) 
        dot_w = new_y * (new_y - theta) * input - self._epsilon * w
        new_w = w + dot_w
        new_theta = np.mean(y[-self._window_len:]/self._y_0)
        return new_w, new_y, new_theta


    def _init_w(self, n_presynaptic):
        """Initializes weight vector, sampling from U[0, 1]

        Args:
            n_presynaptic (int): Number of presynaptic neurons

        Returns:
            np.array: Weight vector 
        """
        return np.random.uniform(size=(n_presynaptic))

    def _init_theta(self):
        """Returns initial value of theta
        Returns:
            (float): Theta value
        """
        return 0.5 

    def forward(self, input):
        """Forward pass through input sequence . 

        Args:
            input (np.array): Matrix of presynaptic neuron activations
        Returns:
            w (np.array): Matrix of synaptic weights over time
            y (np.array): Vector of output
            theta (np.array): Vector of thresholds
        """
        n_timesteps = input.shape[0]
        n_presynaptic = input.shape[1]
        w = np.zeros((n_timesteps, n_presynaptic))
        y = np.zeros((n_timesteps))
        theta = np.zeros((n_timesteps))
        theta[0] = self._init_theta()
        w[0] = self._init_w(n_presynaptic)
        for t in range(1, n_timesteps):
            step_input = input[t]
            step_w, step_y, step_theta = self._step(
                input=step_input, 
                w=w[t-1], 
                y=y[:t], 
                theta=theta[t-1]
            )
            w[t] = step_w
            y[t] = step_y
            theta[t] = step_theta
        return w, y, theta



if __name__ == "__main__":
    # Testing initial implementation
    n_presynaptic = 1
    n_seconds = 5 # length of simulation in seconds
    dt = 0.001 # duration of each timestep in seconds
    n_timesteps = int(n_seconds/dt) # number of timesteps in seconds
    stim_start = int(1/dt) # start of stimulation, in timesteps
    stim_end = int(4/dt) # end of stimulation, in timesteps
    stim_strength = 10  # strength of stimulus
    input = np.zeros((n_timesteps, 1))
    input[stim_start:stim_end] = stim_strength
    syn = BCMSynapse(epsilon=0.01, y_0=0.0001, window_len=1000)
    w, y, theta = syn.forward(input)

    plt.clf()
    ax = plt.gca()
    time = np.cumsum(np.full_like(input, dt)) 
    plot = sns.lineplot(x=time, y=w.squeeze(), ax=ax)
    plot.set_xlabel('time (s)')
    plot.set_ylabel('w (a.u)')
    plt.savefig('images/single_synapse_w', dpi=300)

    plt.clf()
    ax = plt.gca()
    time = np.cumsum(np.full_like(input, dt)) 
    plot = sns.lineplot(x=time, y=input.squeeze(), ax=ax)
    plot.set_xlabel('time (s)')
    plot.set_ylabel('input (spikes/s)')
    plt.savefig('images/single_synapse_input', dpi=300)

    plt.clf()
    ax = plt.gca()
    plot = sns.lineplot(x=time, y=y, ax=ax)
    plot.set_xlabel('time (s)')
    plot.set_ylabel('output (spikes/s)')
    plt.savefig('images/single_synapse_y', dpi=300)

    plt.clf()
    ax = plt.gca()
    plot = sns.lineplot(x=time, y=theta, ax=ax)
    plot.set_xlabel('time (s)')
    plot.set_ylabel('$\\theta$ (spikes/s)')
    plt.savefig('images/single_synapse_theta', dpi=300)





