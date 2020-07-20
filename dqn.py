from absl import app
from absl import flags
from absl import logging


import haiku as hk
import jax
import jax.numpy as np

BATCH_SIZE = 32


class DQN(hk.Module):
    def __init__(self, hidden_size, n_actions):
        super(DQN, self).__init__()
        self.hidden_size = hidden_size
        self.n_actions = n_actions

    def __call__(self, x):
        # cnn hyperparameters are described in the METHODS of
        # Mnih et al.. Human-level control through deep reinforcement learning, 2015
        # https://www.nature.com/articles/nature14236
        cnn = hk.Sequential([
            hk.Conv2D(32, 8, 4),
            jax.nn.relu,
            hk.Conv2D(64, 4, 2),
            jax.nn.relu,
            hk.Conv2D(64, 3, 1),
            jax.nn.relu,
            hk.Linear(self.hidden_size),
            jax.nn.relu,
            hk.Linear(self.n_actions)
            ])
        return cnn(x)


def forward(model, params, x, y):
    y_hat = model.apply(params, x)
    return y_hat


def backward(params, loss):
    return jax.grad(forward)(loss)


def get_loss(params, y_hat, y):
    raise NotImplementedError


def rmsprop_step(params, grads):
    raise NotImplementedError


def training_step(model, x):
    raise NotImplementedError


def main(_):
    print(_)
    model = hk.transform(DQN)
    print(model)

    params = model.init(None, None, np.ones((84, 84, 4)))
    print(params)

    y = model.apply(params, None, np.ones((84, 84, 4)))
    return y

if __name__ == "__main__":
    # RL and DQN training hyperparameters are described in the
    # Extended Data Table 1 | List of Hyperparameters and their values of
    # Mnih et al.. Human-level control through deep reinforcement learning, 2015
    # https://www.nature.com/articles/nature14236
    flags.DEFINE_integer("batch_size", 32, "Number of training cases over which each stochastic gradient descent (SGD) update is computed.")
    flags.DEFINE_integer("replay_memory_size", 1000000, "SGD updates are sampled from this number of most recent frames.")
    flags.DEFINE_integer("agent_history_len", 4, "The number of most recent frames experienced by the agent that are given as input to the Q network.")
    flags.DEFINE_integer("target_network_upate_freq", 10000, "The frequency (measured in number of parameters updates) with which the target network is updated.")
    flags.DEFINE_float("discount", 0.99, "Discount factor gamma used in the Q-Learning update.")
    flags.DEFINE_integer("action_repeat", 4, "Repeat each action selected by the agent this many times." +
                         "Using a value of 4 results in the agent seeing only every 4th input frame.")
    flags.DEFINE_integer("update_frequency", 4, "The number of actions selected by the agent between successive SGD updates." +
                         "Using a value of 4 results in the agent selecting 4 actions between each pair of successive updates.")
    flags.DEFINE_float("learning_rate", 0.00025, "The learning rate used by the RMSProp.")
    flags.DEFINE_float("gradient_momentum", 0.95, "Gradient momentum used by the RMSProp")
    flags.DEFINE_float("squared_gradient_momentum", 0.95, "Squared gradient (denominator) momentum used by the RMSProp")
    flags.DEFINE_float("min_squared_gradient", 0.01, "Constant added to the squared gradient in the denominator of the RMSProp update.")
    flags.DEFINE_float("initial_exploration", 1., "Initial value of ε in ε-greedy exploration.")
    flags.DEFINE_float("final_exploration", 0.1, "Final value of ε in ε-greedy exploration.")
    flags.DEFINE_integer("final_exploration_frame", 1000000, "The number of frames over which the initial value of ε is linearly annealed to its final value.")
    flags.DEFINE_integer("replay_start_size", 50000, "A uniform random policy is run for this number of frames before learning starts" +
                         "and the resulting experience is used to populate the replay memory.")
    flags.DEFINE_integer("no_op_max", 30, "Maximum number of 'do nothing' actions to be performed by the agent at the start of an episode.")

    app.run(main)
