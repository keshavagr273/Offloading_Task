import numpy as np

class RLAgent:
    def __init__(self, n_states, n_actions):
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        return np.argmax(self.q_table[state])
