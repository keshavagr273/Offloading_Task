import numpy as np
import random

class HybridRLAgent:
    def __init__(self, state_bins=None, n_actions=2, learning_rate=0.1, gamma=0.95):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.state_bins = state_bins  # For discretizing continuous states, if needed
        self.q_table = {}  # Q-table stored as a dictionary to handle discrete states
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        q_values = self.q_table.get(state_key, np.zeros(self.n_actions))  # Default to 0 if state is unseen

        # Softmax exploration based on Q-values
        exp_q = np.exp(q_values - np.max(q_values))  # subtract max for numerical stability
        probs = exp_q / np.sum(exp_q)

        if np.random.rand() < self.epsilon:
            # Random action (exploration)
            return np.random.choice(self.n_actions)
        else:
            # Softmax sampling
            return np.random.choice(self.n_actions, p=probs)

    def learn(self, state, action, reward, new_state):
        state_key = self.get_state_key(state)
        new_state_key = self.get_state_key(new_state)

        # Get the current Q-value
        current_q = self.q_table.get(state_key, np.zeros(self.n_actions))[action]

        # Max Q-value for the next state (for the Bellman equation)
        next_max_q = np.max(self.q_table.get(new_state_key, np.zeros(self.n_actions)))

        # Update Q-value using the Bellman equation
        new_q = current_q + self.learning_rate * (reward + self.gamma * next_max_q - current_q)

        # Update Q-table
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        self.q_table[state_key][action] = new_q

        # Decay epsilon (exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_state_key(self, state):
        """
        Converts a state (which might be continuous) to a discrete key for the Q-table.
        Uses state_bins to discretize the state.
        """
        if self.state_bins:
            state_key = tuple([min(int(s * self.state_bins), self.state_bins - 1) for s in state])
        else:
            state_key = str(tuple(state))  # For simple states, use tuple as key
        return state_key

    def save(self):
        # You can implement this method to save the Q-table to a file if needed
        pass

    def get_q_table(self):
        return self.q_table  # Returns the current Q-table for visualization
