# import numpy as np
# import random
# import os
#
# class RLAgent:
#     def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1, q_table_file='q_table.npy'):
#         self.n_states = n_states
#         self.n_actions = n_actions
#         self.alpha = alpha    # Learning rate
#         self.gamma = gamma    # Discount factor
#         self.epsilon = epsilon  # Exploration rate
#         self.q_table_file = q_table_file
#
#         if os.path.exists(self.q_table_file):
#             self.q_table = np.load(self.q_table_file)
#             print("✅ Q-table loaded from file.")
#         else:
#             self.q_table = np.zeros((n_states, n_actions))
#             print("📄 New Q-table initialized.")
#
#     def choose_action(self, state):
#         if random.uniform(0, 1) < self.epsilon:
#             return random.randint(0, self.n_actions - 1)  # Explore
#         else:
#             return np.argmax(self.q_table[state])  # Exploit
#
#     def learn(self, state, action, reward, next_state):
#         predict = self.q_table[state, action]
#         target = reward + self.gamma * np.max(self.q_table[next_state])
#         self.q_table[state, action] += self.alpha * (target - predict)
#
#     def save_q_table(self):
#         np.save(self.q_table_file, self.q_table)
#         print("💾 Q-table saved.")

import numpy as np

class HybridRLAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, gamma=0.95):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.q_table = np.zeros((n_states, n_actions))
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def choose_action(self, state):
        # Softmax exploration based on Q-values
        q_values = self.q_table[state]
        exp_q = np.exp(q_values - np.max(q_values))  # subtract max for numerical stability
        probs = exp_q / np.sum(exp_q)

        if np.random.rand() < self.epsilon:
            # Random action (exploration)
            return np.random.choice(self.n_actions)
        else:
            # Softmax sampling
            return np.random.choice(self.n_actions, p=probs)

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (target - predict)

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename='q_table.npy'):
        np.save(filename, self.q_table)

    def load(self, filename='q_table.npy'):
        self.q_table = np.load(filename)


