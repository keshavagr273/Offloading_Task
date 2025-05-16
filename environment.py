import psutil
import random

class VideoSurveillanceEnv:
    def __init__(self, camera=None, fog_node_capacity=100, network_bandwidth=100):
        self.camera = camera
        self.fog_node_capacity = fog_node_capacity
        self.network_bandwidth = network_bandwidth
        self.previous_rewards = []  # Track historical rewards

    def get_state(self):
        cpu_usage = self.get_cpu_usage()
        network_condition = self.get_network_condition()
        task_queue_length = self.get_task_queue_length()
        energy_level = self.get_energy_level()
        task_size = self.get_task_size()
        computational_demand = self.get_computational_demand(cpu_usage)
        delay_constraint = self.get_delay_constraint()
        fog_load = self.get_fog_load()
        historical_reward = self.get_historical_reward()

        # Return the state as a tuple of 9 values
        state = (
            cpu_usage,
            network_condition,
            task_queue_length,
            energy_level,
            task_size,
            computational_demand,
            delay_constraint,
            historical_reward,
            fog_load
        )

        return state

    def step(self, action):
        """
        Simulate an environment transition based on the action.
        Returns: new_state, reward, done
        """
        # Simulated reward based on action and fog load
        fog_penalty = self.get_fog_load() / 3.0  # normalize to [0, 1]

        if action == 0:  # Process locally
            reward = 1.0 - (psutil.cpu_percent() / 100.0)
        else:  # Offload to fog
            reward = 0.8 - fog_penalty

        reward = max(0.0, min(reward, 1.0))  # clip reward to [0, 1]

        # Store reward history
        self.previous_rewards.append(reward)
        if len(self.previous_rewards) > 50:
            self.previous_rewards.pop(0)

        new_state = self.get_state()
        done = False  # No terminal state in this continuous task

        return new_state, reward, done

    def get_cpu_usage(self):
        usage = psutil.cpu_percent(interval=0.5)
        if usage < 25:
            return 0
        elif usage < 50:
            return 1
        elif usage < 75:
            return 2
        else:
            return 3

    def get_network_condition(self):
        return random.choice([0, 1, 2])

    def get_task_queue_length(self):
        return random.randint(0, 3)

    def get_energy_level(self):
        return random.choice([0, 1, 2])

    def get_task_size(self):
        size = random.uniform(5, 50)
        if size < 15:
            return 0
        elif size < 30:
            return 1
        elif size < 45:
            return 2
        else:
            return 3

    def get_computational_demand(self, cpu_usage):
        return cpu_usage  # Already discretized into 0–3

    def get_delay_constraint(self):
        return random.choice([0, 1, 2])

    def get_fog_load(self):
        return random.randint(0, 3)

    def get_historical_reward(self):
        if self.previous_rewards:
            avg_reward = sum(self.previous_rewards[-5:]) / min(5, len(self.previous_rewards))
            if avg_reward < 0.3:
                return 0
            elif avg_reward < 0.7:
                return 1
            else:
                return 2
        else:
            return 1  # Neutral default
