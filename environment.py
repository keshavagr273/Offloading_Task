import random

class VideoSurveillanceEnv:
    def __init__(self, camera, fog_node_capacity, network_bandwidth):
        self.camera = camera  # Camera capture stream
        self.fog_node_capacity = fog_node_capacity  # Fog node processing power
        self.network_bandwidth = network_bandwidth  # Available network bandwidth

    def get_state(self):
        # Get current state based on system parameters
        cpu_usage = self.get_cpu_usage()
        network_condition = self.get_network_condition()
        return (cpu_usage, network_condition)

    def get_cpu_usage(self):
        # Simulate CPU usage (e.g., based on number of tasks)
        return random.choice([0, 1, 2, 3])  # Simplified

    def get_network_condition(self):
        # Simulate network conditions (e.g., bandwidth availability)
        return random.choice([0, 1, 2])  # Simplified: 0 = low, 1 = medium, 2 = high

    def step(self, action):
        # Perform the action (offload or not) and return new state and reward
        reward = 0
        if action == 0:  # Process locally
            reward = self.process_locally()
        else:  # Offload to fog node
            reward = self.offload_to_fog_node()
        return self.get_state(), reward, False  # Return new state, reward, and done flag

    def process_locally(self):
        # Simulate local processing (e.g., face detection, CPU cost)
        return random.choice([1, 2])  # Simplified reward function

    def offload_to_fog_node(self):
        # Simulate offloading to the fog node (e.g., reduced latency, but energy cost)
        return random.choice([3, 4])  # Higher reward for offloading if bandwidth and CPU allow
