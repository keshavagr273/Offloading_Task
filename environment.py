# import random
#
# class VideoSurveillanceEnv:
#     def __init__(self, camera, fog_node_capacity, network_bandwidth):
#         self.camera = camera  # Camera capture stream
#         self.fog_node_capacity = fog_node_capacity  # Fog node processing power
#         self.network_bandwidth = network_bandwidth  # Available network bandwidth
#
#     def get_state(self):
#         # Get current state based on system parameters
#         cpu_usage = self.get_cpu_usage()
#         network_condition = self.get_network_condition()
#         return (cpu_usage, network_condition)
#
#     def get_cpu_usage(self):
#         # Simulate CPU usage (e.g., based on number of tasks)
#         return random.choice([0, 1, 2, 3])  # Simplified
#
#     def get_network_condition(self):
#         # Simulate network conditions (e.g., bandwidth availability)
#         return random.choice([0, 1, 2])  # Simplified: 0 = low, 1 = medium, 2 = high
#
#     def step(self, action):
#         # Perform the action (offload or not) and return new state and reward
#         reward = 0
#         if action == 0:  # Process locally
#             reward = self.process_locally()
#         else:  # Offload to fog node
#             reward = self.offload_to_fog_node()
#         return self.get_state(), reward, False  # Return new state, reward, and done flag
#
#     def process_locally(self):
#         # Simulate local processing (e.g., face detection, CPU cost)
#         return random.choice([1, 2])  # Simplified reward function
#
#     def offload_to_fog_node(self):
#         # Simulate offloading to the fog node (e.g., reduced latency, but energy cost)
#         return random.choice([3, 4])  # Higher reward for offloading if bandwidth and CPU allow


import random

class VideoSurveillanceEnv:
    def __init__(self, camera, fog_node_capacity, network_bandwidth):
        self.camera = camera  # Currently unused
        self.fog_node_capacity = fog_node_capacity  # Processing power of fog node
        self.network_bandwidth = network_bandwidth  # Available network bandwidth

    def get_state(self):
        """
        Return current CPU usage and network condition.
        State is simplified to small discrete values.
        """
        cpu_usage = self.get_cpu_usage()
        network_condition = self.get_network_condition()
        return (cpu_usage, network_condition)

    def get_cpu_usage(self):
        """
        Simulate CPU usage.
        0: Low CPU usage
        1: Medium CPU usage
        2: High CPU usage
        3: Very High CPU usage
        """
        return random.choice([0, 1, 2, 3])

    def get_network_condition(self):
        """
        Simulate network conditions.
        0: Poor Network
        1: Medium Network
        2: Good Network
        """
        return random.choice([0, 1, 2])

    def step(self, action):
        """
        Take an action and return next_state, reward, done flag.
        action = 0 (process locally), 1 (offload to fog)
        """

        reward = 0

        if action == 0:
            reward = self.process_locally()
        else:
            reward = self.offload_to_fog_node()

        next_state = self.get_state()
        done = False  # Streaming is continuous, so never really 'done'

        return next_state, reward, done

    def process_locally(self):
        """
        Simulate local processing.
        Reward based on CPU usage penalty:
        - Lower CPU usage = better for local processing.
        """
        cpu_usage = self.get_cpu_usage()

        if cpu_usage == 0:
            return 3  # Very efficient
        elif cpu_usage == 1:
            return 2  # Acceptable
        elif cpu_usage == 2:
            return 0  # Risky
        else:
            return -2  # Overloaded, bad to process locally

    def offload_to_fog_node(self):
        """
        Simulate offloading to fog node.
        Reward based on network condition and fog capacity.
        """
        network_condition = self.get_network_condition()

        if network_condition == 2 and self.fog_node_capacity >= 5:
            return 4  # Excellent offloading
        elif network_condition == 1:
            return 2  # Okay offloading
        elif network_condition == 0:
            return -1  # Bad offloading (network poor)

        return 1  # Default small reward

