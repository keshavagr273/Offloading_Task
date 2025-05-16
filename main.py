import cv2
import numpy as np
from face_detector import detect_faces
from rl_agent import HybridRLAgent
from fog_node import process_on_fog
from utils import draw_boxes
from environment import VideoSurveillanceEnv
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize environment and agent
camera = None
fog_node_capacity = 5
network_bandwidth = 5

env = VideoSurveillanceEnv(camera, fog_node_capacity, network_bandwidth)
state_bins = 10  # Discretize states into 10 bins for each parameter (example)
n_actions = 2  # Example: 2 actions (local or offload)

agent = HybridRLAgent(state_bins=state_bins, n_actions=n_actions)  # Corrected instantiation

# Counters for actions
local_count = 0
offload_count = 0

# List to store rewards for plotting
rewards = []
cpu_usages = []
network_conditions = []
task_queue_lengths = []
q_table_data = []

# Real-time plotting setup
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

def update_plots(frame):
    axes[0, 0].clear()
    axes[0, 1].clear()
    axes[1, 0].clear()
    axes[1, 1].clear()

    # Plot CPU usage
    axes[0, 0].plot(cpu_usages, label="CPU Usage", color='r')
    axes[0, 0].set_title("CPU Usage")
    axes[0, 0].set_xlabel("Decision Steps")
    axes[0, 0].set_ylabel("Usage (%)")
    axes[0, 0].legend()

    # Plot network condition
    axes[0, 1].plot(network_conditions, label="Network Condition", color='g')
    axes[0, 1].set_title("Network Condition")
    axes[0, 1].set_xlabel("Decision Steps")
    axes[0, 1].set_ylabel("Condition (0-2)")
    axes[0, 1].legend()

    # Plot task queue length
    axes[1, 0].plot(task_queue_lengths, label="Task Queue Length", color='b')
    axes[1, 0].set_title("Task Queue Length")
    axes[1, 0].set_xlabel("Decision Steps")
    axes[1, 0].set_ylabel("Queue Length")
    axes[1, 0].legend()

    # Plot the Q-table (top 10 states and their action values)
    q_table_values = list(agent.get_q_table().values())[:10]
    if q_table_values:
        q_table_matrix = np.array(q_table_values)
        axes[1, 1].imshow(q_table_matrix, aspect='auto', cmap='coolwarm')
        axes[1, 1].set_title("Top 10 Q-Table States")
        axes[1, 1].set_xlabel("Actions")
        axes[1, 1].set_ylabel("States")
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(["Action 0", "Action 1"])

    plt.tight_layout()

# Initialize animation
ani = animation.FuncAnimation(fig, update_plots, interval=1000)

def start_video_stream():
    global local_count, offload_count

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)
        frame = draw_boxes(frame, faces)

        if len(faces) == 0:
            print("😶 No face detected — skipping decision.")
        else:
            # Get environment state
            cpu_usage, network_condition, task_queue_length, energy_level, task_size, computational_demand, delay_constraint, historical_reward, fog_load = env.get_state()
            state = (cpu_usage, network_condition, task_queue_length, energy_level, task_size, computational_demand, delay_constraint, historical_reward, fog_load)

            # Update tracking data for dynamic plots
            cpu_usages.append(cpu_usage)
            network_conditions.append(network_condition)
            task_queue_lengths.append(task_queue_length)

            # Agent chooses action
            action = agent.choose_action(state)

            if action == 1:
                print("🌐 Offloading to fog node for further processing (face detected).")
                process_on_fog(frame)
                offload_count += 1
            else:
                print("🔍 Processing locally (face detected).")
                print("😃 Human detected.")
                local_count += 1

            # Get new state after taking action
            new_state, reward, _ = env.step(action)
            agent.learn(state, action, reward, new_state)

            rewards.append(reward)

        # Show counts on frame
        text = f"Local: {local_count} | Offload: {offload_count}"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Smart Surveillance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    agent.save()  # Save agent's learned Q-table
    cap.release()
    cv2.destroyAllWindows()

    # Show dynamic plot of rewards and Q-table
    plt.show()

if __name__ == "__main__":
    start_video_stream()
