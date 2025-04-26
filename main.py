# import cv2
# import psutil
# import random
# from face_detector import detect_faces
# from rl_agent import RLAgent
# from fog_node import process_on_fog
# from utils import draw_boxes
# from environment import VideoSurveillanceEnv
#
# agent = RLAgent(n_states=4, n_actions=2)  # States: CPU levels (0-3), Actions: local(0) or offload(1)
# env = VideoSurveillanceEnv(camera=None, fog_node_capacity=100, network_bandwidth=100)
#
# def get_current_state():
#     cpu_usage = psutil.cpu_percent(interval=0.1)
#     if cpu_usage < 25:
#         return 0
#     elif cpu_usage < 50:
#         return 1
#     elif cpu_usage < 75:
#         return 2
#     else:
#         return 3
#
# def start_video_stream():
#     cap = cv2.VideoCapture(0)
#
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             faces = detect_faces(frame)
#             frame = draw_boxes(frame, faces)
#
#             if len(faces) == 0:
#                 print("😶 No face detected — skipping decision.")
#             else:
#                 state = get_current_state()
#                 action = agent.choose_action(state)
#
#                 if action == 0:
#                     print("🔍 Processing locally (face detected).")
#                     reward = 1
#                 else:
#                     print("🌐 Offloading to fog node for further processing (face detected).")
#                     process_on_fog(frame)
#                     reward = 2  # Slightly better reward for offloading
#
#                 next_state = get_current_state()
#                 agent.learn(state, action, reward, next_state)
#
#             cv2.imshow("Smart Surveillance", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         agent.save_q_table()
#
# if __name__ == "__main__":
#     start_video_stream()

import cv2
import psutil
import numpy as np
from face_detector import detect_faces
from rl_agent import HybridRLAgent
from fog_node import process_on_fog
from utils import draw_boxes
from environment import VideoSurveillanceEnv
import matplotlib.pyplot as plt

# Initialize environment and agent
camera = None
fog_node_capacity = 5
network_bandwidth = 5

env = VideoSurveillanceEnv(camera, fog_node_capacity, network_bandwidth)
agent = HybridRLAgent(n_states=4, n_actions=2)

# Counters for actions
local_count = 0
offload_count = 0

# List to store rewards for plotting
rewards = []

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
            cpu_usage, network_condition = env.get_state()
            state = cpu_usage

            action = agent.choose_action(state)

            if action == 1:
                print("🌐 Offloading to fog node for further processing (face detected).")
                process_on_fog(frame)
                offload_count += 1
            else:
                print("🔍 Processing locally (face detected).")
                print("😃 Human detected.")
                local_count += 1

            new_state, reward, _ = env.step(action)
            agent.learn(state, action, reward, new_state[0])

            rewards.append(reward)

        # Show counts on frame
        text = f"Local: {local_count} | Offload: {offload_count}"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Smart Surveillance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    agent.save()
    cap.release()
    cv2.destroyAllWindows()

    # Plotting rewards automatically
    plot_rewards(rewards)

def plot_rewards(rewards):
    # Create moving average
    window = 50
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Reward per step', color='lightblue')
    plt.plot(range(window-1, len(rewards)), moving_avg, label=f'Moving Average (window={window})', color='blue')
    plt.xlabel('Decision Steps')
    plt.ylabel('Reward')
    plt.title('Hybrid RL Agent Learning Curve')
    plt.legend()
    plt.grid(True)

    # Save the plot automatically
    plt.savefig('reward_plot.png')
    print("✅ Reward plot saved as 'reward_plot.png' successfully.")

    # Also show the plot
    plt.show()

if __name__ == "__main__":
    start_video_stream()
