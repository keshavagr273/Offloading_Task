import cv2
import numpy as np
from tkinter import filedialog
from tkinter import Tk
from face_detector import detect_faces
from rl_agent import HybridRLAgent
from environment import VideoSurveillanceEnv
from fog_node import process_on_fog
from utils import draw_boxes

# Setup environment and agent
env = VideoSurveillanceEnv(camera=None, fog_node_capacity=5, network_bandwidth=5)
agent = HybridRLAgent(n_states=4, n_actions=2)

def upload_image():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Failed to read the image.")
        return

    faces = detect_faces(image)
    print(f"👥 Faces detected: {len(faces)}")  # Log face count

    image = draw_boxes(image, faces)

    if len(faces) == 0:
        print("😶 No face detected — skipping decision.")
    else:
        # Get state and make decision
        cpu_usage, network_condition = env.get_state()
        state = cpu_usage
        action = agent.choose_action(state)

        if action == 1:
            print("🌐 Offloading to fog node for further processing (face detected).")
            process_on_fog(image)
        else:
            print("🔍 Processing locally (face detected).")
            print("😃 Human detected.")

        # Train the agent
        next_state, reward, _ = env.step(action)
        agent.learn(state, action, reward, next_state[0])

        print(f"⭐ Reward received: {reward}")

    # Show result
    cv2.imshow("Processed Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    agent.save()

if __name__ == "__main__":
    image_path = upload_image()
    if image_path:
        process_image(image_path)
    else:
        print("⚠️ No image selected.")
