import cv2
import psutil
from face_detector import detect_faces
from rl_agent import RLAgent
from fog_node import process_on_fog
from utils import draw_boxes

agent = RLAgent(n_states=2, n_actions=2)  # Still required for internal learning (optional)

def should_offload(num_faces):
    cpu = psutil.cpu_percent(interval=0.1)
    if cpu > 70 or num_faces > 1:
        return True
    return False

def start_video_stream():
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
            if should_offload(len(faces)):
                print("🌐 Offloading to fog node for further processing (face detected).")
                process_on_fog(frame)
            else:
                print("🔍 Processing locally (face detected).")
                print("😃 Human detected.")

        cv2.imshow("Smart Surveillance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_video_stream()
