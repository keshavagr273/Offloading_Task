import cv2
import numpy as np
import time
import os

def process_on_fog(frame, mode="enhance"):
    start = time.time()
    print(f"📡 Processing frame on fog node (mode={mode})...")

    if mode == "enhance":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        print(" Fog enhancement complete.")

    elif mode == "compress":
        resized = cv2.resize(frame, (320, 240))
        _, processed = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
        print("✅ Compression complete, ready for cloud upload.")

    else:
        raise ValueError("Invalid mode. Use 'enhance' or 'compress'.")

    print(f"⏱️ Processing time: {time.time() - start:.2f}s")
    return processed

def simulate_cloud_upload(data, filename):
    time.sleep(1)
    size = len(data) if isinstance(data, bytes) else data.size
    print(f"☁️ Uploading {filename} ({size} bytes)... Done.")

if __name__ == "__main__":
    img_path = "test_foggy.jpg"
    if not os.path.exists(img_path):
        print("❌ Image not found. Please place test_foggy.jpg in this folder.")
    else:
        frame = cv2.imread(img_path)
        enhanced = process_on_fog(frame, "enhance")
        cv2.imwrite("enhanced_output.jpg", enhanced)
        simulate_cloud_upload(enhanced, "enhanced_output.jpg")
        compressed = process_on_fog(frame, "compress")
        with open("compressed_output.jpg", "wb") as f:
            f.write(compressed)
        simulate_cloud_upload(compressed, "compressed_output.jpg")
        print("🎉 All tasks done successfully.")
