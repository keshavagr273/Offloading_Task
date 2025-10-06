import cv2
import numpy as np

def process_on_fog(frame):
    """
    Simulates fog node image processing by enhancing image contrast
    and reducing haze using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    print("📡 Processing frame on fog node...")

    # Convert to LAB color space for better lightness adjustment
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to improve contrast in foggy images
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge back the LAB channels
    limg = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    print("✅ Fog node processing complete.")
    return enhanced_frame
