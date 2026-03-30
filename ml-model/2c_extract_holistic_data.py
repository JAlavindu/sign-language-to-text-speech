import cv2
import numpy as np
import os
import mediapipe as mp
from tqdm import tqdm

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic

# Configuration - Update these paths once you download the WLASL dataset
VIDEO_DIR = os.path.join("datasets", "wlasl_videos")
OUTPUT_DIR = os.path.join("datasets", "processed_keypoints")

def extract_keypoints(results):
    """
    Extracts and flattens all landmarks into a single numpy array.
    If a landmark is not detected in a frame, it is padded with zeros.
    """
    # Pose: 33 landmarks, 4 values (x, y, z, visibility) -> 132 values
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Face: 468 landmarks, 3 values (x, y, z) -> 1404 values
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    # Left Hand: 21 landmarks, 3 values (x, y, z) -> 63 values
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # Right Hand: 21 landmarks, 3 values (x, y, z) -> 63 values
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Total combined length per frame = 132 + 1404 + 63 + 63 = 1662 values
    return np.concatenate([pose, face, lh, rh])

def process_video(video_path, output_path, holistic_model):
    """Reads a video and saves extracted frame keypoints as a .npy file."""
    cap = cv2.VideoCapture(video_path)
    frames_keypoints = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Recolor feed to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # Enhances performance
        
        # Make Detections
        results = holistic_model.process(image)
        
        # Extract and store coordinates
        keypoints = extract_keypoints(results)
        frames_keypoints.append(keypoints)
        
    cap.release()
    
    # Convert list of frames into a 2D numpy array (Frames x 1662)
    frames_keypoints = np.array(frames_keypoints)
    
    # Save to disk
    np.save(output_path, frames_keypoints)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(VIDEO_DIR):
        print(f"Please create the directory: {VIDEO_DIR} and add your MP4 files.")
        return

    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    print(f"Found {len(video_files)} videos to process.")
    
    # Set up the MediaPipe Model context
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for video_file in tqdm(video_files, desc="Extracting Keypoints"):
            video_path = os.path.join(VIDEO_DIR, video_file)
            
            # Create matching .npy filename
            file_name_without_ext = os.path.splitext(video_file)[0]
            output_path = os.path.join(OUTPUT_DIR, f"{file_name_without_ext}.npy")
            
            # Skip if already processed (allows resuming interrupted extractions)
            if not os.path.exists(output_path):
                process_video(video_path, output_path, holistic)

    print(f"Extraction complete! Data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()