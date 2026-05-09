import cv2
import numpy as np
import os
import mediapipe as mp
from tqdm import tqdm
import concurrent.futures
import multiprocessing

# Configuration - Update these paths if modified for GCP
VIDEO_DIR = os.path.join("datasets", "wlasl_organized")
OUTPUT_DIR = os.path.join("datasets", "processed_keypoints")

def extract_keypoints(results):
    """
    Extracts and flattens all landmarks into a single numpy array.
    If a landmark is not detected in a frame, it is padded with zeros.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])

def process_single_video(args):
    """Worker function that handles a single video extraction on a separate CPU core."""
    video_path, output_path = args
    
    # Skip if already extracted
    if os.path.exists(output_path):
        return True
        
    # Each parallel process requires its own unique MediaPipe instance
    mp_holistic = mp.solutions.holistic
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        cap = cv2.VideoCapture(video_path)
        frames_keypoints = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False # Enhances performance
            
            results = holistic.process(image)
            keypoints = extract_keypoints(results)
            frames_keypoints.append(keypoints)
            
        cap.release()
        
        # Save to disk as .npy if we captured any frames
        if len(frames_keypoints) > 0:
            frames_keypoints = np.array(frames_keypoints)
            np.save(output_path, frames_keypoints)
            
    return True

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(VIDEO_DIR):
        print(f"Please check the directory: {VIDEO_DIR}. Files not found.")
        return

    # 1. Gather all tasks (video path + desired output path) 
    tasks = []
    for class_name in os.listdir(VIDEO_DIR):
        class_path = os.path.join(VIDEO_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
            
        output_class_path = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        # Append every valid video file to our task queue
        for video_file in os.listdir(class_path):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(class_path, video_file)
                file_name_without_ext = os.path.splitext(video_file)[0]
                output_path = os.path.join(output_class_path, f"{file_name_without_ext}.npy")
                
                tasks.append((video_path, output_path))

    print(f"Found {len(tasks)} videos to process overall.")
    
    # 2. Configure Multiprocessing
    # We use all available CPU cores except 2 (leaving some resources so the OS doesn't completely freeze)
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"Starting extraction using {num_cores} parallel CPU cores...")
    
    # 3. Execute in parallel with a single, unified progress bar
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # map binds the array of tasks to our worker function
        list(tqdm(executor.map(process_single_video, tasks), total=len(tasks), desc="Processing Videos"))

    print(f"\nExtraction complete! Data saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()