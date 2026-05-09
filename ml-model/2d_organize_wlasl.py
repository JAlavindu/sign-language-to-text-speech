import os
import json
import shutil
from tqdm import tqdm

# Paths
WLASL_JSON_PATH = os.path.join("datasets", "WLASL_v0.3.json") # Ensure you downloaded this
RAW_VIDEOS_DIR = os.path.join("datasets", "wlasl_videos")     # Where your 5000 mp4s are
ORGANIZED_DIR = os.path.join("datasets", "wlasl_organized")   # Output folder

def organize_dataset():
    if not os.path.exists(WLASL_JSON_PATH):
        print(f"Error: {WLASL_JSON_PATH} not found. Please download it from the WLASL dataset.")
        return
        
    with open(WLASL_JSON_PATH, 'r') as f:
        wlasl_data = json.load(f)

    os.makedirs(ORGANIZED_DIR, exist_ok=True)
    moved_count = 0
    missing_count = 0

    # Parse JSON
    for entry in tqdm(wlasl_data, desc="Organizing Videos"):
        word = entry['gloss']
        
        # WLASL provides multiple instances (videos) for each word
        for instance in entry['instances']:
            video_id = instance['video_id']
            video_filename = f"{video_id}.mp4"
            source_path = os.path.join(RAW_VIDEOS_DIR, video_filename)
            
            # If we successfully downloaded this specific video
            if os.path.exists(source_path):
                # Create the label folder
                word_folder = os.path.join(ORGANIZED_DIR, word)
                os.makedirs(word_folder, exist_ok=True)
                
                # Move or copy the file
                dest_path = os.path.join(word_folder, video_filename)
                shutil.move(source_path, dest_path) # Use shutil.copy if you want to keep originals
                moved_count += 1
            else:
                missing_count += 1

    print(f"Successfully organized {moved_count} videos.")
    print(f"{missing_count} videos from JSON were not found locally.")

if __name__ == "__main__":
    organize_dataset()