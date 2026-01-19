import os
import cv2
import numpy as np

# Constants
VIDEO_DIR = r'C:\Python\IDP\files'  # Use raw string (r'') to avoid escape issues
FRAME_OUTPUT = './video_frames'     # Where extracted frames will be stored
TARGET_FRAMES = 100                 # <-- You can set your desired uniform number of frames

# Create output folder if it doesn't exist
os.makedirs(FRAME_OUTPUT, exist_ok=True)

def extract_frames_uniform(video_path, output_folder, target_frames=100):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"⚠️ Skipping {video_path}: no frames found.")
        cap.release()
        return

    frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=np.int32)

    count = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frame_path = os.path.join(output_folder, f'frame_{count:04}.jpg')
            cv2.imwrite(frame_path, frame)
            count += 1
        else:
            print(f"⚠️ Failed at frame {idx} for {video_path}")

    cap.release()

for category in os.listdir(VIDEO_DIR):
    category_path = os.path.join(VIDEO_DIR, category)
    if not os.path.isdir(category_path):
        continue

    output_category = os.path.join(FRAME_OUTPUT, category)
    os.makedirs(output_category, exist_ok=True)

    for file in os.listdir(category_path):
        if file.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(category_path, file)
            video_name = os.path.splitext(file)[0]
            video_output = os.path.join(output_category, video_name)
            os.makedirs(video_output, exist_ok=True)

            print(f"🎞️ Extracting {TARGET_FRAMES} frames from: {video_path}")
            extract_frames_uniform(video_path, video_output, TARGET_FRAMES)
            print(f"✅ Saved {TARGET_FRAMES} frames to: {video_output}")
