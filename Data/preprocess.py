import os
import cv2
import numpy as np

def extract_frames(video_path, output_dir, frame_interval=5):
    """
    Extract frames from a video and save them to `output_dir`.

    :param video_path: Path to the video file.
    :param output_dir: Directory to save the extracted frames.
    :param frame_interval: Number of frames to skip before extracting a frame.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_id:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_id += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {frame_id} frames from {video_path}, saved to {output_dir}")

def preprocess_images(input_dir, output_dir, size=(224, 224)):
    """
    Resize and normalize images in `input_dir` and save them to `output_dir`.

    :param input_dir: Directory containing raw images.
    :param output_dir: Directory to save preprocessed images.
    :param size: Target size for resizing (width, height).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(('.jpg', '.png')):
            img_path = os.path.join(input_dir, file_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = img.astype(np.float32) / 255.0  # Normalize pixel values

            save_path = os.path.join(output_dir, file_name)
            cv2.imwrite(save_path, (img * 255).astype(np.uint8))

    print(f"Processed all images and saved them to {output_dir}")

# Example usage
if __name__ == "__main__":
    # Extract frames from a sample video
    video_path = "./data/raw/sample_video.mp4"
    extract_frames(video_path, "./data/processed/frames/", frame_interval=10)

    # Preprocess images
    preprocess_images("./data/raw/images/", "./data/processed/resized/")
