import os
import sys

# Allow imports from /scripts
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
sys.path.append(SCRIPTS_DIR)

from scripts.prepare_data import FrameExtractor
from scripts.extract_poses import PoseExtractor

if __name__ == "__main__":
    print("Choose preprocessing mode:")
    print("1- Extract Frames from Video")
    print("2- Extract Poses from Frames")
    print("3- Extract Poses from Dataset (Roboflow-style)")
    choice = input("Enter choice (1/2/3): ").strip()

    raw_dir = os.path.join(ROOT_DIR, "data/raw")
    frames_dir = os.path.join(ROOT_DIR, "data/processed/frames")
    poses_dir = os.path.join(ROOT_DIR, "data/processed/poses")

    dataset_dir = os.path.join(ROOT_DIR, "data/datasets")
    dataset_poses_dir = os.path.join(ROOT_DIR, "data/datasets/poses")

    if choice == "1":
        # Video → Frames
        print(f"[INFO] Actions available in raw/: {os.listdir(raw_dir)}")
        action = input("Enter action folder name (e.g., cross, jab): ").strip()
        video_file = input("Enter video file name (e.g., video1.mp4): ").strip()

        extractor = FrameExtractor(raw_dir, frames_dir)
        extractor.extract(action, video_file, target_fps=None)

    elif choice == "2":
        # Frames → Poses
        print(f"[INFO] Frame directories available: {os.listdir(frames_dir)}")
        frame_subdir = input("Enter frame folder name (e.g., cross_video1): ").strip()

        extractor = PoseExtractor()
        extractor.extract_from_frames(frames_dir, poses_dir, frame_subdir)

    elif choice == "3":
        # Dataset Images → Poses (COCO format with custom skeleton)
        print(f"[INFO] Datasets available: {os.listdir(dataset_dir)}")
        input_folder = input("Enter dataset folder name (e.g., roboflow): ").strip()

        dataset_path = os.path.join(dataset_dir, input_folder)
        if not os.path.exists(dataset_path):
            print(f"[ERROR] Dataset not found: {dataset_path}")
            sys.exit(1)

        print(f"[INFO] Splits available: {os.listdir(dataset_path)}")
        split = input("Enter dataset split (train/valid/test): ").strip()

        extractor = PoseExtractor()
        extractor.extract_from_dataset(dataset_dir, input_folder, split, dataset_poses_dir)

    else:
        print("[ERROR] Invalid choice. Exiting.")
