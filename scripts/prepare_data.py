import os
import cv2
import json

class FrameExtractor:
    def __init__(self, raw_dir, frames_dir):
        self.raw_dir = raw_dir
        self.frames_dir = frames_dir
        self.metadata_path = os.path.join(frames_dir, "metadata.json")
        os.makedirs(frames_dir, exist_ok=True)

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self, metadata):
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def extract(self, action, video_file, target_fps=None):
        input_video = os.path.join(self.raw_dir, action, video_file)
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Video not found: {input_video}")

        video_name = os.path.splitext(video_file)[0]
        output_subdir = f"{action}_{video_name}"
        output_dir = os.path.join(self.frames_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {input_video}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Original FPS: {original_fps:.2f}")

        if target_fps is None:
            try:
                target_fps = int(input("Enter target FPS for extraction: "))
            except ValueError:
                cap.release()
                raise ValueError("Invalid FPS input. Please enter an integer.")

        frame_interval = int(round(original_fps / target_fps))
        if frame_interval <= 0:
            frame_interval = 1

        print(f"[INFO] Extracting at {target_fps} FPS (every {frame_interval} frames).")

        frame_count, saved_count = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
            frame_count += 1

        cap.release()
        print(f"[INFO] Saved {saved_count} frames to {output_dir}")

        # Update metadata
        metadata = self._load_metadata()
        metadata[output_subdir] = {
            "original_fps": original_fps,
            "target_fps": target_fps,
            "total_frames": saved_count,
            "action": action
        }
        self._save_metadata(metadata)
        print(f"[INFO] Metadata updated at {self.metadata_path}")
