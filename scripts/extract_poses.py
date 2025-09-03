import os
import json
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import cv2


# -------------------------------
# Custom 14-point skeleton mapping
# -------------------------------
KEYPOINT_NAMES = [
    'nose', 'neck',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (1, 3),
    (2, 3), (2, 8), (3, 9), (8, 9),
    (2, 4), (4, 6), (3, 5), (5, 7),
    (8, 10), (10, 12), (9, 11), (11, 13)
]

def to_custom_keypoints(yolo_kps):
    """Convert 17 COCO keypoints -> 14 custom keypoints."""
    if yolo_kps.shape[0] < 17:
        return None

    nose = yolo_kps[0]
    left_shoulder, right_shoulder = yolo_kps[5], yolo_kps[6]
    neck = (left_shoulder + right_shoulder) / 2.0

    custom = [
        nose, neck,
        left_shoulder, right_shoulder,
        yolo_kps[7], yolo_kps[8],   # elbows
        yolo_kps[9], yolo_kps[10], # wrists
        yolo_kps[11], yolo_kps[12],# hips
        yolo_kps[13], yolo_kps[14],# knees
        yolo_kps[15], yolo_kps[16] # ankles
    ]
    return np.array(custom)


class PoseExtractor:
    def __init__(self, model_path="yolov8n-pose.pt"):
        self.model = YOLO(model_path)

    def _save_coco(self, annotations, images_info, output_dir):
        """Save in COCO keypoints format for CVAT."""
        os.makedirs(output_dir, exist_ok=True)

        categories = [{
            "id": 1,
            "name": "boxer",
            "supercategory": "person",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": SKELETON_CONNECTIONS
        }]

        coco = {
            "images": images_info,
            "annotations": annotations,
            "categories": categories
        }

        json_path = os.path.join(output_dir, "annotations.json")
        with open(json_path, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"[INFO] COCO annotations saved: {json_path}")

    # -------------------------------
    # Mode 2: Frames → Poses
    # -------------------------------
    def extract_from_frames(self, frames_dir, poses_dir, frame_subdir):
        input_frames = os.path.join(frames_dir, frame_subdir)
        if not os.path.exists(input_frames):
            raise FileNotFoundError(f"Frames directory not found: {input_frames}")

        output_dir = os.path.join(poses_dir, frame_subdir)
        os.makedirs(output_dir, exist_ok=True)

        frame_files = sorted([f for f in os.listdir(input_frames) if f.endswith(".jpg")])

        images_info = []
        annotations = []
        ann_id = 1

        for idx, frame_file in enumerate(tqdm(frame_files, desc="Frames")):
            frame_path = os.path.join(input_frames, frame_file)
            img = cv2.imread(frame_path)
            if img is None:
                print(f"[WARN] Could not read {frame_path}")
                continue

            h, w = img.shape[:2]

            results = self.model(frame_path, verbose=False)
            keypoints = results[0].keypoints.xy.cpu().numpy()
            bboxes = results[0].boxes.xyxy.cpu().numpy()

            images_info.append({
                "id": idx + 1,
                "file_name": frame_file,
                "height": h,
                "width": w
            })

            if keypoints.shape[0] > 0:
                kp = to_custom_keypoints(keypoints[0])  # first person only
                if kp is None:
                    continue

                kp_with_vis = []
                for (x, y) in kp:
                    kp_with_vis.extend([float(x), float(y), 2])

                if len(bboxes) > 0:
                    x1, y1, x2, y2 = bboxes[0]
                    bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                else:
                    xs, ys = kp[:, 0], kp[:, 1]
                    bbox = [float(xs.min()), float(ys.min()),
                            float(xs.max() - xs.min()), float(ys.max() - ys.min())]

                annotations.append({
                    "id": ann_id,
                    "image_id": idx + 1,
                    "category_id": 1,
                    "keypoints": kp_with_vis,
                    "num_keypoints": len(KEYPOINT_NAMES),
                    "bbox": bbox,
                    "iscrowd": 0,
                    "area": bbox[2] * bbox[3]
                })
                ann_id += 1

        self._save_coco(annotations, images_info, output_dir)

    # -------------------------------
    # Mode 3: Dataset → Poses
    # -------------------------------
    def extract_from_dataset(self, dataset_dir, input_folder, split, output_root):
        input_dir = os.path.join(dataset_dir, input_folder, split, "images")
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Dataset split not found: {input_dir}")

        output_dir = os.path.join(output_root, input_folder, split)
        os.makedirs(output_dir, exist_ok=True)

        img_files = sorted([f for f in os.listdir(input_dir) if f.endswith((".jpg", ".png"))])

        images_info = []
        annotations = []
        ann_id = 1

        for idx, img_file in enumerate(tqdm(img_files, desc=f"{input_folder}/{split}")):
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Could not read {img_path}")
                continue

            h, w = img.shape[:2]

            results = self.model(img_path, verbose=False)
            keypoints = results[0].keypoints.xy.cpu().numpy()
            bboxes = results[0].boxes.xyxy.cpu().numpy()

            images_info.append({
                "id": idx + 1,
                "file_name": img_file,
                "height": h,
                "width": w
            })

            if keypoints.shape[0] > 0:
                kp = to_custom_keypoints(keypoints[0])
                if kp is None:
                    continue

                kp_with_vis = []
                for (x, y) in kp:
                    kp_with_vis.extend([float(x), float(y), 2])

                if len(bboxes) > 0:
                    x1, y1, x2, y2 = bboxes[0]
                    bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                else:
                    xs, ys = kp[:, 0], kp[:, 1]
                    bbox = [float(xs.min()), float(ys.min()),
                            float(xs.max() - xs.min()), float(ys.max() - ys.min())]

                annotations.append({
                    "id": ann_id,
                    "image_id": idx + 1,
                    "category_id": 1,
                    "keypoints": kp_with_vis,
                    "num_keypoints": len(KEYPOINT_NAMES),
                    "bbox": bbox,
                    "iscrowd": 0,
                    "area": bbox[2] * bbox[3]
                })
                ann_id += 1

        self._save_coco(annotations, images_info, output_dir)
