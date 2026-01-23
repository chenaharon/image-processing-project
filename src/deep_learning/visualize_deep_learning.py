"""
Deep Learning / 3D-CNN Visualization
=====================================

Visualization that highlights relevant body parts (hands/arms/legs) based on the model's prediction.
Uses motion detection and optional MediaPipe to detect body parts and colors them according to the predicted action.

Method:
- Load video at original resolution
- Get global prediction from 3D CNN model (Hello / Side Wave / Walking)
- For each frame, detect motion and optionally use MediaPipe to detect hands/pose
- Color relevant body parts:
  * Hello/Side Wave: Color hands and arms
  * Walking: Color legs and body movement
- Save visualization at original resolution
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "code"))

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Visualization will use motion detection instead.")

# Colors (BGR for OpenCV)
COLORS = {
    "hand_wave_hello": (255, 0, 0),      # BLUE
    "hand_wave_side": (0, 255, 255),     # YELLOW
    "walking": (128, 0, 128),            # PURPLE
}
GRAY = (128, 128, 128)

MAX_FRAMES_VIS = 120     # limit visualization length per video
WINDOW_SIZE = 16         # temporal window for 3D CNN
MODEL_FRAME_SIZE = 112   # resolution for model input (as trained)


class PreTrained3DCNN(nn.Module):
    """Same backbone definition as in training/evaluation (3D CNN classifier)."""

    def __init__(self, num_classes: int = 3, model_name: str = "r2plus1d_18", pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models.video import r2plus1d_18, r3d_18, mvit_v2_s
        except ImportError:
            self.backbone = self._build_custom_3d_resnet(num_classes)
            return

        if model_name == "r3d_18":
            self.backbone = r3d_18(pretrained=pretrained)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif model_name == "mvit":
            self.backbone = mvit_v2_s(pretrained=pretrained)
            self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)
        else:  # default r2plus1d_18
            self.backbone = r2plus1d_18(pretrained=pretrained)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def _build_custom_3d_resnet(self, num_classes: int):
        class ResNet3D(nn.Module):
            def __init__(self, num_classes: int):
                super().__init__()
                self.conv1 = nn.Conv3d(
                    3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False
                )
                self.bn1 = nn.BatchNorm3d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
                self.layer1 = self._make_layer(64, 64, 2)
                self.layer2 = self._make_layer(64, 128, 2, stride=2)
                self.layer3 = self._make_layer(128, 256, 2, stride=2)
                self.layer4 = self._make_layer(256, 512, 2, stride=2)
                self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
                self.fc = nn.Linear(512, num_classes)

            def _make_layer(self, inplanes, planes, blocks, stride=1):
                layers = [
                    nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm3d(planes),
                    nn.ReLU(inplace=True),
                ]
                for _ in range(1, blocks):
                    layers.extend(
                        [
                            nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm3d(planes),
                            nn.ReLU(inplace=True),
                        ]
                    )
                return nn.Sequential(*layers)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        return ResNet3D(num_classes)

    def forward(self, x):
        return self.backbone(x)


def load_video_rgb(path: Path, max_frames: int = MAX_FRAMES_VIS) -> Tuple[List[np.ndarray], Tuple[int, int], float]:
    """
    Load video as list of RGB frames at ORIGINAL resolution.
    Returns: (frames, (width, height), fps)
    """
    cap = cv2.VideoCapture(str(path))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    rgb_frames: List[np.ndarray] = []

    while len(rgb_frames) < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        # Keep original resolution - no resizing!
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frames.append(frame_rgb)

    cap.release()

    if not rgb_frames:
        raise ValueError(f"Could not read any frames from {path}")

    return rgb_frames, (original_width, original_height), fps


def predict_video_label_from_frames(
    rgb_frames: List[np.ndarray],
    model: nn.Module,
    device: torch.device,
    num_frames_model: int = WINDOW_SIZE,
    model_frame_size: int = MODEL_FRAME_SIZE,
) -> Tuple[int, float]:
    """
    Single global prediction for the whole video (one label per clip).
    Frames are resized to model_frame_size for model input.
    """
    T = len(rgb_frames)
    if T >= num_frames_model:
        indices = np.linspace(0, T - 1, num_frames_model, dtype=int)
        clip = [rgb_frames[i] for i in indices]
    else:
        clip = list(rgb_frames)
        while len(clip) < num_frames_model:
            clip.append(clip[-1])

    # Resize frames to model input size
    clip_resized = []
    for frame in clip:
        frame_resized = cv2.resize(frame, (model_frame_size, model_frame_size))
        clip_resized.append(frame_resized)

    arr = np.stack(clip_resized, axis=0).astype(np.float32) / 255.0  # (T, H, W, 3)
    arr = np.transpose(arr, (3, 0, 1, 2))  # (C, T, H, W)
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)   # (1, C, T, H, W)

    mean = torch.tensor([0.45, 0.45, 0.45], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 3, 1, 1, 1)
    tensor = (tensor - mean) / std

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)

    return int(pred.item()), float(conf.item())


class MediaPipeDetector:
    """Detect hands and pose using MediaPipe (with fallback to motion detection)."""
    
    def __init__(self):
        self.mp_hands = None
        self.mp_pose = None
        self.hands = None
        self.pose = None
        self.use_mediapipe = False
        
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_hands = mp.solutions.hands
                self.mp_pose = mp.solutions.pose
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5
                )
                self.use_mediapipe = True
            except (AttributeError, Exception) as e:
                print(f"  [WARNING] MediaPipe API issue: {e}. Using motion detection fallback.")
                self.use_mediapipe = False
    
    def detect_body_parts(self, frame_rgb: np.ndarray) -> Dict:
        """
        Detect hands and pose landmarks.
        Returns dict with 'hands' and 'pose' landmarks.
        """
        if not self.use_mediapipe or self.hands is None:
            return {'hands': [], 'pose': None}
        
        h, w = frame_rgb.shape[:2]
        
        # Detect hands
        hand_results = self.hands.process(frame_rgb)
        hands_data = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    hand_points.append((x, y))
                hands_data.append(hand_points)
        
        # Detect pose
        pose_results = self.pose.process(frame_rgb)
        pose_landmarks = None
        if pose_results.pose_landmarks:
            pose_points = []
            for landmark in pose_results.pose_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                visibility = landmark.visibility
                pose_points.append((x, y, visibility))
            pose_landmarks = pose_points
        
        return {'hands': hands_data, 'pose': pose_landmarks}
    
    def close(self):
        """Close MediaPipe resources."""
        if self.hands:
            self.hands.close()
        if self.pose:
            self.pose.close()


def create_motion_mask(gray_frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Create motion masks using frame differences.
    Returns list of binary masks (1 = motion, 0 = static).
    Uses adaptive thresholding to focus on significant motion only.
    """
    masks = []
    prev = gray_frames[0].astype(np.float32)
    
    for gray in gray_frames:
        cur = gray.astype(np.float32)
        diff = np.abs(cur - prev)
        prev = cur
        
        # Use higher percentile (95th) to focus on strong motion
        threshold = np.percentile(diff, 95)
        # Minimum threshold to avoid noise
        min_threshold = max(threshold * 0.7, 15.0)
        mask = (diff > min_threshold).astype(np.uint8) * 255
        
        # Morphological operations to clean up and connect nearby regions
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        
        # Close small gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        # Slightly dilate to connect nearby motion regions
        mask = cv2.dilate(mask, kernel_large, iterations=1)
        
        masks.append(mask)
    
    return masks


def draw_hand_region(frame: np.ndarray, hand_points: List[Tuple[int, int]], color: Tuple[int, int, int], thickness: int = 3):
    """Draw filled region around hand landmarks."""
    if len(hand_points) < 5:
        return
    
    # Get bounding box
    xs = [p[0] for p in hand_points]
    ys = [p[1] for p in hand_points]
    x_min, x_max = max(0, min(xs) - 20), min(frame.shape[1], max(xs) + 20)
    y_min, y_max = max(0, min(ys) - 20), min(frame.shape[0], max(ys) + 20)
    
    # Create mask for hand region
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Draw hand landmarks as filled circles
    for x, y in hand_points:
        cv2.circle(mask, (x, y), 15, 255, -1)
    
    # Connect key points to form hand shape
    if len(hand_points) >= 21:  # Full hand landmarks
        # Palm
        palm_indices = [0, 1, 5, 9, 13, 17]
        palm_points = [hand_points[i] for i in palm_indices if i < len(hand_points)]
        if len(palm_points) > 2:
            cv2.fillPoly(mask, [np.array(palm_points)], 255)
        
        # Fingers
        finger_connections = [
            [0, 1, 2, 3, 4],      # Thumb
            [0, 5, 6, 7, 8],      # Index
            [0, 9, 10, 11, 12],   # Middle
            [0, 13, 14, 15, 16],  # Ring
            [0, 17, 18, 19, 20],  # Pinky
        ]
        for connection in finger_connections:
            points = [hand_points[i] for i in connection if i < len(hand_points)]
            if len(points) > 2:
                cv2.polylines(mask, [np.array(points)], False, 255, 8)
    
    # Apply color overlay
    mask_3d = mask[:, :, np.newaxis] / 255.0
    color_bgr = np.array(color, dtype=np.float32)
    frame_float = frame.astype(np.float32)
    colored_region = frame_float * (1 - mask_3d * 0.6) + color_bgr * (mask_3d * 0.6)
    frame[:] = np.clip(colored_region, 0, 255).astype(np.uint8)


def draw_pose_region(frame: np.ndarray, pose_points: List[Tuple[int, int, float]], color: Tuple[int, int, int], action_type: str):
    """Draw colored region for pose (legs/body for walking)."""
    if pose_points is None or len(pose_points) < 10:
        return
    
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # MediaPipe pose landmark indices
    # For walking, focus on legs and lower body
    if action_type == "walking":
        # Leg landmarks: 23, 24 (hips), 25, 26, 27, 28, 29, 30, 31, 32 (legs and feet)
        leg_indices = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        leg_points = []
        for idx in leg_indices:
            if idx < len(pose_points) and pose_points[idx][2] > 0.5:  # visibility > 0.5
                leg_points.append((pose_points[idx][0], pose_points[idx][1]))
        
        if len(leg_points) >= 4:
            # Draw filled region around legs
            for x, y in leg_points:
                cv2.circle(mask, (x, y), 25, 255, -1)
            
            # Connect leg points
            if len(leg_points) >= 6:
                # Left leg
                left_leg = [leg_points[i] for i in [0, 2, 4] if i < len(leg_points)]
                if len(left_leg) >= 2:
                    cv2.polylines(mask, [np.array(left_leg)], False, 255, 20)
                # Right leg
                right_leg = [leg_points[i] for i in [1, 3, 5] if i < len(leg_points)]
                if len(right_leg) >= 2:
                    cv2.polylines(mask, [np.array(right_leg)], False, 255, 20)
    
    # Apply color overlay
    mask_3d = mask[:, :, np.newaxis] / 255.0
    color_bgr = np.array(color, dtype=np.float32)
    frame_float = frame.astype(np.float32)
    colored_region = frame_float * (1 - mask_3d * 0.6) + color_bgr * (mask_3d * 0.6)
    frame[:] = np.clip(colored_region, 0, 255).astype(np.uint8)


def filter_motion_by_region(motion_mask: np.ndarray, predicted_label: str) -> np.ndarray:
    """
    Filter motion mask to focus on relevant regions based on predicted action.
    Returns filtered mask that emphasizes relevant body parts.
    """
    h, w = motion_mask.shape
    filtered_mask = motion_mask.copy()
    
    if predicted_label in ["hand_wave_hello", "hand_wave_side"]:
        # For hand movements, create a soft weight mask that favors upper body
        # but doesn't hard-cut - just reduces weight in lower regions
        weight_mask = np.ones((h, w), dtype=np.float32)
        for y in range(h):
            if y < int(h * 0.3):
                weight_mask[y, :] = 1.0  # Full weight for top 30%
            elif y < int(h * 0.5):
                weight_mask[y, :] = 0.8  # High weight for 30-50%
            elif y < int(h * 0.7):
                weight_mask[y, :] = 0.4  # Medium weight for 50-70%
            else:
                weight_mask[y, :] = 0.1  # Low weight for bottom 30%
        
        filtered_mask = (filtered_mask.astype(np.float32) * weight_mask).astype(np.uint8)
    
    elif predicted_label == "walking":
        # For walking, favor lower body but don't hard-cut
        weight_mask = np.ones((h, w), dtype=np.float32)
        for y in range(h):
            if y < int(h * 0.3):
                weight_mask[y, :] = 0.1  # Low weight for top 30%
            elif y < int(h * 0.5):
                weight_mask[y, :] = 0.4  # Medium weight for 30-50%
            elif y < int(h * 0.7):
                weight_mask[y, :] = 0.8  # High weight for 50-70%
            else:
                weight_mask[y, :] = 1.0  # Full weight for bottom 30%
        
        filtered_mask = (filtered_mask.astype(np.float32) * weight_mask).astype(np.uint8)
    
    return filtered_mask


def visualize_frame(
    frame_rgb: np.ndarray,
    detector: MediaPipeDetector,
    predicted_label: str,
    color: Tuple[int, int, int],
    use_motion_fallback: bool = False,
    motion_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Visualize a single frame by coloring ONLY areas with significant motion.
    No hard cuts - only colors actual moving regions.
    """
    frame_bgr = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    
    # Try MediaPipe first
    body_parts = detector.detect_body_parts(frame_rgb)
    used_mediapipe = False
    
    # Color based on predicted action
    if predicted_label in ["hand_wave_hello", "hand_wave_side"]:
        # Color hands using MediaPipe if available
        for hand_points in body_parts['hands']:
            if len(hand_points) > 0:
                draw_hand_region(frame_bgr, hand_points, color)
                used_mediapipe = True
        
        # Use motion detection - filter to favor upper body but color only actual motion
        if use_motion_fallback and motion_mask is not None:
            # Filter motion mask to favor relevant regions (but no hard cut)
            filtered_mask = filter_motion_by_region(motion_mask, predicted_label)
            
            # Find contours to identify distinct motion regions
            contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter out small contours (noise)
            min_area = (frame_bgr.shape[0] * frame_bgr.shape[1]) * 0.001  # 0.1% of frame area
            significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            # Create mask from significant contours only
            final_mask = np.zeros_like(filtered_mask)
            if significant_contours:
                cv2.drawContours(final_mask, significant_contours, -1, 255, -1)
                # Smooth the mask slightly
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
                final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            
            # Apply color overlay only where there's significant motion
            mask_3d = final_mask[:, :, np.newaxis] / 255.0
            color_bgr = np.array(color, dtype=np.float32)
            frame_float = frame_bgr.astype(np.float32)
            colored = frame_float * (1 - mask_3d * 0.7) + color_bgr * (mask_3d * 0.7)
            frame_bgr = np.clip(colored, 0, 255).astype(np.uint8)
    
    elif predicted_label == "walking":
        # Color legs using MediaPipe if available
        if body_parts['pose']:
            draw_pose_region(frame_bgr, body_parts['pose'], color, "walking")
            used_mediapipe = True
        
        # Use motion detection - filter to favor lower body but color only actual motion
        if use_motion_fallback and motion_mask is not None:
            # Filter motion mask to favor relevant regions (but no hard cut)
            filtered_mask = filter_motion_by_region(motion_mask, predicted_label)
            
            # Find contours to identify distinct motion regions
            contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter out small contours (noise)
            min_area = (frame_bgr.shape[0] * frame_bgr.shape[1]) * 0.001  # 0.1% of frame area
            significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            # Create mask from significant contours only
            final_mask = np.zeros_like(filtered_mask)
            if significant_contours:
                cv2.drawContours(final_mask, significant_contours, -1, 255, -1)
                # Smooth the mask slightly
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
                final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            
            # Apply color overlay only where there's significant motion
            mask_3d = final_mask[:, :, np.newaxis] / 255.0
            color_bgr = np.array(color, dtype=np.float32)
            frame_float = frame_bgr.astype(np.float32)
            colored = frame_float * (1 - mask_3d * 0.7) + color_bgr * (mask_3d * 0.7)
            frame_bgr = np.clip(colored, 0, 255).astype(np.uint8)
    
    return frame_bgr


def load_model(results_dir):
    """Load trained model and config (same as in evaluation pipeline)."""
    model_path = results_dir / "model.pth"
    config_path = results_dir / "training_config.pkl"

    if not model_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Model files not found in {results_dir}")

    checkpoint = torch.load(model_path, map_location="cpu")
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    label_to_id = config["label_to_id"]
    id_to_label = config["id_to_label"]
    num_classes = config["num_classes"]
    model_name = config.get("model_name", "r2plus1d_18")

    model = PreTrained3DCNN(num_classes=num_classes, model_name=model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, label_to_id, id_to_label, config


def main():
    """Main entry point for visualization."""
    print("=" * 60)
    print("DEEP LEARNING / 3D-CNN PIPELINE - Visualization")
    print("=" * 60)

    results_dir = project_root / "results_deep_learning"
    print(f"\n[STEP 1] Loading model from {results_dir}...")
    try:
        model, label_to_id, id_to_label, config = load_model(results_dir)
        print(f"  [OK] Loaded model: {config.get('model_name', 'custom')}")
        print(f"  Classes: {list(label_to_id.keys())}")
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Device: {device}")

    # Initialize MediaPipe detector
    print("\n[STEP 2] Initializing detector...")
    detector = MediaPipeDetector()
    if detector.use_mediapipe:
        print("  [OK] MediaPipe available - will use hand/pose detection")
    else:
        print("  [OK] Using motion detection (MediaPipe not available or API issue)")

    print("\n[STEP 3] Loading test videos...")
    test_df = pd.read_csv(project_root / "data" / "metadata" / "test_labels.csv")
    test_df = test_df[test_df["label"].isin(["hand_wave_hello", "hand_wave_side", "walking"])].copy()

    selected_rows = []
    for label in ["hand_wave_hello", "hand_wave_side", "walking"]:
        subset = test_df[test_df["label"] == label].head(3)
        selected_rows.append(subset)
    selected_df = pd.concat(selected_rows, ignore_index=True)
    print(f"  Selected {len(selected_df)} videos for visualization")

    vis_dir = results_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("\n[STEP 4] Generating visualizations...")
    success = 0

    for idx, row in selected_df.iterrows():
        video_path = Path(row["video_path"])
        if not video_path.exists():
            print(f"  [SKIP] {video_path.name} (file not found)")
            continue

        out_path = vis_dir / f"{video_path.stem}_visualization.mp4"
        print(f"  [{idx+1}/{len(selected_df)}] {video_path.name}...", end=" ")

        try:
            # Load video at original resolution
            rgb_frames, (width, height), fps = load_video_rgb(video_path, max_frames=MAX_FRAMES_VIS)
            print(f"({width}x{height}, {len(rgb_frames)} frames)...", end=" ")

            # Get global prediction
            global_pred, global_conf = predict_video_label_from_frames(rgb_frames, model, device)
            global_label = id_to_label[global_pred]
            class_color = COLORS.get(global_label, GRAY)
            print(f"Predicted: {global_label} ({global_conf:.2f})...", end=" ")

            # Prepare motion masks (always use for better visualization)
            gray_frames = [cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY) for f in rgb_frames]
            motion_masks = create_motion_mask(gray_frames)

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

            # Process each frame
            for t, frame_rgb in enumerate(rgb_frames):
                motion_mask = motion_masks[t] if motion_masks else None
                vis_frame = visualize_frame(
                    frame_rgb,
                    detector,
                    global_label,
                    class_color,
                    use_motion_fallback=True,  # Always use motion as primary or fallback
                    motion_mask=motion_mask
                )
                writer.write(vis_frame)

            writer.release()
            print("[OK]")
            success += 1

        except Exception as e:
            print(f"[FAILED: {e}]")
            import traceback
            traceback.print_exc()
            continue

    # Clean up
    detector.close()

    print(f"\n[OK] Generated {success}/{len(selected_df)} visualizations")
    print(f"  Output directory: {vis_dir}")
    print("\nVisualization complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
