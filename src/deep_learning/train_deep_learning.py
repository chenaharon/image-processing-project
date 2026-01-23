"""
Deep Learning Training Pipeline - Pre-trained 3D CNN
Uses pre-trained 3D CNN models (R2Plus1D-18, R3D-18, or MViT) for video action recognition.
Fine-tuned on our gesture recognition dataset.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import cv2
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'code'))

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("ERROR: PyTorch is required. Install with: pip install torch torchvision")
    sys.exit(1)

# Try to import PyTorch Video
try:
    import pytorchvideo
    from pytorchvideo.models.hub import slowfast_r50, x3d_m
    from pytorchvideo.data import LabeledVideoDataset, RandomClipSampler
    PYTORCHVIDEO_AVAILABLE = True
except ImportError:
    PYTORCHVIDEO_AVAILABLE = False
    print("WARNING: pytorchvideo not available. Using torchvision.models.video instead.")
    try:
        from torchvision.models.video import r3d_18, r2plus1d_18, mvit_v2_s
        TORCHVISION_VIDEO_AVAILABLE = True
    except ImportError:
        TORCHVISION_VIDEO_AVAILABLE = False
        print("WARNING: torchvision video models not available. Using custom 3D ResNet.")


class VideoDataset(Dataset):
    """Dataset for loading and preprocessing videos."""
    
    def __init__(self, video_paths, labels, num_frames=16, frame_size=224, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video frames
        frames = self.load_video(video_path)
        
        # Convert to tensor
        frames_tensor = torch.FloatTensor(frames)
        
        # Apply transforms if provided
        if self.transform:
            frames_tensor = self.transform(frames_tensor)
        
        return frames_tensor, label
    
    def load_video(self, video_path):
        """Load and preprocess video frames."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            frames.append(frame)
        
        cap.release()
        
        # Sample or pad to num_frames
        if len(frames) > self.num_frames:
            # Uniform sampling
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < self.num_frames:
            # Pad with last frame
            last_frame = frames[-1] if frames else np.zeros((self.frame_size, self.frame_size, 3), dtype=np.float32)
            frames.extend([last_frame] * (self.num_frames - len(frames)))
        
        # Convert to numpy array: (T, H, W, C) -> (T, C, H, W)
        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))  # (T, C, H, W)
        
        return frames


class PreTrained3DCNN(nn.Module):
    """Pre-trained 3D CNN model for video action recognition."""
    
    def __init__(self, num_classes=3, model_name='r2plus1d_18', pretrained=True):
        super(PreTrained3DCNN, self).__init__()
        
        if PYTORCHVIDEO_AVAILABLE and model_name == 'slowfast':
            # Use SlowFast from pytorchvideo
            self.backbone = slowfast_r50(pretrained=pretrained)
            self.backbone.blocks[-1].proj = nn.Linear(self.backbone.blocks[-1].proj.in_features, num_classes)
        elif TORCHVISION_VIDEO_AVAILABLE:
            if model_name == 'r2plus1d_18':
                self.backbone = r2plus1d_18(pretrained=pretrained)
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
            elif model_name == 'r3d_18':
                self.backbone = r3d_18(pretrained=pretrained)
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
            elif model_name == 'mvit':
                self.backbone = mvit_v2_s(pretrained=pretrained)
                self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)
            else:
                # Fallback to r2plus1d_18
                self.backbone = r2plus1d_18(pretrained=pretrained)
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        else:
            # Custom 3D ResNet
            self.backbone = self._build_custom_3d_resnet(num_classes)
    
    def _build_custom_3d_resnet(self, num_classes):
        """Build a custom 3D ResNet if pre-trained models are not available."""
        from torchvision.models import resnet18
        import torchvision.models as models
        
        # Use 2D ResNet and adapt to 3D
        model = resnet18(pretrained=True)
        
        # Replace first conv layer to accept 3 channels
        # Adapt for 3D by using Conv3d
        class ResNet3D(nn.Module):
            def __init__(self, num_classes):
                super(ResNet3D, self).__init__()
                self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
                self.bn1 = nn.BatchNorm3d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
                
                # Use ResNet blocks adapted for 3D
                self.layer1 = self._make_layer(64, 64, 2)
                self.layer2 = self._make_layer(64, 128, 2, stride=2)
                self.layer3 = self._make_layer(128, 256, 2, stride=2)
                self.layer4 = self._make_layer(256, 512, 2, stride=2)
                
                self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
                self.fc = nn.Linear(512, num_classes)
            
            def _make_layer(self, inplanes, planes, blocks, stride=1):
                layers = []
                layers.append(nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
                layers.append(nn.BatchNorm3d(planes))
                layers.append(nn.ReLU(inplace=True))
                for _ in range(1, blocks):
                    layers.append(nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))
                    layers.append(nn.BatchNorm3d(planes))
                    layers.append(nn.ReLU(inplace=True))
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
        # x shape: (batch, channels, time, height, width)
        return self.backbone(x)


def main():
    """Main training function."""
    print("=" * 60)
    print("DEEP LEARNING PIPELINE - Training (Pre-trained 3D CNN)")
    print("=" * 60)
    
    if not PYTORCH_AVAILABLE:
        print("\nERROR: PyTorch is required!")
        print("Install with: pip install torch torchvision")
        return 1
    
    # Load training data
    print("\n[STEP 0] Loading training data...")
    train_df = pd.read_csv(project_root / 'data' / 'metadata' / 'train_labels.csv')
    print(f"  Total videos in dataset: {len(train_df)}")
    
    # Filter to 3 classes
    print("\n[STEP 0.1] Filtering to HELLO, WAVE_SIDE, and WALKING categories...")
    train_df = train_df[train_df['label'].isin(['hand_wave_hello', 'hand_wave_side', 'walking'])].copy()
    
    if len(train_df) == 0:
        print("Error: No videos found!")
        return 1
    
    print(f"Loaded {len(train_df)} training videos")
    
    # Balance videos per class
    min_videos = train_df['label'].value_counts().min()
    balanced_videos = []
    for label in sorted(train_df['label'].unique()):
        label_videos = train_df[train_df['label'] == label].copy()
        if len(label_videos) > min_videos:
            np.random.seed(42)
            label_videos = label_videos.sample(n=min_videos, random_state=42)
        balanced_videos.append(label_videos)
    train_df = pd.concat(balanced_videos, ignore_index=True)
    
    print("\nAfter balancing:")
    for label in sorted(train_df['label'].unique()):
        print(f"  {label}: {len(train_df[train_df['label'] == label])}")
    
    # Create label mapping
    unique_labels = sorted(train_df['label'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    print(f"\nLabel mapping: {label_to_id}")
    
    # Prepare video paths and labels
    video_paths = []
    labels = []
    
    for _, row in train_df.iterrows():
        video_path = Path(row['video_path'])
        if video_path.exists():
            video_paths.append(video_path)
            labels.append(label_to_id[row['label']])
    
    print(f"\n[STEP 1] Preparing dataset...")
    print(f"  Total videos: {len(video_paths)}")
    
    # Data augmentation
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])  # Video normalization
    ])
    
    # Create dataset - optimized for speed
    dataset = VideoDataset(video_paths, labels, num_frames=8, frame_size=112, transform=transform)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    
    print(f"  Batch size: 8 (optimized for speed)")
    print(f"  Frames per video: 8 (reduced from 16)")
    print(f"  Frame size: 112x112 (reduced from 224x224)")
    
    # Initialize model
    print(f"\n[STEP 2] Initializing pre-trained 3D CNN model...")
    
    # Try to use best available model
    if TORCHVISION_VIDEO_AVAILABLE:
        print("  Using torchvision.models.video.r2plus1d_18 (pre-trained on Kinetics)")
        model = PreTrained3DCNN(num_classes=len(unique_labels), model_name='r2plus1d_18', pretrained=True)
    else:
        print("  Using custom 3D ResNet (no pre-training)")
        model = PreTrained3DCNN(num_classes=len(unique_labels), model_name='custom', pretrained=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop - optimized for speed
    print(f"\n[STEP 3] Training model...")
    num_epochs = 10  # Reduced from 30 for faster training
    model.train()
    
    best_loss = float('inf')
    patience = 5  # Early stopping after 5 epochs without improvement
    patience_counter = 0
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (frames, labels_batch) in enumerate(train_loader):
            frames = frames.to(device)  # (batch, T, C, H, W)
            labels_batch = labels_batch.to(device)
            
            # Reshape to (batch, C, T, H, W) for 3D CNN
            frames = frames.permute(0, 2, 1, 3, 4)  # (batch, C, T, H, W)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        scheduler.step()
        
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    print("  [OK] Training complete")
    
    # Save model
    print(f"\n[STEP 4] Saving model...")
    results_dir = project_root / 'results_deep_learning'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model state
    model_path = results_dir / 'model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_to_id': label_to_id,
        'id_to_label': id_to_label,
        'num_classes': len(unique_labels),
        'model_name': 'r2plus1d_18' if TORCHVISION_VIDEO_AVAILABLE else 'custom',
    }, model_path)
    
    # Save config
    config = {
        'label_to_id': label_to_id,
        'id_to_label': id_to_label,
        'num_classes': len(unique_labels),
        'model_name': 'r2plus1d_18' if TORCHVISION_VIDEO_AVAILABLE else 'custom',
        'num_frames': 8,
        'frame_size': 112,
        'pretrained': True if TORCHVISION_VIDEO_AVAILABLE else False
    }
    
    with open(results_dir / 'training_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    print(f"Model saved to: {results_dir}")
    print("Training complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
