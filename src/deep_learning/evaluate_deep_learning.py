"""
Deep Learning Evaluation Pipeline
Evaluates trained 3D CNN classifier.
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
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("ERROR: PyTorch is required. Install with: pip install torch torchvision")
    sys.exit(1)

# Try to import PyTorch Video
try:
    from pytorchvideo.models.hub import slowfast_r50, x3d_m
    PYTORCHVIDEO_AVAILABLE = True
except ImportError:
    PYTORCHVIDEO_AVAILABLE = False
    try:
        from torchvision.models.video import r3d_18, r2plus1d_18, mvit_v2_s
        TORCHVISION_VIDEO_AVAILABLE = True
    except ImportError:
        TORCHVISION_VIDEO_AVAILABLE = False


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
            from pytorchvideo.models.hub import slowfast_r50
            self.backbone = slowfast_r50(pretrained=pretrained)
            self.backbone.blocks[-1].proj = nn.Linear(self.backbone.blocks[-1].proj.in_features, num_classes)
        elif TORCHVISION_VIDEO_AVAILABLE:
            if model_name == 'r2plus1d_18':
                from torchvision.models.video import r2plus1d_18
                self.backbone = r2plus1d_18(pretrained=pretrained)
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
            elif model_name == 'r3d_18':
                from torchvision.models.video import r3d_18
                self.backbone = r3d_18(pretrained=pretrained)
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
            elif model_name == 'mvit':
                from torchvision.models.video import mvit_v2_s
                self.backbone = mvit_v2_s(pretrained=pretrained)
                self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)
            else:
                from torchvision.models.video import r2plus1d_18
                self.backbone = r2plus1d_18(pretrained=pretrained)
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        else:
            # Custom 3D ResNet (same as training)
            self.backbone = self._build_custom_3d_resnet(num_classes)
    
    def _build_custom_3d_resnet(self, num_classes):
        """Build a custom 3D ResNet if pre-trained models are not available."""
        class ResNet3D(nn.Module):
            def __init__(self, num_classes):
                super(ResNet3D, self).__init__()
                self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
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
        return self.backbone(x)


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("DEEP LEARNING PIPELINE - Evaluation")
    print("=" * 60)
    
    if not PYTORCH_AVAILABLE:
        print("\nERROR: PyTorch is required!")
        return 1
    
    # Load model
    results_dir = project_root / 'results_deep_learning'
    model_path = results_dir / 'model.pth'
    config_path = results_dir / 'training_config.pkl'
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run train_deep_learning.py first!")
        return 1
    
    print("\n[OK] Loading model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    label_to_id = config['label_to_id']
    id_to_label = config['id_to_label']
    num_classes = config['num_classes']
    model_name = config.get('model_name', 'r2plus1d_18')
    
    print(f"  Model: {model_name}")
    print(f"  Classes: {num_classes}")
    print(f"  Label mapping: {label_to_id}")
    
    # Initialize model
    if TORCHVISION_VIDEO_AVAILABLE:
        model = PreTrained3DCNN(num_classes=num_classes, model_name=model_name, pretrained=False)
    else:
        model = PreTrained3DCNN(num_classes=num_classes, model_name='custom', pretrained=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"  Device: {device}")
    
    # Data transform
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])
    
    # Load validation and test data
    print("\n[STEP 1] Loading validation and test data...")
    val_df = pd.read_csv(project_root / 'data' / 'metadata' / 'val_labels.csv')
    test_df = pd.read_csv(project_root / 'data' / 'metadata' / 'test_labels.csv')
    
    val_df = val_df[val_df['label'].isin(['hand_wave_hello', 'hand_wave_side', 'walking'])].copy()
    test_df = test_df[test_df['label'].isin(['hand_wave_hello', 'hand_wave_side', 'walking'])].copy()
    
    def prepare_dataset(df):
        video_paths = []
        labels = []
        for _, row in df.iterrows():
            video_path = Path(row['video_path'])
            if video_path.exists():
                video_paths.append(video_path)
                labels.append(label_to_id[row['label']])
        return video_paths, labels
    
    val_paths, val_labels = prepare_dataset(val_df)
    test_paths, test_labels = prepare_dataset(test_df)
    
    print(f"  [OK] Loaded {len(val_paths)} validation videos")
    print(f"  [OK] Loaded {len(test_paths)} test videos")
    
    # Evaluation function
    def evaluate_set(video_paths, labels, set_name):
        """Evaluate on a set of videos."""
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {set_name}")
        print(f"{'=' * 60}")
        
        correct = 0
        total = 0
        per_class_correct = {label: 0 for label in label_to_id.values()}
        per_class_total = {label: 0 for label in label_to_id.values()}
        
        for idx, (video_path, true_label) in enumerate(zip(video_paths, labels), 1):
            video_name = video_path.name
            
            try:
                # Load video
                dataset = VideoDataset([video_path], [true_label], num_frames=8, frame_size=112, transform=transform)
                loader = DataLoader(dataset, batch_size=1, shuffle=False)
                
                with torch.no_grad():
                    for frames, _ in loader:
                        frames = frames.to(device)  # (batch, T, C, H, W)
                        frames = frames.permute(0, 2, 1, 3, 4)  # (batch, C, T, H, W)
                        
                        outputs = model(frames)
                        probs = torch.softmax(outputs, dim=1)
                        conf, predicted = torch.max(probs, 1)
                        
                        predicted_label = predicted.item()
                        confidence = conf.item()
                        
                        total += 1
                        per_class_total[true_label] += 1
                        
                        if predicted_label == true_label:
                            correct += 1
                            per_class_correct[true_label] += 1
                            status = "OK"
                        else:
                            status = "X"
                        
                        true_label_name = id_to_label[true_label]
                        pred_label_name = id_to_label[predicted_label]
                        
                        print(f"[{idx}/{len(video_paths)}] {video_name}... {status} "
                              f"true:{true_label_name} -> pred:{pred_label_name} (conf:{confidence:.2f})")
                        break
            except Exception as e:
                print(f"[{idx}/{len(video_paths)}] {video_name}... ERROR: {e}")
                continue
        
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"\n{set_name} Results:")
        print(f"  Video-level accuracy: {accuracy:.2f}%")
        
        # Per-class accuracy
        print(f"\n  Per-class accuracy:")
        for label_id, label_name in id_to_label.items():
            if per_class_total[label_id] > 0:
                class_acc = 100 * per_class_correct[label_id] / per_class_total[label_id]
                print(f"    {label_name}: {class_acc:.2f}% ({per_class_correct[label_id]}/{per_class_total[label_id]})")
        
        return {
            'accuracy': accuracy / 100,
            'correct': correct,
            'total': total,
            'per_class_accuracy': {id_to_label[l]: per_class_correct[l] / per_class_total[l] if per_class_total[l] > 0 else 0 
                                   for l in label_to_id.values()}
        }
    
    # Evaluate on validation and test sets
    val_metrics = evaluate_set(val_paths, val_labels, "Validation Set")
    test_metrics = evaluate_set(test_paths, test_labels, "Test Set")
    
    # Generate plots
    print(f"\n{'=' * 60}")
    print("Generating Plots")
    print(f"{'=' * 60}")
    
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all predictions and labels for plots
    all_val_preds = []
    all_val_labels = []
    all_test_preds = []
    all_test_labels = []
    
    for video_path, true_label in zip(val_paths, val_labels):
        try:
            dataset = VideoDataset([video_path], [true_label], num_frames=8, frame_size=112, transform=transform)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            with torch.no_grad():
                for frames, _ in loader:
                    frames = frames.to(device)
                    frames = frames.permute(0, 2, 1, 3, 4)
                    outputs = model(frames)
                    predicted = torch.argmax(outputs, 1).item()
                    all_val_preds.append(predicted)
                    all_val_labels.append(true_label)
                    break
        except:
            continue
    
    for video_path, true_label in zip(test_paths, test_labels):
        try:
            dataset = VideoDataset([video_path], [true_label], num_frames=8, frame_size=112, transform=transform)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            with torch.no_grad():
                for frames, _ in loader:
                    frames = frames.to(device)
                    frames = frames.permute(0, 2, 1, 3, 4)
                    outputs = model(frames)
                    predicted = torch.argmax(outputs, 1).item()
                    all_test_preds.append(predicted)
                    all_test_labels.append(true_label)
                    break
        except:
            continue
    
    # Combine validation and test for overall metrics
    all_predictions = all_val_preds + all_test_preds
    all_labels_combined = all_val_labels + all_test_labels
    
    # Import plotting libraries
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    
    plt.rcParams['figure.dpi'] = 150
    sns.set_style("whitegrid")
    
    # 1. Accuracy comparison (video-level only for this model)
    fig, ax = plt.subplots(figsize=(8, 6))
    video_acc = (val_metrics['accuracy'] + test_metrics['accuracy']) / 2
    ax.bar(['Video-level'], [video_acc * 100], color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Video-level Accuracy', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([0, 100])
    ax.text(0, video_acc * 100, f'{video_acc*100:.2f}%', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / 'accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] accuracy_comparison.png")
    
    # 2. Confusion matrix
    if len(all_predictions) > 0:
        cm = confusion_matrix(all_labels_combined, all_predictions, labels=list(range(num_classes)))
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = [id_to_label[i] for i in range(num_classes)]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Count'}, ax=ax)
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(plots_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] confusion_matrix.png")
        
        # Save confusion matrix CSV
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv(plots_dir / 'confusion_matrix_detailed.csv')
        print(f"  [OK] confusion_matrix_detailed.csv")
    
    # 3. Per-class metrics
    if len(all_predictions) > 0:
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels_combined, all_predictions, labels=list(range(num_classes)), zero_division=0
        )
        metrics_data = []
        for i in range(num_classes):
            label_name = id_to_label[i]
            metrics_data.append({
                'Class': label_name,
                'Precision': precision[i],
                'Recall': recall[i],
                'F1-Score': f1[i],
                'Support': support[i]
            })
        metrics_df = pd.DataFrame(metrics_data)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(metrics_df))
        width = 0.25
        ax.bar(x - width, metrics_df['Precision'] * 100, width, label='Precision', color='#3498db', alpha=0.8)
        ax.bar(x, metrics_df['Recall'] * 100, width, label='Recall', color='#2ecc71', alpha=0.8)
        ax.bar(x + width, metrics_df['F1-Score'] * 100, width, label='F1-Score', color='#e74c3c', alpha=0.8)
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['Class'])
        ax.legend()
        ax.set_ylim([0, 100])
        plt.tight_layout()
        plt.savefig(plots_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] per_class_metrics.png")
        
        # Save CSV
        metrics_df.to_csv(plots_dir / 'per_class_metrics.csv', index=False)
        print(f"  [OK] per_class_metrics.csv")
    
    # Save results
    print(f"\n{'=' * 60}")
    print("Saving Results")
    print(f"{'=' * 60}")
    
    # Save metrics summary
    with open(results_dir / 'metrics_summary.txt', 'w') as f:
        f.write("Deep Learning Pipeline Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Pre-trained: {config.get('pretrained', False)}\n\n")
        
        f.write("Validation Set:\n")
        f.write(f"  Video-level accuracy: {val_metrics['accuracy']*100:.2f}%\n")
        f.write(f"  Correct: {val_metrics['correct']}/{val_metrics['total']}\n")
        f.write("  Per-class accuracy:\n")
        for label, acc in val_metrics['per_class_accuracy'].items():
            f.write(f"    {label}: {acc*100:.2f}%\n")
        
        f.write("\nTest Set:\n")
        f.write(f"  Video-level accuracy: {test_metrics['accuracy']*100:.2f}%\n")
        f.write(f"  Correct: {test_metrics['correct']}/{test_metrics['total']}\n")
        f.write("  Per-class accuracy:\n")
        for label, acc in test_metrics['per_class_accuracy'].items():
            f.write(f"    {label}: {acc*100:.2f}%\n")
    
    print(f"[OK] Saved metrics summary to {results_dir / 'metrics_summary.txt'}")
    
    print("\nEvaluation complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
