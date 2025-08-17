import argparse
import io
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import requests
import scipy.signal
import soundfile as sf
import torch
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
# Removed matplotlib and seaborn - no plotting needed
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

try:
    import librosa
except Exception as e:  # pragma: no cover - helpful message if user lacks deps
    raise RuntimeError(
        "librosa is required. Please install dependencies from ml/requirements.txt"
    ) from e


FSDD_GITHUB_ZIP = (
    "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
)


@dataclass
class PreprocessingConfig:
    sample_rate: int = 8000
    target_duration_seconds: float = 1.0
    n_fft: int = 256  # Reduced for short signals
    hop_length: int = 64   # Better temporal resolution
    n_mels: int = 32       # Fewer mel bins, focus on key formants
    fmin: int = 50
    fmax: int = 3800


class MelFeatureExtractor:
    def __init__(self, config: PreprocessingConfig):
        self.cfg = config

    def _pad_or_trim(self, y: np.ndarray) -> np.ndarray:
        target_len = int(self.cfg.sample_rate * self.cfg.target_duration_seconds)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        elif len(y) > target_len:
            y = y[:target_len]
        return y

    def extract(self, wav_path: str) -> np.ndarray:
        y, sr = librosa.load(wav_path, sr=self.cfg.sample_rate, mono=True)
        y = self._pad_or_trim(y)
        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
            power=2.0,
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db.astype(np.float32)

    def extract_from_waveform(self, y: np.ndarray) -> np.ndarray:
        y = self._pad_or_trim(y)
        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
            power=2.0,
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db.astype(np.float32)


def generate_realistic_noise(length: int, sr: int, noise_type: str = "mixed") -> np.ndarray:
    """Generate realistic environmental noise patterns"""
    
    if noise_type == "white":
        # White noise - equal energy across all frequencies
        noise = np.random.randn(length).astype(np.float32)
    
    elif noise_type == "pink":
        # Pink noise - 1/f noise, more energy at lower frequencies
        # Generate white noise first
        white = np.random.randn(length).astype(np.float32)
        # Apply pink noise filter using cumulative sum approximation
        noise = np.cumsum(white)
        noise = noise - np.mean(noise)
        noise = noise / np.std(noise)
    
    elif noise_type == "brown":
        # Brown noise - 1/f² noise, even more low-frequency energy
        white = np.random.randn(length).astype(np.float32)
        noise = np.cumsum(np.cumsum(white))
        noise = noise - np.mean(noise)
        noise = noise / np.std(noise)
    
    elif noise_type == "hum":
        # Electrical hum (50/60 Hz and harmonics)
        t = np.linspace(0, length/sr, length, False)
        noise = (0.3 * np.sin(2 * np.pi * 50 * t) +  # 50 Hz
                0.2 * np.sin(2 * np.pi * 100 * t) +  # 100 Hz harmonic
                0.1 * np.sin(2 * np.pi * 150 * t))   # 150 Hz harmonic
        noise = noise.astype(np.float32)
    
    elif noise_type == "fan":
        # Fan/air conditioning noise - low frequency rumble with harmonics
        t = np.linspace(0, length/sr, length, False)
        base_freq = random.uniform(30, 80)  # Variable fan speed
        noise = (0.4 * np.sin(2 * np.pi * base_freq * t) +
                0.2 * np.sin(2 * np.pi * base_freq * 2 * t) +
                0.1 * np.sin(2 * np.pi * base_freq * 3 * t))
        # Add some randomness
        noise += 0.1 * np.random.randn(length)
        noise = noise.astype(np.float32)
    
    elif noise_type == "traffic":
        # Traffic noise - mix of low frequency rumble and mid-frequency content
        # Low frequency rumble
        t = np.linspace(0, length/sr, length, False)
        rumble = 0.3 * np.sin(2 * np.pi * random.uniform(20, 60) * t)
        # Mid frequency content
        mid_noise = np.random.randn(length) * 0.2
        # Apply bandpass-like effect for traffic
        mid_noise = np.convolve(mid_noise, np.ones(5)/5, mode='same')
        noise = (rumble + mid_noise).astype(np.float32)
    
    elif noise_type == "keyboard":
        # Keyboard typing - short bursts of mid-to-high frequency noise
        noise = np.zeros(length)
        # Add random clicks/taps
        num_clicks = random.randint(1, 8)
        for _ in range(num_clicks):
            click_pos = random.randint(0, length - 100)
            click_duration = random.randint(20, 80)
            click_freq = random.uniform(800, 3000)
            t_click = np.linspace(0, click_duration/sr, click_duration, False)
            click = 0.3 * np.sin(2 * np.pi * click_freq * t_click) * np.exp(-3 * t_click)
            noise[click_pos:click_pos + click_duration] += click
        noise = noise.astype(np.float32)
    
    else:  # "mixed" - random combination
        noise_types = ["white", "pink", "brown", "hum", "fan", "traffic"]
        chosen_type = random.choice(noise_types)
        noise = generate_realistic_noise(length, sr, chosen_type)
    
    return noise


def augment_waveform(y: np.ndarray, sr: int, noise_intensity: float = 0.8, enable_enhanced_noise: bool = True) -> np.ndarray:
    """Enhanced augmentation with realistic noise patterns for robustness"""
    
    # Random time shift up to 100 ms
    if random.random() < 0.6:
        max_shift = int(0.1 * sr)
        shift = random.randint(-max_shift, max_shift)
        y = np.roll(y, shift)
    
    # Random time stretch (reduced n_fft to avoid warnings)
    if random.random() < 0.4:
        rate = random.uniform(0.9, 1.1)
        n_fft = min(512, len(y) // 4)
        try:
            y = librosa.effects.time_stretch(y, rate=rate, n_fft=n_fft)
        except:
            pass  # Skip if time stretch fails
    
    # Random pitch shift +/- 1.5 semitones (reduced n_fft)
    if random.random() < 0.4:
        steps = random.uniform(-1.5, 1.5)
        n_fft = min(512, len(y) // 4)
        try:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps, n_fft=n_fft)
        except:
            pass  # Skip if pitch shift fails
    
    # ENHANCED NOISE AUGMENTATION - Much more realistic and varied
    if enable_enhanced_noise and random.random() < (0.8 * noise_intensity):  # Controlled by parameters
        # Calculate signal RMS for SNR-based mixing
        signal_rms = np.sqrt(np.mean(y**2) + 1e-8)
        
        # Random SNR between 5-30 dB (wider range, including challenging conditions)
        snr_db = random.uniform(5, 30)
        
        # Choose noise type
        noise_types = ["mixed", "white", "pink", "brown", "hum", "fan", "traffic", "keyboard"]
        noise_type = random.choice(noise_types)
        
        # Generate realistic noise
        noise = generate_realistic_noise(len(y), sr, noise_type)
        
        # Scale noise to desired SNR
        noise_rms = signal_rms / (10 ** (snr_db / 20.0))
        noise = noise / (np.std(noise) + 1e-8) * noise_rms
        
        # Mix signal with noise
        y = y + noise
    
    # Volume variation (simulate different recording distances/volumes)
    if random.random() < 0.5:
        volume_factor = random.uniform(0.3, 1.5)
        y = y * volume_factor
    
    # Simulate microphone artifacts
    if random.random() < 0.3:
        # Add slight DC offset
        dc_offset = random.uniform(-0.05, 0.05)
        y = y + dc_offset
        
        # Add slight clipping simulation
        if random.random() < 0.5:
            clip_threshold = random.uniform(0.7, 0.95)
            y = np.clip(y, -clip_threshold, clip_threshold)
    
    # Simulate recording quality variations (slight filtering)
    if random.random() < 0.3:
        # Simple high-pass filter to simulate phone/poor microphone quality
        from scipy import signal
        try:
            # High-pass filter around 200-400 Hz
            cutoff = random.uniform(200, 400)
            sos = signal.butter(2, cutoff, btype='high', fs=sr, output='sos')
            y = signal.sosfilt(sos, y).astype(np.float32)
        except:
            pass  # Skip if filtering fails
    
    return y.astype(np.float32)


class FSDDDataset(Dataset):
    def __init__(self, filepaths: List[str], labels: List[int], mean: float, std: float, is_train: bool = False, 
                 enable_enhanced_noise: bool = True, noise_intensity: float = 0.8):
        assert len(filepaths) == len(labels)
        self.filepaths = filepaths
        self.labels = labels
        self.extractor = MelFeatureExtractor(PreprocessingConfig())
        self.mean = mean
        self.std = std if std > 1e-12 else 1.0
        self.is_train = is_train
        self.enable_enhanced_noise = enable_enhanced_noise
        self.noise_intensity = noise_intensity

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        path = self.filepaths[idx]
        label = self.labels[idx]
        y, _ = librosa.load(path, sr=self.extractor.cfg.sample_rate, mono=True)
        if self.is_train:
            y = augment_waveform(y, sr=self.extractor.cfg.sample_rate, 
                               noise_intensity=self.noise_intensity, 
                               enable_enhanced_noise=self.enable_enhanced_noise)
        feat = self.extractor.extract_from_waveform(y)  # (n_mels, time)
        feat = (feat - self.mean) / self.std
        # add channel dim for CNN: (1, n_mels, time)
        feat_tensor = torch.from_numpy(feat)[None, :, :]
        return feat_tensor, torch.tensor(label, dtype=torch.long)


class DigitCNN2D(nn.Module):
    """Enhanced 2D CNN for digit classification with robust feature extraction"""
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        
        # Input: (batch, 1, n_mels, time) - treat as 2D image
        
        # Feature extraction with residual connections
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
        )
        self.skip1 = nn.Conv2d(1, 32, kernel_size=1)  # Skip connection
        self.pool1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
        )
        self.skip2 = nn.Conv2d(32, 64, kernel_size=1)  # Skip connection
        self.pool2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.7)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
        )
        self.skip3 = nn.Conv2d(64, 128, kernel_size=1)  # Skip connection
        
        # Global pooling instead of attention - more robust and less biased
        self.global_pool = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Dropout2d(dropout_rate)
        )
        
        # Additional feature extraction with different kernel sizes for multi-scale features
        self.multi_scale = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 3), padding=(0, 1)),  # Temporal patterns
            nn.Conv2d(64, 32, kernel_size=(3, 1), padding=(1, 0)),  # Frequency patterns
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),       # Spatial patterns
        )
        
        # Classifier with better regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 + 96, 512),  # 128 from main path + 96 from multi-scale
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, n_mels, time)
        
        # Block 1 with residual connection
        identity1 = self.skip1(x)
        out1 = self.conv_block1(x)
        out1 = out1 + identity1
        out1 = self.pool1(out1)
        
        # Block 2 with residual connection
        identity2 = self.skip2(out1)
        out2 = self.conv_block2(out1)
        out2 = out2 + identity2
        out2 = self.pool2(out2)
        
        # Multi-scale feature extraction from intermediate features
        multi_scale_features = []
        for conv_layer in self.multi_scale:
            multi_scale_features.append(conv_layer(out2))
        multi_scale_out = torch.cat(multi_scale_features, dim=1)  # Concat along channel dim
        multi_scale_pooled = nn.AdaptiveAvgPool2d(1)(multi_scale_out).flatten(1)
        
        # Block 3 with residual connection
        identity3 = self.skip3(out2)
        out3 = self.conv_block3(out2)
        out3 = out3 + identity3
        main_features = self.global_pool(out3).flatten(1)
        
        # Combine main features with multi-scale features
        combined_features = torch.cat([main_features, multi_scale_pooled], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        return output


def download_fsdd(target_dir: str) -> str:
    recordings_dir = os.path.join(target_dir, "recordings")
    if os.path.isdir(recordings_dir) and len(os.listdir(recordings_dir)) > 0:
        return recordings_dir

    os.makedirs(target_dir, exist_ok=True)
    print("Downloading FSDD...")
    with requests.get(FSDD_GITHUB_ZIP, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
            zip_path = tmp.name

    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        # recordings are under free-spoken-digit-dataset-master/recordings
        members = [m for m in zf.namelist() if "/recordings/" in m and m.endswith(".wav")]
        zf.extractall(target_dir, members)

    # Move extracted recordings to target_dir/recordings
    root = None
    for name in os.listdir(target_dir):
        if name.startswith("free-spoken-digit-dataset-"):
            root = os.path.join(target_dir, name)
            break
    assert root is not None, "Failed to locate extracted FSDD directory"
    src_recordings = os.path.join(root, "recordings")
    os.makedirs(recordings_dir, exist_ok=True)
    for fname in os.listdir(src_recordings):
        shutil.move(os.path.join(src_recordings, fname), os.path.join(recordings_dir, fname))
    shutil.rmtree(root, ignore_errors=True)
    os.remove(zip_path)
    return recordings_dir


def parse_fsdd(recordings_dir: str) -> Tuple[List[str], List[int]]:
    filepaths = []
    labels = []
    for fname in os.listdir(recordings_dir):
        if not fname.endswith(".wav"):
            continue
        parts = fname.split("_")
        if not parts:
            continue
        label = int(parts[0])
        filepaths.append(os.path.join(recordings_dir, fname))
        labels.append(label)
    # deterministic order
    combined = list(zip(filepaths, labels))
    combined.sort(key=lambda x: x[0])
    filepaths, labels = zip(*combined)
    return list(filepaths), list(labels)


def compute_dataset_norm(filepaths: List[str]) -> Tuple[float, float]:
    extractor = MelFeatureExtractor(PreprocessingConfig())
    stats = []
    for p in tqdm(filepaths, desc="Scanning for normalization"):
        feat = extractor.extract(p)
        stats.append(feat)
    all_feats = np.concatenate([f.reshape(1, -1) for f in stats], axis=1)
    mean = float(all_feats.mean())
    std = float(all_feats.std() + 1e-6)
    return mean, std


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples"""
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def compute_class_weights(labels: List[int], num_classes: int = 10) -> torch.Tensor:
    """Compute class weights for balanced training"""
    from collections import Counter
    label_counts = Counter(labels)
    total_samples = len(labels)
    
    # Inverse frequency weighting
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)  # Avoid divisionl by zero
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)


# Evaluation moved to ml/evaluate.py to keep training script focused


def train_model(train_ds: Dataset, val_ds: Dataset | None, epochs: int = 15, lr: float = 1e-3, 
                device: str = "cpu", dropout_rate: float = 0.3, batch_size: int = 32, 
                train_labels: List[int] = None, use_focal_loss: bool = True):
    model = DigitCNN2D(num_classes=10, dropout_rate=dropout_rate).to(device)
    
    # Enhanced optimizer with better regularization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3, betas=(0.9, 0.999))
    
    # More aggressive learning rate scheduling
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr * 3, epochs=epochs, 
        steps_per_epoch=len(DataLoader(train_ds, batch_size=batch_size, shuffle=True)),
        pct_start=0.1, div_factor=10, final_div_factor=100
    )
    
    # Choose loss function based on class balance
    if use_focal_loss:
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
        print("Using Focal Loss for better handling of hard examples")
    else:
        # Use weighted CrossEntropy if class weights available
        if train_labels is not None:
            class_weights = compute_class_weights(train_labels).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print("Using Weighted CrossEntropy Loss for class balance")
        else:
            criterion = nn.CrossEntropyLoss()
            print("Using standard CrossEntropy Loss")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0) if val_ds is not None else None

    best_val = 0.0
    best_state = None
    patience_counter = 0
    best_epoch = 0
    
    # Training loop with early stopping
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        
        # Training phase with progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}")
        for xb, yb in train_pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()  # Step per batch without arguments
            
            running_loss += loss.item() * xb.size(0)
            running_acc += accuracy(logits.detach(), yb) * xb.size(0)
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        train_loss = running_loss / len(train_ds)
        train_acc = running_acc / len(train_ds)

        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f}")

    return model


def export_onnx(model: nn.Module, mean: float, std: float, export_dir: str):
    os.makedirs(export_dir, exist_ok=True)
    cfg = PreprocessingConfig()
    time_steps = math.ceil(cfg.sample_rate * cfg.target_duration_seconds / cfg.hop_length) + 1
    dummy = torch.zeros(1, 1, cfg.n_mels, time_steps)
    onnx_path = os.path.join(export_dir, "digit_cnn.onnx")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["mel_spectrogram"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes={"mel_spectrogram": {0: "batch", 3: "time"}, "logits": {0: "batch"}},
    )

    meta = {
        "mean": mean,
        "std": std,
        "labels": list(range(10)),
        "preprocessing": {
            "sample_rate": cfg.sample_rate,
            "target_duration_seconds": cfg.target_duration_seconds,
            "n_fft": cfg.n_fft,
            "hop_length": cfg.hop_length,
            "n_mels": cfg.n_mels,
            "fmin": cfg.fmin,
            "fmax": cfg.fmax,
        },
    }
    with open(os.path.join(export_dir, "preprocess.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Exported ONNX model to {onnx_path}")


def split_data(filepaths: List[str], labels: List[int], seed: int = 42) -> Tuple[List[str], List[int], List[str], List[int]]:
    from sklearn.model_selection import train_test_split
    # Select exactly 300 test samples, stratified
    train_files, test_files, train_labels, test_labels = train_test_split(
        filepaths, labels, test_size=300, random_state=seed, stratify=labels
    )
    return train_files, train_labels, test_files, test_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join("data", "fsdd"))
    parser.add_argument("--epochs", type=int, default=50)  # More epochs with early stopping
    parser.add_argument("--lr", type=float, default=1e-3)  # Will be scaled by OneCycleLR
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--export_to", default="assets")
    parser.add_argument("--dropout", type=float, default=0.3)  # Higher dropout for better regularization
    parser.add_argument("--batch_size", type=int, default=32)  # Smaller batch size for better generalization
    parser.add_argument("--noise_aug", action="store_true", default=True, help="Enable enhanced noise augmentation for robustness")
    parser.add_argument("--noise_intensity", type=float, default=0.9, help="Noise augmentation intensity (0.0-1.0)")
    args = parser.parse_args()

    recordings_dir = download_fsdd(args.data_dir)
    filepaths, labels = parse_fsdd(recordings_dir)

    # compute normalization on train set only
    train_files, train_labels, test_files, test_labels = split_data(filepaths, labels)
    mean, std = compute_dataset_norm(train_files)

    train_ds = FSDDDataset(train_files, train_labels, mean, std, is_train=True, 
                          enable_enhanced_noise=args.noise_aug, noise_intensity=args.noise_intensity)
    test_ds = FSDDDataset(test_files, test_labels, mean, std, is_train=False)

    model = train_model(
        train_ds, None,  # Force no val
        epochs=args.epochs, 
        lr=args.lr, 
        device=args.device, 
        dropout_rate=args.dropout, 
        batch_size=args.batch_size,
        train_labels=train_labels,
        use_focal_loss=True
    )

    # Set export directory first
    export_dir = os.path.abspath(args.export_to)

    # Comprehensive evaluation with confusion matrix and per-class metrics
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    print("\n" + "="*60)
    print("STARTING COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # Import and run simple evaluation
    from evaluate import evaluate_model_simple
    eval_results = evaluate_model_simple(
        model=model.to("cpu"), 
        test_loader=DataLoader(test_ds, batch_size=64, shuffle=False),
        device="cpu"
    )
    
    test_acc = eval_results['overall_accuracy']
    export_onnx(model.to("cpu"), mean, std, export_dir)

    # Save simple metrics file
    with open(os.path.join(export_dir, "metrics.json"), "w", encoding="utf-8") as f:
        payload = {
            "test_acc": float(test_acc),
            "per_class_recall": eval_results['per_class_recall'].tolist(),
            "per_class_precision": eval_results['per_class_precision'].tolist(),
            "per_class_f1": eval_results['per_class_f1'].tolist(),
            "per_class_support": eval_results['per_class_support'].tolist(),
            "confusion_matrix": eval_results['confusion_matrix'].tolist()
        }
        json.dump(payload, f, indent=2)
    
    print(f"\nDetailed metrics saved to: {os.path.join(export_dir, 'metrics.json')}")
    print("Confusion matrix and per-class visualizations saved to export directory.")


if __name__ == "__main__":
    main()


