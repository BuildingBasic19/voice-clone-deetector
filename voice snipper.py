import os
import random
import argparse
import re
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
DURATION = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

N_MELS = 96
N_FFT = 1024
HOP_LENGTH = 256
FMIN = 20
FMAX = 7600

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

MODEL_PATH = "voice_authenticity_model.pt"

CLASS_NAMES = {
    0: "NATURAL_HUMAN_SPEECH",
    1: "AI_GENERATED_OR_SYNTHETIC_SPEECH",
}

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def load_audio(path):
    """
    Loads mono audio and resamples it to SAMPLE_RATE.

    librosa supports common formats such as WAV and MP3.
    """

    audio, _ = librosa.load(
        str(path),
        sr=SAMPLE_RATE,
        mono=True,
    )

    audio = np.nan_to_num(audio).astype(np.float32)

    if len(audio) == 0:
        raise ValueError(f"Empty audio file: {path}")

    return audio


def normalize_audio(audio):
    """
    Peak normalization.
    """

    peak = np.max(np.abs(audio))

    if peak > 1e-8:
        audio = audio / peak

    return audio


def pad_or_crop(audio, training=False):
    """
    Converts every sample to exactly DURATION seconds.

    During training, random crops are used when possible.
    During validation/inference, a centered crop is used.
    """

    if len(audio) < NUM_SAMPLES:
        padding = NUM_SAMPLES - len(audio)

        if training:
            left = random.randint(0, padding)
        else:
            left = padding // 2

        right = padding - left

        audio = np.pad(
            audio,
            (left, right),
            mode="constant",
        )

    elif len(audio) > NUM_SAMPLES:

        if training:
            start = random.randint(
                0,
                len(audio) - NUM_SAMPLES,
            )
        else:
            start = (len(audio) - NUM_SAMPLES) // 2

        audio = audio[start:start + NUM_SAMPLES]

    return audio.astype(np.float32)


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment_audio(audio):
    """
    Conservative audio augmentation.

    The purpose is to make the model less dependent on
    recording conditions while preserving speech characteristics.
    """

    if random.random() < 0.5:
        noise_level = random.uniform(0.001, 0.008)

        noise = np.random.normal(
            0,
            noise_level,
            size=audio.shape,
        ).astype(np.float32)

        audio = audio + noise

    if random.random() < 0.3:
        gain = random.uniform(0.75, 1.25)
        audio = audio * gain

    if random.random() < 0.25:
        rate = random.uniform(0.95, 1.05)

        try:
            audio = librosa.effects.time_stretch(
                audio,
                rate=rate,
            )
        except Exception:
            pass

    return audio.astype(np.float32)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def audio_to_log_mel(audio):
    """
    Converts audio into a normalized log-Mel spectrogram.
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    # Per-sample standardization
    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-6

    log_mel = (log_mel - mean) / std

    return log_mel.astype(np.float32)


# ============================================================
# DATASET DISCOVERY
# ============================================================

def _source_group_id(path, class_name):
    """
    Keep chunks originating from the same source together.

    Gary Stafford's dataset uses names such as:
        yt_0001_part_001.flac
        el_0001_c_part_002.flac
        po_0001_part_003.flac
        hg_0001_p2_part_001.flac

    The part/chunk suffix is removed so related chunks do not leak
    across train/validation/test.
    """
    stem = Path(path).stem

    # Remove known chunking suffixes.
    base = re.split(
        r"_(?:p2_)?part_\d+$|_c_part_\d+$",
        stem,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    # Fallback for unexpected naming.
    if not base:
        base = Path(path).parent.name

    return f"{class_name}:{base}"


def discover_dataset(dataset_dir):
    """
    Discover the Hugging Face dataset after it has been downloaded
    to a local directory.

    Expected structure for:
    garystafford/deepfake-audio-detection

        dataset/
            real/
                yt_*.flac

            fake/
                el_*.flac
                hg_*.flac
                hu_*.flac
                lv_*.flac
                po_*.flac
                sp_*.flac

    Also accepts the older/local aliases:
        natural/ -> label 0
        synthetic/ or ai/ -> label 1

    Returns a list of dictionaries:
        {"path": ..., "label": 0/1, "group": ...}
    """

    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_dir}"
        )

    if not dataset_dir.is_dir():
        raise NotADirectoryError(
            f"Dataset path is not a directory: {dataset_dir}"
        )

    # Order matters only for display; labels are explicit.
    class_directories = {
        "real": 0,
        "natural": 0,
        "fake": 1,
        "synthetic": 1,
        "ai": 1,
    }

    samples = []
    counts = {0: 0, 1: 0}

    for class_name, label in class_directories.items():
        class_dir = dataset_dir / class_name

        if not class_dir.exists():
            continue

        for path in class_dir.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue

            group_id = _source_group_id(path, class_name)

            samples.append(
                {
                    "path": str(path),
                    "label": label,
                    "group": group_id,
                }
            )
            counts[label] += 1

    if not samples:
        raise RuntimeError(
            "\nNo audio files were found.\n"
            f"Dataset directory checked: {dataset_dir.resolve()}\n\n"
            "For the Gary Stafford dataset, the directory should contain:\n"
            "  real/   -> 933 FLAC files\n"
            "  fake/   -> 933 FLAC files\n"
        )

    print("\nDataset discovery:")
    print(f"  Natural/real : {counts[0]}")
    print(f"  Synthetic/fake: {counts[1]}")
    print(f"  Total         : {len(samples)}")

    if counts[0] == 0 or counts[1] == 0:
        raise RuntimeError(
            "\nOnly one class was found.\n"
            f"Found real/natural={counts[0]}, fake/synthetic={counts[1]}.\n\n"
            "This dataset must contain BOTH 'real' and 'fake' folders.\n"
            "Your previous run found 933 files because the old code did "
            "not recognize the 'fake' directory."
        )

    return samples


# ============================================================
# GROUPED TRAIN/VALIDATION/TEST SPLIT
# ============================================================

def split_dataset(samples):
    """
    Split into approximately:
        80% training
        10% validation
        10% testing

    Chunks from the same original recording/source stay in the same
    split to reduce source leakage.

    Returns actual sample dictionaries, not integer indices.
    """

    if len(samples) < 20:
        raise ValueError(
            f"Dataset is too small to create reliable splits: {len(samples)} samples."
        )

    labels = np.array([sample["label"] for sample in samples])
    groups = np.array([sample["group"] for sample in samples])
    indices = np.arange(len(samples))

    if len(np.unique(labels)) != 2:
        raise ValueError(
            "Training requires both classes (real and fake), but the "
            f"dataset contains classes: {sorted(np.unique(labels).tolist())}"
        )

    # Prefer StratifiedGroupKFold. Ten folds gives approximately 10%
    # validation and 10% test while keeping source groups together.
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(
            n_splits=10,
            shuffle=True,
            random_state=RANDOM_SEED,
        )

        folds = list(splitter.split(indices, labels, groups))

        # Fold 0 -> test, fold 1 -> validation, remaining -> training.
        test_indices = folds[0][1]
        val_indices = folds[1][1]

        train_indices = np.concatenate(
            [folds[i][1] for i in range(2, len(folds))]
        )

    except (ImportError, ValueError):
        # Compatibility fallback for older scikit-learn versions.
        from sklearn.model_selection import GroupShuffleSplit

        found_split = False

        for seed in range(RANDOM_SEED, RANDOM_SEED + 100):
            first = GroupShuffleSplit(
                n_splits=1,
                test_size=0.20,
                random_state=seed,
            )

            train_indices, temp_indices = next(
                first.split(indices, labels, groups)
            )

            if len(np.unique(labels[temp_indices])) < 2:
                continue
            if len(np.unique(labels[train_indices])) < 2:
                continue

            second = GroupShuffleSplit(
                n_splits=1,
                test_size=0.50,
                random_state=seed,
            )

            val_indices, test_indices = next(
                second.split(
                    temp_indices,
                    labels[temp_indices],
                    groups[temp_indices],
                )
            )

            val_indices = temp_indices[val_indices]
            test_indices = temp_indices[test_indices]

            if (
                len(np.unique(labels[val_indices])) == 2
                and len(np.unique(labels[test_indices])) == 2
            ):
                found_split = True
                break

        if not found_split:
            raise RuntimeError(
                "Could not create a group-aware split containing both "
                "real and fake classes in train/validation/test."
            )

    # Convert indices into the dictionaries VoiceDataset expects.
    train_samples = [samples[int(i)] for i in train_indices]
    validation_samples = [samples[int(i)] for i in val_indices]
    test_samples = [samples[int(i)] for i in test_indices]

    def print_distribution(name, split):
        split_labels = [sample["label"] for sample in split]
        real_count = split_labels.count(0)
        fake_count = split_labels.count(1)

        print(
            f"{name:<12}: {len(split):4d} "
            f"(real={real_count:4d}, fake={fake_count:4d})"
        )

        if real_count == 0 or fake_count == 0:
            raise RuntimeError(
                f"{name} split contains only one class. "
                "The split must contain both real and fake samples."
            )

    print("\nDataset split:")
    print_distribution("Train", train_samples)
    print_distribution("Validation", validation_samples)
    print_distribution("Test", test_samples)

    return train_samples, validation_samples, test_samples


class VoiceDataset(Dataset):

    def __init__(
        self,
        samples,
        training=False,
    ):
        self.samples = samples
        self.training = training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        item = self.samples[index]

        audio = load_audio(item["path"])

        audio = normalize_audio(audio)

        if self.training:
            audio = augment_audio(audio)

        audio = pad_or_crop(
            audio,
            training=self.training,
        )

        features = audio_to_log_mel(audio)

        features = torch.from_numpy(features)

        # CNN expects [channels, height, width]
        features = features.unsqueeze(0)

        label = torch.tensor(
            item["label"],
            dtype=torch.long,
        )

        return features, label


# ============================================================
# CNN MODEL
# ============================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class VoiceAuthenticityCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            ConvBlock(
                1,
                32,
                dropout=0.10,
            ),

            ConvBlock(
                32,
                64,
                dropout=0.15,
            ),

            ConvBlock(
                64,
                128,
                dropout=0.20,
            ),

            ConvBlock(
                128,
                256,
                dropout=0.25,
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.35),

            nn.Linear(
                128,
                2,
            ),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(samples):

    labels = np.array(
        [sample["label"] for sample in samples]
    )

    classes = np.unique(labels)

    if len(classes) != 2 or set(classes.tolist()) != {0, 1}:
        raise ValueError(
            "Training data must contain both class 0 (real) and "
            "class 1 (fake)."
        )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=labels,
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(features)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        total_loss += (
            loss.item() * labels.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / max(total, 1),
        correct / max(total, 1),
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_loss(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    total = 0

    for features, labels in loader:

        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)

        loss = criterion(
            logits,
            labels,
        )

        total_loss += (
            loss.item() * labels.size(0)
        )

        total += labels.size(0)

    return total_loss / max(total, 1)


# ============================================================
# FULL EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for features, labels in loader:

        features = features.to(device)

        logits = model(features)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        all_labels.extend(
            labels.numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
            .tolist()
        )

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_probability = np.array(all_probabilities)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(
            y_true,
            y_probability,
        )
    else:
        roc_auc = float("nan")

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    print("\n==============================")
    print("EVALUATION")
    print("==============================")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    if not np.isnan(roc_auc):
        print(f"ROC-AUC  : {roc_auc:.4f}")
    else:
        print("ROC-AUC  : unavailable")

    print("\nConfusion Matrix:")
    print(
        "                 Predicted"
    )
    print(
        "                 Natural  Synthetic"
    )
    print(
        f"Actual Natural   {matrix[0][0]:8d} {matrix[0][1]:10d}"
    )
    print(
        f"Actual Synthetic {matrix[1][0]:8d} {matrix[1][1]:10d}"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    path,
):

    checkpoint = {
        "model_state_dict": model.state_dict(),

        "sample_rate": SAMPLE_RATE,
        "duration": DURATION,
        "num_samples": NUM_SAMPLES,

        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,

        "fmin": FMIN,
        "fmax": FMAX,

        "class_names": CLASS_NAMES,
        "random_seed": RANDOM_SEED,
    }

    torch.save(
        checkpoint,
        path,
    )

    print(f"\nModel saved to: {path}")


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    path,
    device,
):

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model = VoiceAuthenticityCNN()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model


# ============================================================
# SINGLE AUDIO INFERENCE
# ============================================================

@torch.no_grad()
def predict_audio(
    model,
    audio_path,
    device,
):

    audio = load_audio(audio_path)

    audio = normalize_audio(audio)

    audio = pad_or_crop(
        audio,
        training=False,
    )

    features = audio_to_log_mel(
        audio
    )

    features = torch.from_numpy(
        features
    )

    features = features.unsqueeze(0)
    features = features.unsqueeze(0)

    features = features.to(device)

    logits = model(features)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    predicted_class = torch.argmax(
        probabilities
    ).item()

    confidence = probabilities[
        predicted_class
    ].item()

    natural_probability = probabilities[0].item()
    synthetic_probability = probabilities[1].item()

    print("\n==============================")
    print("AUDIO PREDICTION")
    print("==============================")

    print(
        f"File: {audio_path}"
    )

    print(
        f"Prediction: {CLASS_NAMES[predicted_class]}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print(
        f"Natural speech probability: "
        f"{natural_probability * 100:.2f}%"
    )

    print(
        f"Synthetic speech probability: "
        f"{synthetic_probability * 100:.2f}%"
    )

    return {
        "class": CLASS_NAMES[predicted_class],
        "confidence": confidence,
        "natural_probability": natural_probability,
        "synthetic_probability": synthetic_probability,
    }


# ============================================================
# TRAIN COMMAND
# ============================================================

def train_command(
    dataset_path,
    model_path,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    samples = discover_dataset(
        dataset_path
    )

    print(
        f"Total audio files: {len(samples)}"
    )

    train_samples, validation_samples, test_samples = (
        split_dataset(samples)
    )

    print(
        f"Training samples:   {len(train_samples)}"
    )

    print(
        f"Validation samples: {len(validation_samples)}"
    )

    print(
        f"Test samples:       {len(test_samples)}"
    )

    train_dataset = VoiceDataset(
        train_samples,
        training=True,
    )

    validation_dataset = VoiceDataset(
        validation_samples,
        training=False,
    )

    test_dataset = VoiceDataset(
        test_samples,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = VoiceAuthenticityCNN()

    model.to(device)

    class_weights = calculate_class_weights(
        train_samples
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_validation_loss = float("inf")

    patience = 5
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        validation_loss = evaluate_loss(
            model,
            validation_loader,
            criterion,
            device,
        )

        scheduler.step(validation_loss)

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            epochs_without_improvement = 0

            save_model(
                model,
                model_path,
            )

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:

            print(
                "\nEarly stopping."
            )

            break

    print(
        "\nLoading best model..."
    )

    model = load_model(
        model_path,
        device,
    )

    evaluate_model(
        model,
        test_loader,
        device,
    )


# ============================================================
# PREDICT COMMAND
# ============================================================

def predict_command(
    model_path,
    audio_path,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = load_model(
        model_path,
        device,
    )

    predict_audio(
        model,
        audio_path,
        device,
    )


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AI-generated vs natural speech "
            "classification model."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = subparsers.add_parser(
        "train"
    )

    train_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset directory.",
    )

    train_parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="Output model path.",
    )

    predict_parser = subparsers.add_parser(
        "predict"
    )

    predict_parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="Path to trained model.",
    )

    predict_parser.add_argument(
        "--audio",
        required=True,
        help="Path to audio file.",
    )

    args = parser.parse_args()

    set_seed(42)

    if args.command == "train":

        train_command(
            args.dataset,
            args.model,
        )

    elif args.command == "predict":

        predict_command(
            args.model,
            args.audio,
        )


if __name__ == "__main__":
    main()