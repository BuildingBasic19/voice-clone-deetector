# voice-clone-deetector
# Voice Snipper

## AI-Generated / Synthetic Voice Detection

Voice Snipper is a Python-based machine-learning system designed to classify speech audio into two categories:

* **NATURAL HUMAN** — naturally recorded human speech
* **AI / SYNTHETIC** — speech generated or manipulated using synthetic/AI voice technology

The project uses a convolutional neural network (CNN) trained on Mel-spectrogram representations of speech.

> **Important:** Voice Snipper is a classifier, not a forensic proof system. Its prediction represents what the model learned from its training data and should not be treated as definitive proof that an audio recording is human-made or AI-generated.

---

## Features

* Supports WAV, MP3, FLAC and other common audio formats supported by Librosa.
* Converts audio to mono and resamples it to 16 kHz.
* Normalizes and standardizes audio.
* Handles audio shorter or longer than the model's target duration.
* Uses Mel-spectrogram features.
* CNN-based classification.
* Data augmentation during training.
* Class balancing.
* Training, validation and testing.
* Early stopping.
* Learning-rate scheduling.
* Accuracy measurement.
* Precision measurement.
* Recall measurement.
* F1-score measurement.
* ROC-AUC measurement.
* Confusion matrix.
* Model checkpoint saving.
* Single-file Python implementation.
* CPU training supported.
* GPU training supported when PyTorch detects CUDA.

---

## Project Structure

```text
voice_ai_detector/
│
├── voice snipper.py
│
├── dataset/
│   ├── real/
│   │   └── audio files...
│   │
│   └── fake/
│       └── audio files...
│
├── voice_authenticity_model.pt
│
└── voice_authenticity_model.metrics.json
```

The dataset folders can also contain additional subfolders because the program searches recursively.

---

# Installation

Install the required Python packages:

```bash
pip install numpy librosa torch scikit-learn
```

If you are using a CPU-only computer, the normal PyTorch installation is sufficient.

---

# Dataset

The training dataset should contain two classes.

### Real / Natural

```text
dataset/
└── real/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

### Fake / Synthetic

```text
dataset/
└── fake/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

The program also recognizes alternative folder names such as:

```text
natural/
synthetic/
```

and other common class names implemented in the dataset loader.

---

# Training

Open the terminal in the project directory:

```bash
cd "C:\Users\anant_\.vscode\Python\voice_ai_detector"
```

Then run:

```bash
python "voice snipper.py" train --dataset "dataset"
```

The program will:

1. Find the audio files.
2. Separate natural and synthetic samples.
3. Split the data into training, validation and testing sets.
4. Apply audio augmentation to training samples.
5. Convert audio into Mel-spectrograms.
6. Train the CNN.
7. Monitor validation performance.
8. Save the best model.
9. Evaluate the model on the test set.

The trained model will normally be saved as:

```text
voice_authenticity_model.pt
```

---

# Using an Existing `.pt` Model

If you already have:

```text
voice_authenticity_model.pt
```

you **do not need the training dataset to perform predictions**.

The `.pt` file contains the trained neural-network parameters required by the program.

For example:

```text
voice_ai_detector/
│
├── voice snipper.py
└── voice_authenticity_model.pt
```

is enough for prediction.

You can move the `.pt` file to another computer and use it there, provided the same model code and required Python libraries are available.

---

# Prediction

Put the audio you want to analyze anywhere you like.

For example:

```text
C:\Users\anant_\Desktop\test.wav
```

Then run:

```bash
python "voice snipper.py" predict --model "voice_authenticity_model.pt" --audio "C:\Users\anant_\Desktop\test.wav"
```

The program will produce something similar to:

```text
============================================================
VOICE AUTHENTICITY RESULT
============================================================
Prediction          : AI / SYNTHETIC
Confidence          : 91.43%
Natural probability : 8.57%
Synthetic probability: 91.43%
============================================================
```

---

# Do I Need the Dataset Again?

### For prediction

**No.**

```text
voice snipper.py
        +
voice_authenticity_model.pt
        +
test audio
        ↓
    prediction
```

### For training again

**Yes.**

```text
voice snipper.py
        +
dataset/
        ↓
     training
        ↓
voice_authenticity_model.pt
```

### For improving the model

**Yes.**

If you want to retrain the model with additional human voices, additional AI generators, or a larger dataset, you need the training data.

---

# What Is Inside the `.pt` File?

The model checkpoint contains the trained neural-network parameters and model configuration needed to reconstruct the CNN.

It is **not the original dataset**.

Think of it like:

```text
Dataset
   ↓
Training
   ↓
Learned parameters
   ↓
.pt model
```

After training, the model has learned statistical patterns from the training audio.

You don't need to keep the original audio files just to run inference.

---

# Model Architecture

Voice Snipper uses a convolutional neural network.

The general architecture is:

```text
Audio
  ↓
Resampling
  ↓
Normalization
  ↓
4-second segment
  ↓
Mel Spectrogram
  ↓
CNN
  ↓
Global Average Pooling
  ↓
Fully Connected Layers
  ↓
2 Classes
  ↓
Natural / Synthetic
```

---

# Audio Processing

The model uses:

* Sample rate: **16 kHz**
* Mono audio
* Target duration: **4 seconds**
* Mel bands: **96**
* FFT size: **1024**
* Hop length: **256**

Longer audio is cropped.

Shorter audio is padded.

During training, small augmentations may be applied, including:

* gain changes
* small amounts of noise
* slight time stretching

---

# Evaluation

After training, the model reports:

### Accuracy

Percentage of test samples classified correctly.

### Precision

How often samples predicted as synthetic were actually synthetic in the test set.

### Recall

How many synthetic samples were detected.

### F1 Score

A combined measure of precision and recall.

### ROC-AUC

Measures how well the model separates the two classes across different probability thresholds.

### Confusion Matrix

Shows:

```text
                    Predicted
                 Natural  Synthetic

Actual Natural      TN        FP

Actual Synthetic   FN        TP
```

---

# Important Limitations

Voice Snipper should not be considered a perfect AI-voice detector.

Performance depends heavily on:

* training dataset
* recording quality
* microphones
* compression
* background noise
* speakers
* languages
* accents
* AI generation systems
* voice-cloning systems
* unseen models

A model trained on certain AI generators may perform poorly against an AI generator that was not represented in its training data.

Similarly, naturally recorded audio that has been heavily compressed or processed may sometimes be classified incorrectly.

The confidence percentage is the model's probability estimate, **not a guaranteed probability that the audio is actually AI-generated**.

---

# Human Imitation

Voice Snipper is primarily designed to distinguish:

```text
Natural recording
        vs
AI / synthetic recording
```

It should not be interpreted as a reliable detector of whether a human speaker is intentionally imitating another person's voice.

Human imitation is a different problem and generally requires additional information and specialized modeling.

---

# CPU and GPU

The program automatically checks whether CUDA is available.

If a compatible NVIDIA GPU is available:

```text
Device: cuda
```

Otherwise:

```text
Device: cpu
```

CPU training is supported, although training will generally take longer.

---

# Model File

Default model:

```text
voice_authenticity_model.pt
```

You can specify another filename during training:

```bash
python "voice snipper.py" train --dataset "dataset" --model "my_model.pt"
```

Then use it for prediction:

```bash
python "voice snipper.py" predict --model "my_model.pt" --audio "test.wav"
```

---

# Metrics File

After training, the program also saves:

```text
voice_authenticity_model.metrics.json
```

This contains the final evaluation metrics in JSON format.

---

# Recommended Workflow

### First training

```text
Dataset
   ↓
Train
   ↓
Validation
   ↓
Best model
   ↓
Test evaluation
```

### Later predictions

```text
.pt model
   +
new audio
   ↓
Prediction
```

You can therefore archive the dataset after training if you only want to use the trained model for inference.

---

# License

Add your preferred license here before publishing the project.

---

# Disclaimer

Voice Snipper is an experimental machine-learning project for research and educational purposes. It should not be used as the sole basis for legal, financial, employment, academic, or other high-impact decisions about the authenticity of an audio recording.
