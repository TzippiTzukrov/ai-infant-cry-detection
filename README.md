# Cry-Guard: AI Infant Cry Detection

> Real-time infant cry detection from a live microphone feed, with instant alerts to a parent's browser and phone.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

## Overview

Cry-Guard is a baby-monitor system that listens continuously through a microphone, classifies incoming audio as *crying* or *background* using a convolutional neural network, and notifies a parent when a sustained cry is detected. Alerts are delivered to a live web dashboard and, optionally, by SMS.

To avoid false alarms, the system alerts only after a cry persists beyond a configurable threshold, rather than reacting to every brief noise.

This project was developed as a final project in the Artificial Intelligence track (2024–2025) by:

- Feigy Rosenfeld
- Tzippi Tzukrov
- Avital Cohen

## Screenshots

| Main dashboard | Cry detected | Baby calmed |
|:---:|:---:|:---:|
| ![Main dashboard](screenshots/the_system.png) | ![Active alert](screenshots/alarm.png) | ![Alert cleared](screenshots/finish_alarm.png) |

## Features

- **Real-time listening:** continuous microphone capture, analyzed once per second.
- **Custom CNN:** a deep learning model trained from scratch on a combined audio dataset.
- **Live waveform:** real-time visualization of sound level in the dashboard.
- **Sustained-cry alerts:** an alert is raised only after continuous crying, not on isolated sounds.
- **SMS notifications:** optional alerts to a parent's phone through Twilio.
- **Night mode:** the alert threshold is 5 seconds at night and 10 seconds during the day.
- **Event history:** every cry episode is logged with a timestamp.
- **Priority score:** each episode receives an urgency score from 0 to 100 based on duration, continuity, frequency, and model confidence.

## How It Works

```
Microphone (22,050 Hz)
        ↓
Rolling buffer (2 s window, 1 s step)
        ↓
Mel spectrogram (128 × T)
        ↓
CNN classifier (4 × Conv2D)
        ↓
Prediction: crying / background
        ↓
Episode tracking + priority scoring
        ↓
WebSocket → React dashboard  +  optional SMS
```

### Model Architecture

```
Input(128, T, 1)
Conv2D(32)  → BatchNorm → MaxPool → Dropout(0.25)
Conv2D(64)  → BatchNorm → MaxPool → Dropout(0.25)
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
Conv2D(128) → BatchNorm → GlobalAvgPool
Dense(256)  → Dropout(0.5)
Dense(2)    → Softmax → [crying, background]
```

### Backend Components

| Module | Responsibility |
|--------|----------------|
| `main.py` | FastAPI app, audio pipeline, WebSocket broadcast to all connected clients |
| `episode_tracker.py` | Tracks the lifecycle of each cry episode (start, continuation, end) |
| `night_mode.py` | Switches the alert threshold between day and night settings |
| `priority_scorer.py` | Computes the 0–100 urgency score for an episode |

## Training Data

Audio was converted to 22,050 Hz mono, cut into 2-second windows with a 1-second stride, and transformed into 128-band mel spectrograms.

| Source | Class | Size |
|--------|-------|------|
| Donate-a-Cry | Crying | 457 clips |
| ESC-50 (`crying_baby`) | Crying | 40 clips |
| AudioSet | Crying | 8 clips |
| ESC-50 (other 49 categories) | Background | 1,960 clips |
| AudioSet (speech, music, vacuum, dog, door, and others) | Background | 427 clips |

After windowing, the dataset contains 11,540 training windows and 2,887 test windows.

**Data augmentation (training set, crying class only):** Gaussian noise, time shift, amplitude scaling (0.7–1.3×), and SpecAugment (frequency and time masking). Together with class weighting, this balances the two classes during training (20,308 training samples after augmentation).

**Training setup:** Adam optimizer, categorical cross-entropy, early stopping on validation loss, and learning-rate reduction on plateau. The best checkpoint (epoch 9) was kept.

## Evaluation

The model was evaluated on a held-out test set: 20% of the audio files, split **by file before windowing** (random seed 42), so windows from the same recording never appear in both training and test data. The test set contains 2,887 windows (537 crying, 2,350 background).

| Class | Precision | Recall | F1-score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| Crying | 0.98 | 0.98 | 0.98 | 537 |
| Background | 0.99 | 1.00 | 1.00 | 2,350 |

- **Overall accuracy:** 99.24%
- **ROC AUC (crying):** 0.997

**Limitations:** the test split was also used for early stopping and checkpoint selection, and test clips come from the same public datasets as the training data. These figures should therefore be read as an upper estimate; performance with real household microphones and room acoustics has not been formally measured.

## Tech Stack

- **Backend & ML:** Python, TensorFlow/Keras, librosa, FastAPI, WebSocket, sounddevice, Twilio
- **Frontend:** React 18, JavaScript, Canvas API, WebSocket
- **Training:** Google Colab (GPU), scikit-learn, NumPy, Matplotlib

## Project Structure

```
ai-infant-cry-detection/
├── backend/
│   ├── main.py
│   ├── episode_tracker.py
│   ├── night_mode.py
│   └── priority_scorer.py
├── frontend/
│   └── src/
│       ├── App.js
│       └── App.css
├── models/
│   ├── cryguard_model.keras     # Trained model
│   └── norm_stats.json          # Normalization statistics
├── audio/                       # Sample clips for testing
├── notebooks/                   # Model training and evaluation
├── screenshots/
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A connected microphone
- A Twilio account (optional, for SMS alerts)

### 1. Clone and install

```bash
git clone https://github.com/TzippiTzukrov/ai-infant-cry-detection.git
cd ai-infant-cry-detection

pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
API_KEY=your_api_key
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_FROM_NUMBER=+1234567890
```

Twilio is optional. Without it, the system runs fully except for SMS delivery.

### 3. Start the backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8080
```

### 4. Start the frontend

```bash
cd frontend
npm install
npm start
```

Open `http://localhost:3000` in your browser.

## Usage

1. Press **▶** to start listening.
2. The waveform shows the live sound level.
3. When sustained crying is detected, a red alert appears and an SMS is sent (if configured).
4. When the baby calms down, the alert clears and a green confirmation is shown.

## Built From Scratch

The following components were designed and implemented as part of this project:

- The full CNN architecture and training pipeline
- Priority scoring (0–100) based on duration, continuity, frequency, and confidence
- Episode tracking for the cry lifecycle
- Night/day mode threshold controller
- FastAPI backend with WebSocket broadcasting to all clients

## License

This project was developed for educational purposes.
