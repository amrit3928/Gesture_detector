# Hand Gesture Recognition System

**Team Members:** Amritpal Singh, Hank(Bohan) Fang, Chaoxiang Zhang, Sophie Koehler

## Project Overview

This project implements a live hand gesture recognition system that can run on a standard device with a webcam. The implementation is based on the paper "On-device Real-time Custom Hand Gesture Recognition" from IVCC 2023.

## Features

- **Live Video Processing**: Real-time hand gesture recognition from webcam feed
- **Pre-recorded Video Support**: Process video files for gesture recognition
- **Hand Tracking Visualization**: Display dots and lines showing live tracking of hands and fingers
- **On-device Processing**: Runs entirely on local hardware without cloud dependencies

## Project Structure

```
CSE_Final/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Main entry point
│   ├── hand_detector.py        # Hand landmark detection
│   ├── gesture_classifier.py   # Gesture recognition model
│   ├── video_processor.py      # Video processing utilities
│   └── config.py               # Configuration settings
├── utils/
│   ├── __init__.py
│   ├── visualization.py        # Visualization utilities
│   └── data_utils.py           # Data processing utilities
├── data/
│   ├── raw/                    # Raw video/image data
│   ├── processed/              # Processed training data
│   └── models/                 # Saved model files
├── tests/
│   ├── __init__.py
│   └── test_hand_detector.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CSE_Final
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Live Webcam Processing
```bash
python src/main.py --mode live
```

### Process Pre-recorded Video
```bash
python src/main.py --mode video --input path/to/video.mp4
```

### Train Model
```bash
python src/main.py --mode train --data data/processed/
```

## Milestones

- **Week 1**: Project setup, environment configuration, paper study
- **Week 2**: Image processing, landmark extraction, model training
- **Week 3**: Live video integration, model optimization
- **Week 4**: Testing, paper writing, presentation preparation

## Dependencies

See `requirements.txt` for full list of dependencies.

## License

[Add your license here]

## References

- "On-device Real-time Custom Hand Gesture Recognition" - IVCC 2023

