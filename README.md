# Hand Gesture Recognition System

**Team Members:** Amritpal Singh, Hank(Bohan) Fang, Chaoxiang Zhang, Sophie Koehler

## Project Overview

This project implements a live hand gesture recognition system that can run on a standard device with a webcam. The implementation is based on the paper "On-device Real-time Custom Hand Gesture Recognition" from IVCC 2023.

## Features

- **Live Video Processing**: Real-time hand gesture recognition from webcam feed
- **Pre-recorded Video Support**: Process video files for gesture recognition
- **Hand Tracking Visualization**: Display dots and lines showing live tracking of hands and fingers
- **On-device Processing**: Runs entirely on local hardware without cloud dependencies

## Supported Gestures

The system recognizes 10 hand gestures:

1. **One** - Single finger extended
2. **Peace** - Peace sign (index and middle finger up)
3. **Fist** - All fingers closed
4. **Call** - Phone call gesture
5. **OK** - Thumb and index finger form circle
6. **Like** - Thumbs up
7. **Point** - Index finger extended, others closed
8. **Rock** - Index and pinky extended (rock on)
9. **Three Gun** - Thumb, index, and middle finger extended
10. **Four** - All fingers except thumb extended

## Project Structure

```
Gesture_detector/
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
git clone https://github.com/amrit3928/Gesture_detector.git
cd Gesture_detector
```

2. **Important**: This project requires Python 3.11 (MediaPipe doesn't support Python 3.13+)

3. Install dependencies:
```bash
py -3.11 -m pip install -r requirements.txt
```

Or if Python 3.11 is your default:
```bash
pip install -r requirements.txt
```

## Usage

### Live Webcam Processing
```bash
py -3.11 src/main.py --mode live --model data/models/gesture_model.h5
```

### Process Pre-recorded Video
```bash
py -3.11 src/main.py --mode video --input path/to/video.mp4 --model data/models/gesture_model.h5
```

### Train Model
```bash
py -3.11 src/main.py --mode train --data data/processed/
```

**Note**: The trained model and training data are included in the repository, so you can use the system immediately without training.

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

