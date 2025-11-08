# Setup Guide

## Quick Start

1. **Clone the repository** (when available):
   ```bash
   git clone <repository-url>
   cd CSE_Final
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Test the setup**:
   ```bash
   python src/main.py --mode live
   ```

## Project Structure

```
CSE_Final/
├── src/                    # Source code
│   ├── main.py            # Entry point
│   ├── hand_detector.py   # Hand landmark detection
│   ├── gesture_classifier.py  # Gesture recognition model
│   ├── video_processor.py # Video processing
│   └── config.py          # Configuration
├── utils/                  # Utility functions
├── data/                   # Data directory (created automatically)
│   ├── raw/               # Raw data
│   ├── processed/         # Processed data
│   └── models/            # Saved models
├── tests/                  # Test files
├── requirements.txt        # Dependencies
└── README.md              # Project documentation
```

## Next Steps (Week 2)

1. **Data Collection**: Collect hand gesture images/videos
2. **Landmark Extraction**: Extract landmarks from collected data
3. **Model Training**: Train the gesture classifier
4. **Testing**: Test the model on sample data

## Development Workflow

1. Make changes to source files in `src/`
2. Test changes using `python src/main.py`
3. Run tests: `pytest tests/`
4. Commit changes to git

## Notes

- The model training functionality will be implemented in Week 2
- Make sure your webcam is connected before running live mode
- Video files should be in common formats (mp4, avi, mov)

