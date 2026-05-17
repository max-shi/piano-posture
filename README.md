# Piano Posture Analyzer

Real-time piano hand posture analysis using computer vision and MediaPipe hand tracking.

## Overview

This system analyzes a pianist's right-hand posture using a single RGB webcam. It extracts geometric features from hand landmarks and provides real-time visual feedback with a posture score (0-100).

### Features Analyzed

- **Finger posture** — PIP joint angles classified as curved/neutral/flat/collapsed
- **Wrist height** — Position relative to MCP line (low/neutral/high)
- **Hand arch** — Palm height above keyboard

## Installation

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the main application
python src/main.py

# With custom camera settings
python src/main.py --camera 0 --width 640 --height 480
```

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `c` | Calibrate keyboard line (click to set) |
| `r` | Reset score history |
| `s` | Save screenshot |

## Camera Setup

Position the camera:
- In front of the pianist, above hand height
- Offset towards the right so the right hand is near center
- Tilted downwards to capture the back of the hand and wrist
- Keyboard should be visible in frame

## Project Structure

```
piano-posture/
├── src/
│   ├── main.py              # Main application
│   ├── hand_tracker.py      # MediaPipe hand landmark detection
│   ├── posture_analyzer.py  # Geometric feature extraction & scoring
│   └── visualizer.py        # Visual feedback overlays
├── data/                    # Test recordings (optional)
├── results/                 # Screenshots and outputs
├── requirements.txt
└── README.md
```

## Posture Scoring

The system calculates a score from 0-100 based on:

- **Finger postures (60 points)** — 12 points per finger
  - Curved: 12 pts (ideal)
  - Neutral: 8 pts
  - Collapsed: 4 pts
  - Flat: 0 pts

- **Wrist position (20 points)**
  - Neutral: 20 pts
  - High: 10 pts
  - Low: 5 pts

- **Hand arch (20 points)**
  - Good: 20 pts
  - Excessive: 10 pts
  - Collapsed: 5 pts

## Development

Run individual modules for testing:

```bash
# Test hand tracking only
python src/hand_tracker.py
```
