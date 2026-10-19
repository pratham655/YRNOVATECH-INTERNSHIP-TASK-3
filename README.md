# AI Hand Gesture Recognition

A real-time computer-vision project that uses a webcam to recognize four basic hand gestures and display the predicted label on screen.

## Supported gestures

- Open Hand
- Fist
- Thumbs Up
- Victory / Peace

## Technologies

- Python
- OpenCV for webcam capture and live display
- MediaPipe Hands for hand landmark detection and tracking

This version uses a pretrained hand-landmark model and rule-based gesture classification. The gesture classifier itself is not a separately trained machine-learning model.

## Project status

The application has been run with a webcam, and screenshots have been captured for all four gestures. The code and demo video should be added to this repository by the author before final submission.

## Requirements

Use a Python version compatible with the MediaPipe package. For code that uses the legacy `mp.solutions.hands` API, Python 3.12 and `mediapipe==0.10.21` are recommended.

## Setup on Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python gesture_recognition.py
```

Press **Q** to exit the webcam window.

## How it works

1. OpenCV captures frames from the webcam.
2. Frames are mirrored and converted to RGB.
3. MediaPipe detects hand landmarks.
4. Python logic estimates the gesture from landmark positions.
5. OpenCV overlays the predicted gesture and the hand skeleton on the frame.

## Repository structure

```text
.
├── gesture_recognition.py
├── requirements.txt
├── README.md
├── screenshots/
│   ├── open-hand.png
│   ├── victory-peace.png
│   ├── thumbs-up.png
│   └── fist.png
└── demo/
    └── gesture-demo.mp4
```

Add the actual script, screenshots, and recorded demo to the paths shown above before treating this structure as complete.

## Limitations

- Classification relies on geometric rules and can be sensitive to orientation, lighting, camera angle, and partial visibility.
- No formal accuracy benchmark has been performed.
- A working webcam is required.

## Future improvements

- Improve classification across different hand orientations.
- Add prediction smoothing to reduce label flicker.
- Support multiple hands.
- Extend the system for gesture-based mouse movement and scrolling.

## Author

Pratham S

## Disclaimer

An educational computer-vision project prepared for hands-on learning and demonstration.
