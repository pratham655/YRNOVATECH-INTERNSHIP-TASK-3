# AI Hand Gesture Recognition

A real-time computer-vision project that uses a webcam to recognize four basic hand gestures and display the predicted label on screen.

## Supported gestures

- Open Hand
- Fist
- Thumbs Up
- Victory / Peace

## Technologies

- **Python** — application logic
- **OpenCV** — webcam capture, image processing, and display
- **MediaPipe Hands** — pretrained hand-landmark detection and tracking

This project uses a pretrained hand-landmark model and rule-based gesture classification. The gesture classifier itself is not a separately trained machine-learning model.

## Screenshots

### Open Hand
![Open Hand detection](screenshots/Screenshot%202026-10-19%20132102.png)

### Victory / Peace
![Victory / Peace detection](screenshots/Screenshot%202026-10-19%20132129.png)

### Thumbs Up
![Thumbs Up detection](screenshots/Screenshot%202026-10-19%20132204.png)

### Fist
![Fist detection](screenshots/Screenshot%202026-10-19%20132216.png)

## Demo Video

[Watch or download the project demo](demo/GESTUREDEMO.mp4).

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
5. OpenCV overlays the predicted gesture and hand skeleton on the frame.

## Repository structure

```text
.
├── gesture_recognition.py
├── requirements.txt
├── README.md
├── screenshots/
│   ├── Screenshot 2026-10-19 132102.png
│   ├── Screenshot 2026-10-19 132129.png
│   ├── Screenshot 2026-10-19 132204.png
│   └── Screenshot 2026-10-19 132216.png
└── demo/
    └── GESTUREDEMO.mp4
```

## Limitations

- Classification relies on geometric rules and can be sensitive to orientation, lighting, camera angle, or partial visibility.
- No formal accuracy benchmark has been performed.
- A working webcam is required.

## Future Improvements

- Improve classification across different hand orientations.
- Add prediction smoothing to reduce label flicker.
- Support multiple hands.
- Extend the system into a gesture-based computer-control application.

## Author

Pratham S

## Disclaimer

An educational computer-vision project prepared for hands-on learning and demonstration.
