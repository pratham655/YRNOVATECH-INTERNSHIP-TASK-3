
import os
import urllib.request

import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="AI Hand Gesture Recognition",
    page_icon="✋",
    layout="centered",
)

st.title("✋ AI Hand Gesture Recognition")
st.write(
    "Upload an image of one hand to recognize basic gestures: "
    "Open Hand, Victory / Peace, Fist, and Thumbs Up."
)

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)


@st.cache_resource
def load_hand_detector():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    return mp.tasks.vision.HandLandmarker.create_from_options(
        options
    )


def distance(a, b):
    return (
        (a.x - b.x) ** 2
        + (a.y - b.y) ** 2
        + (a.z - b.z) ** 2
    ) ** 0.5


def classify_gesture(landmarks):
    # Each finger: MCP, PIP, DIP, fingertip
    finger_data = [
        (5, 6, 7, 8),       # Index
        (9, 10, 11, 12),    # Middle
        (13, 14, 15, 16),   # Ring
        (17, 18, 19, 20),   # Little
    ]

    fingers_up = []

    for mcp, pip, dip, tip in finger_data:
        # Compare fingertip distance from the wrist with
        # the middle joint distance. Best for upright hands.
        tip_distance = distance(landmarks[tip], landmarks[0])
        joint_distance = distance(landmarks[pip], landmarks[0])

        fingers_up.append(tip_distance > joint_distance * 1.12)

    index, middle, ring, little = fingers_up

    # Thumb extension relative to its base.
    thumb_extended = (
        distance(landmarks[4], landmarks[2])
        > distance(landmarks[3], landmarks[2]) * 1.15
    )

    if all(fingers_up) and thumb_extended:
        return "Open Hand"

    if index and middle and not ring and not little:
        return "Victory / Peace"

    if not any(fingers_up) and not thumb_extended:
        return "Fist"

    # Thumb pointing upward in an upright image.
    if (
        thumb_extended
        and not any(fingers_up)
        and landmarks[4].y < landmarks[3].y
    ):
        return "Thumbs Up"

    return "Unknown Gesture"


def draw_landmarks(image, landmarks):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = image.size

    points = [
        (int(point.x * width), int(point.y * height))
        for point in landmarks
    ]

    connections = (
        mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    )

    for connection in connections:
        start = connection.start
        end = connection.end

        draw.line(
            [points[start], points[end]],
            fill="lime",
            width=max(2, width // 250),
        )

    radius = max(3, width // 150)

    for x, y in points:
        draw.ellipse(
            (
                x - radius,
                y - radius,
                x + radius,
                y + radius,
            ),
            fill="red",
        )

    return annotated


uploaded = st.file_uploader(
    "Choose a hand image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(
            image,
            caption="Original image",
            width="stretch",
        )

        image_array = np.asarray(image, dtype=np.uint8)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_array,
        )

        with st.spinner("Detecting hand and landmarks..."):
            detector = load_hand_detector()
            results = detector.detect(mp_image)

        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            gesture = classify_gesture(landmarks)

            st.success(f"Predicted gesture: {gesture}")

            annotated = draw_landmarks(image, landmarks)

            st.subheader("Hand Landmark Visualization")
            st.image(
                annotated,
                caption="Detected hand landmarks",
                width="stretch",
            )

            with st.expander("Show classification details"):
                st.write(
                    "Gesture recognition uses geometric rules "
                    "applied to MediaPipe hand landmarks. "
                    "It is not a separately trained classifier."
                )

        else:
            st.warning(
                "No hand detected. Try a well-lit image with "
                "one complete hand clearly visible."
            )

    except Exception as error:
        st.error("Unable to process this image.")
        st.code(f"{type(error).__name__}: {error}")

        st.info(
            "If the error mentions downloading the model, "
            "check whether the hosting environment can access "
            "the MediaPipe model URL."
        )
