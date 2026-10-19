
import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
import urllib.request
import os

st.set_page_config(
    page_title="AI Hand Gesture Recognition",
    page_icon="✋",
    layout="centered"
)

st.title("✋ AI Hand Gesture Recognition")
st.write(
    "Upload an image of one hand to detect basic gestures. "
    "Use a clear image with the entire hand visible."
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

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return HandLandmarker.create_from_options(options)


def classify_gesture(landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    fingers_up = [
        landmarks[tip].y < landmarks[pip].y
        for tip, pip in zip(finger_tips, finger_pips)
    ]

    # Thumb direction depends on which hand and orientation are shown.
    thumb_extended = (
        abs(landmarks[4].x - landmarks[2].x) >
        abs(landmarks[3].x - landmarks[2].x)
    )

    count = sum(fingers_up)

    if count == 4 and thumb_extended:
        return "Open Hand"
    if count == 2 and fingers_up[0] and fingers_up[1]:
        return "Victory / Peace"
    if count == 0 and not thumb_extended:
        return "Fist"
    if thumb_extended and count == 0:
        return "Thumbs Up"

    return "Unknown Gesture"


uploaded = st.file_uploader(
    "Upload a hand image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        image_array = np.asarray(image, dtype=np.uint8)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_array
        )

        with st.spinner("Detecting your hand..."):
            detector = load_hand_detector()
            results = detector.detect(mp_image)

        if results.hand_landmarks:
            hand = results.hand_landmarks[0]
            gesture = classify_gesture(hand)

            st.success(f"Detected gesture: {gesture}")

            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)
            width, height = image.size

            connections = (
                mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
            )

            points = [
                (int(point.x * width), int(point.y * height))
                for point in hand
            ]

            for connection in connections:
                start = connection.start
                end = connection.end
                draw.line(
                    [points[start], points[end]],
                    fill="lime",
                    width=4
                )

            for x, y in points:
                draw.ellipse(
                    (x - 4, y - 4, x + 4, y + 4),
                    fill="red"
                )

            st.image(
                annotated,
                caption="Detected hand landmarks",
                use_container_width=True
            )

            st.caption(
                "Gesture classification uses rules applied to MediaPipe "
                "hand landmarks; it is not a separately trained classifier."
            )

        else:
            st.warning(
                "No hand detected. Try a clear, well-lit image with "
                "your entire hand visible."
            )

    except Exception as error:
        st.error("The image could not be processed.")
        st.caption(f"Technical details: {error}")
