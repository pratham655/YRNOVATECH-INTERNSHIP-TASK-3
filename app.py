import os
import urllib.request

import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

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
st.caption(
    "For best results, keep the hand clearly visible. Detection and gesture "
    "classification are separate steps: landmarks can be detected even when "
    "a gesture is uncertain."
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
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def distance(a, b):
    return (
        (a.x - b.x) ** 2
        + (a.y - b.y) ** 2
        + (a.z - b.z) ** 2
    ) ** 0.5


def classify_gesture(landmarks):
    """
    Lightweight heuristic classifier for upright or near-upright hands.
    This is not a trained gesture classifier and can be uncertain for rotated
    hands, unusual camera angles, or partially occluded fingers.
    """
    # Index, middle, ring and little fingers: PIP joint and fingertip.
    finger_data = [
        (6, 8),    # Index
        (10, 12), # Middle
        (14, 16), # Ring
        (18, 20), # Little
    ]

    fingers_up = []
    for pip, tip in finger_data:
        tip_from_wrist = distance(landmarks[tip], landmarks[0])
        pip_from_wrist = distance(landmarks[pip], landmarks[0])

        # A less strict ratio reduces false "Unknown" results when fingers
        # are extended diagonally rather than straight upward.
        fingers_up.append(tip_from_wrist > pip_from_wrist * 1.05)

    index, middle, ring, little = fingers_up

    # Thumb extension: compare the tip's distance from the thumb base with
    # the IP joint's distance from that same base.
    thumb_extended = (
        distance(landmarks[4], landmarks[2])
        > distance(landmarks[3], landmarks[2]) * 1.05
    )

    # Decide the most specific gestures first.
    if (
        not any(fingers_up)
        and landmarks[4].y < landmarks[3].y
        and thumb_extended
    ):
        return "Thumbs Up"

    if all(fingers_up):
        return "Open Hand"

    if index and middle and not ring and not little:
        return "Victory / Peace"

    if not any(fingers_up):
        return "Fist"

    return "Unknown Gesture"


def draw_landmarks(image, landmarks):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = image.size
    points = [
        (int(point.x * width), int(point.y * height))
        for point in landmarks
    ]

    connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    for connection in connections:
        draw.line(
            [points[connection.start], points[connection.end]],
            fill="lime",
            width=max(2, width // 250),
        )

    radius = max(3, width // 150)
    for x, y in points:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill="red",
        )
    return annotated


def detection_variants(image):
    yield "Original image", image
    yield "Auto contrast", ImageOps.autocontrast(image)
    yield "Slightly brighter", ImageEnhance.Brightness(image).enhance(1.2)
    yield "Higher contrast", ImageEnhance.Contrast(image).enhance(1.25)
    yield "Brighter + contrast", ImageEnhance.Contrast(
        ImageEnhance.Brightness(image).enhance(1.15)
    ).enhance(1.15)


def detect_hand(detector, image):
    for variant_name, variant_image in detection_variants(image):
        image_array = np.asarray(variant_image, dtype=np.uint8)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(image_array),
        )
        results = detector.detect(mp_image)
        if results.hand_landmarks:
            return variant_image, results, variant_name
    return None, None, None


uploaded = st.file_uploader(
    "Choose a hand image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        st.subheader("Uploaded Image")
        st.image(image, caption="Original image", width="stretch")

        with st.spinner("Checking the original image and enhanced versions..."):
            detector = load_hand_detector()
            detected_image, results, variant_name = detect_hand(detector, image)

        if results and results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            gesture = classify_gesture(landmarks)
            st.success(f"Predicted gesture: {gesture}")

            if gesture == "Unknown Gesture":
                st.info(
                    "The hand was detected, but its finger pattern did not "
                    "match one of the four rule-based labels. Try a more "
                    "front-facing pose with fingers clearly separated."
                )

            if variant_name != "Original image":
                st.caption(f"Detection used: {variant_name}")

            annotated = draw_landmarks(detected_image, landmarks)
            st.subheader("Hand Landmark Visualization")
            st.image(
                annotated,
                caption="Detected hand landmarks",
                width="stretch",
            )

            with st.expander("Show classification details"):
                st.write(
                    "The four gesture labels use geometric rules applied to "
                    "MediaPipe landmarks. This is a lightweight heuristic, "
                    "not a separately trained gesture classifier."
                )
                st.write(f"Detection image variant: {variant_name}")
                st.write(f"Hands detected: {len(results.hand_landmarks)}")
        else:
            st.warning(
                "No hand detected after trying the original image and "
                "several mild contrast/brightness adjustments. Try a closer "
                "image with the full hand visible, even lighting, and a "
                "background that contrasts with the hand."
            )
            st.info(
                "These adjustments can help with some images, but cannot "
                "guarantee detection in every condition."
            )

    except Exception as error:
        st.error("Unable to process this image.")
        st.code(f"{type(error).__name__}: {error}")
        st.info(
            "If the error mentions downloading the model, check whether "
            "the hosting environment can access the MediaPipe model URL."
        )
