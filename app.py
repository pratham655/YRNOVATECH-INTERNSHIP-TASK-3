
import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="✋",
    layout="centered"
)

st.title("✋ AI Hand Gesture Recognition")
st.write(
    "Upload an image of one hand to detect basic gestures. "
    "For best results, use a clear, front-facing image."
)

mp_hands = mp.solutions.hands

def classify_gesture(landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    fingers_up = [
        landmarks[tip].y < landmarks[pip].y
        for tip, pip in zip(finger_tips, finger_pips)
    ]

    thumb_up = landmarks[4].x < landmarks[3].x
    count = sum(fingers_up)

    if count == 4 and thumb_up:
        return "Open Hand"
    if count == 2 and fingers_up[0] and fingers_up[1]:
        return "Victory / Peace"
    if count == 0 and not thumb_up:
        return "Fist"
    if thumb_up and count == 0:
        return "Thumbs Up"
    return "Unknown Gesture"

uploaded = st.file_uploader(
    "Upload a hand image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    image_array = np.array(image)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(image_array)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        gesture = classify_gesture(hand.landmark)

        st.success(f"Detected gesture: {gesture}")

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        width, height = image.size

        connections = mp_hands.HAND_CONNECTIONS
        points = [
            (int(point.x * width), int(point.y * height))
            for point in hand.landmark
        ]

        for start, end in connections:
            draw.line([points[start], points[end]], fill="lime", width=4)

        for x, y in points:
            draw.ellipse((x-4, y-4, x+4, y+4), fill="red")

        st.image(
            annotated,
            caption="Detected hand landmarks",
            use_container_width=True
        )
        st.caption(
            "This is a rule-based gesture classifier using MediaPipe "
            "hand landmarks, not a separately trained ML classifier."
        )
    else:
        st.warning(
            "No hand detected. Try a clear image with your entire hand visible."
        )
