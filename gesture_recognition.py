
import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def finger_is_up(landmarks, tip_id, pip_id):
    """Check whether a non-thumb finger is extended."""
    return landmarks[tip_id].y < landmarks[pip_id].y


def thumb_is_extended(landmarks):
    """Estimate whether the thumb is extended sideways."""
    tip = landmarks[4]
    ip = landmarks[3]
    mcp = landmarks[2]

    tip_distance = abs(tip.x - mcp.x)
    ip_distance = abs(ip.x - mcp.x)

    return tip_distance > ip_distance * 1.15


def recognize_gesture(landmarks):
    """
    Classify four basic hand gestures using hand landmarks.
    Returns a gesture label or Unknown.
    """
    index_up = finger_is_up(landmarks, 8, 6)
    middle_up = finger_is_up(landmarks, 12, 10)
    ring_up = finger_is_up(landmarks, 16, 14)
    pinky_up = finger_is_up(landmarks, 20, 18)
    thumb_up = thumb_is_extended(landmarks)

    # Thumbs Up: upright thumb and four curled fingers
    thumb_tip_above_ip = landmarks[4].y < landmarks[3].y
    other_fingers_curled = not any(
        [index_up, middle_up, ring_up, pinky_up]
    )

    if thumb_tip_above_ip and other_fingers_curled:
        return "Thumbs Up"

    # Victory / Peace: index and middle fingers extended
    if (
        index_up
        and middle_up
        and not ring_up
        and not pinky_up
    ):
        return "Victory / Peace"

    # Open Hand: all four fingers and thumb extended
    if index_up and middle_up and ring_up and pinky_up and thumb_up:
        return "Open Hand"

    # Fist: all four fingers curled
    if not any([index_up, middle_up, ring_up, pinky_up]):
        return "Fist"

    return "Unknown"


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        print("Check camera permissions or try camera index 1.")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:

        print("AI Hand Gesture Recognition started.")
        print("Press Q to quit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read from webcam.")
                break

            # Mirror the image for a natural webcam experience
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            gesture = "No Hand Detected"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )

                    gesture = recognize_gesture(
                        hand_landmarks.landmark
                    )

                cv2.rectangle(
                    frame, (10, 10), (440, 75),
                    (35, 35, 35), -1
                )
                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (22, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame,
                    "Show your hand to the camera",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("AI Hand Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
