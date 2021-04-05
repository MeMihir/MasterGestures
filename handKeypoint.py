import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_bounding_box = mp_hands.Hands(min_detection_confidence=0.5)

def detectKeypoints(frame):
    # Flip the frame horizontally for a later selfie-view display, and convert
    # the BGR frame to RGB.
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the frame as not writeable to
    # pass by reference.
    frame.flags.writeable = False
    points = hands.process(frame)
    return points

def drawSkeleton(frame, points):
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    if points.multi_hand_landmarks:
        for hand_landmarks in points.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('MediaPipe Hands', frame)

def test(source=0, display=False):
    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            points = detectKeypoints(frame)
            if display: 
                drawSkeleton(frame, points)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
    finally:
        cap.release()

test(0, True)