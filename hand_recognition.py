import cv2
import numpy as np
import mediapipe as mp

def check_pose(frame, hand_landmarks, hand_type):
    frame_height, frame_width, _ = frame.shape  # Get height and width of the frame
   
    coordinates = (10,50)
    if hand_type == 'Left':
        coordinates=(1050, 50)
    
    if hand_landmarks:
        thumb_tip = hand_landmarks.landmark[4]
        index_finger_tip = hand_landmarks.landmark[8]
        middle_finger_tip = hand_landmarks.landmark[12]
        ring_finger_tip = hand_landmarks.landmark[16]
        little_finger_tip = hand_landmarks.landmark[20]

        # lajk
        if thumb_tip.y < index_finger_tip.y < middle_finger_tip.y < ring_finger_tip.y < little_finger_tip.y:
            cv2.putText(frame, "Thumbs up", coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif thumb_tip.y > index_finger_tip.y > middle_finger_tip.y > ring_finger_tip.y > little_finger_tip.y:
            cv2.putText(frame, "Thumbs down", coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)    
        elif (middle_finger_tip.y < index_finger_tip.y) and abs(thumb_tip.x - little_finger_tip.x) < 0.02 and abs(thumb_tip.x - ring_finger_tip.x) < 0.02:
            cv2.putText(frame, "Peace", coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif middle_finger_tip.y < index_finger_tip.y < ring_finger_tip.y < thumb_tip.y < little_finger_tip.y and abs(ring_finger_tip.x - little_finger_tip.x) < 0.02 and abs(ring_finger_tip.x - little_finger_tip.x) < 0.2:
            cv2.putText(frame, "FUCK YOU", coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
# hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# risanje
mp_drawing = mp.solutions.drawing_utils

# video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(frame_rgb)
    
    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks, hand in zip(results.multi_hand_landmarks, results.multi_handedness):
            #hand_type
            # Draw landmarks on the frame
            hand_type = hand.classification[0].label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            check_pose(frame, hand_landmarks, hand_type)
    
    # Display the frame with hand landmarks
    cv2.imshow('Hand Recognition', frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()