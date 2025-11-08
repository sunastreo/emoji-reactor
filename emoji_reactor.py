#!/usr/bin/env python3
"""
Improved real-time emoji display with better emotion detection.
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Configuration
SURPRISE_THRESHOLD = 0.052  # Порог для удивления (открытый рот)
ANGRY_THRESHOLD = 0.025  # Порог для злости (близость брови к глазу)
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Load emoji images
try:
    smiling_emoji = cv2.imread("smile.jpg")
    straight_face_emoji = cv2.imread("plain.png")
    hands_up_emoji = cv2.imread("air.jpg")
    angry_emoji = cv2.imread("angry.jpg")

    if any(img is None for img in [smiling_emoji, straight_face_emoji, hands_up_emoji]):
        raise FileNotFoundError("Required emoji images not found")
    
    if angry_emoji is None:
        print("Warning: angry.jpg not found. Creating placeholder.")
        angry_emoji = np.ones((200, 200, 3), dtype=np.uint8) * [0, 0, 255]
        cv2.putText(angry_emoji, "ANGRY", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Resize emojis
    smiling_emoji = cv2.resize(smiling_emoji, EMOJI_WINDOW_SIZE)
    straight_face_emoji = cv2.resize(straight_face_emoji, EMOJI_WINDOW_SIZE)
    hands_up_emoji = cv2.resize(hands_up_emoji, EMOJI_WINDOW_SIZE)
    angry_emoji = cv2.resize(angry_emoji, EMOJI_WINDOW_SIZE)
    
except Exception as e:
    print("Error loading emoji images!")
    print(f"Details: {e}")
    exit()

blank_emoji = np.zeros((EMOJI_WINDOW_SIZE[0], EMOJI_WINDOW_SIZE[1], 3), dtype=np.uint8)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
cv2.namedWindow('Camera Feed - Emotion Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed - Emotion Detection', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Emoji Output', WINDOW_WIDTH, WINDOW_HEIGHT)

print("Controls:")
print("  Press 'q' to quit")
print("  Press 'd' to toggle landmark display")
print("  Raise BOTH hands above shoulders for 🙌")
print("  Open mouth for 😲 (surprise)") 
print("  Frown brows for 😠 (angry)")

show_landmarks = True

def detect_angry_expression(face_landmarks):
    """
    Детектор злости: ТОЛЬКО близость внутренних уголков бровей к глазам
    """
    landmarks = face_landmarks.landmark
    
    # Точки бровей
    left_eyebrow_inner = landmarks[65]    # Внутренний угол левой брови
    right_eyebrow_inner = landmarks[295]  # Внутренний угол правой брови
    
    # Точки глаз для сравнения
    left_eye_top = landmarks[159]         # Верх левого глаза
    right_eye_top = landmarks[386]        # Верх правого глаза
    
    # 1. Проверяем положение бровей относительно глаз
    left_brow_to_eye = abs(left_eyebrow_inner.y - left_eye_top.y)
    right_brow_to_eye = abs(right_eyebrow_inner.y - right_eye_top.y)
    
    print(f"Angry detection - Left: {left_brow_to_eye:.3f}, Right: {right_brow_to_eye:.3f}")
    
    # Условия для злости: брови близко к глазам
    eyebrows_low = (left_brow_to_eye < ANGRY_THRESHOLD and 
                   right_brow_to_eye < ANGRY_THRESHOLD)
    
    return eyebrows_low 

def detect_surprise(face_landmarks):
    """
    Детектор удивления: открытый рот
    """
    landmarks = face_landmarks.landmark
    
    # Точки рта для определения открытости
    upper_lip_top = landmarks[0]         # Верхняя губа (верх)
    lower_lip_bottom = landmarks[17]     # Нижняя губа (низ)
    
    # Вычисляем открытость рта
    mouth_openness = abs(lower_lip_bottom.y - upper_lip_top.y)
    
    print(f"Surprise detection - Mouth openness: {mouth_openness:.3f}")
    
    # Условие для удивления: рот открыт достаточно широко
    mouth_open = mouth_openness > SURPRISE_THRESHOLD
    
    return mouth_open

def draw_custom_landmarks(image, pose_landmarks, face_landmarks):
    """Draw custom landmarks with emotion-specific points"""
    h, w, _ = image.shape
    
    # Draw pose landmarks
    if pose_landmarks:
        landmarks = pose_landmarks.landmark
        
        key_points = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_WRIST, 
            mp_pose.PoseLandmark.RIGHT_WRIST
        ]
        
        for point in key_points:
            landmark = landmarks[point]
            x, y = int(landmark.x * w), int(landmark.y * h)
            
            if point in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]:
                color = (0, 255, 0)  # Зеленый для плеч
                size = 8
            else:
                color = (255, 0, 0)  # Красный для запястий
                size = 6
                
            cv2.circle(image, (x, y), size, color, -1)
    
    # Draw face landmarks for emotions
    if face_landmarks:
        landmarks = face_landmarks.landmark
        
        # Точки для удивления (рот) - КРАСНЫЕ
        surprise_points = [0, 17]  # Верх и низ рта
        for point_idx in surprise_points:
            landmark = landmarks[point_idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)  # Красный для удивления
        
        # Точки для злости (брови и глаза) - СИНИЕ  
        angry_points = [65, 295, 159, 386]  # Внутренние брови и верхи глаз
        for point_idx in angry_points:
            landmark = landmarks[point_idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (x, y), 4, (255, 0, 0), -1)  # Синий для злости

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                          min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        current_state = "STRAIGHT_FACE"
        pose_landmarks_result = None
        face_landmarks_result = None

        # Process detection
        results_pose = pose.process(image_rgb)
        results_face = face_mesh.process(image_rgb)
        
        pose_landmarks_result = results_pose.pose_landmarks
        if results_face.multi_face_landmarks:
            face_landmarks_result = results_face.multi_face_landmarks[0]

        # Check for hands up (BOTH hands required)
        if pose_landmarks_result:
            landmarks = pose_landmarks_result.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # ОБЕ руки должны быть выше плеч
            both_hands_up = (left_wrist.y < left_shoulder.y or
                           right_wrist.y < right_shoulder.y)
            
            if both_hands_up:
                current_state = "HANDS_UP"
        
        # Check facial expressions only if hands are down
        if current_state != "HANDS_UP" and face_landmarks_result:
            if detect_angry_expression(face_landmarks_result):
                current_state = "ANGRY"
            elif detect_surprise(face_landmarks_result):
                current_state = "SURPRISE"
            else:
                current_state = "STRAIGHT_FACE"

        # Draw landmarks
        image_rgb.flags.writeable = True
        frame_with_landmarks = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if show_landmarks:
            draw_custom_landmarks(frame_with_landmarks, pose_landmarks_result, face_landmarks_result)

        # Select emoji
        if current_state == "SURPRISE":
            emoji_to_display = smiling_emoji  # Используем smile.jpg для удивления
            emoji_name = "😲"
            state_color = (0, 255, 0)  # Зеленый
        elif current_state == "STRAIGHT_FACE":
            emoji_to_display = straight_face_emoji
            emoji_name = "😐"
            state_color = (255, 255, 255)  # Белый
        elif current_state == "HANDS_UP":
            emoji_to_display = hands_up_emoji
            emoji_name = "🙌"
            state_color = (255, 0, 0)  # Красный
        elif current_state == "ANGRY":
            emoji_to_display = angry_emoji
            emoji_name = "😠"
            state_color = (0, 0, 255)  # Синий
        else:
            emoji_to_display = blank_emoji
            emoji_name = "❓"
            state_color = (255, 255, 255)

        camera_frame_resized = cv2.resize(frame_with_landmarks, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Display status with color coding
        cv2.putText(camera_frame_resized, f'STATE: {current_state} {emoji_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2, cv2.LINE_AA)
        cv2.putText(camera_frame_resized, 'Press "q" to quit, "d" to toggle landmarks', (10, WINDOW_HEIGHT - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(camera_frame_resized, f'Landmarks: {"ON" if show_landmarks else "OFF"}', (10, WINDOW_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('Camera Feed - Emotion Detection', camera_frame_resized)
        cv2.imshow('Emoji Output', emoji_to_display)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_landmarks = not show_landmarks

cap.release()
cv2.destroyAllWindows()
