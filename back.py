import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points
    Parameters:
        a, b, c: Coordinates of points (x, y)
    Returns:
        angle in degrees
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint (vertex)
    c = np.array(c)  # Last point

    ba = a - b  # Vector BA
    bc = c - b  # Vector BC

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)  # In radians
    return np.degrees(angle)  # Convert to degrees

# Open video
cap = cv2.VideoCapture("bbb.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for Mediapipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process with Mediapipe Pose
    results = pose.process(image)

    # Extract pose landmarks
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # Draw the landmarks on the frame
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark

        # Get coordinates for relevant points
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

        # Calculate angles
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        back_angle = calculate_angle(shoulder, hip, knee)

        # Display angles on the video
        cv2.putText(image, f"Elbow Angle: {int(elbow_angle)}", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Back Angle: {int(back_angle)}", (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Resize the frame to 50% of original size
    height, width, _ = image.shape
    resized_image = cv2.resize(image, (width // 4, height // 4))

    # Display the resized frame
    cv2.imshow('Pushup Detection (50% Size, Slow Motion)', resized_image)

    # Add slow motion by increasing the wait time
    if cv2.waitKey(25) & 0xFF == ord('q'):  # 100 ms delay = ~10 FPS
        break

# Release resources

cap.release()
cv2.destroyAllWindows()
