import cv2
import mediapipe as mp
import math

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    """
    Calculates the angle between vectors ab and bc.
    Args:
    a, b, c: Coordinates of points (x, y).

    Returns:
    Angle in degrees.
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    cos_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (
        math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2)
    )
    return round(math.degrees(math.acos(cos_angle)), 2)

# Function to check if the wrist is pointing forward
def is_wrist_forward(shoulder, elbow, wrist, threshold=30):
    """
    Checks if the wrist is pointing forward relative to the elbow.
    Args:
    shoulder, elbow, wrist: Coordinates of points (x, y).
    threshold: Angle deviation allowed (default is 30 degrees).

    Returns:
    True if wrist is pointing forward, False otherwise.
    """
    angle = calculate_angle(shoulder, elbow, wrist)
    return angle <= threshold or angle >= (180 - threshold)

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Open video file
video_path = "pushup_video.mp4"  # Replace with your video file
cap = cv2.VideoCapture(video_path)

# Slow-motion factor (increase to slow down, decrease to speed up)
slow_motion_factor = 5  # Slow down by a factor of 5 (adjust as needed)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (required by Mediapipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract key points for left arm
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Extract key points for right arm
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Calculate elbow angles
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Check if wrists are forward
        left_wrist_forward = is_wrist_forward(left_shoulder, left_elbow, left_wrist)
        right_wrist_forward = is_wrist_forward(right_shoulder, right_elbow, right_wrist)

        # Print elbow angles and wrist alignment status
        print(f"Left Elbow Angle: {left_elbow_angle} degrees")
        print(f"Right Elbow Angle: {right_elbow_angle} degrees")
        print(f"Left Wrist Pointing Forward: {left_wrist_forward}")
        print(f"Right Wrist Pointing Forward: {right_wrist_forward}")

        # Draw landmarks and connections
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display elbow angles and wrist alignment on the frame
        cv2.putText(frame, f"Left Elbow: {left_elbow_angle} deg", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Right Elbow: {right_elbow_angle} deg", 
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Left Wrist Forward: {left_wrist_forward}", 
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Right Wrist Forward: {right_wrist_forward}", 
                    (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Get original dimensions of the frame (height, width)
    height, width, _ = frame.shape

    # Calculate new dimensions (50% of the original dimensions)
    new_width = int(width * 0.5)
    new_height = int(height * 0.5)

    # Resize the frame to 50% of the original size
    frame_resized = cv2.resize(frame, (new_width, new_height))

    # Show the resized frame with the slow-motion effect
    cv2.imshow("Pushup Detection", frame_resized)

    # Introduce the slow-motion delay
    key = cv2.waitKey(10)  # Adjust slow_motion_factor as needed

    # Exit on pressing 'q'
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
