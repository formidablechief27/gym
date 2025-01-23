import os
import io
import cv2
import numpy as np
import mediapipe as mp
import heapq
from flask import Flask, request, Response, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate angles between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # Convert points to numpy arrays
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b  # Vectors
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)  # In radians
    return np.degrees(angle)  # Return the angle in degrees

# Function to analyze angles and movements
def analyze_angles(elbow, back):
    reps = []
    n = len(back)
    i = 0

    while i < n - 1:
        # Initialize the pair with a window of size 29
        p = [i, min(n - 1, i + 29)]
        mn = 300
        
        # Find the minimum elbow angle within the window [i, p[1]]
        for j in range(i, p[1] + 1):
            mn = min(mn, elbow[j])
        
        # If the minimum elbow angle is greater than 150, extend the window size to 59
        if mn > 150:
            p[1] = min(n - 1, i + 59)
        
        max_angle = elbow[p[1]]
        id = p[1]
        idx = p[1] + 1
        prev = i
        queue = []

        while idx < n:
            dd = idx - id
            # If the conditions are met, update the range and reset the queue
            if (dd > 10 and not queue) or elbow[idx] >= max_angle or max_angle < 150:
                d = idx - id
                # If the distance is greater than 20 and the queue has more than 10 elements, break
                if d > 20 and len(queue) > 10:
                    break
                p[1] = idx
                i = idx
                if elbow[idx] >= max_angle:
                    max_angle = elbow[idx]
                id = idx
                queue = []
            elif elbow[idx] < 150:
                heapq.heappush(queue, elbow[idx])
            idx += 1
        
        # If no change in the index, move forward by 29
        if i == prev:
            i += 29

        # Skip small windows where the range length is less than 10
        if p[1] - p[0] < 10:
            continue

        # Append the valid pair (rep) to the list
        reps.append(tuple(p))

    results = []
    for i1, i2 in reps:
        queue_160 = []
        queue_150 = []
        queue_140 = []

        for j in range(i1, i2 + 1):
            if back[j] < 160:
                heapq.heappush(queue_160, back[j])
            if back[j] < 150:
                heapq.heappush(queue_150, back[j])
            if back[j] < 140:
                heapq.heappush(queue_140, back[j])

        queue_100 = []
        queue_90 = []
        queue_80 = []
        queue_70 = []

        for j in range(i1, i2 + 1):
            if elbow[j] < 100:
                heapq.heappush(queue_100, elbow[j])
            if elbow[j] < 90:
                heapq.heappush(queue_90, elbow[j])
            if elbow[j] < 80:
                heapq.heappush(queue_80, elbow[j])
            if elbow[j] < 70:
                heapq.heappush(queue_70, elbow[j])

        score1 = 10
        score2 = 0
        score3 = 0

        if len(queue_160) >= 5:
            if len(queue_140) > 20:
                score1 = 3
            elif len(queue_140) > 15:
                score1 = 4
            elif len(queue_140) > 10:
                score1 = 5
            else:
                score1 = 6
                if len(queue_150) < 10:
                    score1 += 1
                if len(queue_160) < 10:
                    score1 += 1
                if len(queue_160) < 5:
                    score1 += 1

        minimum = float('inf')
        maximum = 0
        flag = False

        for j in range(i1, i2 + 1):
            if minimum > elbow[j]:
                minimum = elbow[j]

        for j in range(i1, i2 + 1):
            if minimum == elbow[j]:
                flag = True
            if flag and maximum < elbow[j]:
                maximum = elbow[j]

        if minimum <= 60:
            score2 = 10
        elif minimum <= 75:
            score2 = 9
        elif minimum <= 90:
            score2 = 8
        elif minimum <= 100:
            score2 = 7
        elif minimum <= 110:
            score2 = 6
        else:
            score2 = 5

        if maximum >= 165:
            score3 = 10
        elif maximum >= 162:
            score3 = 9
        elif maximum >= 160:
            score3 = 8
        elif maximum >= 155:
            score3 = 7
        elif maximum >= 150:
            score3 = 6
        elif maximum >= 145:
            score3 = 5
        else:
            score3 = 4

        results.append((score1, score2, score3))

    return results

@app.route('/data', methods=['GET'])
def get_data():
    # Prepare the response data
    response_body = {
        "message": "Hello, this is a GET response!"
    }
    # Return the response with a 200 OK status
    return jsonify(response_body), 200

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No file part", 400
    video = request.files['video']
    if video.filename == '':
        return "No selected file", 400
    print("im here")
    video_filename = "uploaded_video.mp4"
    video.save(video_filename)

    # Process the video
    cap = cv2.VideoCapture(video_filename)
    
    elbow1 = []
    back1 = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        if results.pose_landmarks:
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
            elbow_angle = int(calculate_angle(shoulder, elbow, wrist))
            back_angle = int(calculate_angle(shoulder, hip, knee))

            # Store angles
            elbow1.append(elbow_angle)
            back1.append(back_angle)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Analyze the angles
    analysis_results = analyze_angles(elbow1, back1)

    # Delete the video file after processing
    os.remove(video_filename)

    # Format the results as a string to return as plain text
    results = []
    for idx, (val1, val2, val3) in enumerate(analysis_results, start=1):
        results.append({
            "rep": idx,
            "back_position": val1,
            "depth": val2,
            "rep_completion": val3
        })

    return jsonify({"data": results})

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug=True, port=5000, threaded=True)
