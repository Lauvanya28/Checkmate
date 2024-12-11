import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import threading
import random
import string
import time
import math
import logging
from collections import deque

app = Flask(__name__)

# logging.basicConfig(
#     level=logging.DEBUG, 
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     handlers=[
#         logging.FileHandler("face_checkmate.log"),
#         logging.StreamHandler()
#     ]
# )
logger = logging.getLogger(__name__)

# Shared states
kyc_started = False
action_sequence = []
action_index = 0
dot_position = None
dot_start_time = None
dot_duration = 2  # seconds per dot
dot_index = 0
dot_positions = []
MAR_SMOOTHING = 5  # frames
MAR_THRESHOLD = 0.6
EAR_THRESHOLD = 0.25
FRAME_WINDOW = 5  # consecutive frames for blink detection
initial_nose_point = None
action_start_time = None
ACTION_TIMEOUT = 20  # seconds

# Code verification
current_code = None
code_verified = False
code_validation_required = False

# Verification result
verification_result = "in_progress"  # "in_progress", "success", "failed"

# Landmark indices
MOUTH_LANDMARKS = [13, 14, 15, 17, 18, 19, 20, 22]
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1

# Lock for thread safety
lock = threading.Lock()

def generate_random_code(length=6):
    """Generate a random alphanumeric code."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def calculate_mar(landmarks):
    """Calculate Mouth Aspect Ratio (MAR)."""
    A = np.linalg.norm(np.array(landmarks[13]) - np.array(landmarks[19]))
    B = np.linalg.norm(np.array(landmarks[14]) - np.array(landmarks[18]))
    C = np.linalg.norm(np.array(landmarks[15]) - np.array(landmarks[17]))
    mar = (A + B) / (2.0 * C)
    return mar

def calculate_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio (EAR)."""
    A = math.dist(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    B = math.dist(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    C = math.dist(landmarks[eye_indices[0]], landmarks[eye_indices[3]])  # width
    ear = (A + B) / (2.0 * C)
    return ear

def detect_head_movement(landmarks, initial_nose):
    """Detect head movement based on initial and current nose positions."""
    current_nose = landmarks[NOSE_TIP]
    dx = current_nose[0] - initial_nose[0]
    dy = current_nose[1] - initial_nose[1]
    return dx, dy

def detect_head_action(current_action, dx, dy):
    """Detect specific head actions based on dx and dy."""
    if current_action == 'turn left' and dx < -30:  # horizontal
        return True
    elif current_action == 'turn right' and dx > 30:
        return True
    elif current_action == 'look up' and dy < -20:  # vertical
        return True
    elif current_action == 'look down' and dy > 20:
        return True
    return False

def detect_smile(landmarks, mar_history):
    """Detect a smile based on MAR."""
    mar = calculate_mar(landmarks)
    mar_history.append(mar)
    avg_mar = sum(mar_history) / len(mar_history)
    return avg_mar > MAR_THRESHOLD

def get_eye_center(landmarks, eye_indices):
    """Calculate the center point of the eyes."""
    try:
        x = [landmarks[i][0] for i in eye_indices]
        y = [landmarks[i][1] for i in eye_indices]
        center = (int(sum(x) / len(x)), int(sum(y) / len(y)))
        return center
    except Exception as e:
        logger.error(f"Error calculating eye center: {e}")
        return None

def generate_dot_positions(frame_width, frame_height):
    """Generate random positions for the follow dot action."""
    return [
        (random.randint(100, frame_width - 100), random.randint(100, frame_height - 100))
        for _ in range(4)
    ]

def video_stream():
    global kyc_started, action_sequence, action_index
    global dot_position, dot_start_time, dot_index, dot_positions, initial_nose_point, action_start_time
    global current_code, code_verified, code_validation_required, verification_result

    # Initialize Mediapipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Unable to access webcam.")
        face_mesh.close()
        verification_result = "failed"
        return

    logger.info("Webcam accessed successfully.")
    initial_nose_point = None
    action_start_time = None
    mar_history = deque(maxlen=MAR_SMOOTHING)
    blink_counter = 0  # Initialize blink counter

    try:
        while kyc_started:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame from webcam.")
                verification_result = "failed"
                break  # Exit the loop if frame capture fails

            frame = cv2.flip(frame, 1)  # Mirror the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_rgb is None:
                logger.warning("Received an invalid frame.")
                continue

            try:
                results = face_mesh.process(frame_rgb)
            except Exception as e:
                logger.error(f"Error processing frame with Mediapipe: {e}")
                break  # Exit the loop if processing fails

            frame_height, frame_width = frame.shape[:2]

            if kyc_started and action_index < len(action_sequence):
                current_action = action_sequence[action_index]
                logger.info(f"Current Action: {current_action}")

                # Initialize action start time
                if action_start_time is None:
                    action_start_time = time.time()

                # Check for action timeout
                elapsed_time = time.time() - action_start_time
                if elapsed_time > ACTION_TIMEOUT:
                    logger.warning(f"Action '{current_action}' timed out. Face verification failed.")
                    verification_result = "failed"
                    kyc_started = False
                    break  # Exit the loop on timeout

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = [
                        (int(pt.x * frame_width), int(pt.y * frame_height))
                        for pt in face_landmarks.landmark
                    ]

                    # Initialize initial_nose_point for head movement detection
                    if current_action in ['turn left', 'turn right', 'look up', 'look down'] and initial_nose_point is None:
                        initial_nose_point = landmarks[NOSE_TIP]
                        logger.info(f"Initial nose position recorded for action '{current_action}'.")

                    if current_action == "smile":
                        if detect_smile(landmarks, mar_history):
                            logger.info("Smile detected!")
                            action_index += 1
                            mar_history.clear()
                            initial_nose_point = None  # Reset for next action
                            action_start_time = None  # Reset timer
                        else:
                            pass  # Waiting for smile

                    elif current_action in ["turn left", "turn right", "look up", "look down"]:
                        dx, dy = detect_head_movement(landmarks, initial_nose_point)
                        action_detected = detect_head_action(current_action, dx, dy)
                        if action_detected:
                            logger.info(f"Action '{current_action}' detected!")
                            action_index += 1
                            initial_nose_point = None  # Reset for next action
                            action_start_time = None  # Reset timer
                        else:
                            pass  # Waiting for head movement

                    elif current_action == "blink":
                        # Blink detection logic
                        left_ear = calculate_ear(landmarks, LEFT_EYE_LANDMARKS)
                        right_ear = calculate_ear(landmarks, RIGHT_EYE_LANDMARKS)
                        avg_ear = (left_ear + right_ear) / 2.0

                        logger.info(f"Left EAR: {left_ear:.3f}, Right EAR: {right_ear:.3f}, Avg EAR: {avg_ear:.3f}")
                        logger.info(f"Blink Counter: {blink_counter}")

                        if avg_ear < EAR_THRESHOLD:
                            blink_counter += 1
                            logger.info(f"Blink Counter Incremented: {blink_counter}")
                            if blink_counter >= FRAME_WINDOW:
                                logger.info("Blink detected!")
                                action_index += 1
                                blink_counter = 0
                                action_start_time = None
                        else:
                            if blink_counter > 0:
                                logger.info(f"Blink Counter Reset: {blink_counter}")
                            blink_counter = max(0, blink_counter - 1)

                    elif current_action == "follow dot":
                        try:
                            if not dot_positions:
                                # Initialize dot positions
                                dot_positions = generate_dot_positions(frame_width, frame_height)
                                if not dot_positions:
                                    logger.error("Failed to generate dot positions.")
                                    verification_result = "failed"
                                    kyc_started = False
                                    break  # Exit the loop if dot generation fails
                                dot_position = dot_positions[0]
                                dot_start_time = time.time()
                                logger.info("Dot positions generated and first dot displayed.")

                            if dot_position:
                                # Change dot color to black (0,0,0)
                                cv2.circle(frame, dot_position, 15, (0, 0, 0), -1)
                                logger.info(f"Dot displayed at position: {dot_position}")

                            elapsed_dot_time = time.time() - dot_start_time
                            logger.info(f"Elapsed Dot Time: {elapsed_dot_time}")

                            if elapsed_dot_time > dot_duration:
                                dot_index += 1
                                if dot_index < len(dot_positions):
                                    dot_position = dot_positions[dot_index]
                                    dot_start_time = time.time()
                                    logger.info(f"Moving to next dot position: {dot_position}")
                                else:
                                    # Validate eye movement towards the last dot
                                    left_eye_center = get_eye_center(landmarks, LEFT_EYE_LANDMARKS)
                                    right_eye_center = get_eye_center(landmarks, RIGHT_EYE_LANDMARKS)
                                    center_eye = None
                                    if left_eye_center and right_eye_center:
                                        center_eye = (
                                            (left_eye_center[0] + right_eye_center[0]) // 2,
                                            (left_eye_center[1] + right_eye_center[1]) // 2
                                        )
                                    else:
                                        logger.error("Invalid eye centers detected.")

                                    if center_eye and dot_position:
                                        distance = math.hypot(center_eye[0] - dot_position[0], center_eye[1] - dot_position[1])
                                        logger.info(f"Distance between eye center and dot: {distance}")

                                        alignment_threshold = 50  # Adjust as needed

                                        if distance < alignment_threshold:
                                            logger.info("Follow dot action completed successfully!")
                                            action_index += 1
                                            dot_positions = []
                                            dot_index = 0
                                            action_start_time = None
                                        else:
                                            logger.warning("Follow dot action failed. Eyes did not follow the dot properly.")
                                            verification_result = "failed"
                                            kyc_started = False
                                            break  # Exit the loop to indicate failure
                                    else:
                                        logger.warning("Invalid eye center or dot position during follow dot validation.")
                                        verification_result = "failed"
                                        kyc_started = False
                                        break
                        except Exception as e:
                            logger.error(f"Exception during 'follow dot' action: {e}")
                            verification_result = "failed"
                            kyc_started = False
                            break  # Exit the loop to prevent further processing

                    elif current_action == "type code":
                        if not code_validation_required:
                            # Generate and display code
                            current_code = generate_random_code()
                            logger.info(f"Code generated for verification: {current_code}")
                            code_validation_required = True
                            code_start_time = time.time()
                            logger.info("Awaiting code input from user.")

                        # Overlay code if within 15 seconds
                        elapsed_code_time = time.time() - code_start_time
                        if elapsed_code_time <= 15:
                            # Center the code on the video feed
                            text = f"Code: {current_code}"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1.2
                            thickness = 3
                            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                            text_x = (frame_width - text_size[0]) // 2
                            text_y = (frame_height + text_size[1]) // 2
                            cv2.putText(frame, text, (text_x, text_y),
                                        font, font_scale, (0, 0, 0), thickness)  # Black color
                        else:
                            # Time expired
                            logger.warning("Code display time expired. Face verification failed.")
                            verification_result = "failed"
                            kyc_started = False
                            break  # Exit the loop on timeout

                    # Display current action (except for 'type code')
                    if current_action != "type code":
                        cv2.putText(frame, f"Action: {current_action.replace('_', ' ').title()}", (10, frame_height - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                else:
                    logger.warning("No face detected.")
                    # Optionally, you can add a timeout or retry mechanism here

                # Encode the frame in JPEG format
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                try:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except GeneratorExit:
                    logger.info("Video stream generator closed.")
                    break  # Exit the loop if the generator is closed
                except Exception as e:
                    logger.error(f"Error in video streaming: {e}")
                    break  # Exit the loop on any other exception

    finally:
        cap.release()
        logger.info("Webcam released.")
        try:
            face_mesh.close()
            cap.close()
            logger.info("Face mesh closed.")
        except Exception as e:
            logger.error(f"Error closing face_mesh: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if not kyc_started:
        return Response("Face Checkmate not started.", mimetype='text/plain')
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_kyc', methods=['POST'])
def start_kyc_route():
    global kyc_started, action_sequence, action_index
    global current_code, code_verified, code_validation_required, verification_result

    with lock:
        if not kyc_started:
            kyc_started = True
            verification_result = "in_progress"
            possible_actions = ["smile", "turn left", "turn right", "look up", "look down", "blink", "follow dot", "type code"]
            # Ensure 'blink' and 'type code' are included
            selected_actions = random.sample([action for action in possible_actions if action not in ["follow dot", "type code"]], 3) + ["follow dot", "type code"]
            action_sequence = selected_actions
            action_index = 0
            current_code = None
            code_verified = False
            code_validation_required = False
            logger.info(f"Face Checkmate process started! Perform the following actions in order: {', '.join(action_sequence)}")
            return jsonify({"status": "Face Checkmate started"})
        else:
            logger.debug("Face Checkmate is already in progress.")
            return jsonify({"status": "Face Checkmate already in progress"})

@app.route('/stop_kyc', methods=['POST'])
def stop_kyc():
    global kyc_started, code_validation_required, verification_result
    global action_sequence, action_index, dot_positions, dot_index
    global current_code, code_verified

    with lock:
        if kyc_started:
            kyc_started = False
            code_validation_required = False
            verification_result = "failed"
            # Reset all relevant variables
            action_sequence = []
            action_index = 0
            dot_positions = []
            dot_index = 0
            current_code = None
            code_verified = False
            logger.info("Face Checkmate process stopped by user. All states have been reset.")
            return jsonify({"status": "Face Checkmate stopped and reset"})
        else:
            logger.debug("Face Checkmate is not running.")
            return jsonify({"status": "Face Checkmate is not running"})

@app.route('/validate_code', methods=['POST'])
def validate_code():
    global current_code, code_verified, code_validation_required, kyc_started, action_index, verification_result

    with lock:
        if not kyc_started or not code_validation_required:
            logger.warning("No code validation required.")
            return jsonify({"status": "No code validation required."}), 400

        data = request.get_json()
        user_code = data.get('code', '').strip().upper()

        if user_code == current_code:
            code_verified = True
            code_validation_required = False
            verification_result = "success"
            logger.info("Code verification successful.")
            # Proceed to the next action
            action_index += 1
            logger.info("Proceeding to next action after successful code verification.")
            # After successful verification, stop KYC
            kyc_started = False
            return jsonify({"status": "Code verified successfully."})
        else:
            code_verified = False
            code_validation_required = False
            verification_result = "failed"
            kyc_started = False
            logger.warning("Code verification failed.")
            return jsonify({"status": "Code verification failed. Face verification failed."})

@app.route('/current_status', methods=['GET'])
def current_status():
    global action_sequence, action_index
    global code_validation_required, code_verified
    global kyc_started, verification_result

    with lock:
        if not kyc_started:
            return jsonify({
                "current_action": "not_started",
                "code_validated": False,
                "verification_result": verification_result
            })

        if action_index < len(action_sequence):
            current_action = action_sequence[action_index]
            if current_action == "type code":
                return jsonify({
                    "current_action": current_action,
                    "code_validated": code_verified,
                    "verification_result": verification_result
                })
            else:
                return jsonify({
                    "current_action": current_action,
                    "code_validated": False,
                    "verification_result": verification_result
                })
        else:
            # All actions completed
            if verification_result == "success":
                return jsonify({
                    "current_action": "all_completed",
                    "code_validated": True,
                    "verification_result": verification_result
                })
            else:
                return jsonify({
                    "current_action": "all_completed",
                    "code_validated": False,
                    "verification_result": verification_result
                })

if __name__ == '__main__':
    try:
        logger.info("Starting Flask server...")
        app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Shutting down application.")
