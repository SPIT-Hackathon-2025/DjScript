import cv2
import time
import tempfile
import os
import numpy as np
import threading
import pyttsx3
from inference_sdk import InferenceHTTPClient

# -----------------------------------------------------------------------------
# Initialize pyttsx3 for text-to-speech (TTS)
# -----------------------------------------------------------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)   # Adjust speech rate (words per minute)
engine.setProperty('volume', 1.0)   # Volume: 0.0 to 1.0

def speak_label(label):
    # Create a new engine instance in this thread.
    local_engine = pyttsx3.init()
    local_engine.setProperty('rate', 150)   # Adjust speech rate (words per minute)
    local_engine.setProperty('volume', 1.0)   # Volume: 0.0 to 1.0
    local_engine.say(label)
    local_engine.runAndWait()
    # Optionally, you can call local_engine.stop() here if needed.


# -----------------------------------------------------------------------------
# Setup the Inference HTTP Client for road sign detection.
#
# Note: This example uses RoboFlow’s classification/detection API endpoint.
# Replace the api_url and MODEL_ID with your own if needed.
# -----------------------------------------------------------------------------
CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",  # Use your appropriate endpoint
    api_key="F3EMGUQMEz5Oj4QmIU6D"
)
MODEL_ID = "traffic-signs-recognition-nl4tf/1"

def infer_from_frame_tempfile(frame, model_id):
    """
    Saves the given frame to a temporary JPEG file,
    runs inference by passing its file path to the CLIENT,
    and then cleans up the temporary file.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_filename = tmp.name
    cv2.imwrite(temp_filename, frame)
    try:
        results = CLIENT.infer(temp_filename, model_id=model_id)
    finally:
        os.remove(temp_filename)
    # In some cases the inference result may be returned as a list.
    if isinstance(results, list) and results:
        return results[0]
    return results

# -----------------------------------------------------------------------------
# Setup Video Capture (0 for webcam or replace with a video file path)
# -----------------------------------------------------------------------------
video_source = "road_sign.mp4"  # Use your video file path if needed, e.g., "road_signs.mp4"
cap = cv2.VideoCapture(video_source)

# Variables to help avoid repeating the same TTS announcement in every frame.
last_announced_label = None
last_announced_time = 0
announcement_cooldown = 3  # seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the current frame.
    try:
        results = infer_from_frame_tempfile(frame, MODEL_ID)
    except Exception as e:
        print("Inference error:", e)
        continue

    # Check if there are predictions in the result.
    if "predictions" in results and results["predictions"]:
        # For this example, we pick the prediction with the highest confidence.
        best_pred = max(results["predictions"], key=lambda pred: pred.get("confidence", 0))
        label = best_pred.get("class", "")
        confidence = best_pred.get("confidence", 0)

        # If bounding box information is available, draw it.
        if all(k in best_pred for k in ["x", "y", "width", "height"]):
            x_center = best_pred.get("x", 0)
            y_center = best_pred.get("y", 0)
            width = best_pred.get("width", 0)
            height = best_pred.get("height", 0)
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare label text with confidence and draw a background for readability.
            label_text = f"{label} {confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label_text, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        else:
            # If no bounding box info is available, display the label in the top-left corner.
            cv2.putText(frame, f"{label} {confidence:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # -----------------------------------------------------------------------------
        # Trigger text-to-speech for the detected road sign.
        # We only announce if it's a new sign or if a cooldown period has passed.
        # -----------------------------------------------------------------------------
        current_time = time.time()
        if (label != last_announced_label) or (current_time - last_announced_time > announcement_cooldown):
            last_announced_label = label
            last_announced_time = current_time
            # Launch TTS in a new thread to avoid blocking the main loop.
            threading.Thread(target=speak_label, args=(label,), daemon=True).start()

    # Display the annotated frame.
    cv2.imshow("Road Signs Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
