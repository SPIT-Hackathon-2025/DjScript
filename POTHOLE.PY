import cv2
import time
import threading
import tempfile
import os
from playsound import playsound

import supervision as sv
from inference_sdk import InferenceHTTPClient

# ---------------------------
# Setup the Inference HTTP Client
# ---------------------------
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="F3EMGUQMEz5Oj4QmIU6D"
)
MODEL_ID = "pothole-detection-project-1dpiq/5"

def infer_from_frame_tempfile(frame, model_id):
    """
    Save the frame to a temporary file and run inference by passing the file path.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_filename = tmp.name
    cv2.imwrite(temp_filename, frame)
    try:
        results = CLIENT.infer(temp_filename, model_id=model_id)
    finally:
        os.remove(temp_filename)
    if isinstance(results, list):
        return results[0]
    return results

# ---------------------------
# Create Annotators for Drawing
# ---------------------------
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# ---------------------------
# Setup Video Capture
# ---------------------------
video_source = "Pothole.mp4"  # Replace with 0 for webcam.
cap = cv2.VideoCapture(video_source)

last_sound_time = 0
sound_cooldown = 2  # seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    try:
        results = infer_from_frame_tempfile(frame, MODEL_ID)
    except Exception as e:
        print("Inference error:", e)
        continue

    detections = sv.Detections.from_inference(results)
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    if any(label.lower() == "pothole" for label in detections.labels):
        current_time = time.time()
        if current_time - last_sound_time > sound_cooldown:
            threading.Thread(target=playsound, args=("Pothole_audio.mp3.mp3",), daemon=True).start()
            last_sound_time = current_time

    cv2.imshow("Pothole Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
