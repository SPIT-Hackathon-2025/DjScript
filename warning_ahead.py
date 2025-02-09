import cv2
import torch
import numpy as np
import pygame  # For playing alert sounds

# Initialize pygame mixer
pygame.mixer.init()
alert_sound = "Danger_audio.mp3.mp3"  # Replace with your alert sound file
pygame.mixer.music.load(alert_sound)

# Load YOLOv5 Object Detector
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load MiDaS Depth Estimation Model
midas_model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Load MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Use live camera feed instead of a video file
cap = cv2.VideoCapture("Rear_car1.mp4")

# Global variables for calibration
calibration_scale = None  
depth_map_global = None  

def click_event(event, x, y, flags, param):
    """
    On mouse click, allow user to calibrate the depth scale.
    Click on a point in the displayed depth map and enter its known real-world distance.
    """
    global calibration_scale, depth_map_global
    if event == cv2.EVENT_LBUTTONDOWN and depth_map_global is not None:
        relative_value = depth_map_global[y, x]
        known_distance = float(input("Enter the known distance (in meters): "))
        calibration_scale = known_distance / relative_value
        print(f"Calibration complete. Scale factor: {calibration_scale:.4f}")

cv2.namedWindow("Combined")
cv2.setMouseCallback("Combined", click_event)

# Define thresholds (in arbitrary units, adjust as needed)
# For example purposes, these thresholds are tuned for objects at closer distances.
# Adjust these values based on your calibration and application.
medium_threshold = 500.0   # e.g., metric_depth > 500 means "Medium"
near_threshold = 800.0     # e.g., metric_depth > 800 means "Near" (alert)

print("Processing live camera feed. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object Detection with YOLOv5
    results = yolo_model(frame)
    detections = results.xyxy[0].cpu().numpy()

    # Depth Estimation with MiDaS
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth_map = prediction.cpu().numpy()
        depth_map_global = depth_map  # update the global depth map

    # Normalize depth map for display purposes
    normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_display = np.uint8(normalized_depth)
    depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

    # Flag to check if any detection is in the "Near" category (alert condition)
    alert_flag = False

    # Process detections and annotate the frame
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)

        # Ensure the region is valid (non-empty)
        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            continue

        avg_depth = np.median(region)
        # Use calibrated metric depth if available
        metric_depth = avg_depth * calibration_scale if calibration_scale is not None else avg_depth

        # Classify the depth into one of two categories:
        # - "Near": if the metric_depth is above the near_threshold, alert is triggered.
        # - "Medium": if metric_depth is greater than medium_threshold but not high enough to be "Near".
        if metric_depth > near_threshold:
            category = "Near"
            color = (0, 0, 255)   # Red
            alert_flag = True
        elif metric_depth > medium_threshold:
            category = "Medium"
            color = (0, 255, 255)  # Yellow
        else:
            category = "Far"
            color = (0, 255, 0)    # Green

        class_name = yolo_model.names[int(cls)]
        label = f"{class_name} {category}: {metric_depth:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # If an object is detected as "Near", overlay a warning and play the alert sound.
    # If not, ensure the sound is stopped.
    if alert_flag:
        cv2.putText(frame, "KEEP SAFE DISTANCE!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()
    else:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

    # Combine the annotated frame and the depth map visualization side by side.
    combined = np.hstack((frame, depth_display))
    cv2.imshow("Combined", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()