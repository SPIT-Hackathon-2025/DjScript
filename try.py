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

# Load Video File
video_path = "Video3.mp4"
cap = cv2.VideoCapture(video_path)

# Global variables
calibration_scale = None  
depth_map_global = None  

def click_event(event, x, y, flags, param):
    global calibration_scale, depth_map_global
    if event == cv2.EVENT_LBUTTONDOWN and depth_map_global is not None:
        relative_value = depth_map_global[y, x]
        known_distance = float(input("Enter the known distance (in meters): "))
        calibration_scale = known_distance / relative_value
        print(f"Calibration complete. Scale factor: {calibration_scale:.4f}")

cv2.namedWindow("Combined")
cv2.setMouseCallback("Combined", click_event)

def check_collision(detections, depth_map):
    """
    Detects potential collisions based on bounding box overlap and depth similarity.
    """
    collision_detected = False
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            x1a, y1a, x2a, y2a, conf1, cls1 = detections[i]
            x1b, y1b, x2b, y2b, conf2, cls2 = detections[j]

            # Convert to int
            x1a, y1a, x2a, y2a = map(int, [x1a, y1a, x2a, y2a])
            x1b, y1b, x2b, y2b = map(int, [x1b, y1b, x2b, y2b])

            # Check bounding box overlap
            overlap_x = max(0, min(x2a, x2b) - max(x1a, x1b))
            overlap_y = max(0, min(y2a, y2b) - max(y1a, y1b))
            overlap_area = overlap_x * overlap_y

            if overlap_area > 0:
                # Get depth values
                depth1 = np.median(depth_map[y1a:y2a, x1a:x2a])
                depth2 = np.median(depth_map[y1b:y2b, x1b:x2b])

                # Check depth closeness
                if abs(depth1 - depth2) < 0.5:  # Adjust threshold based on real-world scenario
                    collision_detected = True
                    return True  # Stop checking further

    return collision_detected

print("Processing video. Press 'q' to quit.")

while cap.isOpened():
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
        depth_map_global = depth_map  

    # Normalize depth map for display
    normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_display = np.uint8(normalized_depth)
    depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

    # Process detections and draw bounding boxes
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)

        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            continue

        avg_depth = np.median(region)
        metric_depth = avg_depth * calibration_scale if calibration_scale is not None else avg_depth

        class_name = yolo_model.names[int(cls)]
        label = f"{class_name}: {metric_depth:.2f}m"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Check for collisions
    if check_collision(detections, depth_map):
        cv2.putText(frame, "COLLISION AHEAD!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()

    # Combine the frame and depth display
    combined = np.hstack((frame, depth_display))
    cv2.imshow("Combined", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
