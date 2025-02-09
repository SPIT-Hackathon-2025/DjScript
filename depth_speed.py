import cv2
import torch
import numpy as np
import pygame
import argparse
from collections import defaultdict, deque

# Import YOLO from ultralytics and Supervision for tracking/annotation
from ultralytics import YOLO
import supervision as sv

# ------------------------------------------------------------------------------
# Global variables for depth calibration via mouse click
calibration_scale = None  
depth_map_global = None  

def click_event(event, x, y, flags, param):
    """
    When the user clicks on the depth map display, allow calibration of the
    relative depth value (e.g. click a point whose true distance is known).
    """
    global calibration_scale, depth_map_global
    if event == cv2.EVENT_LBUTTONDOWN and depth_map_global is not None:
        relative_value = depth_map_global[y, x]
        known_distance = float(input("Enter the known distance (in meters): "))
        calibration_scale = known_distance / relative_value
        print(f"Calibration complete. Scale factor: {calibration_scale:.4f}")

# ------------------------------------------------------------------------------
# Define a simple perspective transformer for speed estimation (from supervision)
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
TARGET_WIDTH = 25
TARGET_HEIGHT = 250
TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# ------------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combined Vehicle Speed Estimation and Depth-based Alert System"
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the output video file",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for detection",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", 
        default=0.7, 
        help="IOU threshold for non-max suppression", 
        type=float
    )
    return parser.parse_args()

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_arguments()

    # ------------------------------------------------------------------------------
    # Initialize pygame mixer for audio alerts.
    pygame.mixer.init()
    alert_sound = "Danger_audio.mp3.mp3"  # Replace with your audio file path
    pygame.mixer.music.load(alert_sound)

    # ------------------------------------------------------------------------------
    # Load the object detection model (YOLOv8) using ultralytics.
    det_model = YOLO("yolov8x.pt")

    # ------------------------------------------------------------------------------
    # Load the MiDaS depth estimation model and associated transform.
    midas_model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    # ------------------------------------------------------------------------------
    # Use Supervision to read the input video and obtain video properties.
    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    # ------------------------------------------------------------------------------
    # Initialize ByteTrack for tracking.
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=args.confidence_threshold
    )

    # Calculate annotation parameters.
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(video_info.fps * 2),
        position=sv.Position.BOTTOM_CENTER,
    )

    # ------------------------------------------------------------------------------
    # Define a polygon zone (e.g. a region of interest) and view transformer.
    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    # ------------------------------------------------------------------------------
    # Dictionary to hold recent y–coordinates (in the transformed view) for each tracker.
    coordinates = defaultdict(lambda: deque(maxlen=int(video_info.fps)))

    # ------------------------------------------------------------------------------
    # Set depth thresholds (arbitrary units; adjust after calibration).
    medium_threshold = 500.0   # e.g. metric_depth > 500 → "Medium"
    near_threshold = 800.0     # e.g. metric_depth > 800 → "Near" (alert condition)

    print("Processing video. Press 'q' in the display window to quit.")

    # ------------------------------------------------------------------------------
    # Set display size for the output window.
    DISPLAY_WIDTH = 800
    DISPLAY_HEIGHT = 600

    # Create a resizable window for display.
    cv2.namedWindow("Combined", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Combined", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    cv2.setMouseCallback("Combined", click_event)

    # ------------------------------------------------------------------------------
    # Initialize VideoWriter for output video.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.target_video_path, fourcc, video_info.fps, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    for frame in frame_generator:
        alert_flag = False  # Will be set True if any detection is too near.

        # ===================== Object Detection & Tracking =====================
        results = det_model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.confidence > args.confidence_threshold]
        detections = detections[polygon_zone.trigger(detections)]
        detections = detections.with_nms(threshold=args.iou_threshold)
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_points = view_transformer.transform_points(points)
        for tracker_id, (_, y) in zip(detections.tracker_id, transformed_points.astype(int)):
            coordinates[tracker_id].append(y)

        labels = []
        for tracker_id in detections.tracker_id:
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                label_text = f"#{tracker_id}"
            else:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time_elapsed = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time_elapsed * 3.6  # convert m/s to km/h
                label_text = f"#{tracker_id} {int(speed)} km/h"
            labels.append(label_text)

        # ============================ Depth Estimation ============================
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
            depth_map_global = depth_map  # for the calibration callback

        for i, bbox in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, bbox[:4])
            region = depth_map[y1:y2, x1:x2]
            if region.size == 0:
                continue
            avg_depth = np.median(region)
            metric_depth = avg_depth * calibration_scale if calibration_scale is not None else avg_depth
            if metric_depth > near_threshold:
                depth_category = "Near"
                alert_flag = True
            elif metric_depth > medium_threshold:
                depth_category = "Medium"
            else:
                depth_category = "Far"
            labels[i] += f", {depth_category}: {metric_depth:.2f}"

        # ======================== Annotation & Alerting =========================
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        if alert_flag:
            cv2.putText(annotated_frame, "KEEP SAFE DISTANCE!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
        else:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()

        # ===================== Depth Visualization (Colorized) =====================
        normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_display = np.uint8(normalized_depth)
        depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        # Combine the annotated frame and depth visualization side by side.
        combined = np.hstack((annotated_frame, depth_display))
        
        # Resize the combined output to fit the display window.
        combined_resized = cv2.resize(combined, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow("Combined", combined_resized)
        
        # Write the resized frame to the output video.
        out.write(combined_resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    out.release()
    cv2.destroyAllWindows()
