import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- CONFIGURATION ---
VIDEO_PATH = "road_scene.mp4"  # Your recorded video
TARGET_CLASSES = ['car', 'bus', 'truck', 'person']  # Objects we want to track

# Load YOLOv5 model (make sure yolov5 repo is cloned)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', source='local')
model.conf = 0.4  # Confidence threshold

# Class names from COCO dataset
class_names = model.names

# Deep SORT Tracker
tracker = DeepSort(max_age=40)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

# Track unique IDs per class
unique_ids_by_class = {cls: set() for cls in TARGET_CLASSES}

# Video writer (optional: save output)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_tracked.mp4", fourcc, int(cap.get(5)), 
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv5 detection
    results = model(frame)
    detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, cls]

    dets = []
    classes_in_frame = []

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_name = class_names[int(cls_id)]

        # Filter only target classes
        if cls_name in TARGET_CLASSES:
            # Format detection for DeepSort
            bbox = [x1.item(), y1.item(), x2.item(), y2.item()]
            dets.append([bbox, conf.item(), int(cls_id)])
            classes_in_frame.append(cls_name)

    # Ensure there are detections to process
    if len(dets) > 0:
        # Track the filtered detections
        tracks = tracker.update_tracks(dets, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            class_id = track.get_det_class()  # class from detection
            cls_name = class_names[class_id]

            if cls_name in TARGET_CLASSES:
                unique_ids_by_class[cls_name].add(track_id)

                # Draw bounding box and label
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(frame, f'{cls_name} ID:{track_id}', (l, t - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Write to output video
    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Final results
print("\n--- Unique Object Counts ---")
for cls in TARGET_CLASSES:
    print(f"{cls.capitalize()}: {len(unique_ids_by_class[cls])}")

