import os
import torch
import cv2
import random
from ultralytics import YOLO
import torchvision.ops

# ---- Set Up Fallback for MPS ----
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ---- Custom CPU NMS Implementation ----
def nms_cpu(boxes, scores, iou_threshold):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(descending=True)
    keep = []
    
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        order = order[1:]
        xx1 = torch.max(x1[i], x1[order])
        yy1 = torch.max(y1[i], y1[order])
        xx2 = torch.min(x2[i], x2[order])
        yy2 = torch.min(y2[i], y2[order])
        w, h = (xx2 - xx1 + 1).clamp(min=0), (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order] - inter)
        order = order[(iou <= iou_threshold).nonzero(as_tuple=True)[0]]
    return torch.tensor(keep, dtype=torch.long)

def custom_nms(boxes, scores, iou_threshold):
    return nms_cpu(boxes.cpu(), scores.cpu(), iou_threshold)

torchvision.ops.nms = custom_nms  # Patch torchvision NMS

# ---- Device Selection ----
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ---- Load Models ----
general_model = YOLO("yolov8x.pt").to(device)  # Load YOLOv8n model
stairs_model_path = "/Users/tripathd/Downloads/Manual Library/Projects/Accesbility_Device/runs/detect/train6/weights/best.pt"
stairs_model = YOLO(stairs_model_path).to(device)  # Load stairs model

# ---- Define Vibrant Colors for Each Class ----
class_colors = {}  # Store assigned colors for each class
vibrant_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),  # Red, Green, Blue
    (255, 165, 0), (255, 255, 0), (0, 255, 255),  # Orange, Yellow, Cyan
    (255, 0, 255), (128, 0, 128), (0, 128, 0),  # Magenta, Purple, Dark Green
    (0, 128, 255), (128, 128, 0), (128, 0, 0)   # Light Blue, Olive, Maroon
]

def get_color(class_name):
    """Assigns a consistent, vibrant color to each class."""
    if class_name not in class_colors:
        class_colors[class_name] = random.choice(vibrant_colors)
    return class_colors[class_name]

# ---- Open Webcam ----
cap = cv2.VideoCapture(0
                       )  # Change source to 0 or 1 as needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on both models
    results_general = general_model(frame, device=device)[0]
    results_stairs = stairs_model(frame, device=device)[0]

    # Merge both detections
    all_boxes = []
    all_classes = []
    all_scores = []

    # Process general model detections
    if results_general.boxes is not None:
        for box in results_general.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            score = float(box.conf[0])
            class_name = general_model.names[class_id]
            
            all_boxes.append((x1, y1, x2, y2, class_name, score))

    # Process stairs model detections
    if results_stairs.boxes is not None:
        for box in results_stairs.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = float(box.conf[0])
            class_name = "Stairs"  # Override class name for stairs model
            
            all_boxes.append((x1, y1, x2, y2, class_name, score))

    # ---- Draw Annotations on Frame ----
    for (x1, y1, x2, y2, class_name, score) in all_boxes:
        color = get_color(class_name)
        label = f"{class_name} {score:.2f}"

        # Draw thicker bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)  # Increased thickness

        # Get text size
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        text_w, text_h = text_size

        # Draw black background rectangle
        cv2.rectangle(frame, (x1, y1 - text_h - 12), (x1 + text_w + 10, y1), (0, 0, 0), -1)

        # Put class label with background
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Show output
    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
