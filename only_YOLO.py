import os
# IMPORTANT: Set the fallback flag before any torch import!
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import cv2
from ultralytics import YOLO
import torchvision.ops

# ---- Custom CPU NMS Implementation ----
def nms_cpu(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression on CPU.
    
    boxes: Tensor of shape [N, 4] in (x1, y1, x2, y2) format.
    scores: Tensor of shape [N] with the confidence scores.
    iou_threshold: IoU threshold for suppression.
    
    Returns a tensor of indices to keep.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by scores (highest first)
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
        
        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order] - inter)
        # Keep boxes with IoU less than or equal to the threshold
        inds = (iou <= iou_threshold).nonzero(as_tuple=True)[0]
        order = order[inds]
    
    return torch.tensor(keep, dtype=torch.long)

def custom_nms(boxes, scores, iou_threshold):
    # Force boxes and scores to CPU before applying NMS
    boxes_cpu = boxes.cpu()
    scores_cpu = scores.cpu()
    return nms_cpu(boxes_cpu, scores_cpu, iou_threshold)

# Monkey-patch torchvision's NMS with our custom CPU version.
torchvision.ops.nms = custom_nms

# ---- End of NMS patch ----

# Select device: use MPS if available; otherwise, use CPU.
if torch.backends.mps.is_available():
    device = "mps"
    print("Using MPS device with fallback enabled.")
else:
    device = "cpu"
    print("Using CPU device.")

# Load YOLOv8 extra large model and move it to the selected device.
model = YOLO("yolov8x.pt")
model.to(device)

# Use streaming inference from the webcam.
results = model.predict(source=1, stream=True, device=device, show=True)              # To change camera


# Loop through frames and display the annotated output.
for result in results:
    frame = result.plot()  # Returns the frame with drawn predictions.
    cv2.imshow("YOLOv8 Inference", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
