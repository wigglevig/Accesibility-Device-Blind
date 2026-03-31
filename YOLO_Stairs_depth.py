import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import torchvision.ops
from PIL import Image
import depth_pro

# ---- Set Up Fallback for MPS (if needed) ----
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
        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order] - inter)
        order = order[(iou <= iou_threshold).nonzero(as_tuple=True)[0]]
    return torch.tensor(keep, dtype=torch.long)

def custom_nms(boxes, scores, iou_threshold):
    return nms_cpu(boxes.cpu(), scores.cpu(), iou_threshold)

torchvision.ops.nms = custom_nms  # Patch torchvision NMS

# ---- Device Selection: Prefer CUDA, then MPS, then CPU ----
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ---- Helper: Draw Text with a Background Box ----
def draw_text_with_background(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale=1.0, thickness=2, text_color=(255,255,255), bg_color=(0,0,0)):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    # Ensure text is not drawn over header bar
    if y < 40:
        y = 40 + text_height + 5
    cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

# ---- Enhanced Color Mapping for Classes ----
def get_detection_color(label):
    # Predefined colors for some common classes.
    color_map = {
        "person": (0, 255, 0),         # green
        "bicycle": (255, 165, 0),      # orange
        "car": (0, 0, 255),            # red
        "motorcycle": (128, 0, 128),   # purple
        "airplane": (255, 0, 255),     # magenta
        "bus": (0, 255, 255),          # yellow
        "train": (0, 128, 255),        # blue-ish
        "truck": (128, 128, 0),        # olive
        "boat": (0, 128, 128),         # teal
        "traffic light": (255, 255, 0) # cyan
    }
    key = label.lower()
    if key in color_map:
        return color_map[key]
    # For any label not predefined, generate a consistent color using its hash.
    h = hash(label) % 360  # Hue value between 0-359.
    hsv_color = np.uint8([[[h, 255, 255]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(x) for x in bgr_color)

# For our custom stairs, always use red.
STAIRS_COLOR = (0, 0, 255)

# ---- Helper: Create Panel with a Top Header Bar ----
BAR_HEIGHT = 40  # Header bar height

def create_panel(image, target_width, target_height, header_text):
    # Resize content image to fixed size.
    resized = cv2.resize(image, (target_width, target_height))
    # Create a new panel with extra space for header.
    panel = np.zeros((target_height + BAR_HEIGHT, target_width, 3), dtype=np.uint8)
    panel[BAR_HEIGHT:BAR_HEIGHT + target_height, :] = resized
    # Draw header bar.
    cv2.rectangle(panel, (0, 0), (target_width, BAR_HEIGHT), (0, 0, 0), -1)
    # Center header text.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(header_text, font, font_scale, thickness)
    text_x = (target_width - text_width) // 2
    text_y = (BAR_HEIGHT + text_height) // 2
    cv2.putText(panel, header_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return panel

# ---- Load YOLO Models ----
general_model = YOLO("yolov8x.pt")
stairs_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect", "train6", "weights", "best.pt")
stairs_model = YOLO(stairs_model_path)
# We'll force inference on GPU by using predict() with device parameter.

# ---- Load Depth Model from ml-depth-pro ----
depth_model, transform = depth_pro.create_model_and_transforms()
depth_model = depth_model.to(device)
depth_model.eval()

# ---- Dashboard Settings (fixed 16:9 panels) ----
TOP_WIDTH, TOP_HEIGHT = 1280, 720       # YOLO detection content
BOTTOM_WIDTH, BOTTOM_HEIGHT = 640, 360    # Each bottom panel

dashboard_window = "Dashboard"
cv2.namedWindow(dashboard_window, cv2.WINDOW_NORMAL)

# Store the most recent depth info as a tuple: (annotated_frame, depth_colormap)
depth_info = None

# ---- Open Webcam ----
cap = cv2.VideoCapture(1)  # Adjust camera index as needed
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Press 'd' to run depth estimation and update dashboard; press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on current frame.
    results_general = general_model.predict(source=frame, device=device)[0]
    results_stairs = stairs_model.predict(source=frame, device=device)[0]

    all_boxes = []
    all_classes = []
    all_scores = []
    if results_general.boxes is not None:
        for box in results_general.boxes:
            all_boxes.append(box.xyxy.cpu().numpy()[0])
            cls_idx = int(box.cls.cpu().numpy()[0])
            all_classes.append(general_model.names[cls_idx])
            all_scores.append(float(box.conf.cpu().numpy()[0]))
    if results_stairs.boxes is not None:
        for box in results_stairs.boxes:
            all_boxes.append(box.xyxy.cpu().numpy()[0])
            all_classes.append("Stairs")
            all_scores.append(float(box.conf.cpu().numpy()[0]))

    display_frame = frame.copy()
    for i, box in enumerate(all_boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = all_scores[i]
        label = all_classes[i]
        if label.lower() == "stairs":
            color = STAIRS_COLOR
        else:
            color = get_detection_color(label)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {conf:.2f}"
        draw_text_with_background(display_frame, text, (x1, y1 - 5))
    
    # Create top panel for YOLO detection.
    top_panel_raw = cv2.resize(display_frame, (TOP_WIDTH, TOP_HEIGHT))
    top_panel = create_panel(top_panel_raw, TOP_WIDTH, TOP_HEIGHT, "YOLO Detection")
    
    dashboard = top_panel.copy()
    
    # Add bottom panels if depth info is available.
    if depth_info is not None:
        annotated_frame, depth_colormap = depth_info
        annotated_resized = cv2.resize(annotated_frame, (BOTTOM_WIDTH, BOTTOM_HEIGHT))
        depth_resized = cv2.resize(depth_colormap, (BOTTOM_WIDTH, BOTTOM_HEIGHT))
        annotated_panel = create_panel(annotated_resized, BOTTOM_WIDTH, BOTTOM_HEIGHT, "Annotated Image")
        depth_panel = create_panel(depth_resized, BOTTOM_WIDTH, BOTTOM_HEIGHT, "Depth Map")
        bottom_row = np.hstack((annotated_panel, depth_panel))
        dashboard = np.vstack((top_panel, bottom_row))
    
    cv2.imshow(dashboard_window, dashboard)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("d"):
        # Run depth estimation on current frame.
        depth_frame = frame.copy()
        frame_rgb = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        f_px = torch.tensor([1000], dtype=torch.float32, device=device)  # Adjust focal length as needed.
        depth_input = transform(pil_img).unsqueeze(0).to(device)
        prediction = depth_model.infer(depth_input, f_px=f_px)
        depth_tensor = prediction["depth"]
        depth_np = depth_tensor.squeeze().cpu().numpy()
        h_depth, w_depth = depth_np.shape
        center_depth = depth_np[h_depth // 2, w_depth // 2]
        print(f"Estimated depth at center: {center_depth:.2f} m")
        
        annotated_frame = depth_frame.copy()
        frame_h, frame_w = annotated_frame.shape[:2]
        center_x, center_y = frame_w // 2, frame_h // 2
        detection_annotated = False
        for box in all_boxes:
            x1, y1, x2, y2 = map(int, box)
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                draw_text_with_background(annotated_frame, f"Depth: {center_depth:.2f} m", (x1, y1 - 10),
                                          font_scale=1.2, thickness=2)
                detection_annotated = True
                break
        if not detection_annotated:
            draw_text_with_background(annotated_frame, f"Depth: {center_depth:.2f} m", (center_x, center_y),
                                      font_scale=1.2, thickness=2)
        
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        inv_depth_norm = 1.0 - depth_norm
        depth_colormap = cv2.applyColorMap((inv_depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        draw_text_with_background(depth_colormap, f"Depth: {center_depth:.2f} m", (10, 30),
                                  font_scale=1.2, thickness=2)
        depth_info = (annotated_frame, depth_colormap)
    
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
