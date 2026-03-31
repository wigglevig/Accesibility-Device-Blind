import os
import torch
import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
import torchvision.ops
from PIL import Image
import depth_pro
import concurrent.futures

# -------------------------
# TTS Initialization
# -------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# -------------------------
# ANSI Terminal Colors for Output
# -------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

def get_terminal_color(label):
    mapping = {
        "person": GREEN,
        "bicycle": YELLOW,
        "car": RED,
        "motorcycle": MAGENTA,
        "airplane": CYAN,
        "bus": BLUE,
        "train": BLUE,
        "truck": YELLOW,
        "boat": CYAN,
        "traffic light": WHITE,
        "stairs": RED,
    }
    return mapping.get(label.lower(), WHITE)

# -------------------------
# Custom CPU NMS Implementation (Patched into torchvision.ops.nms)
# -------------------------
def nms_cpu(boxes, scores, iou_threshold):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
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

torchvision.ops.nms = custom_nms

# -------------------------
# Helper: Draw Text with Background (for image annotations)
# -------------------------
def draw_text_with_background(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale=1.0, thickness=2, text_color=(255,255,255), bg_color=(0,0,0)):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    if y < 40:
        y = 40 + text_height + 5
    cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

# -------------------------
# Helper: Create Panel with Top Header Bar
# -------------------------
BAR_HEIGHT = 40

def create_panel(image, target_width, target_height, header_text):
    resized = cv2.resize(image, (target_width, target_height))
    panel = np.zeros((target_height + BAR_HEIGHT, target_width, 3), dtype=np.uint8)
    panel[BAR_HEIGHT:BAR_HEIGHT+target_height, :] = resized
    cv2.rectangle(panel, (0,0), (target_width, BAR_HEIGHT), (0,0,0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(header_text, font, font_scale, thickness)
    text_x = (target_width - text_width) // 2
    text_y = (BAR_HEIGHT + text_height) // 2
    cv2.putText(panel, header_text, (text_x, text_y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    return panel

# -------------------------
# Enhanced Color Mapping for Detections
# -------------------------
def get_detection_color(label):
    color_map = {
        "person": (0, 255, 0),
        "bicycle": (255, 165, 0),
        "car": (0, 0, 255),
        "motorcycle": (128, 0, 128),
        "airplane": (255, 0, 255),
        "bus": (0, 255, 255),
        "train": (0, 128, 255),
        "truck": (128, 128, 0),
        "boat": (0, 128, 128),
        "traffic light": (255, 255, 0)
    }
    key = label.lower()
    if key in color_map:
        return color_map[key]
    h = hash(label) % 360
    hsv_color = np.uint8([[[h,255,255]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(x) for x in bgr_color)

STAIRS_COLOR = (0,0,255)

# -------------------------
# Model Loading & Device Setup
# -------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

general_model = YOLO("yolov8n.pt")
stairs_model_path = "/Users/tripathd/Downloads/Manual Library/Projects/Accesbility_Device/runs/detect/train6/weights/best.pt"
stairs_model = YOLO(stairs_model_path)

depth_model, transform = depth_pro.create_model_and_transforms()
depth_model = depth_model.to(device)
depth_model.eval()

# -------------------------
# Dashboard Settings (Fixed 16:9 Panels)
# -------------------------
TOP_WIDTH, TOP_HEIGHT = 1280, 720       # YOLO detection content area
BOTTOM_WIDTH, BOTTOM_HEIGHT = 640, 360    # Each bottom panel (Annotated Image & Depth Map)
dashboard_window = "Dashboard"
cv2.namedWindow(dashboard_window, cv2.WINDOW_NORMAL)

depth_info = None  # (annotated_frame, depth_colormap)
previous_detections = set()

# -------------------------
# Asynchronous Full-Frame Depth Inference for Selected Detection
# -------------------------
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

def run_depth_full_frame(frame, device, transform, depth_model, box):
    # Run full-frame depth inference on the entire frame.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    f_px = torch.tensor([1000], dtype=torch.float32, device=device)
    depth_input = transform(pil_img).unsqueeze(0).to(device)
    prediction = depth_model.infer(depth_input, f_px=f_px)
    depth_tensor = prediction["depth"]
    depth_np = depth_tensor.squeeze().cpu().numpy()
    # Get the center of the detection box.
    x1, y1, x2, y2 = map(int, box)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    detection_depth = depth_np[cy, cx]
    return depth_np, detection_depth, box

# -------------------------
# Open Camera (Adjust index as needed)
# -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TOP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TOP_HEIGHT)

print("Press a digit key (1-9) for that object's depth; press 'd' for center depth; press 'v' for verbal description; press 'm' for item counts; press 'q' to quit.")

depth_future = None
selected_detection_idx = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

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
    # Draw detection boxes, labels, and index numbers.
    for i, box in enumerate(all_boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = all_scores[i]
        label = all_classes[i]
        color = STAIRS_COLOR if label.lower() == "stairs" else get_detection_color(label)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {conf:.2f}"
        draw_text_with_background(display_frame, text, (x1, y1 - 5))
        idx_text = str(i + 1)
        (w_text, h_text), _ = cv2.getTextSize(idx_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.circle(display_frame, (x1 + 15, y1 + 15), 15, (0,0,0), -1)
        cv2.putText(display_frame, idx_text, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    top_panel_raw = cv2.resize(display_frame, (TOP_WIDTH, TOP_HEIGHT))
    top_panel = create_panel(top_panel_raw, TOP_WIDTH, TOP_HEIGHT, "YOLO Detection")
    dashboard = top_panel.copy()

    if depth_info is not None:
        annotated_frame, depth_colormap = depth_info
        annotated_resized = cv2.resize(annotated_frame, (BOTTOM_WIDTH, BOTTOM_HEIGHT))
        depth_resized = cv2.resize(depth_colormap, (BOTTOM_WIDTH, BOTTOM_HEIGHT))
        annotated_panel = create_panel(annotated_resized, BOTTOM_WIDTH, BOTTOM_HEIGHT, "Annotated Image")
        depth_panel = create_panel(depth_resized, BOTTOM_WIDTH, BOTTOM_HEIGHT, "Depth Map")
        bottom_row = np.hstack((annotated_panel, depth_panel))
        dashboard = np.vstack((top_panel, bottom_row))

    cv2.imshow(dashboard_window, dashboard)

    os.system('clear')
    print(f"{BOLD}Current YOLO Detections:{RESET}")
    if all_boxes:
        for i, box in enumerate(all_boxes):
            x1, y1, x2, y2 = map(int, box)
            label = all_classes[i]
            conf = all_scores[i]
            term_color = get_terminal_color(label)
            print(f"{BOLD}{term_color}{i+1}. {label}{RESET}: Confidence {conf:.2f}, Box=({x1},{y1},{x2},{y2})")
    else:
        print("No detections.")

    key = cv2.waitKey(1) & 0xFF

    # If a digit key (1-9) is pressed, run asynchronous full-frame depth inference
    if key >= ord('1') and key <= ord('9'):
        idx = key - ord('1')
        if idx < len(all_boxes):
            selected_detection_idx = idx
            chosen_label = all_classes[idx]
            box = all_boxes[idx]
            depth_future = executor.submit(run_depth_full_frame, frame, device, transform, depth_model, box)
    elif key == ord('d'):
        # 'd' key: Use center of full frame for depth.
        selected_detection_idx = None
        depth_frame = frame.copy()
        frame_rgb = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        f_px = torch.tensor([1000], dtype=torch.float32, device=device)
        depth_input = transform(pil_img).unsqueeze(0).to(device)
        prediction = depth_model.infer(depth_input, f_px=f_px)
        depth_tensor = prediction["depth"]
        depth_np = depth_tensor.squeeze().cpu().numpy()
        h_depth, w_depth = depth_np.shape
        center_depth = depth_np[h_depth//2, w_depth//2]
        speak_text = f"Distance is {center_depth:.2f} meters"
        print(f"\n{BOLD}Depth (center):{RESET} {center_depth:.2f} m")
        engine.say(speak_text)
        engine.runAndWait()
        annotated_frame = depth_frame.copy()
        frame_h, frame_w = annotated_frame.shape[:2]
        center_x, center_y = frame_w//2, frame_h//2
        draw_text_with_background(annotated_frame, f"Depth: {center_depth:.2f} m", (center_x, center_y), font_scale=1.2, thickness=2)
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        inv_depth_norm = 1.0 - depth_norm
        depth_colormap = cv2.applyColorMap((inv_depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        draw_text_with_background(depth_colormap, f"Depth: {center_depth:.2f} m", (10, 30), font_scale=1.2, thickness=2)
        depth_info = (annotated_frame, depth_colormap)
    elif key == ord('v'):
        if all_classes:
            unique_items = []
            for item in all_classes:
                if item not in unique_items:
                    unique_items.append(item)
            description = ", ".join(unique_items)
            speak_desc = f"Detected objects: {description}"
            print(f"\n{BOLD}Verbal Description:{RESET} {speak_desc}")
            engine.say(speak_desc)
            engine.runAndWait()
        else:
            engine.say("No objects detected")
            engine.runAndWait()
    elif key == ord('m'):
        if all_classes:
            counts = {}
            for item in all_classes:
                counts[item] = counts.get(item, 0) + 1
            description_list = []
            for label, count in counts.items():
                if count > 1:
                    description_list.append(f"{count} {label}s")
                else:
                    description_list.append(f"{count} {label}")
            description = ", ".join(description_list)
            speak_message = f"Detected items: {description}"
            print(f"\n{BOLD}Item Counts:{RESET} {speak_message}")
            engine.say(speak_message)
            engine.runAndWait()
        else:
            engine.say("No objects detected")
            engine.runAndWait()
    elif key == ord('q'):
        break

    # Check if asynchronous full-frame depth inference has completed.
    if depth_future is not None and depth_future.done():
        try:
            full_depth_map, detection_depth, box = depth_future.result()
            chosen_label = all_classes[selected_detection_idx] if selected_detection_idx is not None else "Unknown"
            speak_text = f"Distance for {chosen_label} is {detection_depth:.2f} meters"
            print(f"\n{BOLD}Depth for {chosen_label}:{RESET} {detection_depth:.2f} m")
            engine.say(speak_text)
            engine.runAndWait()
            # Create an annotated copy of the full frame depth map.
            annotated_frame = frame.copy()
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,0,0), 2)
            draw_text_with_background(annotated_frame, f"Depth: {detection_depth:.2f} m ({chosen_label})", ((x1+x2)//2, y1), font_scale=1.2, thickness=2)
            depth_norm = (full_depth_map - full_depth_map.min()) / (full_depth_map.max() - full_depth_map.min())
            inv_depth_norm = 1.0 - depth_norm
            depth_colormap = cv2.applyColorMap((inv_depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (0,0,0), 2)
            draw_text_with_background(depth_colormap, f"Depth: {detection_depth:.2f} m", (x1, y1-5), font_scale=1.2, thickness=2)
            depth_info = (annotated_frame, depth_colormap)
        except Exception as e:
            print("Error during depth inference:", e)
        finally:
            depth_future = None

cap.release()
cv2.destroyAllWindows()
executor.shutdown()
