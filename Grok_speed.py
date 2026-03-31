import os
import time
import torch
import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
import torchvision.ops
from PIL import Image
import depth_pro
import concurrent.futures
import random
import speech_recognition as sr
from collections import deque
from scipy.spatial.distance import cdist

# Use MPS fallback if needed
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# TTS Initialization
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)
def speak(text):
    engine.say(text)
    engine.runAndWait()

# ANSI Terminal Colors for Output
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
        "door_handle": (255, 0, 255)
    }
    return mapping.get(label.lower(), WHITE)

# Custom CPU NMS Implementation
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
torchvision.ops.nms = custom_nms

# Helper Functions
def draw_text_with_background(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale=1.0, thickness=2, text_color=(255,255,255), bg_color=(0,0,0)):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    if y < 40:
        y = 40 + text_height + 5
    cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

BAR_HEIGHT = 40
def create_panel(image, target_width, target_height, header_text):
    resized = cv2.resize(image, (target_width, target_height))
    panel = np.zeros((target_height + BAR_HEIGHT, target_width, 3), dtype=np.uint8)
    panel[BAR_HEIGHT:BAR_HEIGHT+target_height, :] = resized
    cv2.rectangle(panel, (0, 0), (target_width, BAR_HEIGHT), (0,0,0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(header_text, font, font_scale, thickness)
    text_x = (target_width - text_width) // 2
    text_y = (BAR_HEIGHT + text_height) // 2
    cv2.putText(panel, header_text, (text_x, text_y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    return panel

# Color Mapping
class_colors = {}
vibrant_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 165, 0), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 0, 128), (0, 128, 0),
    (0, 128, 255), (128, 128, 0), (128, 0, 0)
]
def get_detection_color(label):
    if label.lower() == "stairs":
        return (0, 0, 255)
    if label.lower() == "door_handle":
        return (255, 0, 255)
    if label not in class_colors:
        class_colors[label] = random.choice(vibrant_colors)
    return class_colors[label]
STAIRS_COLOR = (0, 0, 255)

# Device & Model Loading
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

# Dashboard Settings
TOP_WIDTH, TOP_HEIGHT = 1280, 720
BOTTOM_WIDTH, BOTTOM_HEIGHT = 640, 360
dashboard_window = "Dashboard"
cv2.namedWindow(dashboard_window, cv2.WINDOW_NORMAL)

depth_info = None
last_yolo_results = None
frame_count = 0

# Asynchronous Depth Inference
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
def run_depth_full_frame(frame, device, transform, depth_model, box=None):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    f_px = torch.tensor([1000], dtype=torch.float32, device=device)
    depth_input = transform(pil_img).unsqueeze(0).to(device)
    prediction = depth_model.infer(depth_input, f_px=f_px)
    depth_tensor = prediction["depth"]
    depth_np = depth_tensor.squeeze().cpu().numpy()
    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        detection_depth = depth_np[cy, cx]
        return depth_np, detection_depth, box
    return depth_np, None, None

# Speech Recognition
def listen_for_command(duration=3):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for voice command...")
        audio = recognizer.listen(source, phrase_time_limit=duration)
    try:
        command = recognizer.recognize_google(audio)
        print("Heard command:", command)
        return command.lower()
    except sr.UnknownValueError:
        print("Voice command not understood")
        return ""
    except sr.RequestError as e:
        print("Speech recognition error:", e)
        return ""
def parse_distance_command(command, detected_labels):
    for label in detected_labels:
        if label.lower() in command:
            return label.lower()
    return None

# Tracking Class
class Track:
    def __init__(self, id, label, box, depth, timestamp):
        self.id = id
        self.label = label
        self.box = box
        self.history = deque([(timestamp, self.get_centroid(box), depth)], maxlen=10)
        self.missed_frames = 0
        self.last_alert_time = 0

    def get_centroid(self, box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2

    def update(self, box, depth, timestamp):
        centroid = self.get_centroid(box)
        self.box = box
        self.history.append((timestamp, centroid, depth))
        self.missed_frames = 0

    @property
    def is_moving(self):
        if len(self.history) < 2:
            return False
        first_centroid = self.history[0][1]
        last_centroid = self.history[-1][1]
        distance = np.linalg.norm(np.array(first_centroid) - np.array(last_centroid))
        return distance > 20

    def get_speed(self):
        depth_entries = [(t, d) for t, _, d in self.history if d is not None]
        if len(depth_entries) < 2:
            return None
        t1, d1 = depth_entries[0]
        t2, d2 = depth_entries[-1]
        if t2 - t1 > 0.1:
            speed = (d2 - d1) / (t2 - t1)
            return speed
        return None

# Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TOP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TOP_HEIGHT)

# Initialize Variables
print("Controls:")
print("  Digit keys (1-9): Get that object's depth manually")
print("  'd': Get center depth")
print("  'v': Verbal summary of detections")
print("  'V': Verbal item counts")
print("  'c': Record a distance query")
print("  's': Get speed of moving objects")
print("  'q': Quit")

depth_future = None
selected_detection_idx = None
current_frame = None
hazard_last_alert = 0
hazard_cooldown = 5
hazard_labels = ["stairs", "door_handle"]
target_classes = ["car", "bicycle", "motorcycle", "airplane", "bus", "truck"]
depth_interval = 5
track_id_counter = 0
tracks = []
periodic_depth_future = None
latest_depth_map = None

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_frame = frame.copy()
    frame_count += 1

    # Periodic Depth Computation
    if frame_count % depth_interval == 0:
        if periodic_depth_future is None or periodic_depth_future.done():
            periodic_depth_future = executor.submit(run_depth_full_frame, current_frame, device, transform, depth_model, None)

    if periodic_depth_future is not None and periodic_depth_future.done():
        try:
            depth_np, _, _ = periodic_depth_future.result()
            latest_depth_map = depth_np
        except Exception as e:
            print("Error during depth inference:", e)
        periodic_depth_future = None

    # YOLO Inference
    if frame_count % 2 == 0:
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
        last_yolo_results = (all_boxes, all_classes, all_scores)
    else:
        if last_yolo_results is not None:
            all_boxes, all_classes, all_scores = last_yolo_results
        else:
            all_boxes, all_classes, all_scores = [], [], []

    # Tracking
    current_time = time.time()
    det_centroids = [((x1 + x2)/2, (y1 + y2)/2) for x1, y1, x2, y2 in all_boxes]
    track_centroids = [track.history[-1][1] for track in tracks] if tracks else []
    assignment = []
    if det_centroids and track_centroids:
        distances = cdist(det_centroids, track_centroids)
        for i in range(len(det_centroids)):
            min_dist = min(distances[i])
            if min_dist < 50:
                j = np.argmin(distances[i])
                assignment.append((i, j))

    # Update Assigned Tracks
    for det_idx, track_idx in assignment:
        box = all_boxes[det_idx]
        label = all_classes[det_idx]
        if label == tracks[track_idx].label:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            depth = latest_depth_map[cy, cx] if latest_depth_map is not None else None
            tracks[track_idx].update(box, depth, current_time)

    # Create New Tracks
    assigned_dets = [det_idx for det_idx, _ in assignment]
    for i in range(len(all_boxes)):
        if i not in assigned_dets:
            box = all_boxes[i]
            label = all_classes[i]
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            depth = latest_depth_map[cy, cx] if latest_depth_map is not None else None
            new_track = Track(track_id_counter, label, box, depth, current_time)
            tracks.append(new_track)
            track_id_counter += 1

    # Handle Missed Tracks
    assigned_tracks = [track_idx for _, track_idx in assignment]
    for i in range(len(tracks)):
        if i not in assigned_tracks:
            tracks[i].missed_frames += 1
    tracks = [t for t in tracks if t.missed_frames < 5]

    # Alerts for Target Objects
    for track in tracks:
        if track.label.lower() in target_classes:
            status = "moving" if track.is_moving else "stationary"
            if current_time - track.last_alert_time > 5:
                speak(f"{track.label} detected, {status}")
                track.last_alert_time = current_time

    # Draw Detections
    display_frame = frame.copy()
    for i, box in enumerate(all_boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = all_scores[i]
        label = all_classes[i]
        color = STAIRS_COLOR if label.lower() == "stairs" else get_detection_color(label)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {conf:.2f}"
        draw_text_with_background(display_frame, text, (x1, y1 - 5))
        idx_text = str(i + 1)
        cv2.circle(display_frame, (x1 + 15, y1 + 15), 15, (0,0,0), -1)
        cv2.putText(display_frame, idx_text, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Build Dashboard
    top_panel = create_panel(cv2.resize(display_frame, (TOP_WIDTH, TOP_HEIGHT)), TOP_WIDTH, TOP_HEIGHT, "YOLO Detection")
    dashboard = top_panel.copy()
    if depth_info is not None:
        annotated_panel = create_panel(cv2.resize(depth_info[0], (BOTTOM_WIDTH, BOTTOM_HEIGHT)), BOTTOM_WIDTH, BOTTOM_HEIGHT, "Annotated Image")
        depth_panel = create_panel(cv2.resize(depth_info[1], (BOTTOM_WIDTH, BOTTOM_HEIGHT)), BOTTOM_WIDTH, BOTTOM_HEIGHT, "Depth Map")
        dashboard = np.vstack((top_panel, np.hstack((annotated_panel, depth_panel))))

    cv2.imshow(dashboard_window, dashboard)
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

    # Hazard Alerts
    current_time = time.time()
    for i, label in enumerate(all_classes):
        if label.lower() in hazard_labels:
            x1, y1, x2, y2 = map(int, all_boxes[i])
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            if (TOP_WIDTH/3 < x_center < 2*TOP_WIDTH/3) and (y_center > TOP_HEIGHT*0.6):
                if current_time - hazard_last_alert > hazard_cooldown:
                    hazard_last_alert = current_time
                    speak(f"Warning: {label} ahead!")
                    print(f"Warning: {label} ahead!")
    key = cv2.waitKey(1) & 0xFF

    # Key Handlers
    if key >= ord('1') and key <= ord('9'):
        idx = key - ord('1')
        if idx < len(all_boxes):
            selected_detection_idx = idx
            box = all_boxes[idx]
            depth_future = executor.submit(run_depth_full_frame, current_frame, device, transform, depth_model, box)
    elif key == ord('d'):
        selected_detection_idx = None
        depth_frame = current_frame.copy()
        frame_rgb = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        f_px = torch.tensor([1000], dtype=torch.float32, device=device)
        depth_input = transform(pil_img).unsqueeze(0).to(device)
        prediction = depth_model.infer(depth_input, f_px=f_px)
        depth_np = prediction["depth"].squeeze().cpu().numpy()
        center_depth = depth_np[depth_np.shape[0]//2, depth_np.shape[1]//2]
        speak_text = f"Distance is {center_depth:.2f} meters"
        print(f"\n{BOLD}Depth (center):{RESET} {center_depth:.2f} m")
        speak(speak_text)
        annotated_frame = depth_frame.copy()
        draw_text_with_background(annotated_frame, f"Depth: {center_depth:.2f} m", (annotated_frame.shape[1]//2, annotated_frame.shape[0]//2), font_scale=1.2, thickness=2)
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        inverted = ((1 - depth_norm) * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(inverted, cv2.COLORMAP_JET)
        draw_text_with_background(depth_colormap, f"Depth: {center_depth:.2f} m", (10, 30), font_scale=1.2, thickness=2)
        thumb = cv2.resize(depth_colormap, (640, 360))
        depth_info = (annotated_frame, thumb)
    elif key == ord('v'):
        if all_boxes:
            summary = {}
            for i, box in enumerate(all_boxes):
                x1, y1, x2, y2 = map(int, box)
                label = all_classes[i]
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                horiz = "left" if x_center < TOP_WIDTH / 3 else "right" if x_center > 2 * TOP_WIDTH / 3 else "center"
                vert = "top" if y_center < TOP_HEIGHT / 3 else "bottom" if y_center > 2 * TOP_HEIGHT / 3 else "middle"
                pos = f"{vert} {horiz}"
                key_label = f"{label} {pos}"
                summary[key_label] = summary.get(key_label, 0) + 1
            summary_list = [f"{count} {key_label}" for key_label, count in summary.items()]
            final_summary = ", ".join(summary_list)
            speak_desc = f"Detected objects: {final_summary}"
            print(f"\n{BOLD}Verbal Description:{RESET} {speak_desc}")
            speak(speak_desc)
        else:
            speak("No objects detected")
    elif key == ord('V'):
        if all_boxes:
            counts = {}
            for label in all_classes:
                counts[label] = counts.get(label, 0) + 1
            description_list = [f"{count} {label}" for label, count in counts.items()]
            speak_message = f"Detected items: {', '.join(description_list)}"
            print(f"\n{BOLD}Item Counts:{RESET} {speak_message}")
            speak(speak_message)
        else:
            speak("No objects detected")
    elif key == ord('c'):
        print("Recording voice command for distance query...")
        command = listen_for_command(duration=3)
        if command and last_yolo_results is not None:
            _, detected_labels, _ = last_yolo_results
            query_obj = parse_distance_command(command, detected_labels)
            if query_obj:
                found_idx = next((i for i, label in enumerate(detected_labels) if query_obj in label.lower()), None)
                if found_idx is not None:
                    print(f"Voice command: Found {detected_labels[found_idx]}. Fetching its distance.")
                    speak(f"{detected_labels[found_idx]} detected. Fetching its distance.")
                    selected_detection_idx = found_idx
                    box = all_boxes[found_idx]
                    depth_future = executor.submit(run_depth_full_frame, current_frame, device, transform, depth_model, box)
                else:
                    speak("Object not detected in the scene.")
            else:
                speak("Could not understand your distance query.")
        else:
            speak("No detection results available yet.")
    elif key == ord('s'):
        for track in tracks:
            if track.label.lower() in target_classes and track.is_moving:
                speed = track.get_speed()
                if speed is not None:
                    if abs(speed) < 0.1:
                        speak(f"The {track.label} is stationary")
                    elif speed > 0:
                        speak(f"The {track.label} is moving away at {speed:.2f} meters per second")
                    else:
                        speak(f"The {track.label} is approaching at {-speed:.2f} meters per second")
    elif key == ord('q'):
        break

    # Depth Future Handling
    if depth_future is not None and depth_future.done():
        try:
            full_depth_map, detection_depth, box = depth_future.result()
            chosen_label = all_classes[selected_detection_idx] if selected_detection_idx is not None else "Unknown"
            speak_text = f"Distance for {chosen_label} is {detection_depth:.2f} meters"
            print(f"\n{BOLD}Depth for {chosen_label}:{RESET} {detection_depth:.2f} m")
            speak(speak_text)
            annotated_frame = current_frame.copy()
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,0,0), 2)
            draw_text_with_background(annotated_frame, f"Depth: {detection_depth:.2f} m ({chosen_label})", ((x1+x2)//2, y1), font_scale=1.2, thickness=2)
            depth_norm = (full_depth_map - full_depth_map.min()) / (full_depth_map.max() - full_depth_map.min())
            inverted = ((1 - depth_norm) * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(inverted, cv2.COLORMAP_JET)
            cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (0,0,0), 2)
            draw_text_with_background(depth_colormap, f"Depth: {detection_depth:.2f} m", (x1, y1-5), font_scale=1.2, thickness=2)
            thumb = cv2.resize(depth_colormap, (640, 360))
            depth_info = (annotated_frame, thumb)
        except Exception as e:
            print("Error during depth inference:", e)
        finally:
            depth_future = None

cap.release()
cv2.destroyAllWindows()
executor.shutdown()