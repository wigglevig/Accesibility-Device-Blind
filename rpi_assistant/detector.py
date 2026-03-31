"""
Object Detector — YOLOv8 nano for real-time object detection.

Designed for Raspberry Pi 4B:
- Uses yolov8n (nano) — lightest model, ~6MB
- Runs on CPU (RPi doesn't have CUDA)
- Provides spatial awareness (left/center/right, top/mid/bottom)
- Generates natural-language summaries for TTS
"""

import os
import time
import numpy as np
from ultralytics import YOLO

from rpi_assistant.config import (
    YOLO_MODEL,
    YOLO_CONFIDENCE,
    POSITION_THRESHOLD_X,
    POSITION_THRESHOLD_Y,
    RUNNING_ON_RPI,
)


class Detection:
    """A single detected object with spatial information."""

    def __init__(self, label, confidence, box, frame_width, frame_height):
        self.label = label
        self.confidence = confidence
        self.box = box  # (x1, y1, x2, y2)
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Calculate spatial position
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Horizontal position
        if cx < frame_width * POSITION_THRESHOLD_X:
            self.horizontal = "left"
        elif cx > frame_width * (1 - POSITION_THRESHOLD_X):
            self.horizontal = "right"
        else:
            self.horizontal = "center"

        # Vertical position (less useful for blind user, but available)
        if cy < frame_height * POSITION_THRESHOLD_Y:
            self.vertical = "top"
        elif cy > frame_height * (1 - POSITION_THRESHOLD_Y):
            self.vertical = "bottom"
        else:
            self.vertical = "middle"

        # Relative size (rough distance proxy when no ultrasonic sensor)
        box_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_width * frame_height
        self.relative_size = box_area / frame_area  # 0.0 to 1.0

        # Rough distance category based on box size
        if self.relative_size > 0.3:
            self.distance_hint = "very close"
        elif self.relative_size > 0.1:
            self.distance_hint = "nearby"
        elif self.relative_size > 0.03:
            self.distance_hint = "a few meters away"
        else:
            self.distance_hint = "far away"

    @property
    def position(self):
        """Human-readable position string."""
        if self.horizontal == "center":
            return "ahead"
        return f"on the {self.horizontal}"

    def __repr__(self):
        return f"Detection({self.label}, {self.confidence:.2f}, {self.position})"


class Detector:
    """
    YOLOv8 object detector optimized for RPi.
    
    Usage:
        detector = Detector()
        detector.load()
        detections = detector.detect(frame)
        summary = detector.summarize(detections)
    """

    def __init__(self, model_path=None):
        self._model_path = model_path or YOLO_MODEL
        self._model = None
        self._device = "cpu"  # RPi doesn't have GPU; Mac can use MPS

    def load(self):
        """Load the YOLO model."""
        print(f"[DETECTOR] Loading model: {self._model_path}")
        start = time.time()

        # Determine device
        try:
            import torch
            if not RUNNING_ON_RPI and torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
        except ImportError:
            pass

        # Check if model file exists, download if needed
        model_file = self._model_path
        if not os.path.isabs(model_file):
            # Check in project models directory first
            project_model = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "rpi_assistant", "models", model_file
            )
            if os.path.exists(project_model):
                model_file = project_model

        self._model = YOLO(model_file)

        elapsed = time.time() - start
        print(f"[DETECTOR] Model loaded on '{self._device}' in {elapsed:.1f}s")
        return True

    def detect(self, frame):
        """
        Run object detection on a frame.
        
        Args:
            frame: numpy array (BGR format from OpenCV)
            
        Returns:
            List of Detection objects, sorted by confidence (highest first).
        """
        if self._model is None:
            print("[DETECTOR] Model not loaded!")
            return []

        h, w = frame.shape[:2]

        # Run inference
        results = self._model.predict(
            source=frame,
            device=self._device,
            conf=YOLO_CONFIDENCE,
            verbose=False,  # Suppress YOLO output spam
        )[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                xyxy = box.xyxy.cpu().numpy()[0]
                cls_idx = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                label = self._model.names[cls_idx]

                det = Detection(
                    label=label,
                    confidence=conf,
                    box=xyxy,
                    frame_width=w,
                    frame_height=h,
                )
                detections.append(det)

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def summarize(self, detections):
        """
        Generate a natural-language summary of detections for TTS.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            Human-readable string suitable for speaking aloud.
            
        Examples:
            "I see 2 people ahead and 1 car on the left."
            "There's a chair on the right, nearby."
            "The path looks clear."
        """
        if not detections:
            return "I don't see any objects right now."

        # Group by label and position
        groups = {}
        for det in detections:
            key = (det.label, det.position)
            if key not in groups:
                groups[key] = {
                    "count": 0,
                    "label": det.label,
                    "position": det.position,
                    "distance_hint": det.distance_hint,
                    "max_conf": det.confidence,
                }
            groups[key]["count"] += 1
            groups[key]["max_conf"] = max(groups[key]["max_conf"], det.confidence)

        # Build sentence parts
        parts = []
        for key, info in groups.items():
            count = info["count"]
            label = info["label"]
            position = info["position"]
            distance = info["distance_hint"]

            if count > 1:
                label_text = f"{count} {label}s"
            else:
                # Use article
                article = "an" if label[0].lower() in "aeiou" else "a"
                label_text = f"{article} {label}"

            parts.append(f"{label_text} {position}, {distance}")

        if len(parts) == 1:
            return f"I see {parts[0]}."
        elif len(parts) == 2:
            return f"I see {parts[0]} and {parts[1]}."
        else:
            # Join all but last with commas, then "and" for last
            return f"I see {', '.join(parts[:-1])}, and {parts[-1]}."

    def summarize_brief(self, detections):
        """
        Generate a very brief count summary.
        
        Example: "2 people, 1 car, 1 chair"
        """
        if not detections:
            return "No objects detected."

        counts = {}
        for det in detections:
            counts[det.label] = counts.get(det.label, 0) + 1

        parts = []
        for label, count in counts.items():
            if count > 1:
                parts.append(f"{count} {label}s")
            else:
                parts.append(f"{count} {label}")

        return f"Detected: {', '.join(parts)}."

    def get_hazards(self, detections):
        """
        Filter detections for potential hazards (objects the user should be warned about).
        
        Returns list of hazard detections that are in the user's path (center).
        """
        hazard_labels = {
            "car", "truck", "bus", "motorcycle", "bicycle",
            "stairs", "dog", "horse", "cow",
            "fire hydrant", "stop sign", "traffic light",
        }

        hazards = []
        for det in detections:
            if det.label.lower() in hazard_labels and det.horizontal == "center":
                hazards.append(det)

        return hazards


# Quick test
if __name__ == "__main__":
    import cv2

    print("=== Detector Test ===")
    detector = Detector()
    detector.load()

    # Capture from camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Frame: {frame.shape}")
            detections = detector.detect(frame)
            print(f"Found {len(detections)} objects:")
            for d in detections:
                print(f"  {d}")
            print(f"\nSummary: {detector.summarize(detections)}")
            print(f"Brief: {detector.summarize_brief(detections)}")
            hazards = detector.get_hazards(detections)
            if hazards:
                print(f"Hazards: {[str(h) for h in hazards]}")
        cap.release()
    else:
        print("No camera available for test")

    print("=== Test Complete ===")
