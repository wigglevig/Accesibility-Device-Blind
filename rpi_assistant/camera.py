"""
Camera Module — Capture frames from USB webcam or Pi Camera.

Provides:
- Single frame capture (for Gemini API / YOLO inference)
- Frame-to-JPEG encoding (for sending to Gemini)
- Automatic camera detection and fallback
"""

import cv2
import base64
import time
import threading

from rpi_assistant.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT


class Camera:
    """
    Camera interface for capturing frames.
    
    Usage:
        cam = Camera()
        cam.start()
        frame = cam.capture()          # Get a numpy frame
        jpg_b64 = cam.capture_base64() # Get base64-encoded JPEG (for APIs)
        cam.stop()
    """

    def __init__(self, index=None, width=None, height=None):
        self._index = index if index is not None else CAMERA_INDEX
        self._width = width or CAMERA_WIDTH
        self._height = height or CAMERA_HEIGHT
        self._cap = None
        self._lock = threading.Lock()
        self._last_frame = None
        self._running = False

    def start(self):
        """Open the camera."""
        if self._running:
            return True

        print(f"[CAMERA] Opening camera {self._index}...")
        self._cap = cv2.VideoCapture(self._index)

        if not self._cap.isOpened():
            # Try alternate indices
            for alt_index in [0, 1, 2]:
                if alt_index != self._index:
                    print(f"[CAMERA] Trying alternate index {alt_index}...")
                    self._cap = cv2.VideoCapture(alt_index)
                    if self._cap.isOpened():
                        self._index = alt_index
                        break

        if not self._cap.isOpened():
            print("[CAMERA] ERROR: Could not open any camera!")
            return False

        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        # Read actual resolution (camera may not support requested size)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAMERA] Opened camera {self._index} at {actual_w}×{actual_h}")

        self._running = True

        # Warm up — first few frames may be dark
        for _ in range(5):
            self._cap.read()

        return True

    def stop(self):
        """Release the camera."""
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        print("[CAMERA] Camera released")

    def capture(self):
        """
        Capture a single frame.
        
        Returns:
            numpy array (BGR format), or None on failure.
        """
        if not self._running or self._cap is None:
            print("[CAMERA] Camera not started")
            return None

        with self._lock:
            ret, frame = self._cap.read()
            if ret:
                self._last_frame = frame
                return frame
            else:
                print("[CAMERA] Failed to capture frame")
                return self._last_frame  # Return last good frame as fallback

    def capture_base64(self, quality=80):
        """
        Capture a frame and return as base64-encoded JPEG string.
        Useful for sending to Gemini API.
        
        Args:
            quality: JPEG quality (0-100). Lower = smaller payload, faster upload.
            
        Returns:
            Base64 string, or None on failure.
        """
        frame = self.capture()
        if frame is None:
            return None

        # Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, jpeg_data = cv2.imencode('.jpg', frame, encode_params)

        # Convert to base64
        b64_string = base64.b64encode(jpeg_data.tobytes()).decode('utf-8')
        return b64_string

    def capture_jpeg_bytes(self, quality=80):
        """
        Capture a frame and return as raw JPEG bytes.
        
        Returns:
            bytes object, or None on failure.
        """
        frame = self.capture()
        if frame is None:
            return None

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, jpeg_data = cv2.imencode('.jpg', frame, encode_params)
        return jpeg_data.tobytes()

    @property
    def is_opened(self):
        return self._running and self._cap is not None and self._cap.isOpened()


# Quick test
if __name__ == "__main__":
    print("=== Camera Test ===")
    cam = Camera()
    if cam.start():
        # Capture a frame
        frame = cam.capture()
        if frame is not None:
            print(f"Frame shape: {frame.shape}")
            print(f"Frame dtype: {frame.dtype}")

            # Test base64 encoding
            b64 = cam.capture_base64()
            if b64:
                print(f"Base64 length: {len(b64)} chars")

            # Save a test frame
            cv2.imwrite("/tmp/camera_test.jpg", frame)
            print("Saved test frame to /tmp/camera_test.jpg")
        else:
            print("Failed to capture frame!")

        cam.stop()
    else:
        print("Failed to open camera!")

    print("=== Test Complete ===")
