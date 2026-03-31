"""
Safety Monitor — Continuous background obstacle detection and alerts.

Combines:
- Ultrasonic sensor readings (if available) for accurate close-range distance
- Periodic YOLO detections for identifying WHAT is nearby
- Camera-based distance estimation (fallback when no ultrasonic sensor)

Runs in a background thread. Provides:
- Escalating alerts: caution → warning → STOP
- Smart cooldown to prevent alert spam
- Only alerts for objects in the user's PATH (center of frame)
- Vibration motor feedback for urgent alerts (if available)
"""

import time
import threading

from rpi_assistant.config import (
    DISTANCE_DANGER,
    DISTANCE_CAUTION,
    ALERT_COOLDOWN,
    SAFETY_CHECK_INTERVAL,
    VIBRATION_MOTOR_PIN,
    RUNNING_ON_RPI,
)
from rpi_assistant.audio_engine import AudioEngine, Priority
from rpi_assistant.distance import UltrasonicSensor, CameraDistanceEstimator


class SafetyMonitor:
    """
    Background safety monitor that warns the user about obstacles.
    
    Usage:
        monitor = SafetyMonitor(audio_engine, camera, detector)
        monitor.start()
        ...
        monitor.stop()
    """

    def __init__(self, audio, camera=None, detector=None):
        """
        Args:
            audio: AudioEngine instance for speaking alerts
            camera: Camera instance (optional, for camera-based distance)
            detector: Detector instance (optional, for identifying obstacles)
        """
        self.audio = audio
        self.camera = camera
        self.detector = detector

        # Ultrasonic sensor
        self.ultrasonic = UltrasonicSensor()

        # Vibration motor
        self._vibration_gpio = None
        self._setup_vibration()

        # State
        self._running = False
        self._thread = None
        self._last_alert_time = 0
        self._last_alert_level = None
        self._last_detection_time = 0
        self._detection_interval = 5.0  # Run YOLO every N seconds (save CPU + reduce spam)
        self._camera_alert_cooldown = 8.0  # Longer cooldown for camera-based alerts

    def _setup_vibration(self):
        """Set up vibration motor GPIO pin."""
        if RUNNING_ON_RPI and VIBRATION_MOTOR_PIN:
            try:
                import RPi.GPIO as GPIO
                GPIO.setwarnings(False)
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(VIBRATION_MOTOR_PIN, GPIO.OUT)
                GPIO.output(VIBRATION_MOTOR_PIN, False)
                self._vibration_gpio = GPIO
                print(f"[SAFETY] Vibration motor ready (GPIO {VIBRATION_MOTOR_PIN})")
            except (ImportError, Exception) as e:
                print(f"[SAFETY] Vibration motor not available: {e}")

    def start(self):
        """Start the background safety monitoring thread."""
        if self._running:
            return

        if not self.ultrasonic.is_available and self.camera is None:
            print("[SAFETY] No distance source available — safety monitor disabled")
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("[SAFETY] Background monitoring started")

    def stop(self):
        """Stop the safety monitor."""
        self._running = False
        self._vibrate(False)
        if self._thread:
            self._thread.join(timeout=5)
        self.ultrasonic.cleanup()
        print("[SAFETY] Background monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop — runs in background thread."""
        while self._running:
            try:
                distance = None
                object_label = "obstacle"

                # Method 1: Ultrasonic sensor (preferred — accurate, fast)
                if self.ultrasonic.is_available:
                    distance = self.ultrasonic.measure_averaged(samples=2)

                # Method 2: Camera + YOLO (fallback — slower, but identifies objects)
                if distance is None and self.camera and self.detector:
                    now = time.time()
                    if now - self._last_detection_time >= self._detection_interval:
                        self._last_detection_time = now
                        frame = self.camera.capture()
                        if frame is not None:
                            detections = self.detector.detect(frame)
                            if detections:
                                closest, est_distance = CameraDistanceEstimator.get_closest_object(detections)
                                if closest:
                                    distance = est_distance
                                    object_label = closest.label

                # Process distance reading
                if distance is not None:
                    self._process_distance(distance, object_label)

            except Exception as e:
                print(f"[SAFETY] Monitor error: {e}")

            time.sleep(SAFETY_CHECK_INTERVAL)

    def _process_distance(self, distance_cm, object_label="obstacle"):
        """
        Process a distance reading and issue appropriate alerts.
        
        Alert levels:
          🔴 DANGER  (< 30cm):  "Stop! [object] very close!" + vibration
          🟡 CAUTION (< 100cm): "Caution, [object] ahead at X centimeters"
          🟢 SAFE    (> 100cm): No alert
        """
        now = time.time()
        effective_cooldown = ALERT_COOLDOWN if self.ultrasonic.is_available else self._camera_alert_cooldown
        cooldown_ok = (now - self._last_alert_time) >= effective_cooldown

        if distance_cm < DISTANCE_DANGER:
            # 🔴 DANGER — always alert (override cooldown for danger)
            if cooldown_ok or self._last_alert_level != "danger":
                self._vibrate(True)
                distance_rounded = int(distance_cm)
                self.audio.speak(
                    f"Stop! {object_label} very close, {distance_rounded} centimeters!",
                    Priority.ALERT
                )
                self._last_alert_time = now
                self._last_alert_level = "danger"
                print(f"[SAFETY] 🔴 DANGER: {object_label} at {distance_cm}cm")

                # Short vibration pulse
                time.sleep(0.3)
                self._vibrate(False)

        elif distance_cm < DISTANCE_CAUTION:
            # 🟡 CAUTION
            self._vibrate(False)  # Stop vibration if was active
            if cooldown_ok:
                distance_rounded = int(distance_cm)
                self.audio.speak(
                    f"Caution, {object_label} ahead at {distance_rounded} centimeters.",
                    Priority.INFO
                )
                self._last_alert_time = now
                self._last_alert_level = "caution"
                print(f"[SAFETY] 🟡 CAUTION: {object_label} at {distance_cm}cm")

        else:
            # 🟢 SAFE — path clear
            self._vibrate(False)
            if self._last_alert_level in ("danger", "caution"):
                # Only announce "path clear" if we were previously alerting
                if cooldown_ok:
                    self.audio.speak("Path ahead is clear.", Priority.LOW)
                    self._last_alert_level = "safe"

    def _vibrate(self, on):
        """Turn vibration motor on/off."""
        if self._vibration_gpio and VIBRATION_MOTOR_PIN:
            try:
                self._vibration_gpio.output(VIBRATION_MOTOR_PIN, on)
            except Exception:
                pass


# Quick test
if __name__ == "__main__":
    print("=== Safety Monitor Test ===")
    print("This test requires either:")
    print("  - A Raspberry Pi with HC-SR04 connected, OR")
    print("  - A camera for camera-based fallback")
    print()

    audio = AudioEngine()
    audio.start()

    # Test without ultrasonic (camera fallback)
    from rpi_assistant.camera import Camera
    from rpi_assistant.detector import Detector

    camera = Camera()
    detector = Detector()

    if camera.start():
        detector.load()

        monitor = SafetyMonitor(audio, camera=camera, detector=detector)
        monitor.start()

        print("Safety monitor running for 15 seconds...")
        print("Move objects close to the camera to trigger alerts.")
        time.sleep(15)

        monitor.stop()
        camera.stop()

    audio.stop()
    print("=== Test Complete ===")
