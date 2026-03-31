"""
Distance Module — Ultrasonic sensor (HC-SR04) with software fallback.

On Raspberry Pi with HC-SR04 connected:
  - Measures real distance in centimeters using ultrasonic pulses
  - Accurate range: 2cm to 400cm

Without ultrasonic sensor:
  - Falls back to camera-based rough estimation using YOLO bounding box sizes
  - Less accurate but requires no extra hardware

The module auto-detects whether the sensor is available.
"""

import time
import threading

from rpi_assistant.config import (
    ULTRASONIC_TRIGGER_PIN,
    ULTRASONIC_ECHO_PIN,
    RUNNING_ON_RPI,
    SAFETY_CHECK_INTERVAL,
)


class UltrasonicSensor:
    """
    HC-SR04 ultrasonic distance sensor via GPIO.
    
    Wiring:
      VCC  → 5V
      Trig → GPIO 23
      Echo → GPIO 24 (through voltage divider: 1kΩ + 2kΩ to protect 3.3V GPIO)
      GND  → GND
    """

    def __init__(self):
        self._gpio = None
        self._available = False
        self._last_distance = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        if RUNNING_ON_RPI and ULTRASONIC_TRIGGER_PIN and ULTRASONIC_ECHO_PIN:
            try:
                import RPi.GPIO as GPIO
                self._gpio = GPIO
                self._setup()
                self._available = True
                print(f"[DISTANCE] Ultrasonic sensor ready (Trig={ULTRASONIC_TRIGGER_PIN}, Echo={ULTRASONIC_ECHO_PIN})")
            except ImportError:
                print("[DISTANCE] RPi.GPIO not available — ultrasonic sensor disabled")
            except Exception as e:
                print(f"[DISTANCE] Ultrasonic setup failed: {e}")
        else:
            print("[DISTANCE] Ultrasonic sensor not configured (not on RPi or pins not set)")

    def _setup(self):
        """Configure GPIO pins."""
        GPIO = self._gpio
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(ULTRASONIC_TRIGGER_PIN, GPIO.OUT)
        GPIO.setup(ULTRASONIC_ECHO_PIN, GPIO.IN)
        GPIO.output(ULTRASONIC_TRIGGER_PIN, False)
        time.sleep(0.1)  # Let sensor settle

    @property
    def is_available(self):
        return self._available

    def measure_once(self):
        """
        Take a single distance measurement.
        
        Returns:
            Distance in centimeters, or None on failure.
        """
        if not self._available:
            return None

        GPIO = self._gpio

        try:
            # Send 10µs trigger pulse
            GPIO.output(ULTRASONIC_TRIGGER_PIN, True)
            time.sleep(0.00001)  # 10 microseconds
            GPIO.output(ULTRASONIC_TRIGGER_PIN, False)

            # Wait for echo to go HIGH (start of echo pulse)
            pulse_start = time.time()
            timeout = pulse_start + 0.1  # 100ms timeout

            while GPIO.input(ULTRASONIC_ECHO_PIN) == 0:
                pulse_start = time.time()
                if pulse_start > timeout:
                    return None  # Timeout — no echo received

            # Wait for echo to go LOW (end of echo pulse)
            pulse_end = time.time()
            timeout = pulse_end + 0.1

            while GPIO.input(ULTRASONIC_ECHO_PIN) == 1:
                pulse_end = time.time()
                if pulse_end > timeout:
                    return None  # Timeout — echo stuck high

            # Calculate distance
            # Speed of sound = 34300 cm/s, divide by 2 for round trip
            pulse_duration = pulse_end - pulse_start
            distance_cm = (pulse_duration * 34300) / 2

            # Sanity check (HC-SR04 range: 2cm to 400cm)
            if 2 <= distance_cm <= 400:
                return round(distance_cm, 1)
            else:
                return None

        except Exception as e:
            print(f"[DISTANCE] Measurement error: {e}")
            return None

    def measure_averaged(self, samples=3, delay=0.05):
        """
        Take multiple measurements and return the median.
        More reliable than single measurements.
        
        Args:
            samples: Number of measurements to take
            delay: Delay between measurements in seconds
            
        Returns:
            Median distance in cm, or None on failure.
        """
        measurements = []
        for _ in range(samples):
            dist = self.measure_once()
            if dist is not None:
                measurements.append(dist)
            time.sleep(delay)

        if not measurements:
            return None

        # Return median
        measurements.sort()
        mid = len(measurements) // 2
        return measurements[mid]

    def start_continuous(self, callback=None):
        """
        Start continuous distance monitoring in a background thread.
        
        Args:
            callback: Function called with (distance_cm) on each reading.
                     distance_cm is None if measurement failed.
        """
        if not self._available:
            print("[DISTANCE] Cannot start continuous mode — sensor not available")
            return

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._continuous_loop, args=(callback,), daemon=True)
        self._thread.start()
        print("[DISTANCE] Continuous monitoring started")

    def stop_continuous(self):
        """Stop continuous monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        print("[DISTANCE] Continuous monitoring stopped")

    def _continuous_loop(self, callback):
        """Background loop for continuous distance measurement."""
        while self._running:
            distance = self.measure_averaged(samples=3)

            with self._lock:
                self._last_distance = distance

            if callback and distance is not None:
                try:
                    callback(distance)
                except Exception as e:
                    print(f"[DISTANCE] Callback error: {e}")

            time.sleep(SAFETY_CHECK_INTERVAL)

    @property
    def last_distance(self):
        """Get the most recent distance reading (from continuous mode)."""
        with self._lock:
            return self._last_distance

    def cleanup(self):
        """Clean up GPIO resources."""
        self.stop_continuous()
        if self._gpio and self._available:
            try:
                self._gpio.cleanup([ULTRASONIC_TRIGGER_PIN, ULTRASONIC_ECHO_PIN])
            except Exception:
                pass


class CameraDistanceEstimator:
    """
    Fallback distance estimation using YOLO detection bounding box sizes.
    
    This is a rough estimation — NOT accurate for absolute distances.
    But useful for relative warnings (something is getting closer/farther).
    
    Uses the principle: larger bounding box = closer object.
    """

    # Rough calibration: object relative size → approximate distance
    # These are very approximate and vary by object type
    SIZE_TO_DISTANCE = [
        (0.50, 30),    # 50%+ of frame → ~30cm (very close)
        (0.30, 60),    # 30-50% → ~60cm
        (0.15, 120),   # 15-30% → ~1.2m
        (0.08, 200),   # 8-15% → ~2m
        (0.03, 350),   # 3-8% → ~3.5m
        (0.01, 500),   # 1-3% → ~5m
        (0.00, 800),   # <1% → ~8m (far away)
    ]

    @staticmethod
    def estimate_distance(detection):
        """
        Estimate distance from a Detection object based on its bounding box size.
        
        Args:
            detection: A Detection object from detector.py
            
        Returns:
            Estimated distance in centimeters (very rough).
        """
        relative_size = detection.relative_size

        for size_threshold, distance in CameraDistanceEstimator.SIZE_TO_DISTANCE:
            if relative_size >= size_threshold:
                return distance

        return 800  # Default: far away

    @staticmethod
    def get_closest_object(detections):
        """
        Find the closest detected object (largest bounding box in center).
        
        Args:
            detections: List of Detection objects
            
        Returns:
            (detection, estimated_distance_cm) or (None, None)
        """
        if not detections:
            return None, None

        # Prefer center objects (in the user's path)
        center_detections = [d for d in detections if d.horizontal == "center"]

        # If nothing in center, use all detections
        candidates = center_detections or detections

        # Find the largest (closest) one
        closest = max(candidates, key=lambda d: d.relative_size)
        distance = CameraDistanceEstimator.estimate_distance(closest)

        return closest, distance


# Quick test
if __name__ == "__main__":
    print("=== Distance Module Test ===")

    # Test ultrasonic sensor
    sensor = UltrasonicSensor()
    if sensor.is_available:
        print("\nUltrasonic sensor test:")
        for i in range(5):
            dist = sensor.measure_averaged()
            print(f"  Reading {i+1}: {dist} cm")
            time.sleep(0.5)
        sensor.cleanup()
    else:
        print("Ultrasonic sensor not available (expected on non-RPi)")

    # Test camera-based estimation
    print("\nCamera distance estimation test:")
    print("  (Requires detector.py test to produce Detection objects)")
    print("  CameraDistanceEstimator is a static utility class")

    print("\n=== Test Complete ===")
