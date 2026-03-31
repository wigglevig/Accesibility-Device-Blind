"""
Main Controller — Smart Vision Assistant for Visually Impaired.

This is the entry point. It ties together all components:
- Audio (TTS + Speech Recognition)
- Camera
- Object Detection (YOLO)
- AI Assistant (Gemini API)
- Button/Keyboard controls

Everything is audio-driven. No visual output.

Usage:
    python -m rpi_assistant.main

Controls (keyboard in dev mode, GPIO buttons on RPi):
    1 = Describe Scene (Gemini AI)
    2 = Ask AI a Question (voice)
    3 = What Objects Are Around Me? (YOLO)
    q = Quit
"""

import time
import threading
import signal
import sys

from rpi_assistant.config import DEV_MODE, RUNNING_ON_RPI
from rpi_assistant.audio_engine import AudioEngine, Priority
from rpi_assistant.button_handler import ButtonHandler
from rpi_assistant.camera import Camera
from rpi_assistant.detector import Detector
from rpi_assistant.ai_assistant import AIAssistant
from rpi_assistant.safety_monitor import SafetyMonitor


class VisionAssistant:
    """
    Main application controller.
    
    Manages the lifecycle of all components and handles user interactions.
    """

    def __init__(self):
        self._running = False

        # Initialize components
        print("=" * 50)
        print("  SMART VISION ASSISTANT")
        print("  For Visually Impaired")
        print("=" * 50)

        # Phase 1: Audio
        print("\n[INIT] Starting audio engine...")
        self.audio = AudioEngine()

        # Phase 1: Buttons
        print("[INIT] Setting up controls...")
        self.buttons = ButtonHandler()

        # Phase 2: Camera
        print("[INIT] Initializing camera...")
        self.camera = Camera()

        # Phase 2: Object Detection
        print("[INIT] Loading object detection model...")
        self.detector = Detector()

        # Phase 4: AI Assistant
        print("[INIT] Setting up AI assistant...")
        self.ai = AIAssistant()

        # Phase 3: Safety Monitor
        print("[INIT] Setting up safety monitor...")
        self.safety = SafetyMonitor(
            audio=self.audio,
            camera=self.camera,
            detector=self.detector,
        )

        # Lock to prevent overlapping actions
        self._action_lock = threading.Lock()
        self._busy = False

    def start(self):
        """Initialize all components and start the assistant."""
        self._running = True

        # Start audio engine
        self.audio.start()

        # Start camera
        if not self.camera.start():
            self.audio.speak_and_wait(
                "Warning: Camera could not be opened. "
                "Some features will not work.",
                Priority.ALERT
            )

        # Load detection model
        try:
            self.detector.load()
        except Exception as e:
            print(f"[MAIN] Detection model load error: {e}")
            self.audio.speak(
                "Warning: Object detection model could not be loaded.",
                Priority.ALERT
            )

        # Wire up button callbacks
        self.buttons.on_describe = self._handle_describe
        self.buttons.on_ask = self._handle_ask
        self.buttons.on_summary = self._handle_summary
        self.buttons.start()

        # Start background safety monitoring
        self.safety.start()

        # Announce ready
        ready_message = "Vision assistant is ready. "
        if DEV_MODE:
            ready_message += (
                "Press 1 to describe your surroundings. "
                "Press 2 to ask me a question. "
                "Press 3 to know what objects are around you. "
                "Press q to quit."
            )
        else:
            ready_message += (
                "Press the first button to describe your surroundings. "
                "Press the second button to ask me a question. "
                "Press the third button to know what objects are around you."
            )

        self.audio.speak_and_wait(ready_message)
        print("\n[MAIN] Assistant is running. Waiting for input...")

    def run(self):
        """Main run loop. Blocks until stopped."""
        self.start()

        # Keep the main thread alive
        try:
            while self._running and self.buttons._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[MAIN] Interrupted by user")

        self.stop()

    def stop(self):
        """Gracefully shut down all components."""
        if not self._running:
            return
        self._running = False

        print("\n[MAIN] Shutting down...")
        self.audio.speak_and_wait("Shutting down. Goodbye.", Priority.ALERT)

        self.safety.stop()
        self.buttons.stop()
        self.camera.stop()
        self.audio.stop()

        print("[MAIN] Assistant stopped.")

    # ================================================================
    # ACTION HANDLERS
    # ================================================================

    def _handle_describe(self):
        """Button 1: AI-powered scene description using Gemini."""
        if not self._try_acquire_lock():
            return

        try:
            if not self.ai.is_available:
                self.audio.speak(
                    "AI assistant is not available. "
                    "Please set up the Gemini API key.",
                    Priority.INFO
                )
                return

            self.audio.speak("Describing your surroundings...", Priority.INFO)

            # Capture frame
            frame_b64 = self.camera.capture_base64()
            if frame_b64 is None:
                self.audio.speak("Camera is not available.", Priority.ALERT)
                return

            # Get AI description
            description = self.ai.describe_scene(frame_b64)
            self.audio.speak(description, Priority.INFO)

        except Exception as e:
            print(f"[MAIN] Describe error: {e}")
            self.audio.speak("Sorry, I couldn't describe the scene.", Priority.INFO)
        finally:
            self._release_lock()

    def _handle_ask(self):
        """Button 2: Voice question + AI answer."""
        if not self._try_acquire_lock():
            return

        try:
            if not self.ai.is_available:
                self.audio.speak(
                    "AI assistant is not available. "
                    "Please set up the Gemini API key.",
                    Priority.INFO
                )
                return

            # Listen for the question
            question = self.audio.listen(prompt="What would you like to know?")

            if not question:
                return  # audio.listen already speaks error message

            self.audio.speak("Let me check...", Priority.INFO)

            # Capture frame
            frame_b64 = self.camera.capture_base64()
            if frame_b64 is None:
                self.audio.speak("Camera is not available.", Priority.ALERT)
                return

            # Get AI answer
            answer = self.ai.ask(frame_b64, question)
            self.audio.speak(answer, Priority.INFO)

        except Exception as e:
            print(f"[MAIN] Ask error: {e}")
            self.audio.speak("Sorry, I couldn't answer that question.", Priority.INFO)
        finally:
            self._release_lock()

    def _handle_summary(self):
        """Button 3: YOLO object detection summary."""
        if not self._try_acquire_lock():
            return

        try:
            self.audio.speak("Scanning for objects...", Priority.INFO)

            # Capture frame
            frame = self.camera.capture()
            if frame is None:
                self.audio.speak("Camera is not available.", Priority.ALERT)
                return

            # Run detection
            detections = self.detector.detect(frame)

            # Generate and speak summary
            summary = self.detector.summarize(detections)
            self.audio.speak(summary, Priority.INFO)

            # Check for hazards
            hazards = self.detector.get_hazards(detections)
            if hazards:
                hazard_names = ", ".join(set(h.label for h in hazards))
                self.audio.speak(
                    f"Caution: {hazard_names} detected ahead.",
                    Priority.ALERT
                )

        except Exception as e:
            print(f"[MAIN] Summary error: {e}")
            self.audio.speak("Sorry, I couldn't scan for objects.", Priority.INFO)
        finally:
            self._release_lock()

    # ================================================================
    # HELPERS
    # ================================================================

    def _try_acquire_lock(self):
        """Try to start an action. Returns False if already busy."""
        if self._busy:
            self.audio.speak("Please wait, I'm still processing.", Priority.LOW)
            return False
        self._busy = True
        return True

    def _release_lock(self):
        """Release the action lock."""
        self._busy = False


def main():
    """Entry point."""
    assistant = VisionAssistant()

    # Handle SIGINT/SIGTERM gracefully
    def signal_handler(sig, frame):
        print("\n[MAIN] Signal received, shutting down...")
        assistant.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    assistant.run()


if __name__ == "__main__":
    main()
