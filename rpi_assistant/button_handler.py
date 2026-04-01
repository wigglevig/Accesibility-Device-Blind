"""
Button Handler — GPIO buttons on RPi, keyboard fallback on Mac/PC.

Provides a unified interface for user input regardless of platform:
- On Raspberry Pi: Physical GPIO buttons with hardware debounce
- On Mac/PC (dev mode): Keyboard keys 1, 2, 3 as button substitutes

Three buttons:
  Button 1: "Describe Scene"  (Gemini AI scene description)
  Button 2: "Ask AI"          (Voice question + Gemini answer)
  Button 3: "What's Around?"  (YOLO object detection summary)
"""

import threading
import time

from rpi_assistant.config import (
    BUTTON_DESCRIBE_PIN,
    BUTTON_ASK_PIN,
    BUTTON_SUMMARY_PIN,
    RUNNING_ON_RPI,
    USE_KEYBOARD_INPUT,
)


class ButtonHandler:
    """
    Manages button input from GPIO pins or keyboard.
    
    Usage:
        buttons = ButtonHandler()
        buttons.on_describe = lambda: print("Describe pressed!")
        buttons.on_ask = lambda: print("Ask pressed!")
        buttons.on_summary = lambda: print("Summary pressed!")
        buttons.start()
        ...
        buttons.stop()
    """

    def __init__(self):
        # Callbacks — set these to your handler functions
        self.on_describe = None     # Button 1: Scene description
        self.on_ask = None          # Button 2: Ask AI question
        self.on_summary = None      # Button 3: Object summary

        self._running = False
        self._thread = None
        self._debounce_time = 0.3   # 300ms debounce
        self._last_press = {}       # Track last press time per button

        # GPIO setup (only when on RPi AND not using keyboard input)
        self._gpio = None
        self._use_keyboard = USE_KEYBOARD_INPUT
        if RUNNING_ON_RPI and not self._use_keyboard:
            try:
                import RPi.GPIO as GPIO
                self._gpio = GPIO
                self._setup_gpio()
            except ImportError:
                print("[BUTTONS] RPi.GPIO not available, falling back to keyboard mode")
                self._use_keyboard = True
            except Exception as e:
                print(f"[BUTTONS] GPIO setup failed: {e}, falling back to keyboard mode")
                self._use_keyboard = True

    def _setup_gpio(self):
        """Configure GPIO pins for button input."""
        GPIO = self._gpio
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)

        pins = [
            (BUTTON_DESCRIBE_PIN, "describe"),
            (BUTTON_ASK_PIN, "ask"),
            (BUTTON_SUMMARY_PIN, "summary"),
        ]

        for pin, name in pins:
            if pin is not None:
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                self._last_press[name] = 0
                print(f"[BUTTONS] GPIO {pin} configured for '{name}'")

    def start(self):
        """Start listening for button presses."""
        if self._running:
            return
        self._running = True

        if self._gpio and not self._use_keyboard:
            self._start_gpio()
        else:
            self._start_keyboard()

        print("[BUTTONS] Handler started")

    def stop(self):
        """Stop listening for button presses."""
        self._running = False
        if self._gpio and not self._use_keyboard:
            try:
                self._gpio.cleanup()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=3)
        print("[BUTTONS] Handler stopped")

    def _start_gpio(self):
        """Set up GPIO interrupt-driven button detection."""
        GPIO = self._gpio

        def _make_callback(name, callback):
            def _cb(channel):
                now = time.time()
                if now - self._last_press.get(name, 0) > self._debounce_time:
                    self._last_press[name] = now
                    print(f"[BUTTONS] '{name}' pressed (GPIO {channel})")
                    if callback:
                        # Run callback in a separate thread to not block GPIO
                        threading.Thread(target=callback, daemon=True).start()
            return _cb

        if BUTTON_DESCRIBE_PIN is not None:
            GPIO.add_event_detect(
                BUTTON_DESCRIBE_PIN, GPIO.FALLING,
                callback=_make_callback("describe", self.on_describe),
                bouncetime=300
            )

        if BUTTON_ASK_PIN is not None:
            GPIO.add_event_detect(
                BUTTON_ASK_PIN, GPIO.FALLING,
                callback=_make_callback("ask", self.on_ask),
                bouncetime=300
            )

        if BUTTON_SUMMARY_PIN is not None:
            GPIO.add_event_detect(
                BUTTON_SUMMARY_PIN, GPIO.FALLING,
                callback=_make_callback("summary", self.on_summary),
                bouncetime=300
            )

    def _start_keyboard(self):
        """Start keyboard listener for development mode."""
        self._thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._thread.start()
        print("[BUTTONS] Keyboard mode — press 1, 2, or 3 then Enter:")
        print("  1 = Describe Scene")
        print("  2 = Ask AI Question")
        print("  3 = Objects Around Me")
        print("  q = Quit")

    def _keyboard_loop(self):
        """Poll keyboard for button-equivalent key presses (dev mode)."""
        while self._running:
            try:
                key = input().strip().lower()

                if key == "1" and self.on_describe:
                    print("[BUTTONS] 'describe' triggered (keyboard)")
                    threading.Thread(target=self.on_describe, daemon=True).start()
                elif key == "2" and self.on_ask:
                    print("[BUTTONS] 'ask' triggered (keyboard)")
                    threading.Thread(target=self.on_ask, daemon=True).start()
                elif key == "3" and self.on_summary:
                    print("[BUTTONS] 'summary' triggered (keyboard)")
                    threading.Thread(target=self.on_summary, daemon=True).start()
                elif key == "q":
                    print("[BUTTONS] Quit requested")
                    self._running = False
                    break
                elif key:
                    print(f"[BUTTONS] Unknown key '{key}'. Use 1, 2, 3, or q.")

            except EOFError:
                # Input stream closed
                break
            except Exception as e:
                print(f"[BUTTONS] Keyboard error: {e}")
                time.sleep(0.5)


# Quick test
if __name__ == "__main__":
    print("=== Button Handler Test ===")

    def on_desc():
        print(">>> DESCRIBE SCENE callback fired!")

    def on_ask():
        print(">>> ASK AI callback fired!")

    def on_summary():
        print(">>> OBJECT SUMMARY callback fired!")

    buttons = ButtonHandler()
    buttons.on_describe = on_desc
    buttons.on_ask = on_ask
    buttons.on_summary = on_summary
    buttons.start()

    # Keep alive until quit
    try:
        while buttons._running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        buttons.stop()

    print("=== Test Complete ===")
