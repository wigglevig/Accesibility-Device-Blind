"""
Audio Engine — Thread-safe TTS and Speech Recognition.

This is the core audio interface for the blind user.
All output goes through TTS (no screen). Input comes via microphone.

Features:
- Priority-based speech queue (ALERT interrupts, INFO queues, LOW skips if busy)
- Thread-safe: can be called from any thread
- Non-blocking speak() — returns immediately, audio plays in background
- Blocking listen() — records and returns transcribed text
"""

import threading
import queue
import time
import pyttsx3
import speech_recognition as sr
from enum import IntEnum

from rpi_assistant.config import (
    TTS_RATE, TTS_VOLUME,
    VOICE_LISTEN_TIMEOUT, VOICE_PHRASE_LIMIT,
)


class Priority(IntEnum):
    """Speech priority levels."""
    LOW = 0       # Skip if queue is busy (e.g., background info)
    INFO = 1      # Normal priority, queued in order
    ALERT = 2     # Urgent — clears queue, speaks immediately


class AudioEngine:
    """
    Thread-safe audio engine for TTS output and speech input.
    
    Usage:
        audio = AudioEngine()
        audio.start()
        audio.speak("Hello world")
        audio.speak("DANGER!", priority=Priority.ALERT)
        text = audio.listen()   # blocks until user speaks
        audio.stop()
    """

    def __init__(self):
        self._speech_queue = queue.PriorityQueue()
        self._tts_thread = None
        self._running = False
        self._speaking = False
        self._seq_counter = 0  # For stable priority queue ordering
        self._lock = threading.Lock()

        # Initialize TTS engine (created in the TTS thread for thread safety)
        self._engine = None

        # Initialize speech recognizer
        self._recognizer = sr.Recognizer()

        # Callback for when speech starts/stops (for LED feedback etc.)
        self.on_speaking_start = None
        self.on_speaking_stop = None
        self.on_listening_start = None
        self.on_listening_stop = None

    def start(self):
        """Start the TTS background thread."""
        if self._running:
            return
        self._running = True
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()
        print("[AUDIO] Engine started")

    def stop(self):
        """Stop the TTS background thread."""
        self._running = False
        # Put a sentinel to unblock the queue
        self._speech_queue.put((-1, -1, None))
        if self._tts_thread:
            self._tts_thread.join(timeout=5)
        print("[AUDIO] Engine stopped")

    def speak(self, text, priority=Priority.INFO):
        """
        Queue text to be spoken. Non-blocking.
        
        Args:
            text: The text to speak
            priority: Priority.ALERT (immediate), Priority.INFO (queued), Priority.LOW (skip if busy)
        """
        if not text or not self._running:
            return

        if priority == Priority.LOW and (self._speaking or not self._speech_queue.empty()):
            # Skip low-priority messages if already busy
            return

        if priority == Priority.ALERT:
            # Clear the queue for urgent messages
            self._clear_queue()

        with self._lock:
            self._seq_counter += 1
            # Negative priority so higher priority = lower number = dequeued first
            self._speech_queue.put((-int(priority), self._seq_counter, text))

    def speak_and_wait(self, text, priority=Priority.INFO):
        """
        Speak text and BLOCK until it's done. Use for critical announcements.
        """
        done_event = threading.Event()
        original_callback = self.on_speaking_stop

        def _on_done():
            done_event.set()
            self.on_speaking_stop = original_callback

        self.on_speaking_stop = _on_done
        self.speak(text, priority)
        done_event.wait(timeout=30)  # Max 30 seconds

    def listen(self, prompt=None):
        """
        Listen for voice input. BLOCKS until the user speaks or timeout.
        
        Args:
            prompt: Optional text to speak before listening (e.g., "What would you like to know?")
            
        Returns:
            Transcribed text (lowercase), or empty string on failure.
        """
        if prompt:
            self.speak_and_wait(prompt)

        if self.on_listening_start:
            self.on_listening_start()

        print("[AUDIO] Listening...")
        try:
            with sr.Microphone() as source:
                # Brief ambient noise adjustment
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self._recognizer.listen(
                    source,
                    timeout=VOICE_LISTEN_TIMEOUT,
                    phrase_time_limit=VOICE_PHRASE_LIMIT,
                )

            # Transcribe using Google Speech Recognition (requires internet)
            text = self._recognizer.recognize_google(audio)
            print(f"[AUDIO] Heard: {text}")

            if self.on_listening_stop:
                self.on_listening_stop()

            return text.lower().strip()

        except sr.WaitTimeoutError:
            print("[AUDIO] No speech detected (timeout)")
            self.speak("I didn't hear anything. Please try again.", Priority.INFO)
        except sr.UnknownValueError:
            print("[AUDIO] Could not understand speech")
            self.speak("I couldn't understand that. Please try again.", Priority.INFO)
        except sr.RequestError as e:
            print(f"[AUDIO] Speech recognition error: {e}")
            self.speak("Speech recognition is not available. Check your internet connection.", Priority.ALERT)
        except OSError as e:
            print(f"[AUDIO] Microphone error: {e}")
            self.speak("Microphone not found. Please check the connection.", Priority.ALERT)

        if self.on_listening_stop:
            self.on_listening_stop()

        return ""

    @property
    def is_speaking(self):
        """Whether TTS is currently speaking."""
        return self._speaking

    def _tts_worker(self):
        """Background thread that processes the speech queue."""
        # Create TTS engine in this thread (pyttsx3 is not thread-safe across threads)
        self._engine = pyttsx3.init()
        self._engine.setProperty('rate', TTS_RATE)
        self._engine.setProperty('volume', TTS_VOLUME)

        # Try to set a natural voice
        voices = self._engine.getProperty('voices')
        for voice in voices:
            # Prefer English voices
            if 'english' in voice.name.lower() or 'en' in voice.languages[0].lower() if voice.languages else False:
                self._engine.setProperty('voice', voice.id)
                break

        print("[AUDIO] TTS engine initialized")

        while self._running:
            try:
                neg_priority, seq, text = self._speech_queue.get(timeout=1)
                if text is None:  # Sentinel value
                    continue

                self._speaking = True
                if self.on_speaking_start:
                    self.on_speaking_start()

                print(f"[AUDIO] Speaking: {text[:80]}...")
                self._engine.say(text)
                self._engine.runAndWait()

                self._speaking = False
                if self.on_speaking_stop:
                    self.on_speaking_stop()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AUDIO] TTS error: {e}")
                self._speaking = False

    def _clear_queue(self):
        """Clear all pending messages from the queue."""
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break


# ============================================================
# Module-level convenience functions
# ============================================================

_engine = None

def get_audio_engine():
    """Get or create the global audio engine instance."""
    global _engine
    if _engine is None:
        _engine = AudioEngine()
        _engine.start()
    return _engine


# Quick test
if __name__ == "__main__":
    print("=== Audio Engine Test ===")
    audio = AudioEngine()
    audio.start()

    # Test TTS
    audio.speak("Vision assistant audio test. This is an info message.")
    time.sleep(1)
    audio.speak("This is a low priority message.", Priority.LOW)
    audio.speak("Alert! This is an urgent message.", Priority.ALERT)

    time.sleep(8)

    # Test listening
    text = audio.listen(prompt="Say something to test the microphone.")
    if text:
        audio.speak(f"You said: {text}")
        time.sleep(5)
    else:
        time.sleep(2)

    audio.stop()
    print("=== Test Complete ===")
