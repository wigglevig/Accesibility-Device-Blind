"""
AI Assistant — Gemini API integration for scene description and Q&A.

Provides:
- Scene description: Camera frame → Gemini → natural language description
- Q&A mode: User asks question + camera frame → Gemini → contextual answer
- Conversation memory: Follow-up questions work naturally
"""

import base64
import time
import json

from rpi_assistant.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_MAX_TOKENS,
    SCENE_DESCRIPTION_PROMPT,
    QA_SYSTEM_PROMPT,
    CONVERSATION_MEMORY_SIZE,
)


class AIAssistant:
    """
    Gemini-powered AI assistant for scene understanding and Q&A.
    
    Usage:
        ai = AIAssistant()
        
        # Scene description
        description = ai.describe_scene(frame_base64)
        
        # Q&A
        answer = ai.ask(frame_base64, "Is there a door nearby?")
        answer = ai.ask(frame_base64, "What color is it?")  # Follow-up works!
    """

    def __init__(self, api_key=None):
        self._api_key = api_key or GEMINI_API_KEY
        self._conversation_history = []
        self._client = None
        self._available = False

        if not self._api_key:
            print("[AI] WARNING: No Gemini API key set!")
            print("[AI] Set GEMINI_API_KEY environment variable or update config.py")
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
            self._available = True
            print(f"[AI] Gemini client initialized (model: {GEMINI_MODEL})")
        except ImportError:
            print("[AI] google-genai package not installed. Run: pip install google-genai")
        except Exception as e:
            print(f"[AI] Failed to initialize Gemini client: {e}")

    @property
    def is_available(self):
        return self._available

    def describe_scene(self, frame_base64):
        """
        Generate a natural scene description from a camera frame.
        
        Args:
            frame_base64: Base64-encoded JPEG image string
            
        Returns:
            Description string, or error message on failure.
        """
        if not self._available:
            return "AI assistant is not available. Please check the API key and internet connection."

        try:
            from google.genai import types

            # Build the request with image
            image_part = types.Part.from_bytes(
                data=base64.b64decode(frame_base64),
                mime_type="image/jpeg",
            )

            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(SCENE_DESCRIPTION_PROMPT),
                            image_part,
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=GEMINI_MAX_TOKENS,
                    temperature=0.3,  # Lower = more factual
                ),
            )

            description = response.text.strip()

            # Store in conversation history
            self._add_to_history("user", "Describe what you see.")
            self._add_to_history("assistant", description)

            print(f"[AI] Scene description: {description[:100]}...")
            return description

        except Exception as e:
            error_msg = str(e)
            print(f"[AI] Scene description error: {error_msg}")
            if "api key" in error_msg.lower() or "401" in error_msg:
                return "The AI API key is invalid. Please check your configuration."
            elif "quota" in error_msg.lower() or "429" in error_msg:
                return "AI service is temporarily busy. Please try again in a moment."
            else:
                return f"I couldn't describe the scene right now. Error: {error_msg[:50]}"

    def ask(self, frame_base64, question):
        """
        Answer a question about the current camera view.
        Uses conversation history for follow-up questions.
        
        Args:
            frame_base64: Base64-encoded JPEG image string
            question: User's question as text
            
        Returns:
            Answer string.
        """
        if not self._available:
            return "AI assistant is not available. Please check the API key."

        try:
            from google.genai import types

            image_part = types.Part.from_bytes(
                data=base64.b64decode(frame_base64),
                mime_type="image/jpeg",
            )

            # Build conversation with history for context
            contents = []

            # System instruction via first message
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(QA_SYSTEM_PROMPT)],
                )
            )
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text("Understood. I'll help describe and answer questions about what's in the image.")],
                )
            )

            # Add conversation history for follow-up context
            for entry in self._conversation_history[-CONVERSATION_MEMORY_SIZE:]:
                role = "model" if entry["role"] == "assistant" else entry["role"]
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(entry["content"])],
                    )
                )

            # Add current question with image
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(f"Looking at this image, {question}"),
                        image_part,
                    ],
                )
            )

            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=GEMINI_MAX_TOKENS,
                    temperature=0.3,
                ),
            )

            answer = response.text.strip()

            # Update conversation history
            self._add_to_history("user", question)
            self._add_to_history("assistant", answer)

            print(f"[AI] Q: {question}")
            print(f"[AI] A: {answer[:100]}...")
            return answer

        except Exception as e:
            error_msg = str(e)
            print(f"[AI] Q&A error: {error_msg}")
            return f"I couldn't answer that right now. Please try again."

    def clear_history(self):
        """Clear conversation history."""
        self._conversation_history.clear()
        print("[AI] Conversation history cleared")

    def _add_to_history(self, role, content):
        """Add an exchange to conversation history, trimming old entries."""
        self._conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })

        # Trim to max size
        max_entries = CONVERSATION_MEMORY_SIZE * 2  # user + assistant pairs
        if len(self._conversation_history) > max_entries:
            self._conversation_history = self._conversation_history[-max_entries:]


# Quick test
if __name__ == "__main__":
    import os
    import sys

    print("=== AI Assistant Test ===")

    if not GEMINI_API_KEY:
        print("Set GEMINI_API_KEY environment variable first!")
        print("  export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)

    ai = AIAssistant()

    if ai.is_available:
        # Test with a camera frame
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    _, jpeg = cv2.imencode('.jpg', frame)
                    b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

                    print("\n--- Scene Description ---")
                    desc = ai.describe_scene(b64)
                    print(desc)

                    print("\n--- Q&A ---")
                    answer = ai.ask(b64, "What objects do you see?")
                    print(answer)

                    print("\n--- Follow-up ---")
                    answer = ai.ask(b64, "Is there anything dangerous?")
                    print(answer)
                cap.release()
            else:
                print("No camera available")
        except ImportError:
            print("OpenCV not available for camera test")
    else:
        print("AI assistant not available")

    print("=== Test Complete ===")
