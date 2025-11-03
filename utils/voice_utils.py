import logging
import re
import speech_recognition as sr
import pyttsx3

logger = logging.getLogger(__name__)

def extract_voice_segment(raw: str) -> str:
    """Extract only the VOICE portion from a combined VOICE|||TEXT message.

    Falls back to raw string if pattern not found.
    """
    if not isinstance(raw, str):
        raw = str(raw)
    match = re.match(r'^VOICE:([\s\S]*?)\|\|\|TEXT:', raw)
    if match:
        # Clean whitespace/newlines
        voice_part = match.group(1).strip()
        # Remove any accidental embedded JSON braces
        voice_part = re.sub(r'[{}\[\]]', '', voice_part)
        return re.sub(r'\s{2,}', ' ', voice_part)
    return raw.strip()

def speak(text: str):
    """Converts text to speech and plays ONLY the voice summary (sanitized)."""
    try:
        pattern_match = re.match(r'^VOICE:([\s\S]*?)\|\|\|TEXT:', text or '')
        if not pattern_match:
            # Hard guard: do NOT speak full diagnostic content accidentally.
            logger.warning("TTS invoked without VOICE|||TEXT pattern – aborting full report speech. Provide combined pattern. Returning fallback line.")
            fallback = "I have an update. Check the diagnostic report for detailed steps."
            engine = pyttsx3.init()
            engine.setProperty("rate", 180)
            engine.say(fallback)
            engine.runAndWait()
            return
        voice_only = extract_voice_segment(text)
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)
        logger.info(f"\n🤖 Assistant (voice summary):\n{voice_only}")
        engine.say(voice_only)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"TTS engine error: {e}")

# NOTE: Ensure upstream caller passes the full raw agent message (with VOICE|||TEXT) into speak().
# If caller currently sends combined diagnostic text, wrap it: speak(f"VOICE:{voice_part}|||TEXT:{text_part}")

def listen_for_command() -> str:
    """Listens for a command from the microphone and returns it as text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        logger.info("\n🎙️ Listening...")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = r.listen(source)
            logger.info("🔍 Recognizing...")
            query = r.recognize_google(audio, language="en-in")
            logger.info(f"🧑 You said: {query}\n")
            return query.lower()
        except sr.UnknownValueError:
            logger.warning("Could not understand audio, please try again.")
            return ""
        except sr.RequestError as e:
            logger.error(f"Speech Recognition request error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return ""
