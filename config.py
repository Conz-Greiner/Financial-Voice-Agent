import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DataAPIConfig:
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
    if not FINNHUB_API_KEY:
        raise ValueError("FINNHUB_API_KEY not found in .env file!")

class LLMConfig:
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "granite3.1-dense:8b")  # Model name
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    TEMPERATURE = 0.0
    CONTEXT_WINDOW = 8192

class AudioConfig:
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 512
    VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.75"))
    MIN_SPEECH_MS = int(os.getenv("MIN_SPEECH_MS", "400"))
    MIN_SILENCE_MS = int(os.getenv("MIN_SILENCE_MS", "800"))

class VoiceConfig:
    WHISPER_MODEL = "base.en"
    WHISPER_DEVICE = "cpu"
    WHISPER_COMPUTE_TYPE = "int8"
    TTS_VOICE = "af_heart"
    TTS_SAMPLE_RATE = 24000