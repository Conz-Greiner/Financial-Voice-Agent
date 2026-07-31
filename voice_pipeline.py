import asyncio
import numpy as np
import sounddevice as sd
import re
import os
import torch
from faster_whisper import WhisperModel
from scipy.io.wavfile import write as write_wav
from config import VoiceConfig
from kokoro import KPipeline


class VoicePipeline:
    """Handles Speech-to-Text and Local Kokoro-TTS (SOTA Quality) with Barge-In."""

    def __init__(self, vad_recorder):
        self.vad = vad_recorder

        # 1. Initialize Whisper STT
        print(f"🧠 [STT] Loading Whisper model ({VoiceConfig.WHISPER_MODEL})...")
        self.whisper = WhisperModel(
            VoiceConfig.WHISPER_MODEL,
            device=VoiceConfig.WHISPER_DEVICE,
            compute_type=VoiceConfig.WHISPER_COMPUTE_TYPE
        )
        print("✅ [STT] Whisper model loaded.")

        # 2. 🚀 THE KOKORO PIVOT: 82M Parameter SOTA TTS
        print("🗣️ [TTS] Loading Kokoro-TTS model... (Downloads ~300MB on first run)")
        # lang_code='a' is American English.
        # Kokoro will automatically download the model from Hugging Face.
        self.pipeline = KPipeline(lang_code='a')
        print("✅ [TTS] Kokoro-TTS loaded. (ElevenLabs-level quality).")

    def transcribe(self, audio_np: np.ndarray) -> str:
        temp_file = "temp_utterance.wav"
        write_wav(temp_file, 16000, audio_np)

        segments, _ = self.whisper.transcribe(
            temp_file,
            beam_size=5,
            vad_filter=True,
            language="en"
        )
        return "".join([segment.text for segment in segments]).strip()

    def _generate_kokoro_audio(self, text: str):
        """Generates audio using Kokoro-TTS."""
        try:
            # Clean text to ensure perfect phonemization
            clean_text = re.sub(r'[\*\_\#\[\]\(\)\$\%\,]', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            if not clean_text:
                return np.array([], dtype=np.int16), 24000

            # Kokoro yields audio in chunks. We collect and concatenate them.
            # voice='af_heart' is a beautiful, professional female voice.
            # Other options: 'am_adam' (male), 'af_bella', 'af_nicole', etc.
            generator = self.pipeline(clean_text, voice='af_heart')

            audio_chunks = []
            for i, (gs, ps, audio) in enumerate(generator):
                audio_chunks.append(audio)

            if not audio_chunks:
                return np.array([], dtype=np.int16), 24000

            # Concatenate all chunks into a single continuous audio array
            full_audio = np.concatenate(audio_chunks)

            # Kokoro outputs float32 (-1.0 to 1.0). Convert to int16 for sounddevice.
            audio_int16 = (full_audio * 32767).astype(np.int16)

            return audio_int16, 24000

        except Exception as e:
            print(f"❌ [TTS] Kokoro synthesis failed: {repr(e)}")
            return np.array([], dtype=np.int16), 24000

    async def speak_with_barge_in(self, text: str) -> bool:
        if not text or not text.strip():
            return False

        print(f"🔊 [TTS] Generating Kokoro neural audio for: '{text[:50]}...'")

        audio_np, actual_sample_rate = self._generate_kokoro_audio(text)

        if len(audio_np) == 0:
            print("⚠️ [TTS] Aborting playback due to empty audio array.")
            return False

        max_audio_amp = np.max(np.abs(audio_np))
        print(f"🔊 [TTS DEBUG] Audio length: {len(audio_np)} samples. Max Amplitude: {max_audio_amp}")

        # Clear VAD queue to prevent false barge-ins
        while not self.vad.audio_queue.empty():
            self.vad.audio_queue.get()

        print(f"🔊 [TTS] Playing audio via sounddevice at {actual_sample_rate}Hz...")
        sd.play(audio_np, samplerate=actual_sample_rate)

        is_interrupted = False
        while sd.get_stream().active:
            if not self.vad.audio_queue.empty():
                sd.stop()  # RUTHLESS: Kill audio instantly
                is_interrupted = True
                print("⚠️ [TTS] Barge-in detected! Stopping playback.")
                break
            await asyncio.sleep(0.05)

        if not is_interrupted:
            sd.wait()
            print("✅ [TTS DEBUG] Playback finished normally.")

        return is_interrupted