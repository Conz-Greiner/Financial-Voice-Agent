import torch
import numpy as np
import sounddevice as sd
from queue import Queue
from config import AudioConfig

sd.default.device[0] = 1


class SileroVADRecorder:
    """
    Production-grade Voice Activity Detection using Silero AI.
    Runs the audio callback in a background C-thread to prevent blocking.
    """

    def __init__(self):
        self.sample_rate = AudioConfig.SAMPLE_RATE
        self.chunk_size = AudioConfig.CHUNK_SIZE
        self.threshold = AudioConfig.VAD_THRESHOLD

        # Convert milliseconds to frames (each frame is 32ms)
        self.min_speech_frames = int((AudioConfig.MIN_SPEECH_MS / 1000) / 0.032)
        self.min_silence_frames = int((AudioConfig.MIN_SILENCE_MS / 1000) / 0.032)

        # Load Silero Model
        print("🧠 [VAD] Loading Silero VAD model...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True
        )
        self.model.eval()

        # State Machine Variables
        self.state = "IDLE"
        self.speech_frames = 0
        self.silence_frames = 0
        self.audio_buffer = []

        # Thread-safe queue to pass completed utterances to the main loop
        self.audio_queue = Queue()
        self.stream = None

    def _audio_callback(self, indata, frames, time, status):
        """Runs in a background thread. Now handles 2 channels and averages them."""
        if status:
            pass

            # indata shape is now (frames, 2). We average them to create a true mono signal.
        # This ensures we don't miss the voice if it's only on the right channel.
        audio_chunk = indata.mean(axis=1).astype(np.int16)

        # Convert to float32 for Silero VAD
        tensor_chunk = torch.tensor(audio_chunk, dtype=torch.float32) / 32768.0

        with torch.no_grad():
            speech_prob = self.model(tensor_chunk, self.sample_rate).item()

        # --- STATE MACHINE LOGIC ---
        if self.state == "IDLE":
            if speech_prob > self.threshold:
                self.speech_frames += 1
                self.audio_buffer.append(audio_chunk)

                if self.speech_frames >= self.min_speech_frames:
                    self.state = "SPEAKING"
                    self.silence_frames = 0
            else:
                self.speech_frames = 0
                self.audio_buffer = []

        elif self.state == "SPEAKING":
            self.audio_buffer.append(audio_chunk)

            if speech_prob < self.threshold:
                self.silence_frames += 1
                if self.silence_frames >= self.min_silence_frames:
                    full_audio = np.concatenate(self.audio_buffer)
                    self.audio_queue.put(full_audio)

                    self.state = "IDLE"
                    self.audio_buffer = []
                    self.speech_frames = 0
                    self.silence_frames = 0
            else:
                self.silence_frames = 0

    def start(self):
        """Starts the background microphone stream."""
        print("🎙️ [VAD] Starting microphone stream...")
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=2,  # 🚨 CRITICAL FIX: Request both channels
            blocksize=self.chunk_size,
            dtype='int16',
            callback=self._audio_callback
        )
        self.stream.start()

    def stop(self):
        """Stops the microphone stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("🎙️ [VAD] Microphone stream stopped.")

    def get_next_utterance(self):
        """
        Blocks the main thread until the user finishes speaking.
        Returns the raw numpy audio array.
        """
        return self.audio_queue.get()