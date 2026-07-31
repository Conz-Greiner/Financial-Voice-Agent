# 🎙️ Financial Voice Agent (Local AI Analyst)

A production-grade, 100% local voice AI agent that acts as your personal financial analyst. It listens to your voice, retrieves real-time market data and economic calendars, and speaks back a concise, professional briefing. 

Built with a focus on **ultra-low latency, privacy, and robustness**, featuring advanced Voice Activity Detection (VAD), seamless "barge-in" interruption, and state-of-the-art local Text-to-Speech (TTS).

---

## 🌟 Key Features

*   **100% Local & Private:** The core AI (LLM, VAD, STT, TTS) runs entirely on your machine. No audio data is sent to third-party cloud providers.
*   **State-of-the-Art Voice Pipeline:**
    *   **Ears:** Silero VAD (AI-powered noise rejection) + Faster-Whisper (highly accurate local transcription).
    *   **Mouth:** Kokoro-TTS (82M parameter neural voice, ElevenLabs-level quality, pure PyTorch).
*   **Smart Barge-In:** Interrupt the AI mid-sentence. The system instantly stops speaking and listens to your new command.
*   **Acoustic Echo Protection:** Built-in playback lockout prevents the microphone from hearing the AI's own voice and triggering infinite feedback loops.
*   **Structured Financial Data:** Uses the Finnhub API for reliable, structured real-time asset prices, news, and high-impact economic calendar events (no brittle web scraping).
*   **Modular Architecture:** Cleanly separated into configuration, data tools, voice pipeline, and LLM orchestration for easy maintenance and extension.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Orchestration** | Python, LangChain, Asyncio |
| **LLM (Brain)** | Ollama (Granite 3.1 8B or Llama 3) |
| **VAD (Ears)** | Silero VAD (PyTorch) |
| **STT (Transcription)** | Faster-Whisper |
| **TTS (Voice)** | Kokoro-TTS (Pure PyTorch/ONNX) |
| **Audio I/O** | SoundDevice, SciPy, NumPy |
| **Data APIs** | Finnhub (Free Tier) |

---

## 📂 Project Structure

```text
financial_voice_agent/
│
├── config.py             # Centralized settings (API keys, model paths, VAD thresholds)
├── tools.py              # Finnhub API wrappers for News, Prices, and Economic Calendar
├── vad_engine.py         # Silero VAD state machine with acoustic echo lockout
├── voice_pipeline.py     # Faster-Whisper STT and Kokoro-TTS with barge-in logic
├── llm_agent.py          # LangChain agent setup and voice-optimized system prompts
├── main.py               # Master asynchronous event loop and orchestrator
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
*   **Python 3.10+** (Recommended: Use a Conda environment)
*   **Ollama** installed and running ([Download here](https://ollama.com/))
*   **eSpeak-NG** installed and added to your system PATH (Required for Kokoro-TTS phonemization). 
    *   *Windows:* Download the `.msi` from [eSpeak-NG Releases](https://github.com/espeak-ng/espeak-ng/releases), install, and add `C:\Program Files\eSpeak NG\` to your Windows Environment Variables `Path`.
    *   *Linux/macOS:* `sudo apt install espeak-ng` or `brew install espeak-ng`

### 2. Clone and Setup Environment
```bash
# Create and activate a conda environment
conda create -n voice_agent python=3.10 -y
conda activate voice_agent

# Install dependencies
pip install -r requirements.txt
```

### 3. Download AI Models
```bash
# Pull the LLM via Ollama
ollama pull granite3.1-dense:8b

# (Optional but recommended) Pull a faster alternative if Granite is slow on your CPU
ollama pull llama3.1:8b
```

### 4. Configure API Keys
1. Get a **free** API key from [Finnhub.io](https://finnhub.io/).
2. Open `config.py` and paste your key:
   ```python
   class DataAPIConfig:
       FINNHUB_API_KEY = "your_finnhub_api_key_here"
   ```

---

## 🎯 Usage

1. Ensure your microphone and speakers (or **headphones**, highly recommended to prevent echo) are connected.
2. Run the main application:
   ```bash
   python main.py
   ```
3. Wait for the models to load (Kokoro-TTS will download ~300MB on the very first run).
4. Speak clearly when prompted. 

**Example Prompts:**
*   *"What is the latest news and price for Gold?"*
*   *"Give me a market report on Bitcoin."*
*   *"What are the high-impact economic calendar events for the US Dollar today?"*
*   *"Tell me about EURUSD."*

*(To exit, simply say "exit", "quit", or "stop", or press `Ctrl+C`)*

---

## ⚙️ Configuration & Tuning

You can fine-tune the agent's behavior in `config.py`:

*   **`VAD_THRESHOLD`**: Increase (e.g., `0.80`) if the agent triggers on background noise. Decrease (e.g., `0.65`) if it misses quiet speech.
*   **`MIN_SILENCE_MS`**: Increase (e.g., `1000`) if the agent cuts you off while you are pausing to think. Decrease (e.g., `500`) for faster, snappier turn-taking.
*   **`OLLAMA_MODEL`**: Change to `"llama3.1:8b"` or `"qwen2.5:7b"` depending on your hardware and preference.

---

## 🔧 Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **Infinite Loop / Agent repeats itself** | The mic is hearing the speakers. **Wear headphones**, or increase `VAD_THRESHOLD` in `config.py`. The built-in lockout helps, but physical isolation is best. |
| **"espeak-ng" not found error** | The phonemizer is missing. Ensure eSpeak-NG is installed and `C:\Program Files\eSpeak NG\` is in your system `PATH`. Restart your terminal after adding it. |
| **Ollama Connection Error** | Ensure the Ollama app is running in your system tray. Run `ollama list` in a separate terminal to verify. |
| **TTS outputs silence / 0 bytes** | Ensure you have a stable internet connection on the first run so Kokoro can download its weights from Hugging Face. |

---

## ⚠️ Disclaimer

*This project is for educational and informational purposes only. The financial data provided via the Finnhub API may be delayed. This agent does not provide financial advice. Always verify market data with official sources before making trading decisions.*

---

## 📜 License

MIT License