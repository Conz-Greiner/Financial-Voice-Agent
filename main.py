import asyncio
import signal
import sys
import numpy as np
from vad_engine import SileroVADRecorder
from voice_pipeline import VoicePipeline
from llm_agent import FinancialAgent


async def main():
    print("=" * 50)
    print("  GRANITE FINANCIAL VOICE AGENT")
    print("=" * 50)
    print("Speak naturally. The AI will listen, analyze, and speak back.")
    print("You can interrupt the AI at any time (Barge-in).")
    print("Say 'exit', 'quit', or 'stop' to end the session.")
    print("=" * 50 + "\n")

    # 1. Initialize Components
    vad = SileroVADRecorder()
    pipeline = VoicePipeline(vad)
    agent = FinancialAgent()

    # 2. Start Audio Engine
    vad.start()

    # Graceful shutdown handler for Ctrl+C
    def signal_handler(sig, frame):
        print("\n\n🛑 Shutting down gracefully...")
        vad.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            # --- PHASE 1: LISTEN ---
            print("👂 Listening...")
            # This blocks until the VAD state machine detects the end of speech
            audio_utterance = vad.get_next_utterance()

            # 🚨 AMPITUDE GATE: Reject quiet background noise instantly
            max_amp = np.max(np.abs(audio_utterance))
            if max_amp < 2000:
                print(f"⚠️ Audio too quiet (Amplitude: {max_amp:.0f}). Ignoring background noise.")
                continue

            # 🚨 ADD THESE TWO LINES FOR DEBUGGING:


            print(f"🔊 DEBUG: Audio captured. Shape: {audio_utterance.shape}, Max Amplitude: {max_amp:.4f}")

            # --- PHASE 2: TRANSCRIBE ---
            print("🧠 Transcribing...")
            user_text = pipeline.transcribe(audio_utterance)
            print(f"👤 You said: '{user_text}'")

            # Check for exit commands
            if user_text.lower() in ["exit", "quit", "stop", "goodbye", "stop listening"]:
                print("👋 Goodbye!")
                break

            # Handle empty transcriptions (happens if VAD triggered on a loud noise)
            if not user_text:
                print("⚠️ Couldn't catch that. Try again.")
                continue

            # --- PHASE 3: THINK & ACT ---
            print("🤖 Analyzing markets and generating report...")
            report_text = agent.run(user_text)

            print(f"📝 [AGENT DEBUG] Raw output from LLM: '{report_text}'\n")

            # --- PHASE 4: SPEAK (WITH BARGE-IN) ---
            print("🔊 Speaking... (Interrupt me if you want!)")
            interrupted = await pipeline.speak_with_barge_in(report_text)

            if interrupted:
                print("⚠️ Interrupted! Processing your new input...\n")
                # The loop naturally continues, and the interrupted audio
                # is already waiting in vad.audio_queue!

    except KeyboardInterrupt:
        pass
    finally:
        vad.stop()
        print("✅ Microphone closed. Session ended.")


if __name__ == "__main__":
    # Run the async event loop
    asyncio.run(main())