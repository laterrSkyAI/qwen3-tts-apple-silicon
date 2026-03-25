"""Бенчмарк 0.6B модели для сравнения с 1.7B."""
import os
import sys
import time
import wave
import shutil
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio

MODEL_DIR = os.path.join(os.getcwd(), "models", "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit")
OUTPUT_DIR = "benchmark_output_lite"
SAMPLE_RATE = 24000

TEST_TEXTS = [
    "Hello! This is a test of the Qwen three text to speech model running on Apple Silicon.",
    "The quick brown fox jumps over the lazy dog. Testing speed and quality of speech synthesis.",
    "Привет! Это тест модели синтеза речи на русском языке. Как звучит голос?",
]

def get_wav_duration(path):
    try:
        with wave.open(path, 'rb') as f:
            return f.getnframes() / f.getframerate()
    except Exception:
        return 0

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Загрузка модели 0.6B из {MODEL_DIR}...")
    t0 = time.time()
    model = load_model(MODEL_DIR)
    load_time = time.time() - t0
    print(f"Модель загружена за {load_time:.1f} сек\n")

    for i, text in enumerate(TEST_TEXTS):
        print(f"--- Тест {i+1} ---")
        print(f"Текст: {text[:60]}...")

        temp_dir = f"bench_lite_temp_{i}"

        t0 = time.time()
        generate_audio(
            model=model, text=text, voice="Vivian",
            instruct="Normal tone", speed=1.0, output_path=temp_dir
        )
        gen_time = time.time() - t0

        audio_path = os.path.join(temp_dir, "audio_000.wav")
        duration = get_wav_duration(audio_path)
        rtf = gen_time / duration if duration > 0 else float('inf')

        print(f"Время генерации: {gen_time:.2f} сек")
        print(f"Длительность аудио: {duration:.2f} сек")
        print(f"RTF: {rtf:.2f}x")
        print()

        if os.path.exists(audio_path):
            os.rename(audio_path, os.path.join(OUTPUT_DIR, f"test_{i+1}.wav"))
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    print("=" * 40)
    print(f"Готово! Аудиофайлы в папке {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
