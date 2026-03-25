"""Бенчмарк Voice Cloning — Qwen3-TTS Base 1.7B."""
import os
import sys
import time
import wave
import shutil
import subprocess
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio

MODEL_DIR = os.path.join(os.getcwd(), "models", "Qwen3-TTS-12Hz-1.7B-Base-8bit")
OUTPUT_DIR = "benchmark_clone_output"
SAMPLE_RATE = 24000

# Референсный голос
REF_AUDIO = os.path.join(os.getcwd(), "ref_voices", "qwen_output_russian.wav")
REF_TEXT = "Привет, меня зовут Нейро! Я искусственный интеллект"

# Текст для озвучки
TEXT = "Привет! Меня зовут Нейро, я - искусственный интеллект. А как зовут тебя, дорогой подписчик?"


def get_wav_duration(path):
    """Получить длительность wav файла в секундах."""
    try:
        with wave.open(path, 'rb') as f:
            return f.getnframes() / f.getframerate()
    except Exception:
        return 0


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print(" Voice Cloning — Qwen3-TTS Base 1.7B")
    print("=" * 60)
    print(f"\nРеференс: {REF_AUDIO}")
    print(f"Транскрипт: {REF_TEXT}")
    print(f"Длительность референса: {get_wav_duration(REF_AUDIO):.2f} сек")
    print(f"\nТекст: {TEXT}")

    # Загрузка модели
    print(f"\nЗагрузка модели...")
    t0 = time.time()
    model = load_model(MODEL_DIR)
    load_time = time.time() - t0
    print(f"Модель загружена за {load_time:.1f} сек\n")

    print("Генерация клонированного голоса...")
    temp_dir = "bench_clone_temp"

    t0 = time.time()
    try:
        generate_audio(
            model=model,
            text=TEXT,
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
            lang_code="russian",
            output_path=temp_dir,
        )
    except Exception as e:
        print(f"ОШИБКА: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return
    gen_time = time.time() - t0

    audio_path = os.path.join(temp_dir, "audio_000.wav")
    duration = get_wav_duration(audio_path)
    rtf = gen_time / duration if duration > 0 else float('inf')

    print(f"\nГенерация: {gen_time:.2f} сек")
    print(f"Аудио: {duration:.2f} сек")
    print(f"RTF: {rtf:.2f}x")

    # Сохраняем
    output_file = os.path.join(OUTPUT_DIR, "clone_neuro.wav")
    if os.path.exists(audio_path):
        shutil.move(audio_path, output_file)
        print(f"\nСохранено: {output_file}")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Воспроизведение
    print("\nВоспроизведение...")
    subprocess.run(["afplay", output_file], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("Готово!")


if __name__ == "__main__":
    main()
