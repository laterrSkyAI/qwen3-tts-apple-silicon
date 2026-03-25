"""Бенчмарк кастомных женских голосов Qwen3-TTS (VoiceDesign 1.7B)."""
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

MODEL_DIR = os.path.join(os.getcwd(), "models", "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit")
OUTPUT_DIR = "benchmark_voices_output"
SAMPLE_RATE = 24000

TEXT = "Привет! Меня зовут Нейро, я - искусственный интеллект. А как зовут тебя, дорогой подписчик?"

# Разные женские голоса с разными интонациями
VOICES = [
    {
        "name": "01_cheerful_young",
        "instruct": "A young woman in her early twenties with a bright, cheerful and energetic voice. She speaks with excitement and enthusiasm, like a friendly streamer greeting her audience.",
    },
    {
        "name": "02_calm_warm",
        "instruct": "A mature woman with a calm, warm and soothing voice. She speaks gently and slowly, like a caring older sister. Her tone is relaxed and comforting.",
    },
    {
        "name": "03_soft_whisper",
        "instruct": "A young woman with a soft, delicate and slightly breathy voice. She speaks quietly, almost whispering, in a very intimate and gentle manner.",
    },
    {
        "name": "04_confident_clear",
        "instruct": "A confident woman with a clear, strong and articulate voice. She speaks with authority and charisma, like a professional news anchor or a public speaker.",
    },
    {
        "name": "05_cute_excited",
        "instruct": "A cute young girl with a high-pitched, sweet and very excited voice. She speaks fast with lots of energy and emotion, like an anime character who is very happy.",
    },
    {
        "name": "06_elegant_slow",
        "instruct": "An elegant woman with a deep, rich and melodic voice. She speaks slowly and gracefully, with a sophisticated and refined tone, like a narrator of a fairy tale.",
    },
    {
        "name": "07_playful_flirty",
        "instruct": "A young woman with a playful, teasing and slightly flirtatious voice. She speaks with a smile in her voice, varying her pitch up and down in a fun, engaging way.",
    },
    {
        "name": "08_serious_low",
        "instruct": "A woman with a low-pitched, serious and dramatic voice. She speaks slowly with gravitas, like a movie trailer narrator. Her tone is deep and commanding.",
    },
]


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
    print(" Бенчмарк кастомных женских голосов — Qwen3-TTS VoiceDesign")
    print("=" * 60)
    print(f"\nТекст: {TEXT}")
    print(f"Голосов: {len(VOICES)}")
    print(f"Модель: VoiceDesign 1.7B (8-bit)")

    # Загрузка модели
    print(f"\nЗагрузка модели...")
    t0 = time.time()
    model = load_model(MODEL_DIR)
    load_time = time.time() - t0
    print(f"Модель загружена за {load_time:.1f} сек\n")

    results = []

    for i, voice in enumerate(VOICES):
        print(f"--- [{i+1}/{len(VOICES)}] {voice['name']} ---")
        print(f"Описание: {voice['instruct'][:80]}...")

        temp_dir = f"bench_voice_temp_{i}"

        t0 = time.time()
        try:
            generate_audio(
                model=model,
                text=TEXT,
                instruct=voice["instruct"],
                lang_code="russian",
                output_path=temp_dir,
            )
        except Exception as e:
            print(f"  ОШИБКА: {e}\n")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            continue
        gen_time = time.time() - t0

        # Длительность аудио
        audio_path = os.path.join(temp_dir, "audio_000.wav")
        duration = get_wav_duration(audio_path)
        rtf = gen_time / duration if duration > 0 else float('inf')

        print(f"  Генерация: {gen_time:.2f} сек | Аудио: {duration:.2f} сек | RTF: {rtf:.2f}x")

        # Сохраняем файл
        output_file = os.path.join(OUTPUT_DIR, f"{voice['name']}.wav")
        if os.path.exists(audio_path):
            shutil.move(audio_path, output_file)
            results.append({
                "name": voice["name"],
                "instruct": voice["instruct"],
                "gen_time": gen_time,
                "duration": duration,
                "rtf": rtf,
                "file": output_file,
            })

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        print()

    # Итоги
    print("=" * 60)
    print(" РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"{'Голос':<25} {'Генерация':>10} {'Аудио':>8} {'RTF':>7}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<25} {r['gen_time']:>8.2f}s {r['duration']:>6.2f}s {r['rtf']:>6.2f}x")

    if results:
        avg_rtf = sum(r["rtf"] for r in results) / len(results)
        avg_gen = sum(r["gen_time"] for r in results) / len(results)
        print("-" * 55)
        print(f"{'Среднее':<25} {avg_gen:>8.2f}s {'':>8} {avg_rtf:>6.2f}x")

    print(f"\nФайлы сохранены в {OUTPUT_DIR}/")

    # Воспроизведение
    print("\nВоспроизвести все голоса? (y/n): ", end="", flush=True)
    try:
        choice = input().strip().lower()
        if choice in ("y", "yes", "д", "да"):
            for r in results:
                print(f"\n▶ {r['name']}")
                print(f"  {r['instruct'][:70]}...")
                subprocess.run(["afplay", r["file"]], check=False,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (KeyboardInterrupt, EOFError):
        pass

    print("\nГотово!")


if __name__ == "__main__":
    main()
