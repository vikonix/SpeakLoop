# SpeakLoop

Голосовой AI-тьютор для практики иностранных языков в живом разговоре.

## О проекте

SpeakLoop — десктопное приложение (Windows), которое позволяет практиковать разговорный иностранный язык с AI-собеседником. Нажимаешь и удерживаешь пробел — говоришь — отпускаешь. Приложение распознаёт речь, отправляет в LLM и озвучивает ответ.

## Технологии

- **GUI** — Tkinter
- **STT** — faster-whisper (Whisper small по умолчанию)
- **LLM** — локальная GGUF-модель через `llm_server/` или LM Studio
- **TTS** — Kokoro (hexgrad/Kokoro-82M)
- **Python** 3.11+

## Установка

```bash
git clone https://github.com/yourusername/speakloop.git
cd speakloop

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

Для `llm_server` — отдельные зависимости:
```bash
pip install -r llm_server/requirements.txt
```

Установка `llama-cpp-python` с поддержкой CUDA — см. [`llm_server/README.md`](llm_server/README.md).

## Настройка

Все параметры в [`config.py`](config.py):

```python
# Выбор LLM-бэкенда
LLM_BACKEND = "local_server"   # рекомендуется
# LLM_BACKEND = "lm-studio"   # если используется LM Studio

# Путь к GGUF-модели (для local_server)
EXTERNAL_MODEL_PATH = "models/llama-3.2-3b-instruct-q4_k_m.gguf"

# Язык обучения
NATIVE_LANGUAGE = "Russian"
TARGET_LANGUAGE = "English"
```

## Запуск

```bash
python main.py
```

При `LLM_BACKEND = "local_server"` сервер запускается автоматически. При `LLM_BACKEND = "lm-studio"` нужно предварительно запустить LM Studio.

## Управление

- **Пробел (удерживать)** — запись речи
- **ESC** — выход

## Структура проекта

```
speakloop/
├── main.py          — GUI, оркестрация потоков
├── stt.py           — Speech-to-Text (Whisper)
├── llm.py           — LLM-клиент (OpenAI-совместимый)
├── tts.py           — Text-to-Speech (Kokoro)
├── config.py        — вся конфигурация
├── models/          — GGUF-файлы моделей
└── llm_server/      — отдельный процесс для локальной LLM
    ├── server.py
    ├── requirements.txt
    └── README.md
```
