# LLM Server

Отдельный HTTP-сервер для запуска локальных GGUF моделей. Работает как самостоятельный процесс, благодаря чему llama_cpp и Kokoro TTS используют независимые CUDA-контексты — без конкуренции за GPU.

Совместим с OpenAI Chat Completions API, поэтому основное приложение общается с ним через стандартный `openai` клиент.

## Установка зависимостей

```bash
cd llm_server
pip install -r requirements.txt
```

### llama-cpp-python с поддержкой CUDA (рекомендуется)

По умолчанию `pip install llama-cpp-python` собирает CPU-версию. Для GPU нужна сборка с CUDA.

**Предварительные требования (Windows):**
- Visual Studio Community с компонентом **"Desktop development with C++"**
- CUDA Toolkit, совместимый с вашей GPU (12.x или 11.x)

**Установка:**
```powershell
# PowerShell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

```cmd
rem Command Prompt
set CMAKE_ARGS=-DGGML_CUDA=on
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

## Запуск сервера вручную

```bash
python server.py --model ../models/llama-3.2-3b-instruct-q4_k_m.gguf
```

Все параметры:

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--model` | — | Путь к GGUF файлу (можно не указывать при старте) |
| `--host` | `127.0.0.1` | Адрес для прослушивания |
| `--port` | `8765` | Порт |
| `--n-gpu-layers` | `20` | Количество слоёв на GPU |
| `--n-ctx` | `2048` | Размер контекстного окна |

## Автоматический запуск из приложения

При `LLM_BACKEND = "local_server"` в `config.py` основное приложение запускает сервер автоматически и ждёт его готовности (до `LOCAL_SERVER_STARTUP_TIMEOUT` секунд).

## Эндпоинты

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/health` | Статус модели и параметры загрузки |
| `GET` | `/v1/models` | Список моделей (OpenAI-совместимый) |
| `POST` | `/v1/chat/completions` | Генерация ответа (streaming и обычный режим) |
| `POST` | `/v1/model/load` | Горячая замена модели без перезапуска сервера |

### Горячая замена модели

```bash
curl -X POST http://127.0.0.1:8765/v1/model/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "../models/qwen2.5-3b-instruct-q4_k_m.gguf",
    "n_gpu_layers": 20,
    "n_ctx": 2048
  }'
```

## Подбор параметров GPU

`n_gpu_layers` — количество слоёв модели, выгружаемых на GPU. Чем больше, тем быстрее генерация, но больше VRAM.

| VRAM | Рекомендуемое значение |
|---|---|
| 4 GB | 10–15 |
| 6 GB | 15–20 |
| 8 GB | 20–25 |
| 12 GB+ | 25+ (полная выгрузка) |

При нехватке VRAM модель автоматически переключается на CPU для оставшихся слоёв.

## Рекомендуемые модели

- `Llama-3.2-3B-Instruct-Q4_K_M.gguf` — хороший баланс качества и скорости
- `Qwen2.5-3B-Instruct-Q4_K_M.gguf` — альтернатива с сильной многоязычностью
