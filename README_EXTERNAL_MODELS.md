# External LLM Models with llama_cpp

This document explains how to use external GGUF models with the `llama_cpp` library in this project.

## Installation

To run external GGUF models locally, you need the `llama-cpp-python` library. 

### CPU-Only Installation
If you do not have a dedicated NVIDIA GPU, or just want a quick setup:
```bash
pip install llama-cpp-python
```

### NVIDIA GPU (CUDA) Acceleration Installation (Highly Recommended)
To run models fast and unload layers to the GPU, you must compile `llama-cpp-python` with CUDA support:

1. **Prerequisites (Windows):**
   - Install **Visual Studio Community** (with the **"Desktop development with C++"** workload selected during install).
   - Install the **CUDA Toolkit** matching your GPU (e.g., CUDA 12.x or 11.x).
2. **Install Command:**
   Open a terminal (PowerShell or Command Prompt) and run:
   ```powershell
   $env:CMAKE_ARGS="-DGGML_CUDA=on"
   pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
   ```
   *(For classic Command Prompt `cmd`, use `set CMAKE_ARGS=-DGGML_CUDA=on` before running pip).*

3. **Verification:**
   Verify CUDA is enabled in Python:
   ```python
   from llama_cpp import Llama
   # If compiled with CUDA, GGML will show CUDA initialization logs on startup
   ```

## Available Models

The following GGUF models are mentioned as examples:

1. `Llama-3.2-3B-Instruct-Q4_K_M.gguf`
2. `Qwen2.5-3B-Instruct-Q4_K_M.gguf`

These are quantized models that can be run locally without requiring a server.

## Usage

### Loading a Model

```python
from llm import ExternalLLMManager

# Create manager instance
llm_manager = ExternalLLMManager()

# Load an external model
model_path = "/path/to/your/model/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
success = llm_manager.load_external_model(
    model_path=model_path,
    n_gpu_layers=20,   # Adjust based on your GPU capabilities (10, 20, 25...)
    n_ctx=2048         # Context window size
)

if success:
    print("Model loaded successfully")
```

### Generating Text

```python
# Generate text using the loaded model
prompt = "Объясни, что такое Docker простыми словами."
response = llm_manager.generate_with_external_model(
    prompt=prompt,
    max_tokens=300,
    temperature=0.7
)

print(response)
```

### Model Configuration Parameters

- `n_gpu_layers`: Number of layers to load on GPU (10, 20, 25...). Set based on your GPU memory.
- `n_ctx`: Context window size. Larger values allow more context but require more memory.
- `max_tokens`: Maximum number of tokens to generate.
- `temperature`: Controls randomness in generation (0.0 to 1.0).

## Model Paths

You need to specify the actual path to your GGUF model files. Common locations:

- Local downloads: `/home/user/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf`
- Windows: `C:\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf`
- Relative paths: `./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf`

## Example Code

See `example_external_model.py` for a complete working example.

## Notes

1. Make sure your system has sufficient RAM and GPU memory for the models you're loading.
2. The `n_gpu_layers` parameter should be adjusted based on your hardware capabilities:
   - 10-20 layers for smaller GPUs
   - 20-25 layers for mid-range GPUs
   - 25+ layers for high-end GPUs
3. Models will automatically fall back to CPU processing if GPU memory is insufficient.