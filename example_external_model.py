#!/usr/bin/env python3
"""
Example script demonstrating how to use the unified ExternalLLMManager with llama_cpp
"""

import logging
import os
import sys
from queue import Queue
from threading import Event

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s'
)

def main():
    """Main function demonstrating external model usage with streaming and history"""

    try:
        from llm import ExternalLLMManager, LLAMA_CPP_AVAILABLE

        if not LLAMA_CPP_AVAILABLE:
            logging.error("llama_cpp is not installed. Please install llama-cpp-python first.")
            return

        # Create an instance of the extended LLM manager
        llm_manager = ExternalLLMManager()
        logging.info("External LLM Manager initialized successfully")

        # Use model path from config or fallback
        model_path = config.EXTERNAL_MODEL_PATH
        if not model_path or not os.path.exists(model_path):
            logging.warning("Please configure EXTERNAL_MODEL_PATH in config.py to point to a valid GGUF file.")
            logging.info("Example usage demonstration (Dry run):")
            logging.info("  1. Configure GGUF model path in config.py")
            logging.info("  2. Run this script to test loading and generation")
            return

        logging.info(f"Loading external model from: {model_path}")
        success = llm_manager.load_external_model(
            model_path=model_path,
            n_gpu_layers=config.EXTERNAL_N_GPU_LAYERS,
            n_ctx=config.EXTERNAL_N_CTX
        )

        if not success:
            logging.error("Failed to load the model.")
            return

        # Show model info
        model_info = llm_manager.get_external_model_info()
        logging.info(f"Model info: {model_info}")

        # Setup mock queues and events to test streaming pipeline
        tts_queue = Queue()
        stop_event = Event()

        user_input = "Hello Emma! How are you doing today? Can you explain what Docker is in one short sentence?"
        logging.info(f"Sending prompt: {user_input!r}")

        print("\n--- Streaming Response Start ---")
        
        # We run the generator. It will print tokens as they arrive and push sentences to the tts_queue
        response_text = llm_manager.stream_and_queue_tts(
            user_text=user_input,
            tts_queue=tts_queue,
            stop_event=stop_event
        )
        
        print("\n--- Streaming Response End ---\n")

        logging.info(f"Full reply saved in history: {response_text!r}")

        # Drain the tts_queue to see parsed sentences
        logging.info("Sentences parsed and sent to TTS queue:")
        while not tts_queue.empty():
            sentence = tts_queue.get()
            logging.info(f"  - {sentence}")

        # Unload model to release VRAM/RAM
        llm_manager.unload_external_model()
        logging.info("Model unloaded successfully.")

    except ImportError as e:
        logging.error(f"Failed to import ExternalLLMManager: {e}")
        logging.info("Make sure llama-cpp-python is installed.")

if __name__ == "__main__":
    main()