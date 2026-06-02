import re
import logging
from queue import Queue
from threading import Event
from openai import OpenAI
import config

# Technical configuration parameters
LLM_TIMEOUT = 30.0


class LLMManager:
    def __init__(self):
        self.client = None
        # Chat history buffer starting with the system instructions
        self.messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]

    def init_client(self):
        """Configure OpenAI compatible client mapped to look at local LM Studio instance."""
        logging.info("Initializing LM Studio client...")
        self.client = OpenAI(
            base_url=config.LM_STUDIO_URL,
            api_key=config.LM_STUDIO_API_KEY,
            timeout=LLM_TIMEOUT,
        )

    def check_connection(self) -> bool:
        """Validates connectivity to the local LLM server."""
        try:
            assert self.client is not None
            self.client.models.list()
            logging.info("Successfully connected to LM Studio.")
            return True
        except Exception as error:
            logging.error(f"LM Studio not available: {error}")
            return False

    def stream_and_queue_tts(self, user_text: str, tts_queue: Queue, stop_event: Event, token_callback=None) -> str:
        """
        Streams text from the LLM, parses sentences using regex on-the-fly,
        and pushes completed strings into the TTS queue.
        """
        assert self.client is not None

        logging.info(f"LLM request started for user input: {user_text!r}")
        self.messages.append({"role": "user", "content": user_text})
        self._trim_history()

        try:
            stream_response = self.client.chat.completions.create(
                model=config.LM_STUDIO_MODEL,
                messages=self.messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
                top_p=config.LLM_TOP_P,
                stream=True,
                timeout=LLM_TIMEOUT,
            )

            full_reply = ""
            sentence_buffer = ""
            # Regex tracking common punctuation (. ! ?) followed by spaces to detect completed sentences
            sentence_end = re.compile(r'(?<=[.!?])\s+')

            for chunk in stream_response:
                if stop_event.is_set():
                    logging.info("LLM streaming interrupted by user stop event.")
                    break

                token = chunk.choices[0].delta.content or ""
                if not token:
                    continue

                if token_callback:
                    token_callback(token)
                else:
                    print(token, end="", flush=True)
                full_reply += token
                sentence_buffer += token

                parts = sentence_end.split(sentence_buffer)
                if len(parts) > 1:
                    sentence_buffer = parts.pop()
                    for item in parts:
                        text_to_speak = item.strip()
                        if text_to_speak:
                            logging.info(f"Queued sentence to TTS: {text_to_speak!r}")
                            tts_queue.put(text_to_speak)

            # Flush any residual text remaining inside the buffer
            remaining_text = sentence_buffer.strip()
            if remaining_text and not stop_event.is_set():
                logging.info(f"Queued final sentence segment to TTS: {remaining_text!r}")
                tts_queue.put(remaining_text)

            final_reply = full_reply.strip() if full_reply.strip() else "Sorry, I did not get a response."
            self.messages.append({"role": "assistant", "content": final_reply})
            self._trim_history()

            logging.info(f"LLM full response: {final_reply!r}")
            return final_reply

        except Exception as error:
            logging.error(f"LLM Stream error: {error}")
            return "Sorry, please try again."

    def _trim_history(self, max_pairs: int = None):
        """Prunes conversation state list sizes to conserve system context window bounds."""
        if max_pairs is None:
            max_pairs = config.LLM_HISTORY_MAX_PAIRS
        system_message = self.messages[0]
        conversation = self.messages[1:]
        max_messages = max_pairs * 2
        conversation = conversation[-max_messages:]
        self.messages = [system_message] + conversation


# External LLM support (using llama_cpp)
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True

    class ExternalLLMManager(LLMManager):
        """Extended LLM Manager that supports external GGUF models with chat completion streaming"""

        def __init__(self):
            super().__init__()
            self.external_model = None
            self.external_model_path = None
            self.is_external_loaded = False

        def load_external_model(self, model_path: str, n_gpu_layers: int = 20, n_ctx: int = 2048, **kwargs) -> bool:
            """Load an external GGUF model using llama_cpp"""
            try:
                import os
                if not os.path.exists(model_path):
                    logging.error(f"Model file not found: {model_path}")
                    return False

                logging.info(f"Loading external model from: {model_path} with {n_gpu_layers} GPU layers, {n_ctx} context...")

                # Initialize the Llama model
                self.external_model = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                    **kwargs
                )

                self.external_model_path = model_path
                self.is_external_loaded = True
                logging.info("External model loaded successfully")
                return True

            except Exception as e:
                logging.error(f"Failed to load external model: {e}")
                return False

        def stream_and_queue_tts(self, user_text: str, tts_queue: Queue, stop_event: Event, token_callback=None) -> str:
            """
            Streams text from the local GGUF model, parses sentences using regex on-the-fly,
            and pushes completed strings into the TTS queue.
            """
            if not self.is_external_loaded or self.external_model is None:
                logging.error("No external model loaded for stream_and_queue_tts")
                return "Sorry, please wait for the local model to finish loading."

            logging.info(f"Local GGUF LLM request started for user input: {user_text!r}")
            self.messages.append({"role": "user", "content": user_text})
            self._trim_history()

            try:
                # Use llama_cpp's create_chat_completion which formats prompts using model templates
                stream_response = self.external_model.create_chat_completion(
                    messages=self.messages,
                    temperature=config.LLM_TEMPERATURE,
                    max_tokens=config.LLM_MAX_TOKENS,
                    top_p=config.LLM_TOP_P,
                    stream=True
                )

                full_reply = ""
                sentence_buffer = ""
                sentence_end = re.compile(r'(?<=[.!?])\s+')

                for chunk in stream_response:
                    if stop_event.is_set():
                        logging.info("Local LLM streaming interrupted by user stop event.")
                        break

                    if not chunk:
                        continue

                    # Safe extraction of token text
                    try:
                        delta = chunk['choices'][0]['delta'] if isinstance(chunk, dict) else chunk.choices[0].delta
                        token = delta.get('content', '') if isinstance(delta, dict) else (getattr(delta, 'content', '') or '')
                    except (KeyError, IndexError, AttributeError) as e:
                        logging.debug(f"Failed to extract token from chunk {chunk}: {e}")
                        continue

                    if not token:
                        continue

                    if token_callback:
                        token_callback(token)
                    else:
                        print(token, end="", flush=True)

                    full_reply += token
                    sentence_buffer += token

                    parts = sentence_end.split(sentence_buffer)
                    if len(parts) > 1:
                        sentence_buffer = parts.pop()
                        for item in parts:
                            text_to_speak = item.strip()
                            if text_to_speak:
                                logging.info(f"Queued sentence to TTS (local model): {text_to_speak!r}")
                                tts_queue.put(text_to_speak)

                # Flush any residual text remaining inside the buffer
                remaining_text = sentence_buffer.strip()
                if remaining_text and not stop_event.is_set():
                    logging.info(f"Queued final sentence segment to TTS (local model): {remaining_text!r}")
                    tts_queue.put(remaining_text)

                final_reply = full_reply.strip() if full_reply.strip() else "Sorry, I did not get a response."
                self.messages.append({"role": "assistant", "content": final_reply})
                self._trim_history()

                logging.info(f"Local LLM full response: {final_reply!r}")
                return final_reply

            except Exception as error:
                logging.error(f"Local GGUF LLM Stream error: {error}")
                return "Sorry, please try again."

        def unload_external_model(self):
            """Unload the current external model and free resources"""
            if self.external_model is not None:
                try:
                    if hasattr(self.external_model, 'close'):
                        self.external_model.close()
                except Exception as e:
                    logging.error(f"Error while closing external model: {e}")
                self.external_model = None
                self.external_model_path = None
                self.is_external_loaded = False
                logging.info("External GGUF model unloaded successfully")

        def get_external_model_info(self) -> dict:
            """Get information about the currently loaded external model"""
            if not self.is_external_loaded or self.external_model is None:
                return {"status": "not_loaded"}

            try:
                return {
                    "status": "loaded",
                    "model_path": self.external_model_path,
                    "n_gpu_layers": getattr(self.external_model, 'n_gpu_layers', 'unknown'),
                    "n_ctx": getattr(self.external_model, 'n_ctx', 'unknown')
                }
            except Exception as e:
                logging.error(f"Failed to get external model info: {e}")
                return {"status": "error", "error": str(e)}

except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logging.warning("llama_cpp not installed. External model loading will be disabled.")
