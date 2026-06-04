import re
import logging
from queue import Queue
from threading import Event
from openai import OpenAI
import config

# Technical configuration parameters
LLM_TIMEOUT = 30.0


class LLMManager:
    def __init__(self, model: str = None):
        self.client = None
        # Model name sent in API requests; defaults to LM Studio value from config
        self.model = model or config.LM_STUDIO_MODEL
        # Chat history buffer starting with the system instructions
        self.messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]

    def init_client(self, base_url: str = None, api_key: str = None):
        """
        Configure OpenAI-compatible client.

        Defaults to LM Studio settings from config when arguments are omitted,
        so existing "lm-studio" backend usage is unchanged.
        """
        url = base_url or config.LM_STUDIO_URL
        key = api_key or config.LM_STUDIO_API_KEY
        logging.info(f"Initializing LLM client → {url}")
        self.client = OpenAI(
            base_url=url,
            api_key=key,
            timeout=LLM_TIMEOUT,
        )

    def check_connection(self) -> bool:
        """Validates connectivity to the local LLM server."""
        try:
            if self.client is None:
                raise RuntimeError("LLM client not initialized. Call init_client() first.")
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

        if self.client is None:
            raise RuntimeError("LLM client not initialized. Call init_client() first.")

        logging.info(f"LLM request started for user input: {user_text!r}")

        try:
            # Append user message only inside try — so we can roll back if streaming fails
            self.messages.append({"role": "user", "content": user_text})

            stream_response = self.client.chat.completions.create(
                model=self.model,
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
            # Roll back the user message so history stays consistent (user/assistant pairs)
            if self.messages and self.messages[-1].get("role") == "user":
                self.messages.pop()
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
