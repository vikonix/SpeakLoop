import re
from queue import Queue
from threading import Event
from openai import OpenAI
import config

# Technical configuration parameters
LLM_TIMEOUT = 5.0


class LLMManager:
    def __init__(self):
        self.client = None
        # Chat history buffer starting with the system instructions
        self.messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]

    def init_client(self):
        """Configure OpenAI compatible client mapped to look at local LM Studio instance."""
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
            return True
        except Exception as error:
            print(f"LM Studio not available: {error}")
            return False

    def stream_and_queue_tts(self, user_text: str, tts_queue: Queue, stop_event: Event) -> str:
        """
        Streams text from the LLM, parses sentences using regex on-the-fly,
        and pushes completed strings into the TTS queue.
        """
        assert self.client is not None

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
                    break

                token = chunk.choices[0].delta.content or ""
                if not token:
                    continue

                print(token, end="", flush=True)
                full_reply += token
                sentence_buffer += token

                parts = sentence_end.split(sentence_buffer)
                if len(parts) > 1:
                    for i in range(len(parts) - 1):
                        text_to_speak = parts[i].strip()
                        if text_to_speak:
                            tts_queue.put(text_to_speak)
                    sentence_buffer = parts[-1]

            # Flush any residual text remaining inside the buffer
            remaining_text = sentence_buffer.strip()
            if remaining_text and not stop_event.is_set():
                tts_queue.put(remaining_text)

            final_reply = full_reply.strip() if full_reply.strip() else "Sorry, I did not get a response."
            self.messages.append({"role": "assistant", "content": final_reply})
            self._trim_history()

            return final_reply

        except Exception as error:
            print(f"\nLLM Stream error: {error}")
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
		