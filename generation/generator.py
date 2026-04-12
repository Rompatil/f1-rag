"""
Generation Layer — sends retrieved context + user query to Claude.
Supports both streaming (for the frontend) and non-streaming (for CLI/API).
"""

import re
from collections.abc import Generator as GenType

import httpx

from utils.config import settings
from utils.logger import logger


SYSTEM_PROMPT = """You are an expert Formula 1 analyst and historian with deep knowledge of the sport.

You have been given retrieved context data below. Use it as your PRIMARY source of truth.

Guidelines:
1. PRIORITIZE the retrieved context for specific stats, results, and standings.
2. If the context covers the question well, base your answer on it and cite the source chunk IDs.
3. If the context only partially covers the question, use it for what it covers and supplement with your general F1 knowledge. Clearly distinguish between data-backed facts and general knowledge.
4. If the context has nothing relevant, answer from your own F1 expertise — but note that the answer is from general knowledge rather than the database.
5. ALWAYS give a complete, helpful answer. Never refuse to answer.
6. Be concise but thorough. Use bullet points, comparisons, or brief tables when they help.
7. Add analytical insight — don't just list raw numbers, explain what they mean.

End your answer with exactly one line:
CONFIDENCE: [high/medium/low]

- high = context directly answered the question with data
- medium = partial context + supplemented with general knowledge
- low = mostly general knowledge, minimal context match"""


class Generator:
    """Generates grounded answers using Claude. Supports streaming."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.generation.anthropic_api_key
        self.model = model or settings.generation.model
        self._client: httpx.Client | None = None

        if not self.api_key:
            logger.warning("No ANTHROPIC_API_KEY set. Generation will fail.")

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url="https://api.anthropic.com",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                timeout=60.0,
            )
        return self._client

    def _build_payload(self, query: str, context: str, stream: bool = False) -> dict:
        user_message = (
            f"RETRIEVED CONTEXT FROM DATABASE:\n{context}\n\n---\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Give a complete, helpful answer. Use the retrieved context as your primary source. "
            f"If context is insufficient, supplement with your F1 expertise. "
            f"End with CONFIDENCE: [high/medium/low]"
        )
        return {
            "model": self.model,
            "max_tokens": settings.generation.max_tokens,
            "temperature": settings.generation.temperature,
            "system": SYSTEM_PROMPT,
            "stream": stream,
            "messages": [{"role": "user", "content": user_message}],
        }

    def generate(self, query: str, context: str) -> dict:
        """Non-streaming generation. Returns complete answer."""
        if not self.api_key:
            return {"answer": "Error: ANTHROPIC_API_KEY not configured.", "confidence": 0.0, "raw_response": None}

        payload = self._build_payload(query, context, stream=False)

        try:
            response = self.client.post("/v1/messages", json=payload)
            response.raise_for_status()
            data = response.json()

            answer_text = "".join(
                block["text"] for block in data.get("content", [])
                if block.get("type") == "text"
            )

            confidence = self._parse_confidence(answer_text)
            answer_clean = self._strip_confidence_line(answer_text)

            return {"answer": answer_clean, "confidence": confidence, "raw_response": data}

        except httpx.HTTPStatusError as e:
            logger.error(f"Claude API error: {e.response.status_code}")
            return {"answer": f"Claude API error: {e.response.status_code}", "confidence": 0.0, "raw_response": None}
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {"answer": f"Generation failed: {e}", "confidence": 0.0, "raw_response": None}

    def generate_stream(self, query: str, context: str) -> GenType[str, None, None]:
        """
        Streaming generation. Yields text chunks as they arrive.
        The final yield is a JSON metadata line: {"confidence": 0.95}
        """
        if not self.api_key:
            yield "Error: ANTHROPIC_API_KEY not configured."
            return

        payload = self._build_payload(query, context, stream=True)

        try:
            with self.client.stream("POST", "/v1/messages", json=payload) as response:
                response.raise_for_status()
                full_text = ""
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        event = __import__("json").loads(data_str)
                    except Exception:
                        continue

                    event_type = event.get("type", "")

                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            full_text += text
                            yield text

                    elif event_type == "message_stop":
                        break

                # After stream ends, emit confidence as metadata
                confidence = self._parse_confidence(full_text)
                yield f"\n__META__:{confidence}"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"\n\nStreaming error: {e}"

    @staticmethod
    def _parse_confidence(text: str) -> float:
        match = re.search(r"CONFIDENCE:\s*\[?(high|medium|low)\]?", text, re.IGNORECASE)
        if match:
            return {"high": 0.95, "medium": 0.70, "low": 0.40}.get(match.group(1).lower(), 0.5)
        return 0.5

    @staticmethod
    def _strip_confidence_line(text: str) -> str:
        return re.sub(r"\n*CONFIDENCE:\s*\[?(high|medium|low)\]?\s*$", "", text, flags=re.IGNORECASE).strip()

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
