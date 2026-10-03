"""Deepgram Aura streaming TTS over the raw WebSocket API (``/v1/speak``).

Text goes in as ``Speak`` messages; audio comes back as binary frames in the
requested telephony format (μ-law 8 kHz), so it can be forwarded to Twilio
without transcoding. Deepgram only synthesises buffered text on ``Flush``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus

from app.core.logging import get_logger
from app.voice.providers.base import AudioFormat, TTSConnectionError, TTSError, TTSProvider

log = get_logger(__name__)


@dataclass
class _StreamState:
    flushes_sent: int = 0
    flushes_acked: int = 0
    text_done: bool = False
    send_error: Exception | None = None
    finished: asyncio.Event = field(default_factory=asyncio.Event)

    def ack_flush(self) -> None:
        self.flushes_acked += 1
        if self.text_done and self.flushes_acked >= self.flushes_sent:
            self.finished.set()


class DeepgramTTSProvider(TTSProvider):
    name = "deepgram"

    def __init__(
        self,
        api_key: str,
        *,
        url: str = "wss://api.deepgram.com/v1/speak",
        model: str = "aura-2-thalia-en",
        connect_timeout_s: float = 5.0,
        flush_timeout_s: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._url = url
        self._model = model
        self._connect_timeout_s = connect_timeout_s
        self._flush_timeout_s = flush_timeout_s

    def build_url(self, audio_format: AudioFormat) -> str:
        params: list[tuple[str, str | int]] = [
            ("model", self._model),
            ("encoding", audio_format.encoding),
            ("sample_rate", audio_format.sample_rate),
        ]
        return f"{self._url}?{urlencode(params)}"

    async def synthesize_stream(
        self, text: AsyncIterator[str], audio_format: AudioFormat
    ) -> AsyncIterator[bytes]:
        try:
            ws = await connect(
                self.build_url(audio_format),
                additional_headers={"Authorization": f"Token {self._api_key}"},
                open_timeout=self._connect_timeout_s,
                max_size=2**20,
            )
        except InvalidStatus as exc:
            status = exc.response.status_code
            raise TTSConnectionError(f"Deepgram rejected connection (HTTP {status})") from exc
        except (OSError, TimeoutError, InvalidHandshake) as exc:
            raise TTSConnectionError(f"Could not connect to Deepgram: {exc!r}") from exc

        state = _StreamState()
        sender = asyncio.create_task(self._send_text(ws, text, state), name="dg-tts-send")
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    yield raw
                else:
                    self._on_message(raw, state)
                if state.finished.is_set():
                    break
        except ConnectionClosed as exc:
            raise TTSError(f"Deepgram connection closed unexpectedly: {exc}") from exc
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            with contextlib.suppress(ConnectionClosed):
                await ws.send(json.dumps({"type": "Close"}))
            await ws.close()

        if state.send_error is not None:
            raise TTSError(
                f"Deepgram text sender failed: {state.send_error!r}"
            ) from state.send_error
        if not state.finished.is_set():
            raise TTSError(
                f"Deepgram closed before all audio was returned: code={ws.close_code} "
                f"{ws.close_reason!r}"
            )

    async def _send_text(
        self, ws: ClientConnection, text: AsyncIterator[str], state: _StreamState
    ) -> None:
        unflushed = False
        try:
            async for chunk in text:
                if not chunk.strip():
                    continue
                await ws.send(json.dumps({"type": "Speak", "text": chunk + " "}))
                unflushed = True
                if state.flushes_sent == 0:
                    # Flush the first sentence on its own so speech starts while the
                    # rest of the reply is still being written. Later sentences are
                    # flushed together: Deepgram rate-limits Flush messages.
                    await self._flush(ws, state)
                    unflushed = False
        except Exception as exc:
            # The receive loop is waiting on this socket; closing it ends that loop.
            state.send_error = exc
            await ws.close()
            return

        state.text_done = True
        if unflushed:
            await self._flush(ws, state)
        elif state.flushes_acked >= state.flushes_sent:
            state.finished.set()
        try:
            await asyncio.wait_for(state.finished.wait(), self._flush_timeout_s)
        except TimeoutError:
            log.warning("deepgram_tts.flush_timeout", timeout_s=self._flush_timeout_s)
        # Closing from our side ends the receive loop whether or not audio completed.
        await ws.close()

    @staticmethod
    async def _flush(ws: ClientConnection, state: _StreamState) -> None:
        state.flushes_sent += 1
        await ws.send(json.dumps({"type": "Flush"}))

    @staticmethod
    def _on_message(raw: str, state: _StreamState) -> None:
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("deepgram_tts.invalid_json", raw=raw[:200])
            return
        match data.get("type"):
            case "Flushed":
                state.ack_flush()
            case "Metadata":
                log.debug("deepgram_tts.connected", request_id=data.get("request_id"))
            case "Warning" | "Error":
                log.warning("deepgram_tts.warning", payload=data)
            case other:
                log.debug("deepgram_tts.unhandled_message", type=other)
