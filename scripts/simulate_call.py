"""Simulate a Twilio phone call against a locally running API — no phone or ngrok needed.

It exercises the real flow end to end:
  1. POST /twilio/voice (signed like Twilio, if TWILIO_AUTH_TOKEN is set)
  2. Read the <Stream> URL + per-call token from the returned TwiML
  3. Open the Media Stream WebSocket and send connected/start/media/stop frames
     in real time (20 ms μ-law frames), just like Twilio
  4. Stay on the line (sending silence) until the assistant's reply has been
     received, echoing its mark back the way Twilio does after playback
  5. Print the transcript and latency the API recorded for the call

Audio input (pick one):
  --text "I've had a cough for five days"   synthesised with macOS `say`
  --wav caller.wav                          16-bit mono PCM WAV at 8 kHz
  --ulaw caller.ulaw                        raw 8 kHz μ-law

Usage (from repo root, API running on :8000):
  uv run --project apps/api python scripts/simulate_call.py --text "I have a headache"
  ... --save-reply reply.wav      also write the assistant's speech to a WAV file
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import uuid
import wave
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx2 as httpx
from twilio.request_validator import RequestValidator
from websockets.asyncio.client import ClientConnection, connect

from app.core.config import Settings
from app.voice.audio import MULAW_SILENCE, mulaw_to_pcm16, pcm16_to_mulaw
from app.voice.twilio_protocol import outbound_mark, outbound_media

FRAME_BYTES = 160  # 20 ms at 8 kHz μ-law
TRAILING_SILENCE_S = 2.0  # lets endpointing fire before hang-up, as on a real call


def load_audio(args: argparse.Namespace) -> bytes:
    if args.ulaw:
        return Path(args.ulaw).read_bytes()
    wav_path = args.wav
    if args.text:
        if sys.platform != "darwin":
            sys.exit("--text uses macOS `say`; pass --wav or --ulaw on other platforms")
        wav_path = str(Path(tempfile.mkdtemp()) / "utterance.wav")
        subprocess.run(
            ["say", "-o", wav_path, "--file-format=WAVE", "--data-format=LEI16@8000", args.text],
            check=True,
        )
    with wave.open(wav_path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (8000, 1, 2):
            sys.exit(
                "WAV must be 8 kHz mono 16-bit. Convert with:\n"
                "  ffmpeg -i in.wav -ar 8000 -ac 1 -sample_fmt s16 out.wav"
            )
        return pcm16_to_mulaw(w.readframes(w.getnframes()))


class ReplyListener:
    """The caller's ear: collects the audio the API sends back over the stream."""

    def __init__(self) -> None:
        self.audio = bytearray()
        self.first_audio_at: float | None = None
        self.finished = asyncio.Event()  # set when a reply's mark arrives

    async def run(self, ws: ClientConnection, stream_sid: str) -> None:
        async for raw in ws:
            message = json.loads(raw)
            if message["event"] == "media":
                if self.first_audio_at is None:
                    self.first_audio_at = time.monotonic()
                self.audio.extend(base64.b64decode(message["media"]["payload"]))
            elif message["event"] == "mark":
                # Twilio echoes a mark once the audio queued before it has played.
                played_at = (self.first_audio_at or time.monotonic()) + len(self.audio) / 8000
                await asyncio.sleep(max(0.0, played_at - time.monotonic()))
                await ws.send(outbound_mark(stream_sid, message["mark"]["name"]))
                self.finished.set()

    def save(self, path: str) -> None:
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(mulaw_to_pcm16(bytes(self.audio)))


def signed_headers(settings: Settings, url: str, params: dict[str, str]) -> dict[str, str]:
    if settings.twilio_auth_token is None:
        return {}
    token = settings.twilio_auth_token.get_secret_value()
    return {"X-Twilio-Signature": RequestValidator(token).compute_signature(url, params)}


async def simulate(args: argparse.Namespace) -> None:
    settings = Settings()
    speech = load_audio(args)
    audio = speech + MULAW_SILENCE * int(8000 * TRAILING_SILENCE_S)
    call_sid = f"CAsim{uuid.uuid4().hex[:28]}"
    stream_sid = f"MZsim{uuid.uuid4().hex[:28]}"
    form = {
        "CallSid": call_sid,
        "From": "+15550100123",
        "To": "+15550100999",
        "AccountSid": "ACsim",
    }

    # Twilio signs the public URL it called; mirror that so signature checks pass.
    signed_url = f"{settings.public_base_url or args.api}/twilio/voice"
    async with httpx.AsyncClient(base_url=args.api, timeout=10) as http:
        resp = await http.post(
            "/twilio/voice", data=form, headers=signed_headers(settings, signed_url, form)
        )
        resp.raise_for_status()
        stream = ET.fromstring(resp.text).find("Connect/Stream")
        assert stream is not None, resp.text
        params = {p.get("name", ""): p.get("value", "") for p in stream.findall("Parameter")}
        # Connect locally even when the TwiML advertises the public (ngrok) URL.
        ws_url = args.api.replace("http", "ws", 1) + "/twilio/media-stream"

        print(f"call {call_sid}: streaming {len(audio) / 8000:.1f}s of audio to {ws_url}")
        async with connect(ws_url) as ws:
            await ws.send(
                json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"})
            )
            await ws.send(
                json.dumps(
                    {
                        "event": "start",
                        "sequenceNumber": "1",
                        "streamSid": stream_sid,
                        "start": {
                            "streamSid": stream_sid,
                            "accountSid": "ACsim",
                            "callSid": call_sid,
                            "tracks": ["inbound"],
                            "customParameters": params,
                            "mediaFormat": {
                                "encoding": "audio/x-mulaw",
                                "sampleRate": 8000,
                                "channels": 1,
                            },
                        },
                    }
                )
            )
            listener = ReplyListener()
            listening = asyncio.create_task(listener.run(ws, stream_sid))
            replies_expected = (await http.get("/health")).json().get("llm", {}).get("configured")
            t0 = time.monotonic()
            silence = MULAW_SILENCE * FRAME_BYTES
            i = 0
            # Send the caller's speech, then stay on the line in silence until the
            # reply has finished playing (or --wait seconds have passed).
            while True:
                offset = i * FRAME_BYTES
                if offset < len(audio):
                    frame = audio[offset : offset + FRAME_BYTES]
                elif not replies_expected or listener.finished.is_set():
                    break
                elif offset > len(audio) + args.wait * 8000:
                    print(f"no complete reply within {args.wait:.0f}s of the caller's speech")
                    break
                else:
                    frame = silence
                message = json.loads(outbound_media(stream_sid, frame))
                message["media"] |= {"track": "inbound", "chunk": str(i + 1)}
                await ws.send(json.dumps(message))
                i += 1
                # Pace like a real call: frame i is due at i * 20 ms.
                await asyncio.sleep(max(0.0, t0 + i * 0.02 - time.monotonic()))
            listening.cancel()
            await ws.send(
                json.dumps(
                    {"event": "stop", "streamSid": stream_sid, "stop": {"callSid": call_sid}}
                )
            )

        for _ in range(50):
            call = (await http.get(f"/api/calls/{call_sid}")).json()
            if call.get("status") == "ended":
                break
            await asyncio.sleep(0.2)

    print("\ntranscript:")
    for msg in call["transcript"]:
        print(f"  [{msg['role']}] {msg['content']}  (confidence={msg['stt_confidence']})")
    if listener.first_audio_at is not None:
        speech_end = t0 + len(speech) / 8000
        print(
            f"\nreply audio: {len(listener.audio) / 8000:.1f}s, first audio "
            f"{(listener.first_audio_at - speech_end) * 1000:.0f} ms after the caller stopped speaking"
        )
        if args.save_reply:
            listener.save(args.save_reply)
            print(f"  saved to {args.save_reply}")
    print("\nlatency:")
    for name, stats in call["latency"].items():
        print(f"  {name}: n={stats['count']} p50={stats['p50_ms']}ms p95={stats['p95_ms']}ms")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="speak this text with macOS `say`")
    source.add_argument("--wav", help="8 kHz mono 16-bit PCM WAV file")
    source.add_argument("--ulaw", help="raw 8 kHz μ-law file")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument(
        "--wait", type=float, default=20.0, help="max seconds to wait for the assistant's reply"
    )
    parser.add_argument("--save-reply", metavar="WAV", help="write the assistant's speech here")
    asyncio.run(simulate(parser.parse_args()))


if __name__ == "__main__":
    main()
