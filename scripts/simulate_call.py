"""Simulate a Twilio phone call against a locally running API — no phone or ngrok needed.

It exercises the real flow end to end:
  1. POST /twilio/voice (signed like Twilio, if TWILIO_AUTH_TOKEN is set)
  2. Read the <Stream> URL + per-call token from the returned TwiML
  3. Open the Media Stream WebSocket and send connected/start/media/stop frames
     in real time (20 ms μ-law frames), just like Twilio
  4. Print the transcript and latency the API recorded for the call

Audio input (pick one):
  --text "I've had a cough for five days"   synthesised with macOS `say`
  --wav caller.wav                          16-bit mono PCM WAV at 8 kHz
  --ulaw caller.ulaw                        raw 8 kHz μ-law

Usage (from repo root, API running on :8000):
  uv run --project apps/api python scripts/simulate_call.py --text "I have a headache"
"""

from __future__ import annotations

import argparse
import asyncio
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
from websockets.asyncio.client import connect

from app.core.config import Settings
from app.voice.audio import MULAW_SILENCE, pcm16_to_mulaw
from app.voice.twilio_protocol import outbound_media

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


def signed_headers(settings: Settings, url: str, params: dict[str, str]) -> dict[str, str]:
    if settings.twilio_auth_token is None:
        return {}
    token = settings.twilio_auth_token.get_secret_value()
    return {"X-Twilio-Signature": RequestValidator(token).compute_signature(url, params)}


async def simulate(args: argparse.Namespace) -> None:
    settings = Settings()
    audio = load_audio(args) + MULAW_SILENCE * int(8000 * TRAILING_SILENCE_S)
    call_sid = f"CAsim{uuid.uuid4().hex[:28]}"
    stream_sid = f"MZsim{uuid.uuid4().hex[:28]}"
    form = {"CallSid": call_sid, "From": "+15550100123", "To": "+15550100999", "AccountSid": "ACsim"}

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
            await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
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
            t0 = time.monotonic()
            for i, offset in enumerate(range(0, len(audio), FRAME_BYTES)):
                frame = audio[offset : offset + FRAME_BYTES]
                message = json.loads(outbound_media(stream_sid, frame))
                message["media"] |= {"track": "inbound", "chunk": str(i + 1)}
                await ws.send(json.dumps(message))
                # Pace like a real call: frame i is due at i * 20 ms.
                await asyncio.sleep(max(0.0, t0 + (i + 1) * 0.02 - time.monotonic()))
            await ws.send(json.dumps({"event": "stop", "streamSid": stream_sid, "stop": {"callSid": call_sid}}))

        for _ in range(50):
            call = (await http.get(f"/api/calls/{call_sid}")).json()
            if call.get("status") == "ended":
                break
            await asyncio.sleep(0.2)

    print("\ntranscript:")
    for msg in call["transcript"]:
        print(f"  [{msg['role']}] {msg['content']}  (confidence={msg['stt_confidence']})")
    print("\nlatency:")
    for name, stats in call["latency"].items():
        print(f"  {name}: n={stats['count']} p50={stats['p50_ms']}ms p95={stats['p95_ms']}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="speak this text with macOS `say`")
    source.add_argument("--wav", help="8 kHz mono 16-bit PCM WAV file")
    source.add_argument("--ulaw", help="raw 8 kHz μ-law file")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    asyncio.run(simulate(parser.parse_args()))


if __name__ == "__main__":
    main()
