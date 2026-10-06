"""Simulate the website's onboarding page against a locally running API — no browser or mic.

It plays the part of the page end to end:
  1. POST /web/session for a token and the stream URL
  2. Open the WebSocket, send `start`, and wait for Roopiee's opening words
  3. Say each --text line as 20 ms linear16 frames at 16 kHz, in real time
  4. Answer every `tool_call` the way the page would, keeping a tiny form in memory
  5. Print the transcript, the tool calls and the form that was filled

Usage (from repo root, API running on :8000):
  uv run --project apps/api python scripts/simulate_web_onboarding.py \\
      --text "My name is Ramesh Kumar and I am sixty two years old"
  ... --save-reply reply.wav      also write Roopiee's speech to a WAV file

Uses the real Deepgram and Claude keys of the running API. Speech is made with
macOS `say`.
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
from pathlib import Path
from typing import Any

import httpx2 as httpx
from websockets.asyncio.client import ClientConnection, connect

FRAME_BYTES = 640  # 20 ms of linear16 at 16 kHz
SILENCE = b"\x00" * FRAME_BYTES
REPLY_RATE = 24000
ORIGIN = "http://localhost:3000"


def speak(text: str) -> bytes:
    if sys.platform != "darwin":
        sys.exit("--text uses macOS `say`")
    wav_path = str(Path(tempfile.mkdtemp()) / "utterance.wav")
    subprocess.run(
        ["say", "-o", wav_path, "--file-format=WAVE", "--data-format=LEI16@16000", text],
        check=True,
    )
    with wave.open(wav_path, "rb") as wav:
        return wav.readframes(wav.getnframes())


class Page:
    """The few things the real page does that the agent depends on."""

    def __init__(self) -> None:
        self.fields: dict[str, Any] = {}
        self.medications: list[dict[str, Any]] = []
        self.permissions: dict[str, Any] = {}
        self.family: list[dict[str, Any]] = []
        self.calls: list[str] = []

    def handle(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append(name)
        match name:
            case "set_field":
                self.fields[arguments["field"]] = arguments["value"]
            case "add_medication":
                self.medications.append(arguments)
            case "set_permission":
                self.permissions[arguments["permission"]] = arguments["value"]
            case "add_family_member":
                self.family.append(arguments)
            case "get_form_state":
                return json.dumps(self.state())
        return "ok"

    def state(self) -> dict[str, Any]:
        wanted = ["full_name", "age", "gender", "height_cm", "weight_kg", "phone", "conditions"]
        return {
            "language": "en",
            "fields": self.fields,
            "medications": self.medications,
            "empty_fields": [f for f in wanted if f not in self.fields],
        }


async def run(args: argparse.Namespace) -> None:
    onboarding_id = f"sim-{uuid.uuid4().hex[:12]}"
    async with httpx.AsyncClient(base_url=args.api, timeout=15) as http:
        response = await http.post(
            "/web/session",
            json={"onboarding_id": onboarding_id, "language": args.language},
            headers={"Origin": ORIGIN},
        )
        if response.status_code != 200:
            sys.exit(f"POST /web/session -> {response.status_code}: {response.text}")
        session = response.json()

    page = Page()
    reply_audio = bytearray()
    state = {"status": "", "replies": 0}
    started = time.monotonic()

    def stamp() -> str:
        return f"{time.monotonic() - started:6.2f}s"

    async with connect(
        f"{session['ws_url']}?token={session['token']}", additional_headers={"Origin": ORIGIN}
    ) as ws:
        await ws.send(
            json.dumps({"type": "start", "onboarding_id": onboarding_id, "language": args.language})
        )
        receiver = asyncio.create_task(receive(ws, page, reply_audio, state, stamp))
        try:
            await say_nothing_until_listening(ws, state, replies=1, timeout_s=args.timeout)
            for index, text in enumerate(args.text, start=2):
                print(f"{stamp()}  (speaking) {text}")
                await send_speech(ws, speak(text))
                await say_nothing_until_listening(ws, state, replies=index, timeout_s=args.timeout)
            await ws.send(json.dumps({"type": "stop"}))
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)

    print("\nTool calls:", ", ".join(page.calls) or "none")
    print("Form:", json.dumps(page.state(), indent=2))
    print(f"Reply audio: {len(reply_audio) / (REPLY_RATE * 2):.1f} s")
    if args.save_reply:
        with wave.open(args.save_reply, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(REPLY_RATE)
            wav.writeframes(bytes(reply_audio))
        print(f"Saved {args.save_reply}")


async def receive(
    ws: ClientConnection,
    page: Page,
    reply_audio: bytearray,
    state: dict[str, Any],
    stamp: Any,
) -> None:
    async for raw in ws:
        if isinstance(raw, bytes):
            reply_audio.extend(raw)
            continue
        message = json.loads(raw)
        match message["type"]:
            case "transcript" if message["final"]:
                who = "Roopiee" if message["role"] == "assistant" else "Person"
                print(f"{stamp()}  {who}: {message['text']}")
            case "status":
                if message["value"] == "listening" and state["status"] == "speaking":
                    state["replies"] += 1
                state["status"] = message["value"]
            case "tool_call":
                result = page.handle(message["name"], message["arguments"])
                print(f"{stamp()}  tool {message['name']} {json.dumps(message['arguments'])}")
                await ws.send(
                    json.dumps({"type": "tool_result", "id": message["id"], "content": result})
                )
            case "clear_audio":
                print(f"{stamp()}  (reply cut off)")
            case "error":
                sys.exit(f"error from server: {message}")


async def send_speech(ws: ClientConnection, pcm: bytes) -> None:
    for offset in range(0, len(pcm), FRAME_BYTES):
        await ws.send(pcm[offset : offset + FRAME_BYTES].ljust(FRAME_BYTES, b"\x00"))
        await asyncio.sleep(0.02)


async def say_nothing_until_listening(
    ws: ClientConnection, state: dict[str, Any], *, replies: int, timeout_s: float
) -> None:
    """Send silence, as an open mic does, until Roopiee has finished her next reply."""
    deadline = time.monotonic() + timeout_s
    while state["replies"] < replies:
        if time.monotonic() > deadline:
            sys.exit(f"no reply within {timeout_s:.0f} s")
        await ws.send(SILENCE)
        await asyncio.sleep(0.02)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="something the person says; repeat for several turns",
    )
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--save-reply")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
