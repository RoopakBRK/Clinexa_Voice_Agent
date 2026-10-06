"""Set up and check the Exotel connection.

    make exotel-status                       what is filled in, and what to paste into Exotel
    make exotel-check                        sign in to Exotel and list your ExoPhones (sends nothing)
    make exotel-whatsapp TO=98XXXXXXXX       send the dose-reminder template to your own number

Secrets are never printed, except the Voicebot URL, which has to carry the stream
password because that is how Exotel signs in to this server.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from urllib.parse import quote

from app.core.config import Settings, get_settings
from app.notify.exotel import ExotelError, build_exotel_client, to_e164_india
from app.notify.worker import callback_url

REQUIRED = [
    ("EXOTEL_API_KEY", "exotel_api_key", "Exotel dashboard, API settings"),
    ("EXOTEL_API_TOKEN", "exotel_api_token", "Exotel dashboard, API settings"),
    ("EXOTEL_ACCOUNT_SID", "exotel_account_sid", "Exotel dashboard, API settings"),
    ("EXOTEL_CALLER_ID", "exotel_caller_id", "your ExoPhone, for reminder calls"),
    ("EXOTEL_VOICE_APP_ID", "exotel_voice_app_id", "the flow with the Voicebot applet"),
    ("EXOTEL_WHATSAPP_FROM", "exotel_whatsapp_from", "your WhatsApp Business number, +91..."),
    ("EXOTEL_STREAM_USERNAME", "exotel_stream_username", "made up, already generated"),
    ("EXOTEL_STREAM_PASSWORD", "exotel_stream_password", "made up, already generated"),
    ("EXOTEL_CALLBACK_KEY", "exotel_callback_key", "made up, already generated"),
    ("SUPABASE_URL", "supabase_url", "Supabase, Project Settings, API"),
    ("SUPABASE_SERVICE_ROLE_KEY", "supabase_service_role_key", "Supabase, Project Settings, API"),
    ("PUBLIC_BASE_URL", "public_base_url", "the https address of this server"),
]


def status(settings: Settings) -> int:
    print("Settings in VoiceAgent/.env\n")
    missing = 0
    for env, attr, where in REQUIRED:
        filled = bool(getattr(settings, attr))
        missing += not filled
        print(f"  [{'x' if filled else ' '}] {env:<28} {'' if filled else '<- ' + where}")
    print(f"\n  ALERTS_ENABLED = {str(settings.alerts_enabled).lower()}")

    base = settings.public_base_url
    if base and settings.exotel_stream_username and settings.exotel_stream_password:
        host = base.removeprefix("https://").removeprefix("http://")
        user = quote(settings.exotel_stream_username, safe="")
        password = quote(settings.exotel_stream_password.get_secret_value(), safe="")
        print("\nPaste into Exotel\n")
        print("  Voicebot applet URL (in your flow):")
        print(f"    wss://{user}:{password}@{host}/exotel/stream")
        print("  WhatsApp default status callback (optional, it is also sent with each message):")
        print(f"    {callback_url(settings, '/exotel/whatsapp-status')}")
    else:
        print("\nSet PUBLIC_BASE_URL to see the URLs to paste into Exotel.")

    print("\nWhatsApp templates to get approved (category: Utility)\n")
    print(f"  {settings.exotel_dose_template}")
    print("    Hello {{1}}, it is time for your medicine: {{2}}, at {{3}}.")
    print(f"  {settings.exotel_low_stock_template}")
    print("    Hello {{1}}, your {{2}} will run out in about {{3}} days.")
    print(f"\n{missing} setting(s) still empty." if missing else "\nEverything is filled in.")
    return 0


async def check(settings: Settings) -> int:
    client = build_exotel_client(settings)
    if client is None:
        print("Fill in EXOTEL_API_KEY, EXOTEL_API_TOKEN and EXOTEL_ACCOUNT_SID first.")
        return 1
    try:
        numbers = await client.list_exophones()
    except ExotelError as exc:
        print(f"Exotel did not accept the request: {exc}")
        return 1
    finally:
        await client.aclose()
    print("Signed in to Exotel. ExoPhones on this account:")
    for number in numbers or ["(none listed)"]:
        marker = "  <- EXOTEL_CALLER_ID" if number == settings.exotel_caller_id else ""
        print(f"  {number}{marker}")
    return 0


async def whatsapp(settings: Settings, to: str) -> int:
    number = to_e164_india(to)
    client = build_exotel_client(settings)
    if number is None:
        print("That is not a 10-digit Indian mobile number.")
        return 1
    if client is None or not client.can_message:
        print("Fill in the Exotel keys and EXOTEL_WHATSAPP_FROM first.")
        return 1
    try:
        sid = await client.send_whatsapp_template(
            number,
            template=settings.exotel_dose_template,
            body_params=["there", "Test medicine 500 mg", "8:00 AM"],
            custom_data="connection-test",
        )
    except ExotelError as exc:
        print(f"Not sent: {exc}")
        return 1
    finally:
        await client.aclose()
    print(f"Sent. Exotel message id: {sid}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("check")
    commands.add_parser("whatsapp").add_argument("to")
    args = parser.parse_args()
    settings = get_settings()
    if args.command == "status":
        return status(settings)
    if args.command == "check":
        return asyncio.run(check(settings))
    return asyncio.run(whatsapp(settings, args.to))


if __name__ == "__main__":
    sys.exit(main())
