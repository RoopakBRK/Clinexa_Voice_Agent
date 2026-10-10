"""Who may start a web session with Clinexsa: only someone signed in to the website.

``POST /web/session`` hands out a token that opens a conversation, and a conversation
spends Deepgram and Anthropic credit. On a public address that must not be open to
anyone who finds the URL. The website sends the signed-in person's Supabase access
token, and this asks Supabase whether it is real.

Only "is this a signed-in user" is checked. Nothing about the person is read or kept.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from app.core.config import Settings


class SignInUnavailableError(Exception):
    """Supabase could not be asked, so nobody can be let in or turned away for certain."""


class SignInCheck(Protocol):
    async def __call__(self, access_token: str) -> bool: ...


class SupabaseSignIn:
    """Asks Supabase Auth whether an access token belongs to a signed-in user."""

    def __init__(
        self, url: str, publishable_key: str, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self._user_url = f"{url.rstrip('/')}/auth/v1/user"
        self._key = publishable_key
        self._client = client or httpx.AsyncClient(timeout=5.0)

    async def __call__(self, access_token: str) -> bool:
        try:
            response = await self._client.get(
                self._user_url,
                headers={"apikey": self._key, "Authorization": f"Bearer {access_token}"},
            )
        except httpx.HTTPError as exc:
            raise SignInUnavailableError from exc
        if response.status_code >= 500:
            raise SignInUnavailableError
        return response.status_code == 200

    async def aclose(self) -> None:
        await self._client.aclose()


def build_sign_in_check(settings: Settings) -> SignInCheck | None:
    """None until both settings are present. A production server refuses sessions then."""
    if settings.supabase_url and settings.supabase_publishable_key:
        return SupabaseSignIn(
            settings.supabase_url, settings.supabase_publishable_key.get_secret_value()
        )
    return None
