"""
Outbound HTTP for the Listening Genome controller (§1.9, §3.8).

Every external call (Last.fm, ListenBrainz, and the MusicBrainz fallback path) goes through the
:class:`HttpClient` protocol so tests can inject a fixture-backed implementation and never touch
the network. :class:`AiohttpClient` is the real implementation, built on ``mass.http_session`` with
a per-host :class:`~music_assistant.helpers.throttle_retry.Throttler`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from music_assistant.helpers.json import json_loads
from music_assistant.helpers.throttle_retry import Throttler

if TYPE_CHECKING:
    from collections.abc import Mapping

    from music_assistant.mass import MusicAssistant


class HttpClient(Protocol):
    """Minimal async HTTP surface used by every Genome importer/enricher."""

    async def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """
        Issue a ``GET`` request and return the JSON-decoded body.

        :param url: The absolute URL to request.
        :param params: Optional query-string parameters.
        :param headers: Optional extra request headers.
        """
        ...

    async def post_json(
        self,
        url: str,
        *,
        json: Any,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """
        Issue a ``POST`` request with a JSON body and return the JSON-decoded response.

        :param url: The absolute URL to request.
        :param json: The JSON-serializable request body.
        :param headers: Optional extra request headers.
        """
        ...


class AiohttpClient:
    """``HttpClient`` backed by ``mass.http_session`` with a per-host throttler."""

    def __init__(self, mass: MusicAssistant, *, rate_limit: int, period: float) -> None:
        """
        Initialize the client.

        :param mass: The running :class:`MusicAssistant` instance (only ``http_session`` is used).
        :param rate_limit: Maximum number of requests allowed per ``period``.
        :param period: The throttling window, in seconds.
        """
        self.mass = mass
        self._throttler = Throttler(rate_limit=rate_limit, period=period)

    async def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Issue a throttled ``GET`` request and return the JSON-decoded body."""
        async with self._throttler:
            async with self.mass.http_session.get(url, params=params, headers=headers) as resp:
                resp.raise_for_status()
                return await resp.json(loads=json_loads)

    async def post_json(
        self,
        url: str,
        *,
        json: Any,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Issue a throttled ``POST`` request with a JSON body and return the decoded response."""
        async with self._throttler:
            async with self.mass.http_session.post(url, json=json, headers=headers) as resp:
                resp.raise_for_status()
                return await resp.json(loads=json_loads)


__all__ = ["AiohttpClient", "HttpClient"]
