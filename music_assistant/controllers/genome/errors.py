"""
User-facing exceptions for the Listening Genome controller (§3.3, §3.8).

Both subclass ``InvalidDataError`` so an unconfigured Last.fm import and a Last.fm-side
failure serialize exactly like every other genome API-command error (a plain, translated
message the frontend can show inline) — they exist as their own classes only so callers
that need to tell the two apart (the scheduled poll, which treats "not configured" as
"nothing to do" but a real API failure as worth logging) can do so without parsing message
text.
"""

from __future__ import annotations

from music_assistant_models.errors import InvalidDataError


class LastfmNotConfiguredError(InvalidDataError):
    """Raised when a Last.fm operation is attempted without a username and API key set."""


class LastfmApiError(InvalidDataError):
    """
    Raised when Last.fm (or the network) rejects or fails a request.

    The message is always built from the response's status/body only — never from a raw
    exception repr or the request URL — so it can never contain the Last.fm API key, which is
    sent as a query parameter and would otherwise leak through an HTTP client's own exception
    string.
    """


__all__ = ["LastfmApiError", "LastfmNotConfiguredError"]
