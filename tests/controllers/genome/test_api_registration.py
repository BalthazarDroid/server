"""
Guard the API-registration path that Music Assistant runs at startup.

``MusicAssistant._register_api_commands`` resolves every ``@api_command`` handler's
annotations with ``get_type_hints()``. Because this package uses
``from __future__ import annotations``, those annotations are strings at runtime, so any
name imported only under ``TYPE_CHECKING`` raises ``NameError`` there — and MA's fallback
resolver only searches ``music_assistant_models``, so it cannot rescue this package's own
types.

That failure happens at server start, long after every unit test has passed, and it takes
the whole server down. These tests reproduce the registration step directly.
"""

from __future__ import annotations

from typing import get_type_hints

import pytest

from music_assistant.controllers.genome.controller import GenomeController


def _api_command_methods() -> list[tuple[str, object]]:
    """Return every ``(name, function)`` on GenomeController decorated with @api_command."""
    found = []
    for name in dir(GenomeController):
        func = getattr(GenomeController, name, None)
        if callable(func) and getattr(func, "api_cmd", None) is not None:
            found.append((name, func))
    return found


def test_controller_exposes_api_commands() -> None:
    """The discovery helper must actually find the commands, or the guard below is vacuous."""
    names = {name for name, _ in _api_command_methods()}
    assert {
        "get_genome",
        "rebuild",
        "import_apple",
        "import_lastfm",
        "get_settings",
        "set_settings",
    } <= names


@pytest.mark.parametrize(
    ("name", "func"), _api_command_methods(), ids=[n for n, _ in _api_command_methods()]
)
def test_api_command_annotations_resolve_at_runtime(name: str, func: object) -> None:
    """
    Every annotation on every api_command must resolve with only runtime imports.

    This is the exact call MA makes while registering the command. A NameError here is the
    startup crash, caught in CI instead of on the user's server.
    """
    try:
        get_type_hints(func)
    except NameError as err:  # pragma: no cover - the assert message is the point
        pytest.fail(
            f"GenomeController.{name}: {err}. "
            "A name used in an @api_command signature is imported only under TYPE_CHECKING. "
            "Move it into the module-level import in controller.py."
        )
