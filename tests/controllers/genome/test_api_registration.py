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

import inspect
from typing import Any, get_type_hints

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
        "unresolved_artists",
        "retry_artists",
        "dismiss_unresolved",
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


# ---------------------------------------------------------------------------------------
# Argument parsing — resolving the annotations is only half of registration. MA then parses
# every incoming argument against them with helpers/api.py::parse_arguments, which bottoms
# out in `isinstance(value, value_type)`. A TypedDict cannot be used with isinstance at all
# ("TypedDict does not support instance and class checks"), so a TypedDict parameter type
# passes the hint-resolution test above and then fails on every real call.
# ---------------------------------------------------------------------------------------

# one representative payload per command, shaped the way the frontend sends it
_SAMPLE_ARGS: dict[str, dict[str, Any]] = {
    "get_genome": {"listener": "household", "refresh": False},
    "unresolved_artists": {"limit": 50},
    "retry_artists": {"artist_keys": ["artist_a"]},
    "dismiss_unresolved": {},
    "rebuild": {"listener": "household", "enrich": True},
    "import_apple": {
        "upload_id": "u1",
        "seq": 0,
        "chunk_b64": "",
        "final": True,
        "filename": "a.csv",
    },
    "import_lastfm": {"username": "Bob_Baird", "max_pages": 1},
    "get_settings": {},
    "set_settings": {"settings": {"lastfm_username": "Bob_Baird", "lastfm_api_key": "a" * 32}},
}


@pytest.mark.parametrize(
    ("name", "func"), _api_command_methods(), ids=[n for n, _ in _api_command_methods()]
)
def test_api_command_arguments_parse(name: str, func: Any) -> None:
    """
    MA must be able to parse each command's arguments against its own annotations.

    This is the call MA makes on every websocket request. Regression test for a real incident:
    `set_settings` took a TypedDict, which resolved fine at registration and then raised
    "TypedDict does not support instance and class checks" on every invocation.
    """
    from music_assistant.helpers.api import parse_arguments  # noqa: PLC0415

    assert name in _SAMPLE_ARGS, (
        f"No sample arguments for GenomeController.{name}. Add one to _SAMPLE_ARGS so this "
        "command's parameter types are covered."
    )
    signature = inspect.signature(func)
    # MA drops `self` before parsing; mirror that
    params = [p for p in signature.parameters.values() if p.name != "self"]
    signature = signature.replace(parameters=params)
    type_hints = {k: v for k, v in get_type_hints(func).items() if k != "return"}
    try:
        parse_arguments(signature, type_hints, _SAMPLE_ARGS[name], strict=True)
    except TypeError as err:
        if "does not support instance and class checks" in str(err):
            pytest.fail(
                f"GenomeController.{name}: {err}. A parameter is annotated with a TypedDict. "
                "MA parses arguments with isinstance(), which TypedDict forbids — use a "
                "mashumaro dataclass (DataClassDictMixin), which MA handles via from_dict."
            )
        raise
