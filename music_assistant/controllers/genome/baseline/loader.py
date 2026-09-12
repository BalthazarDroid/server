"""
Load and validate the shipped average-listener baseline (§3.7).

The baseline is loaded once at controller setup and never raises: a malformed or missing file
logs a warning and falls back to a uniform distribution over the 59-key genre taxonomy, so
``genome/get`` always has something to divide by. See ``docs/ARCHITECTURE.md`` §3.7.
"""

from __future__ import annotations

import os
from typing import Any, Final

from music_assistant.constants import DEFAULT_GENRE_MAPPING
from music_assistant.controllers.genome.constants import LOGGER
from music_assistant.controllers.genome.models import Baseline
from music_assistant.helpers.json import load_json_dict

BASELINE_V1_PATH: Final[str] = os.path.join(os.path.dirname(__file__), "baseline_v1.json")

_REQUIRED_PERCENTILES: Final[tuple[int, ...]] = (5, 10, 25, 50, 75, 90)
_SHARE_SUM_TOLERANCE: Final[float] = 1e-3
_KNOWN_GENRE_KEYS: Final[frozenset[str]] = frozenset(
    entry["translation_key"] for entry in DEFAULT_GENRE_MAPPING
)


async def load_baseline(path: str = BASELINE_V1_PATH) -> Baseline:
    """
    Load and validate a baseline JSON file, falling back to a uniform baseline on any problem.

    :param path: Absolute path to a baseline JSON file shaped like ``baseline_v1.json`` (§3.7).
    """
    try:
        raw = await load_json_dict(path)
    except (OSError, TypeError, ValueError) as err:
        LOGGER.warning("Could not read genome baseline at %s (%s); using a uniform baseline", path, err)
        return uniform_baseline()
    try:
        return _parse_and_validate(raw)
    except _BaselineValidationError as err:
        LOGGER.warning("Genome baseline at %s failed validation (%s); using a uniform baseline", path, err)
        return uniform_baseline()


def uniform_baseline() -> Baseline:
    """Return a uniform fallback baseline: equal genre shares, a flat era spread, no signal."""
    genre_count = len(_KNOWN_GENRE_KEYS) or 1
    genre_shares = dict.fromkeys(_KNOWN_GENRE_KEYS, 1.0 / genre_count)
    decades = (1960, 1970, 1980, 1990, 2000, 2010, 2020)
    era_shares = dict.fromkeys(decades, 1.0 / len(decades))
    return Baseline(
        version="uniform-fallback",
        genre_shares=genre_shares,
        era_shares=era_shares,
        listener_percentiles={5: 10, 10: 25, 25: 100, 50: 500, 75: 2500, 90: 10000},
        concentration=0.0,
    )


class _BaselineValidationError(ValueError):
    """Raised internally when a loaded baseline document fails validation."""


def _parse_and_validate(raw: dict[str, Any]) -> Baseline:
    """Parse a raw baseline document into a :class:`Baseline`, raising on any violation."""
    version = raw.get("version")
    if not isinstance(version, str) or not version:
        msg = "missing or empty 'version'"
        raise _BaselineValidationError(msg)

    genre_shares = _validate_share_map(raw.get("genre_shares"), field_name="genre_shares")
    unknown = set(genre_shares) - _KNOWN_GENRE_KEYS
    if unknown:
        msg = f"unknown genre keys: {sorted(unknown)}"
        raise _BaselineValidationError(msg)

    era_shares_raw = _validate_share_map(raw.get("era_shares"), field_name="era_shares")
    try:
        era_shares = {int(decade): share for decade, share in era_shares_raw.items()}
    except (TypeError, ValueError) as err:
        msg = "era_shares keys must be decade integers"
        raise _BaselineValidationError(msg) from err

    percentiles_raw = raw.get("listener_percentiles")
    if not isinstance(percentiles_raw, dict):
        msg = "missing 'listener_percentiles'"
        raise _BaselineValidationError(msg)
    try:
        listener_percentiles = {int(k): int(v) for k, v in percentiles_raw.items()}
    except (TypeError, ValueError) as err:
        msg = "listener_percentiles must map integers to integers"
        raise _BaselineValidationError(msg) from err
    missing_percentiles = set(_REQUIRED_PERCENTILES) - set(listener_percentiles)
    if missing_percentiles:
        msg = f"missing required percentiles: {sorted(missing_percentiles)}"
        raise _BaselineValidationError(msg)

    concentration = raw.get("concentration")
    if not isinstance(concentration, (int, float)):
        msg = "missing or non-numeric 'concentration'"
        raise _BaselineValidationError(msg)

    return Baseline(
        version=version,
        genre_shares=genre_shares,
        era_shares=era_shares,
        listener_percentiles=listener_percentiles,
        concentration=float(concentration),
    )


def _validate_share_map(value: Any, *, field_name: str) -> dict[str, float]:
    """Validate a ``{key: float}`` share map sums to 1 within tolerance."""
    if not isinstance(value, dict) or not value:
        msg = f"missing or empty '{field_name}'"
        raise _BaselineValidationError(msg)
    try:
        shares = {str(k): float(v) for k, v in value.items()}
    except (TypeError, ValueError) as err:
        msg = f"'{field_name}' values must be numeric"
        raise _BaselineValidationError(msg) from err
    total = sum(shares.values())
    if abs(total - 1.0) > _SHARE_SUM_TOLERANCE:
        msg = f"'{field_name}' sums to {total}, expected ~1.0"
        raise _BaselineValidationError(msg)
    return shares
