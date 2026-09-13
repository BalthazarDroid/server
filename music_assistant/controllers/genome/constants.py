"""
Constants for the Listening Genome controller.

See ``docs/ARCHITECTURE.md`` sections 3.0, 3.1, 3.2, 3.3, 3.5 and 3.7 for the source
specification these values are drawn from.
"""

from __future__ import annotations

import logging
from typing import Final

from music_assistant.constants import MASS_LOGGER_NAME

LOGGER = logging.getLogger(f"{MASS_LOGGER_NAME}.genome")

# --- database (§3.1) ---------------------------------------------------------------

DB_SCHEMA_VERSION: Final[int] = 1
DB_TABLE_GENOME_LISTENS: Final[str] = "genome_listens"
DB_TABLE_GENOME_ARTIST_META: Final[str] = "genome_artist_meta"
DB_TABLE_GENOME_CACHE: Final[str] = "genome_cache"
DB_TABLE_SETTINGS: Final[str] = "settings"

LISTENER_HOUSEHOLD: Final[str] = "household"

# --- engine / baseline versioning (§3.6, §3.7) --------------------------------------
# `engine.py` (WP-B) is the canonical home for `ENGINE_VERSION` per §3.6, but it does not
# exist yet at this step; it is defined here first and re-exported from `engine.py` once
# that module is written, so nothing downstream needs to change.
ENGINE_VERSION: Final[str] = "1.0.0"
BASELINE_VERSION: Final[str] = "v1-2026-09"
GENOME_RESULT_SCHEMA_VERSION: Final[int] = 1

# --- recency weighting defaults (§3.5) ----------------------------------------------

DEFAULT_HALF_LIFE_DAYS: Final[int] = 548  # ~18 months; 0 disables decay
DEFAULT_TOP_N: Final[int] = 20
DEFAULT_NEW_ARTIST_WINDOW_DAYS: Final[int] = 90

# --- config entry keys (§3.2) --------------------------------------------------------

CONF_RECENCY_HALF_LIFE_DAYS: Final[str] = "recency_half_life_days"
CONF_LASTFM_USERNAME: Final[str] = "lastfm_username"
CONF_LASTFM_API_KEY: Final[str] = "lastfm_api_key"
CONF_LASTFM_POLL_ENABLED: Final[str] = "lastfm_poll_enabled"
CONF_LASTFM_POLL_INTERVAL_HOURS: Final[str] = "lastfm_poll_interval_hours"
CONF_APPLE_IMPORT_DIR: Final[str] = "apple_import_dir"
CONF_ENRICH_ENABLED: Final[str] = "enrich_enabled"
CONF_OBSCURITY_PERCENTILE: Final[str] = "obscurity_percentile"
CONF_MIN_SECONDS_PLAYED: Final[str] = "min_seconds_played"
CONF_REBUILD_SCHEDULE_HOUR: Final[str] = "rebuild_schedule_hour"

# config entry defaults (§3.2)

DEFAULT_LASTFM_USERNAME: Final[str] = ""
DEFAULT_LASTFM_API_KEY: Final[str] = ""
DEFAULT_LASTFM_POLL_ENABLED: Final[bool] = False
DEFAULT_LASTFM_POLL_INTERVAL_HOURS: Final[int] = 6
DEFAULT_APPLE_IMPORT_DIR: Final[str] = ""
DEFAULT_ENRICH_ENABLED: Final[bool] = True
DEFAULT_OBSCURITY_PERCENTILE: Final[int] = 25
DEFAULT_MIN_SECONDS_PLAYED: Final[int] = 30
DEFAULT_REBUILD_SCHEDULE_HOUR: Final[int] = 4

# config actions (§3.2)

CONF_ACTION_REBUILD_NOW: Final[str] = "rebuild_now"
CONF_ACTION_CLEAR_GENOME_DATA: Final[str] = "clear_genome_data"

# --- listen sources (§3.1, §3.8) -----------------------------------------------------

SOURCE_MA_PLAYLOG: Final[str] = "ma_playlog"
SOURCE_MA_BACKFILL: Final[str] = "ma_backfill"
SOURCE_APPLE_EXPORT: Final[str] = "apple_export"
SOURCE_LASTFM: Final[str] = "lastfm"
SOURCE_LISTENBRAINZ: Final[str] = "listenbrainz"

# artist-meta resolve states (§3.8)

RESOLVE_STATE_PENDING: Final[str] = "pending"
RESOLVE_STATE_OK: Final[str] = "ok"
RESOLVE_STATE_NOT_FOUND: Final[str] = "not_found"
RESOLVE_STATE_ERROR: Final[str] = "error"

# re-resolve cooldowns (§3.8)

RESOLVE_OK_COOLDOWN_DAYS: Final[int] = 180
RESOLVE_NOT_FOUND_COOLDOWN_DAYS: Final[int] = 30

# --- Apple Music CSV upload protocol (§3.3) ------------------------------------------

GENOME_REBUILD_TASK_ID: Final[str] = "genome_rebuild"
GENOME_LASTFM_POLL_TASK_ID: Final[str] = "genome_lastfm_poll"

GENOME_UPLOADS_DIRNAME: Final[str] = "genome_uploads"
GENOME_UPLOAD_CHUNK_MAX_B64_BYTES: Final[int] = 512 * 1024
GENOME_UPLOAD_MAX_TOTAL_BYTES: Final[int] = 512 * 1024 * 1024
GENOME_UPLOAD_TTL_SECONDS: Final[int] = 15 * 60

# --- Last.fm polling (§3.8) -----------------------------------------------------------

LASTFM_BASE_URL: Final[str] = "https://ws.audioscrobbler.com/2.0/"
# A Last.fm API key is a 32-character hex string. Validating the shape locally turns a
# mis-pasted value (a URL, a stray password-manager entry, a truncated copy) into an immediate,
# specific error instead of an opaque 403 from Last.fm — and keeps a bad value from ever being
# sent over the wire as a query parameter.
LASTFM_API_KEY_PATTERN: Final[str] = r"^[0-9a-fA-F]{32}$"
LASTFM_PAGE_LIMIT: Final[int] = 200
LASTFM_INTER_PAGE_DELAY_SECONDS: Final[float] = 0.25

# --- ListenBrainz popularity (§3.8) ---------------------------------------------------

LISTENBRAINZ_POPULARITY_URL: Final[str] = "https://api.listenbrainz.org/1/popularity/artist"
LISTENBRAINZ_POPULARITY_BATCH_SIZE: Final[int] = 50
