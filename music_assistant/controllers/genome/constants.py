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
GENOME_RESULT_SCHEMA_VERSION: Final[int] = 7  # 7: added GenreShare.baseline_known

# --- baseline comparability (§3.6) ----------------------------------------------------
# The reference distribution is sampled from ListenBrainz's sitewide most-listened artists,
# so it is inherently mainstream: whole genres (ambient, funk, latin, klezmer, polka, field
# recording) carry a share of exactly 0.0 in a real 993-artist sample, and a dozen more sit
# far below a tenth of a percent. That is an absence of evidence, not evidence of absence -
# but `share / max(baseline_share, 1e-6)` turns it into a clamped 99.0 and the UI renders
# "ambient 99x average", which is a claim about the household's taste made entirely out of
# missing reference data.
#
# The sample is ~1000 artists, so a single artist's entire weight is ~0.001 of it. A genre
# whose baseline share falls below that is backed by less than one artist's worth of
# observations and cannot support a ratio at all; at or above it there is at least one real
# artist standing behind the figure. Genres are therefore split at exactly that point.
GENOME_BASELINE_MIN_COMPARABLE_SHARE: Final[float] = 0.001

# The `ratio` reported for a genre the baseline cannot speak to. Deliberately 0.0 rather than
# the raw quotient: it sorts to the BOTTOM of any ranking that forgets to check
# `GenreShare.baseline_known` and fails every "over-expressed" threshold, so a consumer that
# ignores the flag under-claims instead of inventing a headline chip. `share` and
# `baseline_share` on the same row keep their true measured values either way.
GENOME_INCOMPARABLE_RATIO: Final[float] = 0.0

# Whether a genre the baseline cannot speak to still carries its Jensen-Shannon
# `contribution` (which is what the frontend ranks to pick the lit rungs on the DNA molecule).
#
# True on purpose. Unlike `ratio`, which divides by a baseline that may be zero and so
# explodes on missing data, a zero-baseline genre's JS term is exactly `0.5 * share` - bounded
# by the household's own listening. Every row is then divided by the same `jsd`, so ranking
# genres by `contribution` is exactly ranking them by those bounded terms: a genre only
# outranks another when the household really does listen to more of it. (The normalized value
# CAN still be large for a small genre when the rest of the household happens to match the
# baseline closely - but that is a true statement about where its divergence comes from, not
# an artifact.) The headline divergence score already sums these same terms, so zeroing them
# here would make the published decomposition disagree with the number it decomposes, and
# would strip the molecule of exactly the genres that characterise a household listening
# outside the mainstream reference.
# Flip to False to exclude them; `_genre_shares` is the single place that reads it.
GENOME_INCOMPARABLE_KEEPS_CONTRIBUTION: Final[bool] = True

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
CONF_ACTION_EXPORT_DB: Final[str] = "export_db"

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
# An artist whose lookup RAISES was previously eligible again immediately, so a name that
# fails every time was retried on every pass forever and sat in "still resolving" for good.
# A cooldown lets it be retried a few times a day - often enough to recover from a transient
# outage, rarely enough that a permanently broken name stops churning.
RESOLVE_ERROR_COOLDOWN_HOURS: Final[int] = 6

# --- Apple Music CSV upload protocol (§3.3) ------------------------------------------

GENOME_REBUILD_TASK_ID: Final[str] = "genome_rebuild"
GENOME_LASTFM_POLL_TASK_ID: Final[str] = "genome_lastfm_poll"

GENOME_UPLOADS_DIRNAME: Final[str] = "genome_uploads"

# Candidate destinations for the TEMPORARY genome/export_db command, most useful first. The
# database lives in the app's private /data, which nothing outside the container can reach, so
# the export has to land on a mounted share - and which ones a given app maps is not knowable
# from in here. The DEV app does not map /share, which is what an assumed default cost us. So
# the command probes this list and says what it found rather than failing on one guess.
GENOME_EXPORT_DIRS: Final[tuple[str, ...]] = (
    "/share",
    "/media",
    "/config",
    "/homeassistant",
    "/addon_configs",
    "/addons",
    "/backup",
)
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

# A page fetch that fails with a transient status (429/500/502/503/504) or a network-level
# error (no status at all - a timeout, a dropped connection) is retried this many times with
# exponential backoff before the page is treated as failed (P2, 2026-09-13 real-hardware run:
# a single 500 on page 62/275 killed the whole import). 401/403/404 are permanent and never
# retried - see ``importers/lastfm.py::_PERMANENT_STATUS_CODES``.
LASTFM_RETRY_MAX_ATTEMPTS: Final[int] = 4
LASTFM_RETRY_BASE_DELAY_SECONDS: Final[float] = 0.5
LASTFM_RETRY_MAX_DELAY_SECONDS: Final[float] = 8.0

# --- ListenBrainz popularity (§3.8) ---------------------------------------------------

LISTENBRAINZ_POPULARITY_URL: Final[str] = "https://api.listenbrainz.org/1/popularity/artist"
LISTENBRAINZ_POPULARITY_BATCH_SIZE: Final[int] = 50

# --- MusicBrainz enrichment pacing (§3.8, P3) ------------------------------------------
# MusicBrainz's own documented courtesy limit is ~1 req/sec; MA's shared, throttled client
# (rate_limit=10, period=10) allows bursts of 10 in under a second, which is enough on its own
# to trip the hosted mirror's own rate limiter (observed: a 63s `Retry-After` after a 200-artist
# pass). Genome paces its *own* calls on top of that shared throttler.
#
# The interval is per *artist*, and resolving one artist costs two requests (a search, then a
# lookup) - so 1.1s/artist is ~1.8 req/sec, still over the limit, and a live pass spent its
# whole time collecting 60s penalties rather than resolving anyone. It also shares the client
# with MA's own metadata lookups. 2.5s/artist leaves headroom for both.
GENOME_MB_ENRICHMENT_MIN_INTERVAL_SECONDS: Final[float] = 2.5
GENOME_ENRICHMENT_TASK_ID: Final[str] = "genome_enrichment"
# Per-run ceiling for the continuous background enrichment pass. The pacing above - not this
# number - bounds real-world duration (~2.5s/artist), so this is really "how long may one pass
# run": 500 artists is a little over 20 minutes, comfortably inside the hourly cadence, and a
# backlog simply drains over successive runs.
GENOME_ENRICHMENT_BATCH_LIMIT: Final[int] = 500

# --- discovery: cold corners + Last.fm similar artists (D-16) ---------------------------
# The read path (`genome/discovery`) serves cold corners computed from local data and
# suggestions served straight from the store. Nothing here is ever fetched inline.

GENOME_DISCOVERY_TASK_ID: Final[str] = "genome_discovery"

# Bumped whenever the stored discovery blob's shape changes. Deliberately separate from
# GENOME_RESULT_SCHEMA_VERSION: discovery lives in its own cache row, so the (far more
# expensive) genome result is not discarded because a discovery field moved.
GENOME_DISCOVERY_CACHE_VERSION: Final[int] = 1

# An artist in the library counts as a "cold corner" when the household has played them at
# most this many times. Zero would surface only never-touched imports and would miss the more
# interesting case - the artist tried once or twice, years ago, and never returned to. Three
# or more starts to read as a small habit rather than a corner, so two is the ceiling.
GENOME_DISCOVERY_COLD_MAX_PLAYS: Final[int] = 2
GENOME_DISCOVERY_COLD_LIMIT: Final[int] = 25

# How many of the household's own artists seed one Last.fm pass, and how many similar artists
# are kept per seed. The product bounds one pass's request count (seeds) and result size.
GENOME_DISCOVERY_SEED_LIMIT: Final[int] = 8
GENOME_DISCOVERY_SIMILAR_PER_SEED: Final[int] = 20
GENOME_DISCOVERY_SUGGESTED_LIMIT: Final[int] = 30

# Cadence of the background discovery pass. Similar-artist graphs move on the scale of weeks,
# and the seeds only change when the household's own top artists do, so a daily pass is
# already generous - and it keeps the Last.fm request budget to a few dozen calls a day.
GENOME_DISCOVERY_REFRESH_INTERVAL_HOURS: Final[int] = 24

# Minimum wall-clock spacing between per-seed `artist.getSimilar` calls, mirroring
# GENOME_MB_ENRICHMENT_MIN_INTERVAL_SECONDS. One request per seed per second sits an order of
# magnitude under Last.fm's own ~5 req/sec guidance and leaves the shared throttler headroom
# for a concurrent scrobble import.
GENOME_DISCOVERY_LASTFM_MIN_INTERVAL_SECONDS: Final[float] = 1.0

# A seed whose `artist.getSimilar` call RAISES is recorded with the time it failed and skipped
# until this cooldown expires - the same shape as RESOLVE_ERROR_COOLDOWN_HOURS, and for the
# same reason: a seed that can never succeed (a name Last.fm 404s on) must not be retried on
# every pass forever.
GENOME_DISCOVERY_SEED_ERROR_COOLDOWN_HOURS: Final[int] = 6

LASTFM_SIMILAR_METHOD: Final[str] = "artist.getSimilar"

# `suggested_state` values on the `genome/discovery` result (see models.DiscoveryResult).
DISCOVERY_STATE_READY: Final[str] = "ready"
DISCOVERY_STATE_PENDING: Final[str] = "pending"
DISCOVERY_STATE_UNAVAILABLE: Final[str] = "unavailable"
