r"""
Build ``music_assistant/controllers/genome/baseline/baseline_v1.json`` (§3.7).

Standalone and runnable offline: this script imports nothing from ``music_assistant`` so it
never needs the server's dependencies (or the server to be importable at all) — only stdlib
plus ``aiohttp`` for the live-fetch path. It reads the canonical 59-key genre taxonomy directly
off disk (the same file ``music_assistant/constants.py`` loads at runtime) rather than importing
the package, to keep that "no server import" guarantee.

Three modes, chosen by whichever fixture/dump flag is given (default: live fetch):

- (no flag): sample ListenBrainz's sitewide top-artists + popularity endpoints, resolve genres
  via MusicBrainz artist lookups. **Not reachable from this development workspace** (see
  ``BRIEF.md`` "Network constraints") — exercised on Bob's machine.
- ``--fixture PATH``: read a local JSON file already shaped like the combined
  ListenBrainz-popularity + MusicBrainz-tags records this script needs (see
  ``tests/fixtures/genome/listenbrainz_baseline_sample.json`` for the shape), and run the exact
  same aggregation as the live path. This is how the shipped ``baseline_v1.json`` was produced
  in this workspace — it is therefore marked ``"provisional": true`` in its output until Bob
  regenerates it for real.
- ``--from-dump PATH``: same as ``--fixture``, kept as a separate flag per §3.7 for a raw
  ListenBrainz data-dump export shaped the same way; the two are equivalent in this script.

Usage. Plain ``python3`` on purpose, not ``uv run``: the point of the no-server-import rule
above is that this script runs anywhere the repo is checked out, including a shell that has
none of the project tooling. ``uv`` lives in the dev container; requiring it here would throw
that away for no benefit. Run from the repo root::

    python3 scripts/build_genome_baseline.py --fixture tests/fixtures/genome/listenbrainz_baseline_sample.json \\
        --out music_assistant/controllers/genome/baseline/baseline_v1.json
    python3 scripts/build_genome_baseline.py --out music_assistant/controllers/genome/baseline/baseline_v1.json --sample 1500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ruff: noqa: T201

REPO_ROOT = Path(__file__).resolve().parents[1]
GENRE_MAPPING_PATH = (
    REPO_ROOT / "music_assistant" / "helpers" / "resources" / "genres" / "genre_mapping.json"
)

LISTENBRAINZ_BASE_URL = "https://api.listenbrainz.org"
# The sitewide endpoint's per-request ceiling. Asking for more returns this many without
# complaint, which is why a --sample of 1500 quietly produced a 993-artist baseline.
LISTENBRAINZ_PAGE_LIMIT = 1000
# Every range ListenBrainz will compute a sitewide chart for, most durable first. Each is
# capped at its own top 1000, so they are queried in turn to widen the candidate pool; the
# short ranges are asked last because their tails are the most transient.
LISTENBRAINZ_RANGES = (
    "all_time",
    "year",
    "half_yearly",
    "quarter",
    "month",
    "week",
    "this_year",
    "this_month",
    "this_week",
)
# MusicBrainz proper, not the Music Assistant mirror the SERVER uses.
#
# The mirror exists so that running MA instances can resolve metadata cheaply, and it answers
# 403 to anything not identifying itself as Music Assistant. Setting that User-Agent from a
# script would get past it, and that is exactly why it should not be done: a one-off baseline
# build is ~1500 lookups of bulk traffic the project is hosting for a different purpose, and
# it is not an MA instance asking. The canonical API is the right place to ask, it welcomes
# this with an identifying User-Agent and one request per second, and the pacing below already
# honours that. Override with --mb-base-url if you have your own mirror.
MUSICBRAINZ_BASE_URL = "https://musicbrainz.org/ws/2"
REQUIRED_PERCENTILES = (5, 10, 25, 50, 75, 90)
# MusicBrainz's documented courtesy limit is ~1 req/sec, and this project has already been
# penalised once for treating that as a target rather than a ceiling. 1.3s with jitter keeps
# every run comfortably under it even when several requests land on the same second.
LIVE_REQUEST_DELAY_SECONDS = 1.3
LIVE_REQUEST_JITTER_SECONDS = 0.4
# Retry a transient failure rather than losing the whole run to one bad response. A full run
# is ~1500 requests over ~35 minutes; at that length a single 503 is close to expected, and
# without a retry it costs the entire run.
LIVE_MAX_ATTEMPTS = 4
LIVE_RETRY_BACKOFF_SECONDS = 2.0
# 429 and 5xx are worth another attempt. A 404 or a 400 is not - the request itself is wrong,
# and repeating it just spends someone else's rate limit to get the same answer.
LIVE_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
# How much of the sample may fail to resolve before the run is treated as blocked rather than
# unlucky. MusicBrainz answers 503 when it is rate-limiting, so a burst of them means "slow
# down", but a steady stream of them means something systemic and retrying is just rude.
MAX_LIVE_FAILURE_RATE = 0.2
MIN_LIVE_FAILURES = 20
# MusicBrainz requires a User-Agent identifying the application and giving a contact address,
# and refuses anonymous clients. aiohttp sends only its own by default, which is exactly the
# anonymous client that policy is aimed at.
USER_AGENT = "MusicAssistant-ListeningGenome/1.0 ( https://github.com/music-assistant/server )"


async def _pace() -> None:
    """Wait out one request's share of the courtesy limit, with jitter to avoid lockstep."""
    await asyncio.sleep(LIVE_REQUEST_DELAY_SECONDS + random.uniform(0, LIVE_REQUEST_JITTER_SECONDS))


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and write the baseline JSON file; return a process exit code."""
    args = _parse_args(argv)
    genre_mapping = _load_genre_mapping()
    fixture_path = args.fixture or args.from_dump
    if fixture_path:
        records = _load_fixture(Path(fixture_path))
        provisional = True
        source = "listenbrainz-popularity-fixture"
    else:
        try:
            records = asyncio.run(_fetch_live(args.sample, args.mb_base_url))
        except OSError as err:
            print(f"Live ListenBrainz fetch failed ({err}); see BRIEF.md network constraints.")
            return 1
        provisional = False
        source = "listenbrainz-popularity-sample"

    aggregate = _aggregate(records, genre_mapping)
    document = {
        "version": args.version,
        "built_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": source,
        "provisional": provisional,
        "sample_size": aggregate["sample_size"],
        "genre_shares": aggregate["genre_shares"],
        "era_shares": aggregate["era_shares"],
        "listener_percentiles": aggregate["listener_percentiles"],
        "concentration": aggregate["concentration"],
    }
    if provisional:
        document["note"] = (
            "provisional, fixture-derived — regenerate with "
            "`python3 scripts/build_genome_baseline.py --sample 50000` once ListenBrainz is "
            "reachable, then drop the 'provisional'/'note' fields."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({aggregate['sample_size']} artist samples, source={source})")
    return 0


def _aggregate(
    records: list[dict[str, Any]], genre_mapping: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate raw artist samples into genre/era shares, percentiles and concentration."""
    alias_map = _build_alias_map(genre_mapping)
    genre_weight: dict[str, float] = dict.fromkeys(
        (entry["translation_key"] for entry in genre_mapping), 0.0
    )
    era_weight: dict[int, float] = {}
    listen_counts: list[float] = []
    user_counts: list[int] = []

    for record in records:
        listen_count = float(record.get("total_listen_count") or 0)
        user_count = record.get("total_user_count")
        if isinstance(user_count, int):
            user_counts.append(user_count)
        if listen_count > 0:
            listen_counts.append(listen_count)

        matched = _match_genres(record.get("tags") or [], alias_map)
        if matched and listen_count > 0:
            share = listen_count / len(matched)
            for key in matched:
                genre_weight[key] += share

        year = record.get("first_release_year")
        if year and listen_count > 0:
            decade = (int(year) // 10) * 10
            era_weight[decade] = era_weight.get(decade, 0.0) + listen_count

    return {
        "sample_size": len(records),
        "genre_shares": _normalized(genre_weight),
        "era_shares": _normalized(era_weight),
        "listener_percentiles": _percentiles(user_counts, REQUIRED_PERCENTILES),
        "concentration": _concentration(listen_counts),
    }


def _match_genres(tags: list[dict[str, Any]], alias_map: dict[str, str]) -> list[str]:
    """
    Map an artist's tags to up to 3 distinct genre_mapping.json translation_keys (§3.8).

    Tags are tried in descending popularity ('count') order; the first 3 distinct matches win.
    """
    ordered = sorted(tags, key=lambda tag: tag.get("count", 0), reverse=True)
    matched: list[str] = []
    for tag in ordered:
        key = alias_map.get(_normalize_tag(str(tag.get("name", ""))))
        if key and key not in matched:
            matched.append(key)
        if len(matched) >= 3:
            break
    return matched


def _build_alias_map(genre_mapping: list[dict[str, Any]]) -> dict[str, str]:
    """Build a ``normalized alias -> translation_key`` lookup from genre_mapping.json."""
    alias_map: dict[str, str] = {}
    for entry in genre_mapping:
        key = entry["translation_key"]
        candidates = [entry["genre"], key, *entry.get("aliases", [])]
        for candidate in candidates:
            alias_map[_normalize_tag(candidate)] = key
    return alias_map


def _normalize_tag(value: str) -> str:
    """Case/punctuation-fold a tag or alias for matching (§3.8: '-/_/&/and normalized')."""
    value = value.lower().strip()
    value = value.replace("&", " and ")
    value = re.sub(r"[-_/]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _normalized(weights: dict[Any, float]) -> dict[str, float]:
    """Return ``weights`` scaled to sum to 1, rounded to 4dp; unchanged (zeroed) if empty."""
    total = sum(weights.values())
    if total <= 0:
        return {str(key): 0.0 for key in weights}
    return {str(key): round(value / total, 4) for key, value in weights.items()}


def _percentiles(values: list[int], percentiles: tuple[int, ...]) -> dict[str, int]:
    """Return ``{percentile: value}`` via linear interpolation over the sorted sample."""
    if not values:
        return {str(p): 0 for p in percentiles}
    ordered = sorted(values)
    result: dict[str, int] = {}
    for pct in percentiles:
        rank = (pct / 100) * (len(ordered) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(ordered) - 1)
        fraction = rank - lo
        interpolated = ordered[lo] + (ordered[hi] - ordered[lo]) * fraction
        result[str(pct)] = round(interpolated)
    return result


def _concentration(listen_counts: list[float]) -> float:
    """Compute the normalized Herfindahl index over sampled artists' listen-count shares (0..1)."""
    total = sum(listen_counts)
    if total <= 0 or not listen_counts:
        return 0.0
    shares = [count / total for count in listen_counts]
    n = len(shares)
    if n == 1:
        return 1.0
    herfindahl = sum(share * share for share in shares)
    return round((herfindahl - 1 / n) / (1 - 1 / n), 4)


def _load_genre_mapping() -> list[dict[str, Any]]:
    """Load the 59-key genre taxonomy directly off disk (no ``music_assistant`` import)."""
    with GENRE_MAPPING_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_fixture(path: Path) -> list[dict[str, Any]]:
    """Load a local JSON fixture shaped like the combined ListenBrainz+MusicBrainz records."""
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    records = data["artists"] if isinstance(data, dict) and "artists" in data else data
    if not isinstance(records, list):
        msg = f"Expected a list of artist records (or {{'artists': [...]}}) in {path}"
        raise TypeError(msg)
    return records


async def _fetch_live(sample: int, mb_base_url: str = MUSICBRAINZ_BASE_URL) -> list[dict[str, Any]]:
    """
    Sample ListenBrainz's sitewide popularity plus MusicBrainz artist tags (§3.7).

    Not reachable from this development workspace (BRIEF.md); written to run for real on
    Bob's machine, where both hosts are reachable.
    """
    import aiohttp  # noqa: PLC0415 - optional dependency, only needed for the live-fetch path

    records: list[dict[str, Any]] = []
    async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
        artists = await _fetch_top_artists(session, sample)
        mbids = [a["artist_mbid"] for a in artists if a.get("artist_mbid")]
        popularity_by_mbid: dict[str, dict[str, Any]] = {}
        for batch_start in range(0, len(mbids), 50):
            batch = mbids[batch_start : batch_start + 50]
            popularity = await _post_json(
                session,
                f"{LISTENBRAINZ_BASE_URL}/1/popularity/artist",
                json={"artist_mbids": batch},
            )
            for item in _unwrap(popularity):
                if item.get("artist_mbid"):
                    popularity_by_mbid[item["artist_mbid"]] = item
            await _pace()

        wanted = [a for a in artists if a.get("artist_mbid")]
        skipped = 0
        for index, artist in enumerate(wanted, start=1):
            mbid = artist["artist_mbid"]
            popularity = popularity_by_mbid.get(mbid, {})
            try:
                lookup = await _get_json(
                    session,
                    f"{mb_base_url}/artist/{mbid}",
                    params={"inc": "tags+genres", "fmt": "json"},
                )
            except OSError as err:
                # One artist that will not resolve must not cost the other 1499. A baseline
                # built from most of the sample is still a usable baseline; a run that dies
                # 30 minutes in yields nothing at all, which is what used to happen.
                skipped += 1
                print(f"  skipped {mbid}: {err}")
                if skipped > max(MIN_LIVE_FAILURES, len(wanted) * MAX_LIVE_FAILURE_RATE):
                    msg = (
                        f"Gave up after {skipped} failed artist lookups out of {index} tried. "
                        "That is a rate limit or a block, not bad luck - wait a while, or pass "
                        "--mb-base-url to use a different MusicBrainz host."
                    )
                    raise OSError(msg) from err
                await _pace()
                continue
            await _pace()
            if index % 100 == 0:
                print(f"  {index}/{len(wanted)} artists resolved ({skipped} skipped)")
            life_span = lookup.get("life-span") or {}
            begin = life_span.get("begin")
            records.append(
                {
                    "artist_mbid": mbid,
                    "artist_name": artist.get("artist_name", ""),
                    "tags": lookup.get("tags", []),
                    # The fallback is only safe for an all-time row. Every other range reports
                    # a listen count for that window alone, and averaging those together with
                    # all-time totals would weight a busy week like a busy decade.
                    "total_listen_count": popularity.get("total_listen_count")
                    or (artist.get("listen_count", 0) if artist.get("_range") == "all_time" else 0),
                    "total_user_count": popularity.get("total_user_count"),
                    "first_release_year": int(begin[:4]) if begin and begin[:4].isdigit() else None,
                }
            )
    if skipped:
        print(
            f"Resolved {len(records)} artists; {skipped} could not be looked up and were skipped."
        )
    return records


async def _fetch_top_artists(session: Any, sample: int) -> list[dict[str, Any]]:
    """
    Gather up to ``sample`` distinct artists from ListenBrainz's sitewide statistics.

    ListenBrainz computes only the top 1000 artists PER TIME RANGE. That is a property of the
    data, not a per-request limit, so offset paging within one range stops dead at 1000 and a
    --sample of 5000 came back with 993 looking like a finished run. Widening therefore means
    asking different ranges, not asking the same range harder: the all-time chart and the
    this-week chart share their head but diverge in the tail, so each adds artists the others
    do not have.

    The ranges are a way of finding CANDIDATES only. Every artist's weight still comes from the
    all-time popularity endpoint in :func:`_fetch_live`, so widening the pool this way does not
    let a this-week listen count contaminate an all-time average.

    :param session: An open aiohttp ClientSession.
    :param sample: How many distinct artists are wanted in total.
    """
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for time_range in LISTENBRAINZ_RANGES:
        if len(collected) >= sample:
            break
        before = len(collected)
        offset = 0
        while len(collected) < sample:
            page_size = min(LISTENBRAINZ_PAGE_LIMIT, sample - len(collected))
            payload = await _get_json(
                session,
                f"{LISTENBRAINZ_BASE_URL}/1/stats/sitewide/artists",
                params={
                    "count": str(page_size),
                    "offset": str(offset),
                    "range": time_range,
                },
            )
            await _pace()
            page = _unwrap(payload, "artists")
            if not page:
                break
            fresh = []
            for artist in page:
                key = artist.get("artist_mbid") or artist.get("artist_name")
                if not key or key in seen:
                    continue
                seen.add(key)
                # Remembered so the listen-count fallback in _fetch_live can tell an all-time
                # figure from a range-scoped one.
                fresh.append({**artist, "_range": time_range})
            collected.extend(fresh)
            if len(page) < page_size or not fresh:
                break
            offset += len(page)
        added = len(collected) - before
        print(f"  {time_range}: +{added} new (running total {len(collected)})")

    if len(collected) < sample:
        print(
            f"  ListenBrainz had {len(collected)} distinct artists to give, not {sample}. "
            "It computes only the top 1000 per time range, and the ranges overlap heavily."
        )
    return collected[:sample]


def _unwrap(response: Any, key: str | None = None) -> list[dict[str, Any]]:
    """
    Pull the list of records out of a ListenBrainz response, whatever envelope it arrived in.

    The two endpoints this script uses do not agree with each other. ``/1/stats/sitewide/artists``
    answers ``{"payload": {"artists": [...]}}`` while ``/1/popularity/artist`` answers a bare
    JSON array, and assuming the first shape for both crashed the live fetch on an
    ``AttributeError`` after it had already spent a minute on the network. Accepting all three
    shapes costs nothing and means an envelope change upstream degrades to an empty result
    rather than a traceback.

    :param response: The decoded JSON body.
    :param key: The key holding the list when the payload is an object rather than a list.
    """
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    if not isinstance(response, dict):
        return []
    payload = response.get("payload", response)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and key:
        inner = payload.get(key, [])
        if isinstance(inner, list):
            return [item for item in inner if isinstance(item, dict)]
    return []


async def _get_json(session: Any, url: str, *, params: dict[str, str] | None = None) -> Any:
    """Issue a GET request and return the decoded JSON body, retrying transient failures."""
    return await _request_json(session, "GET", url, params=params)


async def _request_json(
    session: Any,
    method: str,
    url: str,
    *,
    params: dict[str, str] | None = None,
    json_body: Any = None,
) -> Any:
    """
    Issue one request, retrying a transient failure with exponential backoff.

    :param session: An open aiohttp ClientSession.
    :param method: HTTP method.
    :param url: Absolute request URL.
    :param params: Optional query parameters.
    :param json_body: Optional JSON request body.
    """
    import aiohttp  # noqa: PLC0415 - optional dependency, only needed for the live-fetch path

    last_status: int | None = None
    for attempt in range(1, LIVE_MAX_ATTEMPTS + 1):
        try:
            async with session.request(method, url, params=params, json=json_body) as response:
                if response.status in LIVE_RETRY_STATUSES:
                    last_status = response.status
                    # A server telling us exactly how long to wait knows better than our own
                    # backoff curve does; MusicBrainz sends this when it is rate-limiting.
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                    raise _TransientFetchError(response.status, retry_after)
                if response.status >= 400:
                    # Raised directly rather than via raise_for_status(), whose
                    # ClientResponseError is an aiohttp.ClientError and would therefore be
                    # caught by the retry handler below - turning "this request is wrong" into
                    # four identical wrong requests.
                    msg = f"{method} {url} failed (HTTP {response.status})"
                    raise OSError(msg)
                return await response.json()
        except (aiohttp.ClientError, TimeoutError, _TransientFetchError) as err:
            if attempt == LIVE_MAX_ATTEMPTS:
                detail = f"HTTP {last_status}" if last_status else type(err).__name__
                msg = f"{method} {url} failed after {attempt} attempts ({detail})"
                raise OSError(msg) from err
            # Back off further each time: a server shedding load needs longer, not the same
            # interval again. Growing by the square rather than linearly because the linear
            # curve topped out at 6 seconds, which a rate limiter does not even notice.
            hinted = getattr(err, "retry_after", None)
            await asyncio.sleep(hinted or LIVE_RETRY_BACKOFF_SECONDS * attempt**2)
    # Unreachable: the final attempt either returns or raises.
    raise OSError(f"{method} {url} failed")


class _TransientFetchError(Exception):
    """A response worth retrying. Carries only the status - never the URL, which has a key."""

    def __init__(self, status: int, retry_after: float | None = None) -> None:
        """
        Record a retryable response.

        :param status: The HTTP status that triggered the retry.
        :param retry_after: Seconds the server asked us to wait, if it said.
        """
        super().__init__(f"HTTP {status}")
        self.status = status
        self.retry_after = retry_after


def _parse_retry_after(value: str | None) -> float | None:
    """
    Read a ``Retry-After`` header expressed in seconds.

    :param value: The raw header value, or None when the server did not send one.
    """
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        # The header may also carry an HTTP-date. Falling back to our own backoff is better
        # than parsing dates to save a second or two.
        return None
    # A server asking for an implausibly long wait is more likely misconfigured than serious.
    return min(seconds, 60.0) if seconds > 0 else None


async def _post_json(session: Any, url: str, *, json: Any) -> Any:
    """Issue a POST request and return the decoded JSON body, retrying transient failures."""
    return await _request_json(session, "POST", url, json_body=json)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT
            / "music_assistant"
            / "controllers"
            / "genome"
            / "baseline"
            / "baseline_v1.json"
        ),
        help="Output path for the baseline JSON file.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1500,
        # 50000 was the original default and is indefensible: at one MusicBrainz lookup per
        # artist it is a 13-hour run and a genuinely rude load on a donated service. The
        # baseline is a distribution - a well-drawn sample of the top ~1500 artists pins the
        # genre and era shares and the listener percentiles to more precision than the rest
        # of this feature can honestly use.
        help="Live mode: number of top artists to sample (each costs one MusicBrainz lookup).",
    )
    parser.add_argument(
        "--fixture", default=None, help="Offline mode: path to a local artist-sample JSON file."
    )
    parser.add_argument(
        "--from-dump",
        default=None,
        help="Offline mode: path to a raw ListenBrainz dump shaped the same way.",
    )
    parser.add_argument("--version", default="v1-2026-09", help="Baseline version string to embed.")
    parser.add_argument(
        "--mb-base-url",
        default=MUSICBRAINZ_BASE_URL,
        help=(
            "MusicBrainz web-service root. Defaults to the public API; point this at your own "
            "mirror if you run one. Not Music Assistant's mirror - see the note by "
            "MUSICBRAINZ_BASE_URL."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
