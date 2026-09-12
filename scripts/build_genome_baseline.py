"""
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

Usage:
    uv run -m scripts.build_genome_baseline --fixture tests/fixtures/genome/listenbrainz_baseline_sample.json \\
        --out music_assistant/controllers/genome/baseline/baseline_v1.json
    uv run -m scripts.build_genome_baseline --out music_assistant/controllers/genome/baseline/baseline_v1.json --sample 50000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ruff: noqa: T201

REPO_ROOT = Path(__file__).resolve().parents[1]
GENRE_MAPPING_PATH = REPO_ROOT / "music_assistant" / "helpers" / "resources" / "genres" / "genre_mapping.json"

LISTENBRAINZ_BASE_URL = "https://api.listenbrainz.org"
MUSICBRAINZ_BASE_URL = "https://musicbrainz-mirror.music-assistant.io/ws/2"
REQUIRED_PERCENTILES = (5, 10, 25, 50, 75, 90)
LIVE_REQUEST_DELAY_SECONDS = 1.0


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
            records = asyncio.run(_fetch_live(args.sample))
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
            "`uv run -m scripts.build_genome_baseline --sample 50000` once ListenBrainz is "
            "reachable, then drop the 'provisional'/'note' fields."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({aggregate['sample_size']} artist samples, source={source})")
    return 0


def _aggregate(records: list[dict[str, Any]], genre_mapping: list[dict[str, Any]]) -> dict[str, Any]:
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
    """Normalized Herfindahl index over sampled artists' listen-count shares (0..1)."""
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


async def _fetch_live(sample: int) -> list[dict[str, Any]]:
    """
    Sample ListenBrainz's sitewide popularity plus MusicBrainz artist tags (§3.7).

    Not reachable from this development workspace (BRIEF.md); written to run for real on
    Bob's machine, where both hosts are reachable.
    """
    import aiohttp  # noqa: PLC0415 - optional dependency, only needed for the live-fetch path

    records: list[dict[str, Any]] = []
    async with aiohttp.ClientSession() as session:
        top_artists = await _get_json(
            session, f"{LISTENBRAINZ_BASE_URL}/1/stats/sitewide/artists", params={"count": str(sample)}
        )
        await asyncio.sleep(LIVE_REQUEST_DELAY_SECONDS)
        artists = top_artists.get("payload", {}).get("artists", [])
        mbids = [a["artist_mbid"] for a in artists if a.get("artist_mbid")]
        popularity_by_mbid: dict[str, dict[str, Any]] = {}
        for batch_start in range(0, len(mbids), 50):
            batch = mbids[batch_start : batch_start + 50]
            popularity = await _post_json(
                session, f"{LISTENBRAINZ_BASE_URL}/1/popularity/artist", json={"artist_mbids": batch}
            )
            for item in popularity.get("payload", []):
                popularity_by_mbid[item["artist_mbid"]] = item
            await asyncio.sleep(LIVE_REQUEST_DELAY_SECONDS)

        for artist in artists:
            mbid = artist.get("artist_mbid")
            if not mbid:
                continue
            popularity = popularity_by_mbid.get(mbid, {})
            lookup = await _get_json(
                session, f"{MUSICBRAINZ_BASE_URL}/artist/{mbid}", params={"inc": "tags+genres", "fmt": "json"}
            )
            await asyncio.sleep(LIVE_REQUEST_DELAY_SECONDS)
            life_span = lookup.get("life-span") or {}
            begin = life_span.get("begin")
            records.append(
                {
                    "artist_mbid": mbid,
                    "artist_name": artist.get("artist_name", ""),
                    "tags": lookup.get("tags", []),
                    "total_listen_count": popularity.get(
                        "total_listen_count", artist.get("listen_count", 0)
                    ),
                    "total_user_count": popularity.get("total_user_count"),
                    "first_release_year": int(begin[:4]) if begin and begin[:4].isdigit() else None,
                }
            )
    return records


async def _get_json(session: Any, url: str, *, params: dict[str, str] | None = None) -> Any:
    """Issue a throttled GET request and return the decoded JSON body."""
    async with session.get(url, params=params) as response:
        response.raise_for_status()
        return await response.json()


async def _post_json(session: Any, url: str, *, json: Any) -> Any:
    """Issue a throttled POST request and return the decoded JSON body."""
    async with session.post(url, json=json) as response:
        response.raise_for_status()
        return await response.json()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "music_assistant" / "controllers" / "genome" / "baseline" / "baseline_v1.json"),
        help="Output path for the baseline JSON file.",
    )
    parser.add_argument("--sample", type=int, default=50000, help="Live mode: number of top artists to sample.")
    parser.add_argument("--fixture", default=None, help="Offline mode: path to a local artist-sample JSON file.")
    parser.add_argument(
        "--from-dump", default=None, help="Offline mode: path to a raw ListenBrainz dump shaped the same way."
    )
    parser.add_argument("--version", default="v1-2026-09", help="Baseline version string to embed.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
