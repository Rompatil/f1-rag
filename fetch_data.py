#!/usr/bin/env python3
"""
F1 Data Fetcher — Downloads real Formula 1 data from the Jolpica API.

Usage:
    python fetch_data.py                          # Default: 1995-2026
    python fetch_data.py --seasons 2000-2025      # Custom range
    python fetch_data.py --output ./my_data       # Custom output dir

API: https://api.jolpi.ca/ergast/f1/
Rate limit: 4 req/sec, 500 req/hr. This script respects that.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

try:
    import requests
    USE_REQUESTS = True
except ImportError:
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError, URLError
    USE_REQUESTS = False

BASE_URL = "https://api.jolpi.ca/ergast/f1"
OUTPUT_DIR = Path(__file__).parent / "data" / "raw"
DELAY = 0.35  # seconds between requests


def fetch_json(url: str, retries: int = 3) -> dict | None:
    """Fetch JSON with retry + rate limiting."""
    for attempt in range(retries):
        try:
            if USE_REQUESTS:
                resp = requests.get(url, timeout=30, headers={"User-Agent": "F1-RAG/1.0"})
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    print(f"    ⏳ Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                data = resp.json()
            else:
                req = Request(url, headers={"User-Agent": "F1-RAG/1.0"})
                with urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())

            time.sleep(DELAY)
            return data
        except Exception as e:
            if "429" in str(e):
                wait = 2 ** (attempt + 1)
                print(f"    ⏳ Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif "404" in str(e):
                return None
            else:
                print(f"    ⚠️  Attempt {attempt+1}/{retries}: {e}")
                if attempt == retries - 1:
                    return None
                time.sleep(2)
    return None


def fetch_all_pages(base_url: str, table_key: str, list_key: str,
                    page_limit: int = 100) -> list[dict]:
    """Fetch all pages of a paginated endpoint."""
    all_items = []
    offset = 0
    while True:
        url = f"{base_url}?limit={page_limit}&offset={offset}"
        data = fetch_json(url)
        if not data:
            break
        mrdata = data.get("MRData", {})
        total = int(mrdata.get("total", 0))
        items = mrdata.get(table_key, {}).get(list_key, [])
        if not items:
            break
        all_items.extend(items)
        offset += page_limit
        if offset >= total:
            break
    return all_items


def write_csv(filepath: Path, headers: list[str], rows: list[list]) -> int:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    return len(rows)


# ── Table fetchers ────────────────────────────────────────────────────

def fetch_drivers(output: Path) -> int:
    print("📋 Fetching all drivers...")
    url = f"{BASE_URL}/drivers.json"
    items = fetch_all_pages(url, "DriverTable", "Drivers", page_limit=100)
    rows = []
    for i, d in enumerate(items, 1):
        rows.append([
            i, d.get("driverId", ""),
            d.get("permanentNumber", "\\N"), d.get("code", "\\N"),
            d.get("givenName", ""), d.get("familyName", ""),
            d.get("dateOfBirth", ""), d.get("nationality", ""),
            d.get("url", ""),
        ])
    count = write_csv(output / "drivers.csv",
        ["driverId","driverRef","number","code","forename","surname","dob","nationality","url"],
        rows)
    print(f"    ✅ {count} drivers")
    return count


def fetch_constructors(output: Path) -> int:
    print("📋 Fetching all constructors...")
    url = f"{BASE_URL}/constructors.json"
    items = fetch_all_pages(url, "ConstructorTable", "Constructors", page_limit=100)
    rows = []
    for i, c in enumerate(items, 1):
        rows.append([
            i, c.get("constructorId", ""),
            c.get("name", ""), c.get("nationality", ""), c.get("url", ""),
        ])
    count = write_csv(output / "constructors.csv",
        ["constructorId","constructorRef","name","nationality","url"],
        rows)
    print(f"    ✅ {count} constructors")
    return count


def fetch_circuits(output: Path) -> int:
    print("📋 Fetching all circuits...")
    url = f"{BASE_URL}/circuits.json"
    items = fetch_all_pages(url, "CircuitTable", "Circuits", page_limit=100)
    rows = []
    for i, c in enumerate(items, 1):
        loc = c.get("Location", {})
        rows.append([
            i, c.get("circuitId", ""), c.get("circuitName", ""),
            loc.get("locality", ""), loc.get("country", ""),
            loc.get("lat", ""), loc.get("long", ""),
            loc.get("alt", "0"), c.get("url", ""),
        ])
    count = write_csv(output / "circuits.csv",
        ["circuitId","circuitRef","name","location","country","lat","lng","alt","url"],
        rows)
    print(f"    ✅ {count} circuits")
    return count


def fetch_races(seasons: list[int], output: Path) -> int:
    print("📋 Fetching races...")
    all_rows = []
    race_id = 1
    for season in seasons:
        url = f"{BASE_URL}/{season}.json?limit=100"
        data = fetch_json(url)
        if not data:
            continue
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        for r in races:
            circuit_ref = r.get("Circuit", {}).get("circuitId", "")
            all_rows.append([
                race_id, season, r.get("round", ""), circuit_ref,
                r.get("raceName", ""), r.get("date", ""),
                r.get("time", "\\N"), r.get("url", ""),
            ])
            race_id += 1
        sys.stdout.write(f"\r    {season}... ({len(all_rows)} races so far)")
        sys.stdout.flush()
    count = write_csv(output / "races.csv",
        ["raceId","year","round","circuitId","name","date","time","url"],
        all_rows)
    print(f"\r    ✅ {count} races" + " " * 30)
    return count


def fetch_results(seasons: list[int], output: Path) -> int:
    print("📋 Fetching race results...")

    # Build lookup maps from already-fetched CSVs
    race_id_map = _load_race_id_map(output / "races.csv")
    driver_ref_map = _load_ref_map(output / "drivers.csv", "driverRef", "driverId")
    con_ref_map = _load_ref_map(output / "constructors.csv", "constructorRef", "constructorId")

    all_rows = []
    result_id = 1

    for season in seasons:
        # Fetch all results for this season, paginated
        url = f"{BASE_URL}/{season}/results.json"
        races = fetch_all_pages(url, "RaceTable", "Races", page_limit=100)

        for race in races:
            rnd = race.get("round", "0")
            race_id = race_id_map.get((season, int(rnd)), f"{season}_{rnd}")

            for res in race.get("Results", []):
                d_ref = res.get("Driver", {}).get("driverId", "")
                c_ref = res.get("Constructor", {}).get("constructorId", "")
                d_id = driver_ref_map.get(d_ref, d_ref)
                c_id = con_ref_map.get(c_ref, c_ref)

                time_obj = res.get("Time")
                time_str = time_obj.get("time", "\\N") if isinstance(time_obj, dict) else "\\N"
                time_ms = time_obj.get("millis", "\\N") if isinstance(time_obj, dict) else "\\N"

                fl = res.get("FastestLap") or {}
                fl_time_obj = fl.get("Time") or {}
                fl_speed_obj = fl.get("AverageSpeed") or {}

                all_rows.append([
                    result_id, race_id, d_id, c_id,
                    res.get("number", "\\N"), res.get("grid", ""),
                    res.get("position", "\\N"), res.get("positionText", ""),
                    res.get("position", 99) or 99,
                    res.get("points", "0"), res.get("laps", ""),
                    time_str, time_ms,
                    fl.get("lap", "\\N"), fl.get("rank", "\\N"),
                    fl_time_obj.get("time", "\\N"),
                    fl_speed_obj.get("speed", "\\N"),
                    res.get("status", ""),
                ])
                result_id += 1

        sys.stdout.write(f"\r    {season}... ({len(all_rows)} results so far)")
        sys.stdout.flush()

    count = write_csv(output / "results.csv",
        ["resultId","raceId","driverId","constructorId","number","grid",
         "position","positionText","positionOrder","points","laps",
         "time","milliseconds","fastestLap","rank",
         "fastestLapTime","fastestLapSpeed","statusId"],
        all_rows)
    print(f"\r    ✅ {count} results" + " " * 30)
    return count


def fetch_standings(seasons: list[int], output: Path) -> tuple[int, int]:
    print("📋 Fetching championship standings...")

    race_id_map = _load_race_id_map(output / "races.csv")
    driver_ref_map = _load_ref_map(output / "drivers.csv", "driverRef", "driverId")
    con_ref_map = _load_ref_map(output / "constructors.csv", "constructorRef", "constructorId")

    # Find last raceId per season
    season_last_race = {}
    races_file = output / "races.csv"
    if races_file.exists():
        with open(races_file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                yr = int(row["year"])
                rid = int(row["raceId"])
                if yr not in season_last_race or rid > season_last_race[yr]:
                    season_last_race[yr] = rid

    d_rows, c_rows = [], []
    ds_id, cs_id = 1, 1

    for season in seasons:
        last_race = season_last_race.get(season, f"{season}_last")

        # Driver standings
        url = f"{BASE_URL}/{season}/driverStandings.json?limit=100"
        data = fetch_json(url)
        if data:
            sl = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
            if sl:
                for s in sl[0].get("DriverStandings", []):
                    d_ref = s.get("Driver", {}).get("driverId", "")
                    d_rows.append([
                        ds_id, last_race, driver_ref_map.get(d_ref, d_ref),
                        s.get("points", "0"), s.get("position", ""),
                        s.get("positionText", ""), s.get("wins", "0"),
                    ])
                    ds_id += 1

        # Constructor standings
        url = f"{BASE_URL}/{season}/constructorStandings.json?limit=100"
        data = fetch_json(url)
        if data:
            sl = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
            if sl:
                for s in sl[0].get("ConstructorStandings", []):
                    c_ref = s.get("Constructor", {}).get("constructorId", "")
                    c_rows.append([
                        cs_id, last_race, con_ref_map.get(c_ref, c_ref),
                        s.get("points", "0"), s.get("position", ""),
                        s.get("positionText", ""), s.get("wins", "0"),
                    ])
                    cs_id += 1

        sys.stdout.write(f"\r    {season}...")
        sys.stdout.flush()

    dc = write_csv(output / "driver_standings.csv",
        ["driverStandingsId","raceId","driverId","points","position","positionText","wins"],
        d_rows)
    cc = write_csv(output / "constructor_standings.csv",
        ["constructorStandingsId","raceId","constructorId","points","position","positionText","wins"],
        c_rows)
    print(f"\r    ✅ {dc} driver standings, {cc} constructor standings" + " " * 20)
    return dc, cc


# ── Helpers ───────────────────────────────────────────────────────────

def _load_race_id_map(filepath: Path) -> dict:
    """Build (year, round) -> raceId mapping."""
    m = {}
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                m[(int(row["year"]), int(row["round"]))] = int(row["raceId"])
    return m


def _load_ref_map(filepath: Path, ref_col: str, id_col: str) -> dict:
    """Build ref -> id mapping from CSV."""
    m = {}
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                m[row[ref_col]] = row[id_col]
    return m


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch F1 data from Jolpica API")
    parser.add_argument("--seasons", default="1995-2026",
                        help="Season range (default: 1995-2026)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help="Output directory")
    args = parser.parse_args()

    start_str, end_str = args.seasons.split("-")
    seasons = list(range(int(start_str), int(end_str) + 1))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  F1 Data Fetcher — Jolpica API")
    print(f"  Seasons: {seasons[0]}–{seasons[-1]} ({len(seasons)} seasons)")
    print(f"  Output:  {output.resolve()}")
    print(f"  API:     {BASE_URL}")
    print("=" * 60)
    print()

    t0 = time.time()

    # ORDER MATTERS: static tables first, then season-based
    fetch_drivers(output)
    fetch_constructors(output)
    fetch_circuits(output)
    fetch_races(seasons, output)
    fetch_results(seasons, output)
    fetch_standings(seasons, output)

    elapsed = time.time() - t0
    print()
    print(f"🏁 Done in {elapsed:.0f}s")
    print(f"   Files: {output.resolve()}")
    print(f"   Next:  python run.py --ingest")


if __name__ == "__main__":
    main()
