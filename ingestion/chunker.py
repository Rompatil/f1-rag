"""
Chunking Layer — converts structured F1 data into semantic text chunks.

Instead of embedding raw CSV rows, we create meaningful text that captures
context a human would understand:

Chunk types:
1. Race Result Summary — one per race, describes full podium + key stats
2. Driver Season Profile — one per driver per season, aggregates their performance
3. Constructor Season Profile — one per team per season
4. Driver Career Profile — one per driver, career-level stats
5. Circuit Profile — one per circuit
6. Season Championship Summary — one per season, final standings narrative

Each chunk has:
- chunk_id: unique identifier
- category: type of chunk (for filtering)
- content: the text to embed
- metadata: structured fields for source tracing
"""

from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

from ingestion.loader import F1DataLoader
from utils.logger import logger


@dataclass
class Chunk:
    chunk_id: str
    category: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class F1Chunker:
    """Produces semantic text chunks from F1 data."""

    def __init__(self, loader: F1DataLoader):
        self.loader = loader

    def generate_all_chunks(self) -> list[Chunk]:
        """Generate all chunk types and return a flat list."""
        chunks: list[Chunk] = []

        enriched_results = self.loader.get_enriched_results()
        driver_standings = self.loader.get_enriched_driver_standings()
        constructor_standings = self.loader.get_enriched_constructor_standings()

        chunks.extend(self._race_result_chunks(enriched_results))
        chunks.extend(self._driver_season_chunks(enriched_results))
        chunks.extend(self._constructor_season_chunks(enriched_results))
        chunks.extend(self._circuit_chunks())
        chunks.extend(self._season_championship_chunks(driver_standings, constructor_standings))
        chunks.extend(self._driver_career_chunks(enriched_results, driver_standings))

        logger.info(f"Generated {len(chunks)} total chunks across {len(set(c.category for c in chunks))} categories")
        return chunks

    # ------------------------------------------------------------------
    # Chunk generators
    # ------------------------------------------------------------------

    def _race_result_chunks(self, df: pd.DataFrame) -> list[Chunk]:
        """One chunk per race — podium, winner info, fastest lap."""
        chunks = []
        for race_id, group in df.groupby("raceId"):
            group = group.sort_values("positionOrder")
            race_row = group.iloc[0]

            race_name = race_row.get("race_name", "Unknown GP")
            year = int(race_row["year"]) if pd.notna(race_row.get("year")) else "?"
            circuit = race_row.get("circuit_name", "Unknown Circuit")
            country = race_row.get("circuit_country", "")
            date = race_row.get("date", "")

            lines = [f"Race: {race_name} ({year})",
                     f"Circuit: {circuit}, {country}",
                     f"Date: {date}",
                     ""]

            # Podium
            podium = group[group["positionOrder"].isin([1, 2, 3])]
            for _, row in podium.iterrows():
                pos = int(row["positionOrder"])
                name = row.get("driver_name", "Unknown")
                team = row.get("constructor_name", "Unknown")
                pts = row.get("points", 0)
                medal = {1: "Winner", 2: "2nd", 3: "3rd"}.get(pos, f"P{pos}")
                time_str = row.get("time", "")
                time_info = f" — Time: {time_str}" if pd.notna(time_str) and time_str else ""
                lines.append(f"{medal}: {name} ({team}, {pts} pts){time_info}")

            # Full finishing order for non-podium finishers
            rest = group[(group["positionOrder"] > 3) & (group["positionOrder"] <= 10)]
            if not rest.empty:
                lines.append("")
                for _, row in rest.iterrows():
                    pos = int(row["positionOrder"])
                    name = row.get("driver_name", "Unknown")
                    team = row.get("constructor_name", "Unknown")
                    lines.append(f"P{pos}: {name} ({team})")

            # Fastest lap
            fl_rows = group[group["rank"] == 1]
            if not fl_rows.empty:
                fl = fl_rows.iloc[0]
                fl_name = fl.get("driver_name", "Unknown")
                fl_time = fl.get("fastestLapTime", "")
                if pd.notna(fl_time):
                    lines.append(f"\nFastest Lap: {fl_name} — {fl_time}")

            # DNFs
            dnf = group[group["positionText"].isin(["R", "D", "W"])]
            if not dnf.empty:
                dnf_names = [r.get("driver_name", "?") for _, r in dnf.iterrows()]
                lines.append(f"DNF: {', '.join(dnf_names)}")

            laps = group["laps"].max()
            if pd.notna(laps):
                lines.append(f"Total Laps: {int(laps)}")

            content = "\n".join(lines)
            chunks.append(Chunk(
                chunk_id=f"race_{race_id}",
                category="race_result",
                content=content,
                metadata={"raceId": race_id, "year": year, "race_name": race_name,
                          "circuitId": race_row.get("circuitId", "")},
            ))

        logger.debug(f"Generated {len(chunks)} race result chunks")
        return chunks

    def _driver_season_chunks(self, df: pd.DataFrame) -> list[Chunk]:
        """One chunk per driver per season — aggregated stats."""
        chunks = []
        for (year, driver_id), group in df.groupby(["year", "driverId"]):
            row0 = group.iloc[0]
            name = row0.get("driver_name", "Unknown")
            team = row0.get("constructor_name", "Unknown")
            nationality = row0.get("driver_nationality", "")

            total_points = group["points"].sum()
            races = len(group)
            wins = len(group[group["positionOrder"] == 1])
            podiums = len(group[group["positionOrder"].isin([1, 2, 3])])
            dnfs = len(group[group["positionText"].isin(["R", "D", "W"])])
            best_finish = group["positionOrder"].min()

            lines = [
                f"Driver Season Summary: {name} — {int(year)} Season",
                f"Team: {team}",
                f"Nationality: {nationality}",
                f"Races entered: {races}",
                f"Wins: {wins}",
                f"Podiums: {podiums}",
                f"Total Points: {total_points}",
                f"Best Finish: P{int(best_finish)}",
                f"DNFs: {dnfs}",
            ]

            # List individual race results
            lines.append("\nRace-by-race:")
            for _, r in group.sort_values("date").iterrows():
                rname = r.get("race_name", "?")
                pos = r.get("positionText", "?")
                pts = r.get("points", 0)
                lines.append(f"  {rname}: P{pos} ({pts} pts)")

            content = "\n".join(lines)
            chunks.append(Chunk(
                chunk_id=f"driver_season_{driver_id}_{int(year)}",
                category="driver_season",
                content=content,
                metadata={"driverId": driver_id, "year": int(year),
                          "driver_name": name, "team": team},
            ))

        logger.debug(f"Generated {len(chunks)} driver season chunks")
        return chunks

    def _constructor_season_chunks(self, df: pd.DataFrame) -> list[Chunk]:
        """One chunk per constructor per season."""
        chunks = []
        for (year, con_id), group in df.groupby(["year", "constructorId"]):
            row0 = group.iloc[0]
            team_name = row0.get("constructor_name", "Unknown")

            total_points = group["points"].sum()
            wins = len(group[group["positionOrder"] == 1])
            podiums = len(group[group["positionOrder"].isin([1, 2, 3])])
            races = group["raceId"].nunique()

            # Drivers who drove for the team this season
            drivers = group.groupby("driver_name")["points"].sum().sort_values(ascending=False)
            driver_lines = [f"  {name}: {pts} pts" for name, pts in drivers.items()]

            lines = [
                f"Constructor Season Summary: {team_name} — {int(year)} Season",
                f"Races: {races}",
                f"Wins: {wins}",
                f"Podiums: {podiums}",
                f"Total Points: {total_points}",
                f"Drivers:",
                *driver_lines,
            ]

            content = "\n".join(lines)
            chunks.append(Chunk(
                chunk_id=f"constructor_season_{con_id}_{int(year)}",
                category="constructor_season",
                content=content,
                metadata={"constructorId": con_id, "year": int(year),
                          "constructor_name": team_name},
            ))

        logger.debug(f"Generated {len(chunks)} constructor season chunks")
        return chunks

    def _circuit_chunks(self) -> list[Chunk]:
        """One chunk per circuit."""
        circuits = self.loader.tables["circuits"]
        chunks = []
        for _, row in circuits.iterrows():
            cid = row.get("circuitId", row.get("circuitRef", "unknown"))
            name = row.get("name", "Unknown")
            location = row.get("location", "")
            country = row.get("country", "")
            lat = row.get("lat", "")
            lng = row.get("lng", "")

            lines = [
                f"Circuit: {name}",
                f"Location: {location}, {country}",
                f"Coordinates: {lat}, {lng}",
            ]
            alt = row.get("alt", "")
            if pd.notna(alt):
                lines.append(f"Altitude: {alt}m")

            content = "\n".join(lines)
            chunks.append(Chunk(
                chunk_id=f"circuit_{cid}",
                category="circuit",
                content=content,
                metadata={"circuitId": cid, "name": name, "country": country},
            ))

        logger.debug(f"Generated {len(chunks)} circuit chunks")
        return chunks

    def _season_championship_chunks(
        self,
        driver_standings: pd.DataFrame,
        constructor_standings: pd.DataFrame,
    ) -> list[Chunk]:
        """One chunk per season — final championship standings."""
        chunks = []

        races = self.loader.tables.get("races")
        race_year_map = {}
        if races is not None:
            for _, r in races.iterrows():
                race_year_map[r["raceId"]] = int(r["year"])

        # Driver championship
        for race_id, group in driver_standings.groupby("raceId"):
            year = race_year_map.get(race_id, "?")
            group = group.sort_values("position")

            lines = [f"Drivers' Championship Final Standings — {year} Season", ""]

            champion = group.iloc[0]
            champion_name = champion.get("driver_name", "Unknown")
            lines.append(f"Champion: {champion_name}")
            lines.append("")

            for _, row in group.iterrows():
                if pd.isna(row["position"]):
                    continue
                pos = int(row["position"])
                name = row.get("driver_name", "Unknown")
                pts = row.get("points", 0)
                wins = int(row.get("wins", 0))
                lines.append(f"P{pos}: {name} — {pts} pts ({wins} wins)")

            content = "\n".join(lines)
            chunks.append(Chunk(
                chunk_id=f"championship_drivers_{year}",
                category="championship",
                content=content,
                metadata={"year": year, "type": "drivers", "champion": champion_name},
            ))

        # Constructor championship
        for race_id, group in constructor_standings.groupby("raceId"):
            year = race_year_map.get(race_id, "?")
            group = group.sort_values("position")

            lines = [f"Constructors' Championship Final Standings — {year} Season", ""]

            champion_team = group.iloc[0]
            champion_name = champion_team.get("constructor_name", "Unknown")
            lines.append(f"Champion Constructor: {champion_name}")
            lines.append("")

            for _, row in group.iterrows():
                if pd.isna(row["position"]):
                    continue
                pos = int(row["position"])
                name = row.get("constructor_name", "Unknown")
                pts = row.get("points", 0)
                wins = int(row.get("wins", 0))
                lines.append(f"P{pos}: {name} — {pts} pts ({wins} wins)")

            content = "\n".join(lines)
            chunks.append(Chunk(
                chunk_id=f"championship_constructors_{year}",
                category="championship",
                content=content,
                metadata={"year": year, "type": "constructors", "champion": champion_name},
            ))

        logger.debug(f"Generated {len(chunks)} championship chunks")
        return chunks

    def _driver_career_chunks(
        self,
        results: pd.DataFrame,
        standings: pd.DataFrame,
    ) -> list[Chunk]:
        """One chunk per driver — career-level aggregate."""
        chunks = []
        races = self.loader.tables.get("races")
        race_year_map = {}
        if races is not None:
            for _, r in races.iterrows():
                race_year_map[r["raceId"]] = int(r["year"])

        for driver_id, group in results.groupby("driverId"):
            row0 = group.iloc[0]
            name = row0.get("driver_name", "Unknown")
            nationality = row0.get("driver_nationality", "")
            dob = row0.get("dob", "")

            seasons = sorted(group["year"].dropna().unique())
            total_races = len(group)
            total_points = group["points"].sum()
            wins = len(group[group["positionOrder"] == 1])
            podiums = len(group[group["positionOrder"].isin([1, 2, 3])])
            teams = group["constructor_name"].dropna().unique().tolist()

            # Championship wins from standings
            champ_wins = []
            driver_std = standings[standings["driverId"] == driver_id]
            for _, s in driver_std.iterrows():
                if s.get("position") == 1:
                    yr = race_year_map.get(s["raceId"])
                    if yr:
                        champ_wins.append(str(yr))

            lines = [
                f"Driver Career Profile: {name}",
                f"Nationality: {nationality}",
                f"Date of Birth: {dob}",
                f"Active Seasons (in dataset): {', '.join(str(int(s)) for s in seasons)}",
                f"Teams: {', '.join(teams)}",
                f"Total Races (in dataset): {total_races}",
                f"Wins: {wins}",
                f"Podiums: {podiums}",
                f"Total Points: {total_points}",
            ]
            if champ_wins:
                lines.append(f"World Championships: {', '.join(champ_wins)}")

            content = "\n".join(lines)
            chunks.append(Chunk(
                chunk_id=f"driver_career_{driver_id}",
                category="driver_career",
                content=content,
                metadata={"driverId": driver_id, "driver_name": name},
            ))

        logger.debug(f"Generated {len(chunks)} driver career chunks")
        return chunks
