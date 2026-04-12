"""
Data Ingestion Layer — loads Ergast-schema CSVs and produces
denormalized DataFrames ready for chunking.

Handles:
- CSV loading with type coercion
- Joining relational tables (results + drivers + constructors + races + circuits)
- Cleaning nulls and normalizing text
"""

import pandas as pd
from pathlib import Path

from utils.config import settings
from utils.logger import logger


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV, treating \\N as NaN (Ergast convention)."""
    df = pd.read_csv(path, na_values=["\\N", ""], keep_default_na=True)
    logger.debug(f"Loaded {path.name}: {len(df)} rows, {list(df.columns)}")
    return df


def _to_int(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """Coerce columns to nullable Int64 in-place."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


class F1DataLoader:
    """Loads and joins all F1 CSV tables into analysis-ready DataFrames."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or settings.paths.data_raw
        self._tables: dict[str, pd.DataFrame] = {}

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all available CSV files into DataFrames."""
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        for path in csv_files:
            table_name = path.stem
            self._tables[table_name] = _read_csv(path)

        logger.info(f"Loaded {len(self._tables)} tables: {list(self._tables.keys())}")
        return self._tables

    @property
    def tables(self) -> dict[str, pd.DataFrame]:
        if not self._tables:
            self.load_all()
        return self._tables

    def get_enriched_results(self) -> pd.DataFrame:
        """
        Join results with drivers, constructors, races, and circuits
        to produce a fully denormalized race results table.
        """
        t = self.tables
        results = _to_int(t["results"].copy(), "raceId", "driverId", "constructorId")
        drivers = _to_int(
            t["drivers"][["driverId", "forename", "surname", "nationality", "code", "dob"]].copy(),
            "driverId",
        )
        constructors = _to_int(
            t["constructors"][["constructorId", "name", "nationality"]].copy(),
            "constructorId",
        )
        races = _to_int(
            t["races"][["raceId", "year", "round", "circuitId", "name", "date"]].copy(),
            "raceId", "year", "round",
        )
        # circuits.circuitId is a sequential integer; races.circuitId is a string ref
        # (e.g. "albert_park") — join on circuitRef instead.
        circuits = t["circuits"][["circuitRef", "name", "location", "country"]].copy()

        # Rename to avoid column collisions
        drivers = drivers.rename(columns={"nationality": "driver_nationality"})
        constructors = constructors.rename(columns={
            "name": "constructor_name",
            "nationality": "constructor_nationality",
        })
        races = races.rename(columns={"name": "race_name"})
        circuits = circuits.rename(columns={
            "name": "circuit_name",
            "location": "circuit_location",
            "country": "circuit_country",
        })

        # Join chain
        df = results.merge(drivers, on="driverId", how="left")
        df = df.merge(constructors, on="constructorId", how="left")
        df = df.merge(races, on="raceId", how="left")
        # races.circuitId holds the string ref; join to circuits.circuitRef
        df = df.merge(circuits, left_on="circuitId", right_on="circuitRef", how="left")

        # Computed columns
        df["driver_name"] = df["forename"].fillna("") + " " + df["surname"].fillna("")
        df["driver_name"] = df["driver_name"].str.strip()

        logger.info(f"Enriched results: {len(df)} rows, {len(df.columns)} columns")
        return df

    def get_enriched_driver_standings(self) -> pd.DataFrame:
        """Join driver standings with driver info."""
        t = self.tables
        standings = _to_int(t["driver_standings"].copy(), "raceId", "driverId")
        drivers = _to_int(
            t["drivers"][["driverId", "forename", "surname", "code", "nationality"]].copy(),
            "driverId",
        )

        df = standings.merge(drivers, on="driverId", how="left")
        df["driver_name"] = df["forename"].fillna("") + " " + df["surname"].fillna("")
        df["driver_name"] = df["driver_name"].str.strip()
        return df

    def get_enriched_constructor_standings(self) -> pd.DataFrame:
        """Join constructor standings with constructor info."""
        t = self.tables
        standings = _to_int(t["constructor_standings"].copy(), "raceId", "constructorId")
        constructors = _to_int(
            t["constructors"][["constructorId", "name"]].copy(),
            "constructorId",
        )
        constructors = constructors.rename(columns={"name": "constructor_name"})

        df = standings.merge(constructors, on="constructorId", how="left")
        return df
