"""Tests for the ingestion layer — loader and chunker."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from ingestion.loader import F1DataLoader
from ingestion.chunker import F1Chunker, Chunk


@pytest.fixture
def loader():
    ld = F1DataLoader()
    ld.load_all()
    return ld


@pytest.fixture
def chunks(loader):
    chunker = F1Chunker(loader)
    return chunker.generate_all_chunks()


class TestLoader:
    def test_loads_all_tables(self, loader):
        expected = {"circuits", "constructors", "drivers", "races",
                    "results", "driver_standings", "constructor_standings"}
        assert expected == set(loader.tables.keys())

    def test_tables_not_empty(self, loader):
        for name, df in loader.tables.items():
            assert len(df) > 0, f"Table {name} is empty"

    def test_enriched_results_has_driver_name(self, loader):
        df = loader.get_enriched_results()
        assert "driver_name" in df.columns
        assert df["driver_name"].notna().all()

    def test_enriched_results_has_race_info(self, loader):
        df = loader.get_enriched_results()
        assert "race_name" in df.columns
        assert "circuit_name" in df.columns

    def test_enriched_results_joined_correctly(self, loader):
        df = loader.get_enriched_results()
        # Every result should have a constructor name from the join
        assert "constructor_name" in df.columns
        assert df["constructor_name"].notna().sum() > 0


class TestChunker:
    def test_generates_chunks(self, chunks):
        assert len(chunks) > 0

    def test_all_categories_present(self, chunks):
        categories = {c.category for c in chunks}
        expected = {"race_result", "driver_season", "constructor_season",
                    "circuit", "championship", "driver_career"}
        assert expected == categories

    def test_chunk_has_required_fields(self, chunks):
        for chunk in chunks:
            assert chunk.chunk_id, "chunk_id must not be empty"
            assert chunk.category, "category must not be empty"
            assert chunk.content, "content must not be empty"
            assert isinstance(chunk.metadata, dict)

    def test_chunk_ids_unique(self, chunks):
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_race_chunks_contain_winner(self, chunks):
        race_chunks = [c for c in chunks if c.category == "race_result"]
        for rc in race_chunks:
            assert "Winner:" in rc.content, f"Race chunk missing winner: {rc.chunk_id}"

    def test_driver_career_chunks_have_stats(self, chunks):
        career_chunks = [c for c in chunks if c.category == "driver_career"]
        for cc in career_chunks:
            assert "Wins:" in cc.content
            assert "Podiums:" in cc.content
            assert "Total Points:" in cc.content

    def test_championship_chunks_have_champion(self, chunks):
        champ_chunks = [c for c in chunks if c.category == "championship"]
        for cc in champ_chunks:
            assert "Champion" in cc.content

    def test_to_dict(self, chunks):
        d = chunks[0].to_dict()
        assert "chunk_id" in d
        assert "category" in d
        assert "content" in d
        assert "metadata" in d
