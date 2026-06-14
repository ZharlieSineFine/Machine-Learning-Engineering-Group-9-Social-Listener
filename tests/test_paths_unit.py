"""Unit tests for workspace dataset path resolution."""
from __future__ import annotations

from pathlib import Path

from data.paths import (
    DEFAULT_TRIPADVISOR_CSV,
    DEFAULT_YELP_TAR,
    ROOT,
    tripadvisor_csv_path,
    yelp_tar_path,
)


def test_default_paths_are_under_workspace_root():
    workspace = ROOT.parent.parent
    assert DEFAULT_YELP_TAR == (workspace / "Yelp_JSON" / "yelp_dataset" / "yelp_dataset").resolve()
    assert DEFAULT_TRIPADVISOR_CSV == (
        workspace / "TripAdvisor_data_cleaned.csv" / "TripAdvisor_data_cleaned.csv"
    ).resolve()


def test_workspace_datasets_exist_when_checked_in():
    if DEFAULT_YELP_TAR.is_file():
        assert yelp_tar_path() == DEFAULT_YELP_TAR
    if DEFAULT_TRIPADVISOR_CSV.is_file():
        assert tripadvisor_csv_path() == DEFAULT_TRIPADVISOR_CSV
