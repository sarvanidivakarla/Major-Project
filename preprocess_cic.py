"""Basic preprocessing for CIC network-flow CSV data."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


PROBLEMATIC_COLUMNS = {"Flow ID", "Src IP", "Dst IP"}
LABEL_COLUMN = "Label"


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Normalize invalid values before duplicate detection."""
    numeric_columns = chunk.select_dtypes(include=[np.number]).columns
    chunk[numeric_columns] = chunk[numeric_columns].replace([np.inf, -np.inf], np.nan)

    for column in chunk.columns:
        if column == LABEL_COLUMN or pd.api.types.is_numeric_dtype(chunk[column]):
            continue
        chunk[column] = chunk[column].replace(
            {"": np.nan, "inf": np.nan, "-inf": np.nan, "Infinity": np.nan, "-Infinity": np.nan}
        )
    return chunk


def column_fingerprints(input_path: Path, chunk_size: int) -> tuple[list[str], dict[str, str]]:
    """Create stable fingerprints for each cleaned column using two streaming passes."""
    column_hashes = {}
    columns: list[str] = []
    for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
        chunk = clean_chunk(chunk)
        if not columns:
            columns = list(chunk.columns)
            column_hashes = {column: hashlib.sha256() for column in columns}
        for column in columns:
            values = pd.util.hash_pandas_object(chunk[column], index=False).to_numpy()
            column_hashes[column].update(values.tobytes())

    return columns, {column: digest.hexdigest() for column, digest in column_hashes.items()}


def preprocess(input_path: Path, output_path: Path, chunk_size: int = 100_000) -> dict[str, int]:
    """Clean a CIC CSV in chunks and return row-level processing statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    stats = {
        "input_rows": 0,
        "output_rows": 0,
        "removed_rows": 0,
        "removed_columns": 0,
        "duplicate_rows_removed": 0,
        "duplicate_columns_removed": 0,
    }
    columns, fingerprints = column_fingerprints(input_path, chunk_size)
    kept_columns: list[str] = []
    seen_column_fingerprints: set[str] = set()
    for column in columns:
        if column == LABEL_COLUMN or fingerprints[column] not in seen_column_fingerprints:
            kept_columns.append(column)
            seen_column_fingerprints.add(fingerprints[column])
    columns_to_drop = [column for column in columns if column not in kept_columns]
    stats["duplicate_columns_removed"] = len(columns_to_drop)
    seen_rows: set[int] = set()
    wrote_header = False

    for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
        stats["input_rows"] += len(chunk)
        problematic_columns = [column for column in PROBLEMATIC_COLUMNS if column in chunk.columns]
        if not wrote_header:
            stats["removed_columns"] = len(problematic_columns) + len(columns_to_drop)

        chunk = chunk.drop(columns=problematic_columns + columns_to_drop)
        if LABEL_COLUMN not in chunk.columns:
            raise ValueError(f"The input must contain a '{LABEL_COLUMN}' column.")

        chunk = clean_chunk(chunk)

        before = len(chunk)
        chunk = chunk.dropna(axis=0, how="any")
        stats["removed_rows"] += before - len(chunk)

        row_hashes = pd.util.hash_pandas_object(chunk, index=False)
        unique_rows = []
        for row_hash in row_hashes:
            is_unique = row_hash not in seen_rows
            unique_rows.append(is_unique)
            seen_rows.add(row_hash)
        stats["duplicate_rows_removed"] += len(chunk) - sum(unique_rows)
        chunk = chunk.loc[unique_rows]
        stats["output_rows"] += len(chunk)
        chunk.to_csv(output_path, mode="a", index=False, header=not wrote_header)
        wrote_header = True

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the raw CIC CSV")
    parser.add_argument("output", type=Path, help="Path for the cleaned CSV")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    stats = preprocess(args.input, args.output, args.chunk_size)
    print("Preprocessing complete")
    for name, value in stats.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()