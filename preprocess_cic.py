"""Basic preprocessing for CIC network-flow CSV data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROBLEMATIC_COLUMNS = {"Flow ID", "Src IP", "Dst IP", "Timestamp"}
LABEL_COLUMN = "Label"


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
    }
    wrote_header = False

    for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
        stats["input_rows"] += len(chunk)
        columns_to_drop = [column for column in PROBLEMATIC_COLUMNS if column in chunk.columns]
        if not wrote_header:
            stats["removed_columns"] = len(columns_to_drop)

        chunk = chunk.drop(columns=columns_to_drop)
        if LABEL_COLUMN not in chunk.columns:
            raise ValueError(f"The input must contain a '{LABEL_COLUMN}' column.")

        numeric_columns = chunk.select_dtypes(include=[np.number]).columns
        chunk[numeric_columns] = chunk[numeric_columns].replace([np.inf, -np.inf], np.nan)

        # Empty strings and common textual sentinels are invalid in numeric fields.
        for column in chunk.columns:
            if column == LABEL_COLUMN or pd.api.types.is_numeric_dtype(chunk[column]):
                continue
            chunk[column] = chunk[column].replace(
                {"": np.nan, "inf": np.nan, "-inf": np.nan, "Infinity": np.nan, "-Infinity": np.nan}
            )

        before = len(chunk)
        chunk = chunk.dropna(axis=0, how="any")
        stats["removed_rows"] += before - len(chunk)
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