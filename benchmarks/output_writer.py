"""Output writer for different formats (JSON, JSONL, CSV, Parquet, etc.)"""

import csv
import json
import os
from typing import Any, Dict, Optional

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class OutputWriter:
    """Handles writing aggregate benchmark statistics to various output formats."""

    SUPPORTED_FORMATS = ["json", "jsonl", "csv", "parquet"]

    def __init__(
        self, output_format: Optional[str] = None, output_file: Optional[str] = None
    ):
        """
        Initialize output writer.

        Args:
            output_format: Format to write (json, jsonl, csv, parquet, etc.)
            output_file: Path to output file
        """
        self.output_format = output_format.lower() if output_format else None
        self.output_file = output_file
        self.aggregate_results = []  # Store aggregate statistics, not individual records

        if self.output_format and self.output_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported output format: {output_format}. Supported: {self.SUPPORTED_FORMATS}"
            )

        if self.output_format in ["parquet"] and not HAS_PANDAS:
            raise ImportError(
                f"Format '{output_format}' requires pandas. Install: pip install pandas pyarrow"
            )

    def add_aggregate_result(self, result: Dict[str, Any]):
        """Add an aggregate result (statistics for one benchmark run)."""
        if self.output_format:
            self.aggregate_results.append(result)

    def write(self, mode="append"):
        """
        Write accumulated aggregate results to the output file.

        Args:
            mode: 'append' to add to existing file, 'overwrite' to replace
        """
        if not self.output_format or not self.output_file or not self.aggregate_results:
            return

        # Create output directory if it doesn't exist
        out_dir = os.path.dirname(self.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Load existing data if appending
        existing_data = []
        if mode == "append" and os.path.exists(self.output_file):
            existing_data = self._load_existing()

        # Combine existing and new data
        all_data = existing_data + self.aggregate_results

        if self.output_format == "json":
            self._write_json(all_data)
        elif self.output_format == "jsonl":
            self._write_jsonl(all_data, append=(mode == "append"))
        elif self.output_format == "csv":
            self._write_csv(
                all_data, append=(mode == "append" and len(existing_data) > 0)
            )
        elif self.output_format == "parquet":
            self._write_parquet(all_data)

    def _load_existing(self):
        """Load existing data from file for appending."""
        if not os.path.exists(self.output_file):
            return []

        try:
            if self.output_format == "json":
                with open(self.output_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            elif self.output_format == "jsonl":
                data = []
                with open(self.output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                return data
            elif self.output_format == "csv":
                with open(self.output_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    return list(reader)
            elif self.output_format == "parquet" and HAS_PANDAS:
                df = pd.read_parquet(self.output_file)
                return df.to_dict("records")
        except Exception:
            return []
        return []

    def _write_json(self, data):
        """Write data as a single JSON array."""
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _write_jsonl(self, data, append=False):
        """Write data as JSON Lines (one JSON object per line)."""
        mode = "a" if append else "w"
        with open(self.output_file, mode, encoding="utf-8") as f:
            for record in data if not append else self.aggregate_results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_csv(self, data, append=False):
        """Write data as CSV with proper escaping."""
        if not data:
            return

        # Flatten data for CSV
        flat_data = [self._flatten_dict(r) for r in data]

        # Get all unique keys
        all_keys = set()
        for record in flat_data:
            all_keys.update(record.keys())

        fieldnames = sorted(all_keys)

        mode = "a" if append else "w"
        with open(self.output_file, mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            if not append:
                writer.writeheader()
            # Only write new rows if appending
            rows_to_write = (
                flat_data
                if not append
                else [self._flatten_dict(r) for r in self.aggregate_results]
            )
            writer.writerows(rows_to_write)

    def _write_parquet(self, data):
        """Write data as Parquet file using pandas."""
        if not HAS_PANDAS:
            raise ImportError(
                "Parquet format requires pandas. Install: pip install pandas pyarrow"
            )

        # Flatten data for DataFrame
        flat_data = [self._flatten_dict(r) for r in data]
        df = pd.DataFrame(flat_data)
        df.to_parquet(self.output_file, index=False, engine="pyarrow")

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """Recursively flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Skip lists in aggregate data (they're usually not needed for stats)
                continue
            else:
                items.append((new_key, v))
        return dict(items)
