import os
from pathlib import Path
from typing import List
import pandas as pd

def get_directory_structure(root_dir: str,
                        max_files_per_dir: int = 3,
                        sample_rows: int = 5) -> str:
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise ValueError(f"'{root_dir}' is not a valid directory.")

    lines: List[str] = []
    root_name = root_dir.name
    lines.append(f"Data directory structure {root_dir}:")

    csv_paths: List[Path] = []

    for dirpath, _, filenames in os.walk(root_dir):
        dirpath = Path(dirpath)
        # Exclude description.txt files
        filenames = [fname for fname in filenames if fname.lower() != "description.txt"]
        filenames.sort()
        rel_dir = dirpath.relative_to(root_dir)
        rel_dir_str = "" if rel_dir == Path(".") else str(rel_dir)

        # Process only first max_files_per_dir files
        for fname in filenames[:max_files_per_dir]:
            rel_file = (rel_dir / fname) if rel_dir_str else Path(fname)
            lines.append(str(rel_file))

            if fname.lower().endswith(".csv"):
                csv_paths.append(rel_file)

        if len(filenames) > max_files_per_dir:
            prefix = f"{rel_dir_str}\\" if rel_dir_str else ""
            lines.append(f"{prefix}...")

        # Process remaining files for csv collection
        for fname in filenames[max_files_per_dir:]:
            if fname.lower().endswith(".csv"):
                rel_file = (rel_dir / fname) if rel_dir_str else Path(fname)
                csv_paths.append(rel_file)

    if csv_paths:
        lines.append("\n" + "="*60)
        lines.append("SUMMARY OF CSV FILES")
        lines.append("="*60)

        for rel_path in csv_paths:
            abs_path = root_dir / rel_path
            try:
                df = pd.read_csv(abs_path, nrows=sample_rows)
            except Exception as e:
                lines.append(f"\nCould not read '{rel_path}': {e}")
                continue

            lines.append(f"\nStructure of file {rel_path}:")
            lines.append("Columns: " + ", ".join(df.columns.astype(str)))
            lines.append("Some first rows:")
            lines.append(df.to_string(index=False))

    summary_str = "\n".join(lines)
    return summary_str