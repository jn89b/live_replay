#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import pickle
from pymavlink import DFReader



class FlightParser:
    def __init__(self, bin_path: Path) -> None:
        self.bin_path = bin_path
        self.binary_log = DFReader.DFReader_binary(filename=str(bin_path))

        print("Message types present:",
              sorted(fmt.name for fmt in self.binary_log.formats.values()))

        self.binary_log.rewind()

    @staticmethod
    def add_rel_time(df: pd.DataFrame, col: str = "TimeUS") -> pd.DataFrame:
        if df.empty or col not in df:
            return df
        df = df.sort_values(col).copy()
        t0 = df[col].iloc[0]
        df["t"] = (df[col] - t0) * 1e-6
        return df

    def extract(
            self,
            config: Dict[str, List[str]],
            add_relative_time: bool = True,
            ) -> Dict[str, pd.DataFrame]:
        """
        Extract selected message types and columns.
        """
        rows = {msg_type: [] for msg_type in config.keys()}

        while True:
            m = self.binary_log.recv_msg()
            if m is None:
                break

            msg_type = m.get_type()
            if msg_type not in config:
                continue

            d = m.to_dict()
            d["TimeUS"] = getattr(m, "TimeUS", None)
            rows[msg_type].append(d)

        dfs: Dict[str, pd.DataFrame] = {}

        for msg_type, cols in config.items():
            df = pd.DataFrame(rows[msg_type])
            if not df.empty:
                keep_cols = [c for c in cols if c in df.columns]
                df = df[keep_cols]
                if add_relative_time:
                    df = self.add_rel_time(df)
            dfs[msg_type] = df

        return dfs

def ensure_csv_folder(bin_folder: Path) -> Path:
    """
    Create csv_files folder next to BIN folder.
    """
    csv_dir = bin_folder.parent / "csv_files"
    csv_dir.mkdir(exist_ok=True)
    return csv_dir

def save_dataframes(
    dfs: Dict[str, pd.DataFrame],
    output_dir: Path,
    prefix: str,
    ) -> None:
    """
    Save each dataframe to its own CSV.
    """
    for name, df in dfs.items():
        if df.empty:
            continue
        out_file = output_dir / f"{prefix}_{name}.csv"
        df.to_csv(out_file, index=False)

def combine_dataframes(
    dfs: Dict[str, pd.DataFrame],
    on: str = "TimeUS",
    ) -> pd.DataFrame:
    """
    Combine multiple dataframes on a shared time column.
    """
    combined = None
    for name, df in dfs.items():
        if df.empty or on not in df.columns:
            continue
        df = df.add_prefix(f"{name}_")
        df = df.rename(columns={f"{name}_{on}": on})

        combined = df if combined is None else pd.merge_asof(
            combined.sort_values(on),
            df.sort_values(on),
            on=on,
            direction="nearest",
        )

    return combined if combined is not None else pd.DataFrame()

def parse_bin(
    binaries_folder: str,
    bin_name: str,
    data_config: Dict[str, List[str]],
    combine: bool = True,
    ) -> None:
    bin_directory = Path(binaries_folder)
    bin_path = bin_directory / f"{bin_name}.BIN"

    if not bin_path.exists():
        raise FileNotFoundError(bin_path)

    csv_dir = ensure_csv_folder(bin_directory)

    parser = FlightParser(bin_path)
    data = parser.extract(data_config)

    # Save individual CSVs
    save_dataframes(data, csv_dir, bin_name)

    # Save combined CSV
    if combine:
        combined = combine_dataframes(data)
        if not combined.empty:
            combined.to_csv(csv_dir / f"{bin_name}_combined.csv", index=False)


if __name__ == "__main__":
    binaries_folder = "src/live_replay/live_replay/live_replay/binaries"
    bin_name = "00000091"

    data_config: Dict[str, List[str]] = {
        "IMU": ["TimeUS", "AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"],
        "RCOU": ["TimeUS", "C1", "C2", "C3", "C4"],
        "ATT": ["TimeUS", "Roll", "Pitch", "Yaw", "DesRoll", "DesPitch", "DesYaw"],
        "CTUN": ["TimeUS", "ThO"],
        "GPS": ["TimeUS", "Lat", "Lng", "Alt", "Spd"]
    }

    parse_bin(
        binaries_folder=binaries_folder,
        bin_name=bin_name,
        data_config=data_config,
        combine=True,
    )
