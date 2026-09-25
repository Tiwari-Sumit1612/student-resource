"""Shared plumbing: paths, per-country frames with integer ids, timing and memory measurement.

Column names always come from preprocessing.schema (never hard-coded); preprocessing/ is only read.
Integer ids: `s1` = row index in {split}_source1.parquet; `q` = row index in source2, or n(source2) + row index
in source3. Both are internal handles only, never features.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path

import polars as pl
import psutil

from preprocessing import config as PC          # read-only: file locations
from preprocessing import transforms as T       # read-only: on-demand expressions (fold_diacritics)
from preprocessing.schema import COL

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEED = PC.SEED
fold = T.fold_diacritics


def proc(split: str, source: str) -> pl.LazyFrame:
    return pl.scan_parquet(PC.processed_path(split, source))


def collect(lf: pl.LazyFrame) -> pl.DataFrame:
    return lf.collect(engine="streaming")


def n_rows(split: str, source: str) -> int:
    return collect(proc(split, source).select(pl.len())).item()


# --------------------------------------------------------------------------- frames per country block

INDEX_COLS = [COL.entity_id, COL.name_clean, COL.name_base, COL.name_domain, COL.address_numbers,
              COL.address_clean, COL.address_region]
QUERY_COLS = INDEX_COLS + [COL.name_latin, COL.name_script,
                           COL.name_is_domain_only, COL.address_missing]   # last two: evaluation slices only


def index_frame(split: str, country: str) -> pl.DataFrame:
    """S1 records of one country block (the retrieval index)."""
    lf = proc(split, "S1").with_row_index("s1").filter(pl.col(COL.country_clean) == country)
    return collect(lf.select("s1", *INDEX_COLS))


def query_frame(split: str, country: str) -> pl.DataFrame:
    """S2 + S3 records of one country block (the queries)."""
    n2 = n_rows(split, "S2")
    s2 = proc(split, "S2").with_row_index("q").with_columns(src=pl.lit("S2"))
    s3 = (proc(split, "S3").with_row_index("q")
          .with_columns(q=(pl.col("q") + n2).cast(pl.UInt32), src=pl.lit("S3")))
    lf = pl.concat([s2, s3]).filter(pl.col(COL.country_clean) == country)
    return collect(lf.select("q", "src", *QUERY_COLS))


# --------------------------------------------------------------------------- measurement

class Meter:
    """Wall time + peak RSS of a block of work (RSS sampled every 0.1 s in a thread)."""

    def __init__(self):
        self.proc = psutil.Process()
        self.seconds = 0.0
        self.peak_gb = 0.0
        self._stop = threading.Event()

    @contextmanager
    def measure(self):
        self._stop.clear()
        peak = [self.proc.memory_info().rss]

        def run():
            while not self._stop.is_set():
                peak[0] = max(peak[0], self.proc.memory_info().rss)
                time.sleep(0.1)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t0 = time.time()
        try:
            yield
        finally:
            self._stop.set()
            t.join()
            self.seconds += time.time() - t0
            self.peak_gb = max(self.peak_gb, peak[0] / 1e9)


def save(df: pl.DataFrame, name: str, sub: str = "") -> pl.DataFrame:
    out = RESULTS / sub if sub else RESULTS
    out.mkdir(parents=True, exist_ok=True)
    df.write_csv(out / f"{name}.csv")
    return df
