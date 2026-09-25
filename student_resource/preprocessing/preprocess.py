"""Stage 2 preprocessing pipeline (no matching logic).

Run from the student_resource/ directory:

    python -m preprocessing.preprocess            # process -> validate -> measure
    python -m preprocessing.preprocess --clean    # delete processed/ and results/ first (reproducibility run)
    python -m preprocessing.preprocess --skip-measure

Raw TSVs are only ever read. Every source file is streamed through a lazy Polars plan into one
Parquet file (written to a temp name, then atomically renamed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import polars as pl
import psutil

from . import config as C
from . import transforms as T

log = logging.getLogger("preprocess")


# --------------------------------------------------------------------------- infrastructure

class MemoryMonitor:
    """Samples this process's RSS in a background thread. `peak_gb` is the run maximum;
    `window` is the maximum since the last `reset_window()` (used for per-step peaks)."""

    def __init__(self, interval: float = 0.2):
        self.proc, self.interval, self.peak, self.window = psutil.Process(), interval, 0, 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss
            self.peak, self.window = max(self.peak, rss), max(self.window, rss)
            time.sleep(self.interval)

    def reset_window(self):
        self.window = self.proc.memory_info().rss

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    @property
    def peak_gb(self) -> float:
        return round(self.peak / 1e9, 2)


MONITOR: "MemoryMonitor | None" = None
STEP_PEAKS: dict = {}


@contextmanager
def step(name: str, timings: dict):
    t0 = time.time()
    if MONITOR:
        MONITOR.reset_window()
    log.info("START %s", name)
    try:
        yield
    except Exception:
        log.exception("FAILED %s", name)
        raise
    timings[name] = round(time.time() - t0, 1)
    peak = MONITOR.window / 1e9 if MONITOR else float("nan")
    STEP_PEAKS[name] = round(peak, 2)
    log.info("DONE  %s (%.1fs, step peak rss %.2f GB)", name, timings[name], peak)


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def raw_files() -> list[Path]:
    return [C.raw_path(sp, s) for sp in C.SPLITS for s in C.SOURCES] + [C.GT_RAW]


def scan_raw(split: str, source: str) -> pl.LazyFrame:
    """Strict UTF-8, no quoting (fields contain quotes/commas), no type inference."""
    return pl.scan_csv(C.raw_path(split, source), separator="\t", quote_char=None,
                       infer_schema=False, encoding="utf8")


# --------------------------------------------------------------------------- record transformation

def build(lf: pl.LazyFrame, split: str, source: str) -> pl.LazyFrame:
    """Raw source frame -> processed frame. Pure function of its input (deterministic)."""
    missing = lambda c: pl.col(c).str.strip_chars().str.len_chars().fill_null(0) == 0
    lf = lf.select(
        pl.col("entity_id"),
        pl.lit(split).alias("split"),
        pl.lit(source).alias("source"),
        pl.col("business_name").alias("business_name_raw"),
        pl.col("business_address").alias("business_address_raw"),
        pl.col("country").alias("country_raw"),
    )
    lf = lf.with_columns(
        country_clean=T.country_clean("country_raw"),
        name_missing=missing("business_name_raw"),
        address_missing=missing("business_address_raw"),
        country_missing=missing("country_raw"),
        name_norm=T.norm("business_name_raw"),
        address_norm=T.norm("business_address_raw"),
    )
    lf = lf.with_columns(
        name_clean=T.punctuation_stage("name_norm"),
        address_segments=T.address_segments_clean("address_norm"),
    )
    lf = lf.with_columns(
        name_tokens=T.tokens("name_clean"),
        address_clean=pl.when(pl.col("address_segments").list.len() > 0).then(pl.col("address_segments").list.join(" ")),
    )
    lf = lf.with_columns(
        address_tokens=T.tokens("address_clean"),
        name_legal_forms=T.legal_forms("name_tokens"),
        name_base=T.name_base("name_tokens"),
        name_domain=T.domain("name_norm"),
        name_numbers=T.numbers("name_norm"),
        address_numbers=T.numbers(T.address_placeholder_stage("address_norm"), strip_leading_zeros=True),
        postal_code_candidate=T.postal_code_candidate("address_norm", "country_clean"),
        name_script=T.script_label("business_name_raw"),
        address_script=T.script_label("business_address_raw"),
    )
    n_raw, a_raw, a_norm = pl.col("business_name_raw"), pl.col("business_address_raw"), pl.col("address_norm")
    lf = lf.with_columns(
        # ---- name flags (descriptive only)
        name_is_upper=T.is_upper(n_raw),
        name_is_lower=T.is_lower(n_raw),
        name_has_digits=n_raw.str.contains(r"\d"),
        name_has_non_latin=n_raw.str.contains(C.NON_LATIN_LETTER),
        name_has_latin_diacritics=n_raw.str.contains(C.LATIN_DIACRITIC),
        name_has_domain=pl.col("name_domain").is_not_null(),
        name_is_domain_only=pl.col("name_norm").str.contains(T.DOMAIN_ONLY_RE),
        name_has_special_chars=n_raw.str.contains(C.NAME_SPECIAL_CHARS),
        name_has_brackets=n_raw.str.contains(C.BRACKETS),
        name_has_leading_junk=n_raw.str.contains(r"^[^\p{L}\p{N}]"),
        name_has_repeated_whitespace=n_raw.str.contains(r"\s{2,}"),
        name_has_encoding_artifact=T.encoding_artifact(n_raw),
        name_has_legal_form=pl.col("name_legal_forms").list.len() > 0,
        name_legal_at_start=T.is_legal_token(pl.col("name_tokens").list.first()),
        name_legal_only=pl.col("name_clean").is_not_null() & pl.col("name_base").is_null(),
        name_clean_empty=~pl.col("name_missing") & pl.col("name_clean").is_null(),
        # ---- address flags
        address_is_upper=T.is_upper(a_raw),
        address_has_digits=a_raw.str.contains(r"\d"),
        address_has_non_latin=a_raw.str.contains(C.NON_LATIN_LETTER),
        address_has_latin_diacritics=a_raw.str.contains(C.LATIN_DIACRITIC),
        address_has_special_chars=a_raw.str.contains(C.ADDR_SPECIAL_CHARS),
        address_has_encoding_artifact=T.encoding_artifact(a_raw),
        address_has_placeholder=a_norm.str.split(",").list.eval(
            pl.element().str.strip_chars().str.contains(C.ADDRESS_PLACEHOLDER_SEGMENT)).list.any(),
        address_placeholder_only=~pl.col("address_missing") & pl.col("address_clean").is_null(),
        address_has_zero_padded_number=a_norm.str.contains(r"(?:^|\D)0\d"),
        address_has_landmark=a_norm.str.contains(C.LANDMARK),
        address_has_care_of=a_norm.str.contains(C.CARE_OF),
        # ---- counts
        name_n_tokens=pl.col("name_tokens").list.len().cast(pl.UInt16),
        address_n_tokens=pl.col("address_tokens").list.len().cast(pl.UInt16),
        address_n_segments=pl.col("address_segments").list.len().cast(pl.UInt16),
    )
    # ---- auxiliary representations added by PREPROCESSING_AUDIT section 9 (appended; nothing above changes)
    lf = lf.with_columns(
        address_region=T.address_region("address_segments", "country_clean"),
        name_latin=T.name_latin("name_clean", "name_script"),
    )
    lf = lf.with_columns(address_region_found=pl.col("address_region").is_not_null())
    return lf


COLUMN_ORDER = [
    "entity_id", "split", "source",
    "business_name_raw", "business_address_raw", "country_raw",
    "country_clean", "name_missing", "address_missing", "country_missing",
    "name_norm", "name_clean", "name_tokens", "name_base", "name_legal_forms", "name_domain", "name_numbers", "name_script",
    "address_norm", "address_clean", "address_tokens", "address_segments", "address_numbers", "postal_code_candidate", "address_script",
    "name_n_tokens", "address_n_tokens", "address_n_segments",
]


def process_file(split: str, source: str) -> dict:
    out = C.processed_path(split, source)
    tmp = out.with_suffix(".parquet.tmp")
    tmp.unlink(missing_ok=True)
    lf = build(scan_raw(split, source), split, source)
    names = lf.collect_schema().names()
    lf = lf.select(COLUMN_ORDER + [c for c in names if c not in COLUMN_ORDER])
    lf.sink_parquet(tmp, **C.PARQUET)
    os.replace(tmp, out)                                    # atomic: a failed run never leaves a half file
    rows = pl.scan_parquet(out).select(pl.len()).collect().item()
    return {"split": split, "source": source, "output": out.name, "rows_out": rows,
            "size_mb": round(out.stat().st_size / 1e6, 1), "raw_size_mb": round(C.raw_path(split, source).stat().st_size / 1e6, 1)}


def process_ground_truth() -> dict:
    """Format conversion only (TSV -> Parquet with a parsed id list). Not used to derive any rule."""
    tmp = C.GT_PROCESSED.with_suffix(".parquet.tmp")
    ids = (pl.col("matched_entity_ids_raw").str.split(",")
           .list.eval(pl.element().str.strip_chars()).list.eval(pl.element().filter(pl.element().str.len_chars() > 0)))
    lf = (pl.scan_csv(C.GT_RAW, separator="\t", quote_char=None, infer_schema=False, encoding="utf8")
          .rename({"matched_entity_ids": "matched_entity_ids_raw"})
          .with_columns(matched_entity_ids=ids)
          .with_columns(n_matches=pl.col("matched_entity_ids").list.len().fill_null(0).cast(pl.UInt8),
                        n_s2=pl.col("matched_entity_ids").list.eval(pl.element().str.starts_with("S2-")).list.sum().fill_null(0).cast(pl.UInt8),
                        n_s3=pl.col("matched_entity_ids").list.eval(pl.element().str.starts_with("S3-")).list.sum().fill_null(0).cast(pl.UInt8)))
    lf.sink_parquet(tmp, **C.PARQUET)
    os.replace(tmp, C.GT_PROCESSED)
    return {"split": "train", "source": "GT", "output": C.GT_PROCESSED.name,
            "rows_out": pl.scan_parquet(C.GT_PROCESSED).select(pl.len()).collect().item(),
            "size_mb": round(C.GT_PROCESSED.stat().st_size / 1e6, 1), "raw_size_mb": round(C.GT_RAW.stat().st_size / 1e6, 1)}


def fingerprint(path: Path) -> dict:
    """Order-independent content fingerprint of every column (sum of row hashes mod a prime)."""
    lf = pl.scan_parquet(path)
    cols = lf.collect_schema().names()
    r = lf.select([(pl.col(c).hash(C.SEED) % 1_000_000_007).sum().alias(c) for c in cols]).collect(engine="streaming")
    return {c: int(v) for c, v in r.row(0, named=True).items()}


# --------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", action="store_true", help="delete processed/ and preprocessing/results/ first")
    ap.add_argument("--skip-measure", action="store_true")
    args = ap.parse_args(argv)

    if args.clean:
        for d in (C.PROCESSED_DIR, C.RESULTS_DIR):
            shutil.rmtree(d, ignore_errors=True)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(C.RESULTS_DIR / "run.log", mode="w", encoding="utf-8")])
    global T_START
    T_START = time.time()
    timings: dict = {}
    manifest = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(),
                "polars": pl.__version__, "platform": platform.platform(), "clean_start": args.clean}

    global MONITOR
    with MemoryMonitor() as mem:
        MONITOR = mem
        with step("hash raw files (before)", timings):
            manifest["raw_sha256_before"] = {p.name: sha256(p) for p in raw_files()}

        outputs = []
        for sp in C.SPLITS:
            for s in C.SOURCES:
                with step(f"process {sp} {s}", timings):
                    outputs.append(process_file(sp, s))
        with step("process ground truth", timings):
            outputs.append(process_ground_truth())
        pl.DataFrame(outputs).write_csv(C.RESULTS_DIR / "output_files.csv")

        with step("fingerprint outputs", timings):
            manifest["output_fingerprints"] = {o["output"]: fingerprint(C.PROCESSED_DIR / o["output"]) for o in outputs}

        from . import validate
        with step("validate", timings):
            checks = validate.run_all()
        failed = checks.filter(~pl.col("passed"))

        if not args.skip_measure:
            from . import measure
            measure.run_all(step=lambda n: step(f"measure: {n}", timings))

        with step("hash raw files (after)", timings):
            manifest["raw_sha256_after"] = {p.name: sha256(p) for p in raw_files()}
        manifest["raw_files_unchanged"] = manifest["raw_sha256_before"] == manifest["raw_sha256_after"]

    manifest.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), timings_s=timings, step_peak_rss_gb=STEP_PEAKS,
                    total_s=round(time.time() - T_START, 1), peak_rss_gb=mem.peak_gb,
                    validation_failed=failed.height, outputs=outputs)
    (C.RESULTS_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("peak RSS %.2f GB | total %.0fs | raw unchanged: %s | failed checks: %d",
             mem.peak_gb, manifest["total_s"], manifest["raw_files_unchanged"], failed.height)
    if failed.height or not manifest["raw_files_unchanged"]:
        log.error("VALIDATION FAILED:\n%s", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
