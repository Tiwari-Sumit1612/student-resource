"""Sanity checks proving that preprocessing did not lose or alter information.

Every check yields one row: check, split, source, expected, observed, passed.
Results -> results/validation_checks.csv
"""
from __future__ import annotations

import polars as pl

from . import config as C
from . import transforms as T

RAW_TO_PROCESSED = {"business_name": "business_name_raw", "business_address": "business_address_raw", "country": "country_raw"}


def _raw(split, source):
    return pl.scan_csv(C.raw_path(split, source), separator="\t", quote_char=None, infer_schema=False, encoding="utf8")


def _proc(split, source):
    return pl.scan_parquet(C.processed_path(split, source))


def _c(lf):
    return lf.collect(engine="streaming")


def checks_for_file(split: str, source: str) -> list[dict]:
    raw, proc = _raw(split, source), _proc(split, source)
    rows = []

    def add(check, expected, observed, passed=None):
        rows.append({"check": check, "split": split, "source": source, "expected": str(expected),
                     "observed": str(observed), "passed": bool(expected == observed if passed is None else passed)})

    # 1. row counts
    n_raw = _c(raw.select(pl.len())).item()
    n_out = _c(proc.select(pl.len())).item()
    add("row count unchanged", n_raw, n_out)

    # 2. ids: unique, same set, correct prefix
    add("entity_id unique", n_out, _c(proc.select(pl.col("entity_id").n_unique())).item())
    add("raw ids missing from output", 0, _c(raw.select("entity_id").join(proc.select("entity_id"), on="entity_id", how="anti").select(pl.len())).item())
    add("output ids not in raw", 0, _c(proc.select("entity_id").join(raw.select("entity_id"), on="entity_id", how="anti").select(pl.len())).item())
    add("entity_id prefix matches source", 0, _c(proc.select((~pl.col("entity_id").str.starts_with(f"{source}-")).sum())).item())
    add("row order identical to raw file (row i = raw row i)", True,
        _c(raw.select("entity_id"))["entity_id"].equals(_c(proc.select("entity_id"))["entity_id"]))
    add("split/source columns constant", 1,
        _c(proc.select(((pl.col("split") == split) & (pl.col("source") == source)).all().cast(pl.Int8))).item())

    # 3. raw fields byte-identical (row-level join on entity_id, null-safe comparison)
    joined = raw.join(proc.select("entity_id", *RAW_TO_PROCESSED.values()), on="entity_id", how="inner")
    mism = _c(joined.select([(~pl.col(r).eq_missing(pl.col(p))).sum().alias(r) for r, p in RAW_TO_PROCESSED.items()])).row(0, named=True)
    for r, v in mism.items():
        add(f"raw field preserved: {r}", 0, v)
    add("no unicode replacement chars in raw fields (strict UTF-8 read)", 0,
        _c(proc.select(sum(pl.col(c).str.contains("�", literal=True).fill_null(False).sum() for c in RAW_TO_PROCESSED.values()))).item())

    # 4. country distribution unchanged
    d_raw = _c(raw.group_by("country").len()).rename({"country": "k"})
    d_out = _c(proc.group_by("country_clean").len()).rename({"country_clean": "k"})
    diff = d_raw.join(d_out, on="k", how="full", coalesce=True).filter(~pl.col("len").eq_missing(pl.col("len_right"))).height
    add("country distribution unchanged (raw vs country_clean)", 0, diff)
    add("country_clean differs from stripped country_raw", 0,
        _c(proc.select((~pl.col("country_clean").eq_missing(pl.col("country_raw").str.strip_chars())).sum())).item())

    # 5. missingness understood
    miss = _c(raw.select([pl.col(c).is_null().sum().alias(c) for c in RAW_TO_PROCESSED])).row(0, named=True)
    flags = _c(proc.select(pl.col("name_missing").sum(), pl.col("address_missing").sum(), pl.col("country_missing").sum())).row(0)
    for (c, v), f in zip(miss.items(), flags):
        add(f"missing flag == raw nulls: {c}", v, f)

    # 6. no silent information loss between representations
    s = _c(proc.select(
        norm_lost_name=(pl.col("business_name_raw").is_not_null() & pl.col("name_norm").is_null()).sum(),
        norm_lost_addr=(pl.col("business_address_raw").is_not_null() & pl.col("address_norm").is_null()).sum(),
        clean_empty_name_unflagged=(pl.col("name_norm").is_not_null() & pl.col("name_clean").is_null() & ~pl.col("name_clean_empty")).sum(),
        clean_empty_addr_unflagged=(pl.col("address_norm").is_not_null() & pl.col("address_clean").is_null() & ~pl.col("address_placeholder_only")).sum(),
        tokens_mismatch=(pl.col("name_tokens").list.join(" ") != pl.col("name_clean")).sum()
                        + (pl.col("address_tokens").list.join(" ") != pl.col("address_clean")).sum(),
        name_digits_lost=(pl.col("name_numbers").list.join("") != pl.col("name_clean").str.replace_all(r"\D", "")).sum(),
    )).row(0, named=True)
    add("raw name non-null -> name_norm non-null", 0, s["norm_lost_name"])
    add("raw address non-null -> address_norm non-null", 0, s["norm_lost_addr"])
    add("empty name_clean always flagged (name_clean_empty)", 0, s["clean_empty_name_unflagged"])
    add("empty address_clean always flagged (address_placeholder_only)", 0, s["clean_empty_addr_unflagged"])
    add("tokens rejoin exactly to clean string", 0, s["tokens_mismatch"])
    add("all digits of name_norm survive in name_clean (same order)", 0, s["name_digits_lost"])

    # 7. determinism: rebuild a 1% sample twice from the raw file and compare with the stored rows
    from .preprocess import COLUMN_ORDER, build
    sample = raw.filter(pl.col("entity_id").hash(C.SEED) % 100 == 0)
    a = _c(build(sample, split, source)).sort("entity_id")
    b = _c(build(sample, split, source)).sort("entity_id")
    stored = _c(proc.filter(pl.col("entity_id").hash(C.SEED) % 100 == 0)).sort("entity_id").select(a.columns)
    add("deterministic: two rebuilds of 1% sample identical", True, a.equals(b))
    add("deterministic: rebuild equals stored parquet (1% sample)", True, a.equals(stored))
    return rows


def ground_truth_checks() -> list[dict]:
    gt = pl.scan_parquet(C.GT_PROCESSED)
    raw = pl.scan_csv(C.GT_RAW, separator="\t", quote_char=None, infer_schema=False)
    rows = []

    def add(check, expected, observed):
        rows.append({"check": check, "split": "train", "source": "GT", "expected": str(expected),
                     "observed": str(observed), "passed": expected == observed})

    add("GT row count unchanged", _c(raw.select(pl.len())).item(), _c(gt.select(pl.len())).item())
    add("GT raw id lists preserved", 0, _c(raw.join(gt, on="source1_entity_id").select(
        (~pl.col("matched_entity_ids").eq_missing(pl.col("matched_entity_ids_raw"))).sum())).item())
    pairs = gt.select("source1_entity_id", m="matched_entity_ids").explode("m").drop_nulls("m")
    other = pl.concat([_proc("train", "S2").select("entity_id"), _proc("train", "S3").select("entity_id")])
    add("GT S1 ids all present in processed train S1", 0,
        _c(gt.select("source1_entity_id").join(_proc("train", "S1").select(source1_entity_id="entity_id"), on="source1_entity_id", how="anti").select(pl.len())).item())
    add("GT matched ids all present in processed train S2/S3", 0,
        _c(pairs.join(other.select(m="entity_id"), on="m", how="anti").select(pl.len())).item())
    add("GT pair count (raw ids = parsed ids)", _c(raw.select(pl.col("matched_entity_ids").str.split(",").list.len().sum())).item(),
        _c(pairs.select(pl.len())).item())
    return rows


def run_all() -> pl.DataFrame:
    rows = []
    for sp in C.SPLITS:
        for s in C.SOURCES:
            rows += checks_for_file(sp, s)
    rows += ground_truth_checks()
    df = pl.DataFrame(rows)
    df.write_csv(C.RESULTS_DIR / "validation_checks.csv")
    return df
