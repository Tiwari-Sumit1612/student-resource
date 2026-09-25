"""Evaluation for PREPROCESSING_AUDIT.md section 9 (steps 1-4).

Uses the ground truth for EVALUATION ONLY. Nothing in here is imported by the preprocessing
pipeline, and no rule in config.py / transforms.py is derived from these numbers.

    python -m preprocessing.audit_eval rows      # step 1: row-order leakage
    python -m preprocessing.audit_eval region    # step 2: address_region pass bars
    python -m preprocessing.audit_eval latin     # step 3: anyascii vs ITRANS, name_latin
    python -m preprocessing.audit_eval legal     # step 4: legal-form coverage by script
    python -m preprocessing.audit_eval all

Outputs: preprocessing/results/audit/*.csv
"""
from __future__ import annotations

import sys

import polars as pl

from . import config as C
from . import transforms as T

OUT = C.RESULTS_DIR / "audit"
SEED = C.SEED


def _proc(split, source) -> pl.LazyFrame:
    return pl.scan_parquet(C.processed_path(split, source))


def _c(lf: pl.LazyFrame) -> pl.DataFrame:
    return lf.collect(engine="streaming")


def _save(df: pl.DataFrame, name: str) -> pl.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUT / f"{name}.csv")
    print(f"\n== {name}")
    with pl.Config(tbl_rows=80, tbl_cols=20, fmt_str_lengths=80, tbl_width_chars=250, tbl_hide_dataframe_shape=True):
        print(df)
    return df


def _gt_pairs() -> pl.LazyFrame:
    return (pl.scan_parquet(C.GT_PROCESSED).select(s1_id="source1_entity_id", m_id="matched_entity_ids")
            .explode("m_id").drop_nulls("m_id"))


# =========================================================================== step 1: row-order leakage

def row_order_leakage() -> pl.DataFrame:
    """Does the raw row position (or the numeric id) of an S2/S3 record carry information about
    the row position of its S1 entity? Compared with a random pairing of the same size."""
    n = {s: _c(_proc("train", s).select(pl.len())).item() for s in C.SOURCES}
    pos = {s: _proc("train", s).select("entity_id").with_row_index("row") for s in C.SOURCES}
    s1 = pos["S1"].select(s1_id="entity_id", i="row")
    other = pl.concat([pos[s].select(m_id="entity_id", j="row", src=pl.lit(s)) for s in ("S2", "S3")])
    pairs = _c(_gt_pairs().join(s1, on="s1_id").join(other, on="m_id"))
    num = lambda c: pl.col(c).str.extract(r"-(\d+)$", 1).cast(pl.Int64)
    pairs = pairs.with_columns(
        u=pl.col("i") / n["S1"],
        v=pl.col("j") / pl.when(pl.col("src") == "S2").then(n["S2"]).otherwise(n["S3"]),
        id_a=num("s1_id"), id_b=num("m_id"))

    # random baseline: same S1 rows, match positions shuffled within source (deterministic)
    rnd = pairs.with_columns(
        v=pl.col("v").shuffle(seed=SEED).over("src"), j=pl.col("j").shuffle(seed=SEED).over("src"),
        id_b=pl.col("id_b").shuffle(seed=SEED).over("src"))

    def stats(df, kind):
        return df.group_by("src").agg(
            pairing=pl.lit(kind), pairs=pl.len(),
            spearman_row_position=pl.corr("i", "j", method="spearman").round(5),
            spearman_numeric_id=pl.corr("id_a", "id_b", method="spearman").round(5),
            mean_abs_position_gap=(pl.col("u") - pl.col("v")).abs().mean().round(5),
            pct_within_0_1pct_of_file=(((pl.col("u") - pl.col("v")).abs() < 0.001).mean() * 100).round(4),
            pct_within_1pct_of_file=(((pl.col("u") - pl.col("v")).abs() < 0.01).mean() * 100).round(4),
        )

    cross = pl.concat([stats(pairs, "true pairs"), stats(rnd, "random pairing")]).sort("src", "pairing")
    _save(cross, "01_row_order_S1_vs_match")

    # within a source: are the records of ONE entity stored next to each other?
    same = (pairs.select("s1_id", "src", "j").sort("s1_id", "src", "j")
            .with_columns(prev=pl.col("j").shift(1).over("s1_id", "src"))
            .drop_nulls("prev").with_columns(gap=pl.col("j") - pl.col("prev")))
    rnd_same = (pairs.select("s1_id", "src", j=pl.col("j").shuffle(seed=SEED + 1).over("src"))
                .sort("s1_id", "src", "j").with_columns(prev=pl.col("j").shift(1).over("s1_id", "src"))
                .drop_nulls("prev").with_columns(gap=pl.col("j") - pl.col("prev")))

    def gaps(df, kind):
        return df.group_by("src").agg(
            pairing=pl.lit(kind), consecutive_same_entity_pairs=pl.len(),
            median_row_gap=pl.col("gap").median(),
            pct_gap_le_1=((pl.col("gap") <= 1).mean() * 100).round(4),
            pct_gap_le_100=((pl.col("gap") <= 100).mean() * 100).round(4))

    within = pl.concat([gaps(same, "same entity"), gaps(rnd_same, "random")]).sort("src", "pairing")
    _save(within, "01_row_order_within_source")

    # S2 vs S3 records of the same entity
    s2s3 = (pairs.filter(pl.col("src") == "S2").select("s1_id", v2="v", id2="id_b")
            .join(pairs.filter(pl.col("src") == "S3").select("s1_id", v3="v", id3="id_b"), on="s1_id"))
    xs = pl.DataFrame({
        "comparison": ["S2 vs S3 records of the same entity"],
        "pairs": [s2s3.height],
        "spearman_row_position": [round(s2s3.select(pl.corr("v2", "v3", method="spearman")).item(), 5)],
        "spearman_numeric_id": [round(s2s3.select(pl.corr("id2", "id3", method="spearman")).item(), 5)],
    })
    _save(xs, "01_row_order_S2_vs_S3")
    return cross


# =========================================================================== step 2: address_region

def _with_region(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Stored column when the file has been rebuilt, otherwise the identical pipeline expression."""
    if "address_region" in lf.collect_schema().names():
        return lf
    return lf.with_columns(address_region=T.address_region("address_segments", "country_clean"))


def _sample_pairs(pct: int) -> pl.DataFrame:
    return _c(_gt_pairs().filter(pl.col("s1_id").hash(SEED) % 100 < pct))


def _pair_frame(pairs: pl.DataFrame, cols: list[str], add) -> pl.DataFrame:
    """Join S1-side (a_) and match-side (b_) columns onto a pair list (s1_id, m_id)."""
    def side(split_src, ids, prefix):
        lf = add(_proc("train", split_src)).select(["entity_id", *cols]).join(ids.lazy(), on="entity_id", how="semi")
        return _c(lf.select([pl.col(c).alias(f"{prefix}{c}") for c in ["entity_id", *cols]]))
    a = side("S1", pairs.select(entity_id="s1_id").unique(), "a_")
    b = pl.concat([side(s, pairs.select(entity_id="m_id"), "b_").with_columns(b_source=pl.lit(s)) for s in ("S2", "S3")])
    return pairs.join(a, left_on="s1_id", right_on="a_entity_id").join(b, left_on="m_id", right_on="b_entity_id")


def region_eval() -> dict:
    # --- coverage (label-free), every file
    cov = []
    for sp in C.SPLITS:
        for s in C.SOURCES:
            cov.append(_c(_with_region(_proc(sp, s)).group_by("country_clean").agg(
                split=pl.lit(sp), source=pl.lit(s), records=pl.len(),
                pct_found=(pl.col("address_region").is_not_null().mean() * 100).round(2),
                pct_found_nonmissing_address=(pl.col("address_region").is_not_null().filter(~pl.col("address_missing")).mean() * 100).round(2))))
    bars = {"US": 90.0, "India": 85.0}
    cov = (pl.concat(cov).rename({"country_clean": "country"})
           .with_columns(pass_bar=pl.col("country").replace_strict(bars, default=None, return_dtype=pl.Float64))
           .with_columns(passed=pl.when(pl.col("pass_bar").is_not_null()).then(pl.col("pct_found") >= pl.col("pass_bar")))
           .select("split", "source", "country", "records", "pct_found", "pct_found_nonmissing_address", "pass_bar", "passed")
           .sort("country", "split", "source"))
    _save(cov, "02_region_coverage")

    # --- agreement on true pairs (10 % S1 sample, evaluation only)
    cols = ["country_clean", "name_clean", "address_region"]
    tp = _pair_frame(_sample_pairs(10), cols, _with_region)
    both = tp.filter(pl.col("a_address_region").is_not_null() & pl.col("b_address_region").is_not_null())
    agree = (both.group_by(country="a_country_clean", match_source="b_source")
             .agg(pairs=pl.len(), pct_agree=((pl.col("a_address_region") == pl.col("b_address_region")).mean() * 100).round(3))
             .sort("country", "match_source"))
    overall = both.select(pl.lit("ALL").alias("country"), pl.lit("S2+S3").alias("match_source"), pairs=pl.len(),
                          pct_agree=((pl.col("a_address_region") == pl.col("b_address_region")).mean() * 100).round(3))
    agree = pl.concat([agree, overall]).with_columns(pass_bar=pl.lit(97.0), passed=pl.col("pct_agree") >= 97.0)
    agree = agree.with_columns(pct_pairs_both_found=pl.lit(round(both.height / tp.height * 100, 2)))
    _save(agree, "02_region_true_pair_agreement")
    fails = both.filter(pl.col("a_address_region") != pl.col("b_address_region"))
    _save(fails.group_by("a_country_clean", "a_address_region", "b_address_region").len().sort("len", descending=True).head(15),
          "02_region_true_pair_disagreements_top")

    # --- same-name decoys (EDA 7.2 set: 5 % hash sample of train S1, S2/S3 records with equal
    #     normalised name that are NOT a true match of that S1 entity)
    s1 = _c(_with_region(_proc("train", "S1")).filter(pl.col("entity_id").hash(SEED) % 10_000 < 500)
            .select(s1_id="entity_id", k="name_clean", a_country="country_clean", a_region="address_region"))
    oth = pl.concat([_with_region(_proc("train", s)).select(m_id="entity_id", k="name_clean", b_region="address_region") for s in ("S2", "S3")])
    cand = _c(oth.join(s1.lazy().select("k").unique(), on="k", how="semi")).join(s1, on="k")
    truth = _c(_gt_pairs().join(s1.lazy().select("s1_id"), on="s1_id", how="semi")).with_columns(is_true=pl.lit(True))
    cand = cand.join(truth, on=["s1_id", "m_id"], how="left").with_columns(is_true=pl.col("is_true").fill_null(False))
    cb = cand.filter(pl.col("a_region").is_not_null() & pl.col("b_region").is_not_null())
    dec = (cb.group_by(country="a_country", kind=pl.when(pl.col("is_true")).then(pl.lit("same-name TRUE pair")).otherwise(pl.lit("same-name DECOY")))
           .agg(pairs=pl.len(), pct_region_disagree=((pl.col("a_region") != pl.col("b_region")).mean() * 100).round(2))
           .sort("country", "kind"))
    _save(dec, "02_region_decoy_disagreement")

    # --- S1 identity: adding a column cannot merge S1 records; confirm keys over existing columns
    ident = []
    for sp in C.SPLITS:
        g = _c(_with_region(_proc(sp, "S1")).group_by("name_clean", "address_clean").len().filter(pl.col("len") > 1))
        g2 = _c(_with_region(_proc(sp, "S1")).group_by("name_clean", "address_clean", "address_region").len().filter(pl.col("len") > 1))
        ident.append({"split": sp, "S1 groups on name_clean+address_clean": g.height,
                      "S1 groups on name_clean+address_clean+address_region": g2.height,
                      "new S1 merges": max(0, g2.height - g.height)})
    _save(pl.DataFrame(ident), "02_region_s1_identity")
    return {"coverage": cov, "agree": agree, "decoy": dec}


# =========================================================================== main

def main(argv):
    what = argv[1] if len(argv) > 1 else "all"
    steps = {"rows": row_order_leakage, "region": region_eval}
    for k, fn in steps.items():
        if what in (k, "all"):
            fn()


if __name__ == "__main__":
    main(sys.argv)
